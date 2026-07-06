#!/usr/bin/env python3
"""
runpod.py — thin RunPod GraphQL client for the autonomous loop (generalizes the
snippets in deploy_netvol.sh / watchdog.sh). Read-only helpers + deploy/stop.

Auth: RUNPOD_API_KEY env, or ~/.runpod/api_key.
Everything returns plain dicts; no exceptions swallowed silently — the conductor
decides what a failure means.
"""
import json
import os
import subprocess
import time
from pathlib import Path

API = "https://api.runpod.io/graphql"


def _key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        p = Path.home() / ".runpod" / "api_key"
        if p.exists():
            k = p.read_text().strip()
    if not k:
        raise SystemExit("no RUNPOD_API_KEY and no ~/.runpod/api_key")
    return k


def gql(query: str) -> dict:
    out = subprocess.run(
        ["curl", "-s", f"{API}?api_key={_key()}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"query": query})],
        capture_output=True, text=True, timeout=60)
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"errors": [{"message": out.stdout[:200]}]}


def balance() -> float:
    d = gql("query { myself { clientBalance } }")
    return float(d["data"]["myself"]["clientBalance"])


def deploy_student(name, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                   count=4, dc="US-NC-2", vol_gb=700,
                   image="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"):
    q = ('mutation { podFindAndDeployOnDemand(input:{cloudType: SECURE, '
         f'gpuCount: {count}, gpuTypeId: "{gpu}", dataCenterId: "{dc}", '
         'volumeMountPath: "/workspace", containerDiskInGb: 80, '
         f'volumeInGb: {vol_gb}, name: "{name}", imageName: "{image}", '
         'ports: "22/tcp", startSsh: true}) { id costPerHr } }')
    d = gql(q)
    pod = (d.get("data") or {}).get("podFindAndDeployOnDemand")
    if not pod:
        return {"error": (d.get("errors") or [{}])[0].get("message", "deploy failed")}
    return pod


def pod_ssh(pod_id, tries=40, pause=15):
    """Wait for the public SSH port; returns (ip, port) or None."""
    q = (f'query {{ pod(input:{{podId:"{pod_id}"}}) {{ runtime {{ ports '
         '{ ip isIpPublic privatePort publicPort } } } }')
    for _ in range(tries):
        d = gql(q)
        rt = ((d.get("data") or {}).get("pod") or {}).get("runtime")
        for p in (rt or {}).get("ports", []) if rt else []:
            if p["privatePort"] == 22 and p["isIpPublic"]:
                return p["ip"], p["publicPort"]
        time.sleep(pause)
    return None


def stop_pod(pod_id):
    return gql(f'mutation {{ podStop(input:{{podId:"{pod_id}"}}) {{ desiredStatus }} }}')


def resume_pod(pod_id, count=4):
    return gql(f'mutation {{ podResume(input:{{podId:"{pod_id}", gpuCount:{count}}}) '
               '{ desiredStatus } }')


def pod_status(pod_id):
    d = gql(f'query {{ pod(input:{{podId:"{pod_id}"}}) {{ desiredStatus }} }}')
    return ((d.get("data") or {}).get("pod") or {}).get("desiredStatus")


if __name__ == "__main__":
    import sys
    print("balance:", balance()) if len(sys.argv) == 1 else print(gql(sys.argv[1]))
