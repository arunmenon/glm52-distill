#!/usr/bin/env python3
"""
providers.py — GPU-provider abstraction for the autonomous loop.

The conductor depends ONLY on this interface, never on any cloud API. A provider
turns "I need GPUs" into an SSH-reachable Host, and "I'm done" into release. This
is what makes the loop provider-AGNOSTIC: RunPod, GCP, a Slurm cluster, or a
literal list of machines you already own all implement the same three methods.

  Host = {"id", "ip", "port", "cost_per_hr" (0 if unbilled)}

Providers:
  static  — BYO GPUs: you give a list of ssh endpoints (ANY cloud / on-prem /
            bare metal). No deploy, no billing. balance()=None → the money
            guard is skipped; only HALT + wall-clock guards apply. THIS is the
            "just point it at GPUs" mode.
  runpod  — deploy/stop/balance via RunPod GraphQL (wraps runpod.py).
  gcp     — stub: acquire = gcloud instance create; release = stop/delete.
"""
import json
import os
import time
from pathlib import Path


class Provider:
    name = "base"

    def acquire(self, spec: dict) -> dict:
        """Return a Host dict {id, ip, port, cost_per_hr}, or raise."""
        raise NotImplementedError

    def release(self, host: dict):
        raise NotImplementedError

    def balance(self):
        """Remaining spendable balance, or None if unbilled (BYO/on-prem)."""
        return None


class StaticProvider(Provider):
    """A fixed pool of SSH-reachable GPU hosts you already have. Provider-agnostic
    by definition — it doesn't know or care who owns the GPU."""
    name = "static"

    def __init__(self, hosts: list):
        # hosts: [{"ip":..., "port":22, "id":"box1"}...] from config/env
        self._pool = [dict(h) for h in hosts]
        self._busy = set()

    def acquire(self, spec: dict) -> dict:
        for h in self._pool:
            if h["id"] not in self._busy:
                self._busy.add(h["id"])
                return {**h, "cost_per_hr": h.get("cost_per_hr", 0)}
        raise RuntimeError("no free GPU host in the static pool")

    def release(self, host: dict):
        self._busy.discard(host["id"])   # returned to pool, not destroyed

    def balance(self):
        return None                       # unbilled → money guard skipped


class RunPodProvider(Provider):
    name = "runpod"

    def __init__(self, gpu=None, count=4, dc="US-NC-2"):
        import runpod
        self.rp = runpod
        self.gpu = gpu or "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        self.count = count
        self.dc = dc

    def acquire(self, spec: dict) -> dict:
        # reuse a stopped pod if the caller pinned one (cheapest)
        pid = spec.get("pod_id")
        if pid:
            self.rp.resume_pod(pid, self.count)
        else:
            pod = self.rp.deploy_student(
                spec.get("name", "autoloop-student"),
                gpu=spec.get("gpu", self.gpu), count=spec.get("count", self.count),
                dc=spec.get("dc", self.dc))
            if "error" in pod:
                raise RuntimeError(f"runpod deploy failed: {pod['error']}")
            pid = pod["id"]
        addr = self.rp.pod_ssh(pid)
        if not addr:
            raise RuntimeError(f"runpod pod {pid} never exposed SSH")
        ip, port = addr
        return {"id": pid, "ip": ip, "port": port,
                "cost_per_hr": spec.get("cost_per_hr", 8.36)}

    def release(self, host: dict):
        self.rp.stop_pod(host["id"])      # stop (keep volume), not terminate

    def balance(self):
        return self.rp.balance()


class GCPProvider(Provider):
    """Stub for the GCP fleet (RUN phase). acquire = gcloud instance start;
    release = gcloud instance stop. balance()=None (billed out-of-band, so use
    a wall-clock/quota guard instead of a live balance guard)."""
    name = "gcp"

    def __init__(self, **cfg):
        self.cfg = cfg

    def acquire(self, spec: dict) -> dict:
        raise NotImplementedError("GCP backend: implement gcloud compute instances "
                                  "start + describe for the IP; see deploy_netvol.sh "
                                  "for the RunPod shape to mirror.")

    def release(self, host: dict):
        raise NotImplementedError

    def balance(self):
        return None


def make_provider(cfg: dict) -> Provider:
    """cfg = {"kind": "static|runpod|gcp", ...}. For static, hosts may come from
    cfg["hosts"] or the AUTOLOOP_HOSTS env (json list)."""
    kind = cfg.get("kind", "runpod")
    if kind == "static":
        hosts = cfg.get("hosts")
        if not hosts and os.environ.get("AUTOLOOP_HOSTS"):
            hosts = json.loads(os.environ["AUTOLOOP_HOSTS"])
        if not hosts:
            raise SystemExit("static provider needs 'hosts' in the plan or "
                             "AUTOLOOP_HOSTS env (json list of {ip,port,id})")
        return StaticProvider(hosts)
    if kind == "runpod":
        return RunPodProvider(gpu=cfg.get("gpu"), count=cfg.get("count", 4),
                              dc=cfg.get("dc", "US-NC-2"))
    if kind == "gcp":
        return GCPProvider(**cfg)
    raise SystemExit(f"unknown provider kind: {kind}")
