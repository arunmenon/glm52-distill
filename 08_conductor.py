#!/usr/bin/env python3
"""
08_conductor.py — the deterministic FSM spine of the autonomous distillation loop
(autoloop_design.md, CRAWL phase). One invocation drives the sweep to completion
by running trials sequentially on ONE student pod, checkpointing after every
trial so a kill -9 loses nothing (re-run resumes from the ledger).

Guardrails (code-enforced): HALT sentinel, live-balance reserve check before any
deploy, hard spend cap, per-trial cap (from preflight), base-anchor quality floor.
watchdog.sh runs ON the pod as the independent dead-man.

Usage:
  python 08_conductor.py --plan experiments.yaml            # run to completion
  touch runs/<sweep>/HALT                                    # cooperative stop
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
import providers

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]


def ledger_append(run_dir: Path, row: dict):
    with open(run_dir / "ledger.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def ledger_read(run_dir: Path) -> dict:
    """experiment_id -> last row (resume: completed trials skip)."""
    out = {}
    p = run_dir / "ledger.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[r["experiment_id"]] = r
    return out


def ssh(ip, port, cmd, timeout=None):
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(port), "-i", str(Path.home() / ".ssh/id_ed25519"),
         f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout)


def guard(run_dir: Path, plan, provider, spent):
    """Return None to proceed, or a halt-reason string. The MONEY guard applies
    only to billed providers (provider.balance() is not None); for BYO/on-prem
    GPUs the HALT + wall-clock guards carry safety instead."""
    if (run_dir / "HALT").exists():
        return "HALT sentinel present"
    cap = plan["budget"]["hard_cap_usd"]
    reserve = plan["budget"]["reserve_usd"]
    if spent >= cap:
        return f"hard spend cap ${cap} reached (spent ~${spent:.2f})"
    try:
        bal = provider.balance()
    except Exception as e:  # noqa: BLE001
        return f"cannot read balance ({e}) — refusing to spend blind"
    if bal is None:
        return None                       # unbilled provider: money guard N/A
    if bal - reserve <= 0:
        return f"balance ${bal:.2f} within reserve ${reserve}"
    return None


def run_on_pod(ip, port, cfg, run_dir):
    """Push the trial config, run run_experiment.sh, pull result.json."""
    tdir = f"trials/{cfg['experiment_id']}"
    cfgpath = run_dir / f"{cfg['experiment_id']}.json"
    cfgpath.write_text(json.dumps(cfg))
    subprocess.run(["scp", *SSH_OPTS, "-P", str(port),
                    "-i", str(Path.home() / ".ssh/id_ed25519"), str(cfgpath),
                    f"root@{ip}:/workspace/glm52-distill/trial_cfg.json"], check=True)
    r = ssh(ip, port,
            "cd /workspace/glm52-distill && bash run_experiment.sh trial_cfg.json",
            timeout=4 * 3600)
    print(r.stdout[-500:] if r.stdout else "", r.stderr[-300:] if r.returncode else "")
    got = ssh(ip, port, f"cat /workspace/glm52-distill/{tdir}/result.json 2>/dev/null || echo '{{}}'")
    try:
        return json.loads(got.stdout)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="experiments.yaml")
    ap.add_argument("--pod-id", default=None, help="reuse an existing student pod")
    args = ap.parse_args()

    plan = yaml.safe_load(open(args.plan))
    provider = providers.make_provider(plan.get("provider", {"kind": "runpod"}))
    run_dir = Path(f"runs/{plan['sweep']}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # PREFLIGHT (idempotent)
    trials_f = run_dir / "trials.json"
    if not trials_f.exists():
        subprocess.run([sys.executable, "preflight.py", "--plan", args.plan,
                        "--out", str(trials_f)], check=True)
    trials = json.loads(trials_f.read_text())["trials"]

    done = ledger_read(run_dir)
    spent = sum(r.get("cost", 0) for r in done.values())

    reason = guard(run_dir, plan, provider, spent)
    if reason:
        raise SystemExit(f"HALTED before deploy: {reason}")

    # "Don't start what you can't finish" (journal rule that saved ~$16):
    # refuse to deploy unless the live balance covers the WHOLE remaining sweep
    # plus the reserve — not merely clears the reserve.
    projected = json.loads(trials_f.read_text()).get("projected_rung0_cost", 0)
    projected += 3  # anchor
    projected += plan["rung"]["keep_top"] * 6  # promotions to full
    remaining = projected - spent
    bal = provider.balance()
    if bal is not None and bal - plan["budget"]["reserve_usd"] < remaining:
        raise SystemExit(
            f"HALTED before deploy: balance ${bal:.2f} - reserve "
            f"${plan['budget']['reserve_usd']} = ${bal - plan['budget']['reserve_usd']:.2f} "
            f"< projected remaining ${remaining:.2f}. Top up or shrink the grid "
            "(the conductor will not start a sweep it cannot finish).")

    # acquire a GPU host through the provider (deploy, resume, or BYO pool)
    try:
        host = provider.acquire({"name": f"{plan['sweep']}-student",
                                 "pod_id": args.pod_id})
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"acquire failed: {e}")
    pod_id, ip, port = host["id"], host["ip"], host["port"]
    print(f"[{provider.name}] host {pod_id} @ {ip}:{port} "
          f"(${host.get('cost_per_hr', 0)}/hr)")

    # bootstrap the pod: sync repo, pull artifacts, arm watchdog
    subprocess.run(["rsync", "-rlt", "-e",
                    f"ssh {' '.join(SSH_OPTS)} -p {port} -i {Path.home()}/.ssh/id_ed25519",
                    "--exclude", ".state*", "--exclude", "runs/", "--exclude", "*.log",
                    "./", f"root@{ip}:/workspace/glm52-distill/"], check=False)
    ssh(ip, port, "cd /workspace/glm52-distill && source /workspace/.pilot_env && "
                  "export PATH=/workspace/venv/bin:$PATH && "
                  "hf download $PILOT_STORE --repo-type dataset --local-dir . "
                  "--include 'corpus/*' --include 'data/*' --include 'gate_result.json' "
                  "--include 'contamination_report.json' 2>&1 | tail -1", timeout=1200)
    # on-pod dead-man: stop the pod after a wall-clock budget no matter what
    ssh(ip, port, f"cd /workspace/glm52-distill && RUNPOD_API_KEY=$(cat ~/.runpod/api_key 2>/dev/null) "
                  f"setsid bash watchdog.sh 300 {pod_id} < /dev/null > watchdog.log 2>&1 & echo armed", timeout=30)

    # ANCHOR (mandatory quality floor)
    if "ANCHOR" not in done:
        cfg = {"experiment_id": "ANCHOR", "mode": "anchor",
               "proxy_rows": plan["rung"]["proxy_rows"]}
        res = run_on_pod(ip, port, cfg, run_dir)
        m = res.get("metrics", {})
        g = m.get("gsm8k_flexible", 0) or 0
        ok = g >= plan["anchor"]["min_gsm8k_flexible"]
        ledger_append(run_dir, {"experiment_id": "ANCHOR", "state": "done",
                                "metrics": m, "cost": 3, "anchor_ok": ok})
        Path("anchors").mkdir(exist_ok=True)
        (run_dir / "anchor.json").write_text(json.dumps(m))
        # ship anchor to the pod for objective deltas
        subprocess.run(["scp", *SSH_OPTS, "-P", str(port),
                        "-i", str(Path.home() / ".ssh/id_ed25519"), str(run_dir / "anchor.json"),
                        f"root@{ip}:/workspace/glm52-distill/anchors/qwen.json"], check=False)
        if not ok:
            provider.release(host)
            raise SystemExit(
                f"ANCHOR FAILED: base GSM8K flexible {g} < "
                f"{plan['anchor']['min_gsm8k_flexible']} — methodology bug (see "
                "Discovery #6), NOT a green light. Pod stopped. PAUSED_HUMAN.")
        spent += 3
        done = ledger_read(run_dir)

    # SCHEDULE rung-0: run every trial not already in the ledger
    for cfg in trials:
        eid = cfg["experiment_id"]
        if eid in done:
            print(f"[skip] {eid} already in ledger")
            continue
        reason = guard(run_dir, plan, provider, spent)
        if reason:
            print(f"[stop] {reason}; finishing here.")
            break
        cfg = {**cfg, "mode": "train", "proxy_rows": plan["rung"]["proxy_rows"],
               "heldout": plan["corpus"]["heldout"]}
        res = run_on_pod(ip, port, cfg, run_dir)
        ledger_append(run_dir, {"experiment_id": eid, "state": "done", "rung": 0,
                                "config": {k: cfg[k] for k in ("alpha", "lr", "lora_rank", "epochs")},
                                "score": res.get("score", -1), "metrics": res.get("metrics", {}),
                                "cost": cfg.get("est_cost", 3)})
        spent += cfg.get("est_cost", 3)
        done = ledger_read(run_dir)

    # PROMOTE top-K rung-0 winners to the full corpus
    rung0 = [r for r in ledger_read(run_dir).values()
             if r.get("rung") == 0 and r.get("state") == "done"]
    rung0.sort(key=lambda r: r.get("score", -1), reverse=True)
    keep = plan["rung"]["keep_top"]
    for r in rung0[:keep]:
        eid = f"{r['experiment_id']}-full"
        if eid in ledger_read(run_dir):
            continue
        if guard(run_dir, plan, provider, spent):
            break
        cfg = {**r["config"], "experiment_id": eid, "rung": 1, "mode": "train",
               "heldout": plan["corpus"]["heldout"]}
        res = run_on_pod(ip, port, cfg, run_dir)
        ledger_append(run_dir, {"experiment_id": eid, "state": "done", "rung": 1,
                                "config": r["config"], "score": res.get("score", -1),
                                "metrics": res.get("metrics", {}), "cost": 6})
        spent += 6

    # FINALIZE
    final = ledger_read(run_dir)
    ranked = sorted([r for r in final.values() if "score" in r],
                    key=lambda r: r.get("score", -1), reverse=True)
    lb = ["# Leaderboard: " + plan["sweep"], "", "| rank | id | rung | score | config |",
          "|---|---|---|---|---|"]
    for i, r in enumerate(ranked[:15], 1):
        lb.append(f"| {i} | {r['experiment_id']} | {r.get('rung','-')} | "
                  f"{r.get('score')} | {r.get('config','-')} |")
    (run_dir / "leaderboard.md").write_text("\n".join(lb))
    print("\n".join(lb))
    print(f"\napprox spend this sweep: ${spent:.2f}")
    provider.release(host)
    print(f"[{provider.name}] host {pod_id} released. Sweep complete.")


if __name__ == "__main__":
    main()
