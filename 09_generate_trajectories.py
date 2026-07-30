#!/usr/bin/env python3
"""CRAWL trajectory generator (agentic_trajectory_design.md section 10,
trajectory_task_spec.md sections 3+6).

Reuses the rehearsal's proven rollout/verify machinery (09_rehearsal.py,
loaded as a module); adds the CRAWL-specific policy:

  - 50 SWE-Gym-Lite tasks: tier mix 20 easy / 20 medium / 10 hard,
    per-repo cap 6 (12%), rehearsal tasks EXCLUDED (one phase per task),
    Docker Hub image preflight, seed 42, frozen task list.
  - Decontam gate (trajectory_decontam.py) must pass on the frozen list
    BEFORE any rollout (hard precondition).
  - Adaptive N by tier: easy N<=2 stop at 1st verified; medium N<=4 stop
    at 2nd; hard N<=6 never early-stops.
  - Tier-interleaved task order so telemetry accumulates evenly.
  - Guards: phase cost cap $150 (halt); hard-tier pause-not-backfill
    (verified rate <10% after >=30 hard rollouts -> PAUSED_HUMAN halt).
  - Per-task ledger resume; image GC after each task; funnel by
    (tier, repo) -> crawl_report.json.

Usage:
  python3 09_generate_trajectories.py select   # freeze + gate the 50 tasks
  python3 09_generate_trajectories.py run      # resumable generation
  python3 09_generate_trajectories.py report
"""

import importlib.util
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).parent
CRAWL_DIR = REPO_DIR / "runs" / "crawl"
TASKS_FILE = CRAWL_DIR / "crawl_tasks.json"
REPORT_FILE = REPO_DIR / "crawl_report.json"

SEED = 42
TIER_TARGETS = {"easy": 20, "medium": 20, "hard": 10}
PER_REPO_CAP = 6
ADAPTIVE_N = {"easy": 2, "medium": 4, "hard": 6}
EARLY_STOP_AT = {"easy": 1, "medium": 2, "hard": 99}
PHASE_COST_CAP_USD = 150.0
HARD_PAUSE_MIN_ROLLOUTS = 30
HARD_PAUSE_RATE = 0.10

_spec = importlib.util.spec_from_file_location(
    "rehearsal", REPO_DIR / "09_rehearsal.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)   # shared: rollout, verify, image_name, tiers...


FULL_PARQUET = Path("/private/tmp/claude-501/-Users-arunmenon-projects-"
                    "glm52-distill/ad896283-5c39-4a76-bdab-80daf2f812f4/"
                    "scratchpad/swegym_full.parquet")
_full_cache = []


def full_pool():
    if len(_full_cache):
        return _full_cache[0]
    if not FULL_PARQUET.exists():
        print("downloading SWE-Gym full (44MB) ...")
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym/resolve/main/"
            "data/train-00000-of-00001.parquet", FULL_PARQUET)
    df = pd.read_parquet(FULL_PARQUET)
    lite_ids = set(pd.read_parquet(R.PARQUET).instance_id)
    df = df[~df.instance_id.isin(lite_ids)]
    df["tier"] = df.apply(R.tier_of, axis=1)
    print("full-set top-up pool (Lite excluded):",
          df.tier.value_counts().to_dict())
    _full_cache.append(df)
    return df


def select_tasks():
    rng = random.Random(SEED)
    df = pd.read_parquet(R.PARQUET)
    df["tier"] = df.apply(R.tier_of, axis=1)
    rehearsal_ids = {t["instance_id"] for t in
                     json.loads((REPO_DIR / "runs" / "rehearsal" /
                                 "rehearsal_tasks.json").read_text())}
    df = df[~df.instance_id.isin(rehearsal_ids)]
    print("pool after rehearsal exclusion:",
          df.tier.value_counts().to_dict())

    chosen, repo_counts = [], {}
    for tier, want in TIER_TARGETS.items():
        pool = df[df.tier == tier].to_dict("records")
        rng.shuffle(pool)
        pool.sort(key=lambda r: repo_counts.get(r["repo"], 0))
        got = 0
        for row in pool:
            if got == want:
                break
            if repo_counts.get(row["repo"], 0) >= PER_REPO_CAP:
                continue
            if not R.image_exists_on_hub(row["instance_id"]):
                print(f"  preflight MISS: {row['instance_id']}")
                continue
            chosen.append(row)
            repo_counts[row["repo"]] = repo_counts.get(row["repo"], 0) + 1
            got += 1
        if got < want:
            print(f"Lite pool short for {tier}: {got}/{want} - "
                  f"topping up from SWE-Gym FULL (depth quotas are "
                  f"load-bearing; Lite is a strict subset, so exclusion "
                  f"stays clean)")
            full = full_pool()
            pool = full[(full.tier == tier) &
                        (~full.instance_id.isin(
                            {c["instance_id"] for c in chosen}))
                        ].to_dict("records")
            rng.shuffle(pool)
            pool.sort(key=lambda r: repo_counts.get(r["repo"], 0))
            for row in pool:
                if got == want:
                    break
                if repo_counts.get(row["repo"], 0) >= PER_REPO_CAP:
                    continue
                if not R.image_exists_on_hub(row["instance_id"]):
                    continue
                row["source"] = "swegym_full"
                chosen.append(row)
                repo_counts[row["repo"]] = repo_counts.get(row["repo"],
                                                           0) + 1
                got += 1
            if got < want:
                print(f"SHORTFALL tier {tier}: {got}/{want} even with "
                      f"full-set top-up (spec sec 7: warn, not fail)")

    # tier-interleave: e,m,h,e,m,h,... so guards see all tiers early
    by_tier = {t: [c for c in chosen if c["tier"] == t] for t in TIER_TARGETS}
    interleaved, idx = [], {t: 0 for t in TIER_TARGETS}
    while any(idx[t] < len(by_tier[t]) for t in TIER_TARGETS):
        for t in TIER_TARGETS:
            if idx[t] < len(by_tier[t]):
                interleaved.append(by_tier[t][idx[t]])
                idx[t] += 1

    CRAWL_DIR.mkdir(parents=True, exist_ok=True)
    serializable = [{k: (R.as_list(v) if k in ("FAIL_TO_PASS",
                                               "PASS_TO_PASS") else v)
                     for k, v in row.items()} for row in interleaved]
    TASKS_FILE.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"frozen {len(interleaved)} tasks "
          f"({ {t: len(v) for t, v in by_tier.items()} }, "
          f"{len(repo_counts)} repos) -> {TASKS_FILE}")

    gate = subprocess.run(
        [sys.executable, str(REPO_DIR / "trajectory_decontam.py"),
         str(TASKS_FILE)], capture_output=True, text=True)
    print(gate.stdout[-400:])
    if gate.returncode != 0:
        sys.exit("DECONTAM GATE FAILED - do not run")
    print("decontam gate: CLEAN")


def spent_so_far() -> float:
    total = 0.0
    for ledger in CRAWL_DIR.glob("*/result.json"):
        for r in json.loads(ledger.read_text()).get("rollouts", []):
            total += r.get("cost_usd", 0.0)
    return total


def hard_stats():
    rollouts = verified = 0
    for ledger in CRAWL_DIR.glob("*/result.json"):
        d = json.loads(ledger.read_text())
        if d["tier"] == "hard":
            for r in d.get("rollouts", []):
                rollouts += 1
                verified += int(r.get("verify", {}).get("verified", False))
    return rollouts, verified


def run():
    R.load_env_key()
    if not (REPO_DIR / "trajectory_gate.json").exists() or \
            json.loads((REPO_DIR / "trajectory_gate.json").read_text()
                       ).get("verdict") != "CLEAN":
        sys.exit("no CLEAN decontam gate on record - run select first")
    tasks = json.loads(TASKS_FILE.read_text())
    for instance in tasks:
        iid, tier = instance["instance_id"], instance["tier"]
        out_dir = CRAWL_DIR / iid
        out_dir.mkdir(parents=True, exist_ok=True)
        ledger = out_dir / "result.json"
        if ledger.exists():
            continue

        cost = spent_so_far()
        if cost > PHASE_COST_CAP_USD:
            sys.exit(f"PHASE COST CAP: ${cost:.2f} > ${PHASE_COST_CAP_USD}")
        h_roll, h_ver = hard_stats()
        if (h_roll >= HARD_PAUSE_MIN_ROLLOUTS
                and h_ver / h_roll < HARD_PAUSE_RATE):
            sys.exit(f"PAUSED_HUMAN: hard-tier verified {h_ver}/{h_roll} "
                     f"< {HARD_PAUSE_RATE:.0%} - mix decision needed "
                     f"(trajectory_task_spec.md sec 3)")

        print(f"\n=== {iid} [{tier}] {instance['repo']} "
              f"(spent ${cost:.2f}) ===")
        try:
            R.docker_pull(R.image_name(iid))
        except Exception as exc:  # noqa: BLE001
            print(f"TASK {iid}: SKIPPED (pull failed: {exc})")
            ledger.write_text(json.dumps({
                "instance_id": iid, "tier": tier, "repo": instance["repo"],
                "rollouts": [], "task_verified": False,
                "error": f"pull failed: {exc}"}))
            continue

        n_max, stop_at = ADAPTIVE_N[tier], EARLY_STOP_AT[tier]
        task_result = {"instance_id": iid, "tier": tier,
                       "repo": instance["repo"], "rollouts": []}
        n_verified = 0
        for idx in range(1, n_max + 1):
            print(f"rollout {idx}/{n_max} ...")
            rec = R.rollout(instance, idx, out_dir)
            if rec["submitted"] and rec["patch_chars"] > 0:
                rec["verify"] = R.verify(instance, rec.pop("submission"))
            else:
                rec.pop("submission", None)
                rec["verify"] = {"verified": False, "reason": "no submission"}
            ok = rec["verify"].get("verified", False)
            n_verified += int(ok)
            print(f"  r{idx}: exit={rec['exit_status']} "
                  f"steps={rec['n_steps']} ${rec['cost_usd']} verified={ok}")
            task_result["rollouts"].append(rec)
            if n_verified >= stop_at:
                print(f"  early stop ({n_verified} verified)")
                break
        task_result["task_verified"] = n_verified > 0
        ledger.write_text(json.dumps(task_result, indent=2))
        print(f"TASK {iid}: verified={task_result['task_verified']} "
              f"({n_verified} rollouts)")
        subprocess.run(["docker", "rmi", R.image_name(iid)],
                       capture_output=True)
    report()


def report():
    rows = [json.loads(p.read_text())
            for p in sorted(CRAWL_DIR.glob("*/result.json"))]
    funnel = {}
    for row in rows:
        key = row["tier"]
        f = funnel.setdefault(key, {"tasks": 0, "tasks_verified": 0,
                                    "rollouts": 0, "rollouts_verified": 0,
                                    "cost_usd": 0.0, "pull_failed": 0,
                                    "repos": {}})
        f["tasks"] += 1
        f["tasks_verified"] += int(row.get("task_verified", False))
        f["pull_failed"] += int("error" in row)
        rep = f["repos"].setdefault(row["repo"], [0, 0])
        rep[0] += 1
        rep[1] += int(row.get("task_verified", False))
        for r in row.get("rollouts", []):
            f["rollouts"] += 1
            f["rollouts_verified"] += int(
                r.get("verify", {}).get("verified", False))
            f["cost_usd"] += r.get("cost_usd", 0.0)
    for f in funnel.values():
        f["cost_usd"] = round(f["cost_usd"], 2)
    total_verified = sum(f["rollouts_verified"] for f in funnel.values())
    total_cost = round(sum(f["cost_usd"] for f in funnel.values()), 2)
    hard = funnel.get("hard", {})
    out = {"date": time.strftime("%Y-%m-%d"), "phase": "CRAWL",
           "model": R.MODEL_NAME, "provider_pin": R.PROVIDER_PIN,
           "tasks_done": len(rows), "funnel": funnel,
           "verified_total": total_verified, "cost_total_usd": total_cost,
           "cost_per_verified_usd": round(total_cost /
                                          max(1, total_verified), 3),
           "hard_share_of_verified": round(
               hard.get("rollouts_verified", 0) / max(1, total_verified), 3),
           "quota_note": ">=25% hard target (trajectory_task_spec sec 3)"}
    REPORT_FILE.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"select": select_tasks, "run": run, "report": report}[
        sys.argv[1] if len(sys.argv) > 1 else "run"]()
