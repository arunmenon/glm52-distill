#!/usr/bin/env python3
"""WALK bugfix slice (trajectory_task_spec.md sections 4+6, staged launch).

The WALK mix is 50% bugfix / 20% mutation / 15% PR-mirror / 10%
test-writing (+~15% degraded-statement overlay). Only the bugfix
machinery exists today (09_generate_trajectories + 09_rehearsal); the
SWE-smith slices need their loaders + per-strategy spot audit and the
test-writing slice needs the inverted verifier. This wrapper launches the
bugfix slice FIRST so WALK generation starts the moment the rehearsal
verdict allows, while the other slices are built.

Policy (48 tasks = the 50% slice of a ~95-task WALK):
  - tier mix 40/40/20 -> 19 easy / 19 medium / 10 hard
  - per-repo cap 6 (12%), rehearsal AND crawl instance ids excluded
  - adaptive N + guards inherited from 09_generate_trajectories
  - decontam gate on the frozen list is a hard precondition (run select,
    then trajectory_decontam.py runs/walk/bugfix/walk_bugfix_tasks.json)

Usage (on the WALK VM):
  python3 09d_walk_bugfix.py select
  python3 09d_walk_bugfix.py run       # resumable; watchdog-supervised
  python3 09d_walk_bugfix.py report
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).parent
WALK_DIR = REPO_DIR / "runs" / "walk" / "bugfix"
SCRATCH = Path(__file__).parent / "walk_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "gen", REPO_DIR / "09_generate_trajectories.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
R = G.R

# retarget every phase constant BEFORE any command runs
R.PARQUET = SCRATCH / "swegym_lite.parquet"
G.FULL_PARQUET = SCRATCH / "swegym_full.parquet"
G.CRAWL_DIR = WALK_DIR
G.TASKS_FILE = WALK_DIR / "walk_bugfix_tasks.json"
G.REPORT_FILE = REPO_DIR / "walk_bugfix_report.json"
G.TIER_TARGETS = {"easy": 19, "medium": 19, "hard": 10}
G.PER_REPO_CAP = 6
G.PHASE_COST_CAP_USD = 75.0

LITE_URL = ("https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite/resolve/"
            "main/data/train-00000-of-00001.parquet")


def ensure_parquets():
    import urllib.request
    if not R.PARQUET.exists():
        print("downloading SWE-Gym-Lite parquet ...")
        urllib.request.urlretrieve(LITE_URL, R.PARQUET)


def excluded_ids() -> set:
    ids = set()
    for rel in ("runs/rehearsal/rehearsal_tasks.json",
                "runs/crawl/crawl_tasks.json"):
        p = REPO_DIR / rel
        if p.exists():
            ids |= {t["instance_id"] for t in json.loads(p.read_text())}
    return ids


def select_tasks():
    ensure_parquets()
    rng = random.Random(G.SEED)
    df = pd.read_parquet(R.PARQUET)
    df["tier"] = df.apply(R.tier_of, axis=1)
    df = df[~df.instance_id.isin(excluded_ids())]
    # top up thin tiers from SWE-Gym full (Lite is small after exclusions)
    full = G.full_pool()
    full = full[~full.instance_id.isin(excluded_ids())]
    print("lite pool:", df.tier.value_counts().to_dict())

    chosen, repo_counts = [], {}
    for tier, want in G.TIER_TARGETS.items():
        pool = df[df.tier == tier].to_dict("records")
        rng.shuffle(pool)
        pool += full[full.tier == tier].to_dict("records")  # lite first
        got = 0
        for row in pool:
            if got == want:
                break
            if repo_counts.get(row["repo"], 0) >= G.PER_REPO_CAP:
                continue
            if any(c["instance_id"] == row["instance_id"] for c in chosen):
                continue
            if not R.image_exists_on_hub(row["instance_id"]):
                print(f"  preflight MISS: {row['instance_id']}")
                continue
            keep = {k: row[k] for k in
                    ("instance_id", "repo", "problem_statement",
                     "base_commit", "patch", "test_patch", "version")
                    if k in row}
            keep["tier"] = tier
            chosen.append(keep)
            repo_counts[row["repo"]] = repo_counts.get(row["repo"], 0) + 1
            got += 1
        if got < want:
            print(f"WARNING: tier {tier} short {got}/{want}")

    WALK_DIR.mkdir(parents=True, exist_ok=True)
    G.TASKS_FILE.write_text(json.dumps(chosen, indent=1))
    print(f"froze {len(chosen)} tasks -> {G.TASKS_FILE}")
    print("repo mix:", repo_counts)
    print("NEXT: python3 trajectory_decontam.py", G.TASKS_FILE)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "select":
        select_tasks()
    elif cmd == "run":
        G.run()
    else:
        G.report()
