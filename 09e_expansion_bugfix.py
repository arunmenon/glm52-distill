#!/usr/bin/env python3
"""Expansion bugfix slice: ~24 fresh SWE-Gym tasks -> ~10-15 more verified
trajectories (Plan C expansion arm, corpus 45 -> ~60).

Differences from the WALK slice (09d):
  - excludes EVERY instance id ever attempted (all runs/*/result.json plus
    every frozen task list), not just rehearsal+crawl — expansion must be
    100% unseen tasks
  - tier mix skewed to yield: 10 easy / 12 medium / 2 hard. Hard stays
    token-level (2) because the pause-not-backfill guard tripped at 2.8%
    verified; medium leads because it dominated the regen funnel.
  - cost cap $50 (the Plan C expansion budget)
  - runs with the container git-history strip active (09_rehearsal
    STRIP_CMD), so verify-rate forecasts from prior runs are upper bounds:
    82% of pilot successes used history retrieval.

Usage:
  python3 09e_expansion_bugfix.py select    # freeze the task list (local ok)
  python3 trajectory_decontam.py runs/expansion/bugfix/expansion_tasks.json
  python3 09e_expansion_bugfix.py run       # on the env VM
  python3 09e_expansion_bugfix.py report
"""

import glob
import importlib.util
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
EXP_DIR = REPO_DIR / "runs" / "expansion" / "bugfix"
SCRATCH = REPO_DIR / "walk_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "gen", REPO_DIR / "09d_walk_bugfix.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
G = D.G

# retarget the phase constants 09d set for WALK
D.WALK_DIR = EXP_DIR   # select_tasks() mkdirs this module global
G.CRAWL_DIR = EXP_DIR
G.TASKS_FILE = EXP_DIR / "expansion_tasks.json"
G.REPORT_FILE = REPO_DIR / "expansion_report.json"
G.TIER_TARGETS = {"easy": 10, "medium": 12, "hard": 2}
G.PHASE_COST_CAP_USD = 50.0
G.FULL_PARQUET = SCRATCH / "swegym_full.parquet"


def attempted_ids() -> set:
    """Every instance id we have EVER rolled out or frozen into a task
    list, across crawl/rehearsal/walk/regen. Expansion repeats none."""
    ids = set()
    for rf in glob.glob(str(REPO_DIR / "runs" / "**" / "result.json"),
                        recursive=True):
        ids.add(json.loads(Path(rf).read_text())["instance_id"])
    for tf in glob.glob(str(REPO_DIR / "runs" / "**" / "*tasks*.json"),
                        recursive=True):
        try:
            data = json.loads(Path(tf).read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            ids |= {t["instance_id"] for t in data
                    if isinstance(t, dict) and "instance_id" in t}
    return ids


D.excluded_ids = attempted_ids

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "select":
        print(f"excluding {len(attempted_ids())} previously seen ids")
        D.select_tasks()
    elif cmd == "run":
        G.run()
    else:
        G.report()
