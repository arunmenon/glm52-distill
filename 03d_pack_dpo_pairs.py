#!/usr/bin/env python3
"""Pack verified-vs-failed rollout pairs into DPO preference data.

For every task with at least one VERIFIED and one FAILED rollout, emit one
row per (verified, failed) combination: the verified trajectory is `chosen`,
the failed one `rejected`. Both sides share the identical task prompt, so
the pair isolates solution quality — the classic DPO setup, at zero
marginal generation cost (task #3; failures were already paid for).

Failed rollouts that never submitted a patch (LimitsExceeded with no
submission, driver errors) are still legitimate rejections: the preference
being taught is "solve and verify" over "wander and fail".

Input : runs/rehearsal/ + runs/walk_regen/bugfix/ (ledgers + trajectories)
Output: packed/dpo/dpo_pairs_v0.parquet + dpo_pack_report.json
Message cleaning + format contract shared with 03b (fc_bash_v1).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "pk", REPO_DIR / "03b_pack_trajectories.py")
PK = importlib.util.module_from_spec(_spec)
sys.modules["pk"] = PK
_spec.loader.exec_module(PK)

OUT_DIR = REPO_DIR / "packed" / "dpo"
REPORT = REPO_DIR / "dpo_pack_report.json"
MAX_PAIRS_PER_TASK = 4   # cap combinatorics on multi-rollout tasks


def load_rollouts(result_file: Path):
    d = json.loads(result_file.read_text())
    good, bad = [], []
    for i, r in enumerate(d["rollouts"]):
        tf = result_file.parent / f"rollout{i + 1}.traj.json"
        if not tf.exists():
            continue
        traj = json.loads(tf.read_text())
        try:
            msgs = PK.clean_messages(traj["messages"])
        except ValueError:
            continue
        rec = (i + 1, msgs, r)
        (good if r.get("verify", {}).get("verified") else bad).append(rec)
    return d, good, bad


def main():
    rows = []
    report = {"tasks_seen": 0, "tasks_paired": 0, "pairs": 0}
    for result_file in sorted(
            f for dd in PK.RUNS_DIRS for f in dd.glob("*/result.json")):
        d, good, bad = load_rollouts(result_file)
        report["tasks_seen"] += 1
        if not good or not bad:
            continue
        report["tasks_paired"] += 1
        n = 0
        for gi, gmsgs, grec in good:
            for bi, bmsgs, brec in bad:
                if n >= MAX_PAIRS_PER_TASK:
                    break
                rows.append({
                    "instance_id": d["instance_id"],
                    "repo": d["repo"], "tier": d["tier"],
                    "source": result_file.parent.parent.name,
                    "format": PK.FORMAT_ID,
                    "chosen_rollout": gi, "rejected_rollout": bi,
                    "chosen_history_assisted":
                        PK.is_history_assisted(gmsgs),
                    "rejected_exit": brec.get("exit_status"),
                    "chosen_json": json.dumps(gmsgs, ensure_ascii=False),
                    "rejected_json": json.dumps(bmsgs, ensure_ascii=False),
                })
                n += 1
        report["pairs"] += n

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "dpo_pairs_v0.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    report["output"] = str(out.relative_to(REPO_DIR))
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
