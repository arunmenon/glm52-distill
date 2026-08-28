#!/usr/bin/env python3
"""Freeze the gold-smoke queue from manually reviewed live survivors.

The queue is for base-fail/gold-pass validation only.  It is not a teacher
rollout list and grants no spend approval.  Runner-hard tasks are capped at
four; the remaining hard survivors stay recorded as holds rather than being
discarded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).parent
EXP_DIR = REPO_DIR / "runs" / "expansion" / "bugfix"
SURVIVOR_FILE = EXP_DIR / "candidate_survivors.json"
SURVIVOR_GATE_FILE = EXP_DIR / "candidate_survivors_decontam.json"
QUEUE_FILE = EXP_DIR / "smoke_queue.json"
QUEUE_GATE_FILE = EXP_DIR / "smoke_queue_decontam.json"
REPORT_FILE = EXP_DIR / "smoke_queue_report.json"
MARKDOWN_FILE = EXP_DIR / "smoke_queue.md"

MAX_RUNNER_HARD = 4

# These four preserve otherwise-lost Modin coverage and span data typing,
# rendering/build integration, filesystem/SCM, and distributed JSON parsing.
HARD_SELECTED: dict[str, str] = {
    "pandas-dev__pandas-50672": "Arrow dtype correctness; clear reproduction and broad predicate coverage.",
    "bokeh__bokeh-13608": "Rendering/math integration regression from a distinct application domain.",
    "iterative__dvc-1700": "SCM-boundary bug and one of the stronger diagnostic-hard survivors.",
    "modin-project__modin-5946": "Only surviving Modin bug; retains distributed-dataframe diversity.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def interleave(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tier = {
        tier: [row for row in rows if row["tier"] == tier]
        for tier in ("easy", "medium", "hard")
    }
    offsets = Counter()
    output: list[dict[str, Any]] = []
    # Medium leads; easy and hard are distributed through the prefix.  Once a
    # tier is exhausted, the remaining medium candidates continue in source
    # review order.
    pattern = ("medium", "easy", "medium", "hard")
    while len(output) < len(rows):
        progressed = False
        for tier in pattern:
            index = offsets[tier]
            if index < len(by_tier[tier]):
                output.append(by_tier[tier][index])
                offsets[tier] += 1
                progressed = True
        if not progressed:
            break
    return output


def main() -> None:
    for path in (SURVIVOR_FILE, SURVIVOR_GATE_FILE):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    survivors = json.loads(SURVIVOR_FILE.read_text())
    gate = json.loads(SURVIVOR_GATE_FILE.read_text())
    survivor_hash = sha256(SURVIVOR_FILE)
    if gate.get("verdict") != "CLEAN" or gate.get("task_list_sha256") != survivor_hash:
        raise SystemExit("survivor decontamination gate is absent, stale, or not CLEAN")
    if len(survivors) != 42 or any(
        row.get("conversion_disposition") != "bugfix_survivor" for row in survivors
    ):
        raise SystemExit("unexpected survivor input")

    by_id = {row["instance_id"]: row for row in survivors}
    if set(HARD_SELECTED) - set(by_id):
        raise SystemExit("configured hard selection is absent from survivors")
    if any(by_id[iid]["tier"] != "hard" for iid in HARD_SELECTED):
        raise SystemExit("configured hard selection contains a non-hard runner tier")

    selected = [
        row
        for row in survivors
        if row["tier"] != "hard" or row["instance_id"] in HARD_SELECTED
    ]
    hard_holds = [
        row
        for row in survivors
        if row["tier"] == "hard" and row["instance_id"] not in HARD_SELECTED
    ]
    if Counter(row["tier"] for row in selected)["hard"] != MAX_RUNNER_HARD:
        raise SystemExit("runner-hard cap was not enforced exactly")

    queue = []
    for position, original in enumerate(interleave(selected), start=1):
        row = dict(original)
        row["smoke_queue_position"] = position
        row["smoke_status"] = "pending"
        row["smoke_scope"] = "base_fail_then_gold_pass_twice"
        row["teacher_spend_approved"] = False
        queue.append(row)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))

    queue_gate: dict[str, Any] = {
        "path": str(QUEUE_GATE_FILE.relative_to(REPO_DIR)),
        "status": "NOT_RUN",
    }
    if QUEUE_GATE_FILE.exists():
        exact_gate = json.loads(QUEUE_GATE_FILE.read_text())
        current_hash = sha256(QUEUE_FILE)
        queue_gate.update({
            "status": (
                exact_gate.get("verdict", "UNKNOWN")
                if exact_gate.get("task_list_sha256") == current_hash
                else "STALE"
            ),
            "verdict": exact_gate.get("verdict"),
            "checks": exact_gate.get("checks", {}),
            "task_list_sha256": exact_gate.get("task_list_sha256"),
            "current_queue_sha256": current_hash,
        })

    report = {
        "schema_version": "expansion_gold_smoke_queue_v1",
        "approval_state": "GOLD_SMOKE_ONLY_NO_TEACHER_SPEND",
        "queue_count": len(queue),
        "source_survivors": len(survivors),
        "source_survivors_sha256": survivor_hash,
        "by_runner_tier": dict(sorted(Counter(row["tier"] for row in queue).items())),
        "by_heuristic_difficulty": dict(
            sorted(Counter(row["heuristic_difficulty"] for row in queue).items())
        ),
        "by_repo": dict(sorted(Counter(row["repo"] for row in queue).items())),
        "by_bug_family": dict(sorted(Counter(row["bug_family"] for row in queue).items())),
        "runner_hard_policy": {
            "cap": MAX_RUNNER_HARD,
            "selected": [
                {
                    "instance_id": iid,
                    "reason": reason,
                }
                for iid, reason in HARD_SELECTED.items()
            ],
            "held_for_later": [
                {
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "heuristic_difficulty": row["heuristic_difficulty"],
                    "reason": "Held by the four-task runner-hard cap; remains in candidate_survivors.json.",
                }
                for row in hard_holds
            ],
        },
        "decontamination": queue_gate,
        "queue": str(QUEUE_FILE.relative_to(REPO_DIR)),
        "queue_sha256": sha256(QUEUE_FILE),
        "markdown": str(MARKDOWN_FILE.relative_to(REPO_DIR)),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# Expansion gold-smoke queue",
        "",
        "> Base-fail/gold-pass validation only — no teacher rollout spend is approved.",
        "",
        (
            f"**{len(queue)} tasks:** {report['by_runner_tier'].get('easy', 0)} runner-easy / "
            f"{report['by_runner_tier'].get('medium', 0)} runner-medium / "
            f"{report['by_runner_tier'].get('hard', 0)} runner-hard. All are medium or hard "
            "under the diagnostic heuristic."
        ),
        "",
        "| # | Instance | Repo | Runner tier | Heuristic | Family | Smoke status |",
        "|---:|---|---|:---:|:---:|---|:---:|",
    ]
    for row in queue:
        cells = (
            row["smoke_queue_position"], row["instance_id"], row["repo"], row["tier"],
            row["heuristic_difficulty"], row["bug_family"], row["smoke_status"],
        )
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    lines.extend([
        "",
        "## Runner-hard holds",
        "",
        "These remain valid survivors; they are not silently dropped.",
        "",
        "| Instance | Repo | Heuristic | Reason |",
        "|---|---|:---:|---|",
    ])
    for row in hard_holds:
        cells = (
            row["instance_id"], row["repo"], row["heuristic_difficulty"],
            "Held by the four-task runner-hard cap.",
        )
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    MARKDOWN_FILE.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "validation": "PASS",
        "queue_count": len(queue),
        "by_runner_tier": report["by_runner_tier"],
        "by_heuristic_difficulty": report["by_heuristic_difficulty"],
        "runner_hard_held": len(hard_holds),
        "decontamination": queue_gate["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
