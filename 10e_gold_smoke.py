#!/usr/bin/env python3
"""Resumable base-fail/gold-pass smoke runner for the expansion queue.

This script never calls a teacher model.  For each task it uses two fresh
containers:

1. base commit + gold test patch: FAIL_TO_PASS must fail;
2. base commit + gold code patch + gold test patch: F2P plus capped P2P must
   pass twice.

Only a double green gold run is smoke-pass.  Per-task ledgers make the run
resumable and are bound to the exact frozen queue hash.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).parent
EXP_DIR = REPO_DIR / "runs" / "expansion" / "bugfix"
QUEUE_FILE = EXP_DIR / "smoke_queue.json"
GATE_FILE = EXP_DIR / "smoke_queue_decontam.json"
RUNS_DIR = EXP_DIR / "gold_smoke"
REPORT_FILE = EXP_DIR / "gold_smoke_report.json"

VERIFY_TIMEOUT_S = 1800
P2P_CAP = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_name(instance_id: str) -> str:
    return f"xingyaoww/sweb.eval.x86_64.{instance_id.replace('__', '_s_')}".lower()


def docker_pull(image: str) -> None:
    for attempt in range(1, 4):
        proc = subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image],
            capture_output=True,
            timeout=3600,
            text=True,
        )
        if proc.returncode == 0:
            return
        print(f"pull attempt {attempt}/3 failed: {proc.stderr[-500:]}")
        if attempt < 3:
            time.sleep(30 * attempt)
    raise RuntimeError(f"image pull failed three times: {image}")


def make_env(image: str):
    from minisweagent.environments.docker import DockerEnvironment

    return DockerEnvironment(
        image=image,
        cwd="/testbed",
        timeout=VERIFY_TIMEOUT_S,
        run_args=["--rm", "--platform=linux/amd64", "--network=none"],
        env={
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
            "BASH_ENV": "/root/.bashrc",
        },
        container_timeout="3h",
    )


def put_text(env: Any, text: str, destination: str) -> dict[str, Any]:
    encoded = base64.b64encode(text.encode()).decode()
    return env.execute({"command": f"echo {encoded} | base64 -d > {destination}"})


def test_command(test_ids: list[str]) -> str:
    nodes = " ".join(shlex.quote(test_id) for test_id in test_ids)
    return f"cd /testbed && python -m pytest -q --no-header {nodes}"


def run_base(instance: dict[str, Any], image: str) -> dict[str, Any]:
    result: dict[str, Any] = {"test_patch_apply_ok": False, "run": None}
    env = make_env(image)
    try:
        put_text(env, instance["test_patch"], "/tmp/gold_tests.patch")
        applied = env.execute({
            "command": (
                "cd /testbed && git apply --whitespace=nowarn "
                "/tmp/gold_tests.patch"
            )
        })
        result["test_patch_apply_ok"] = applied["returncode"] == 0
        if not result["test_patch_apply_ok"]:
            result["apply_tail"] = applied.get("output", "")[-1200:]
            return result
        out = env.execute({"command": test_command(list(instance["FAIL_TO_PASS"]))})
        result["run"] = {
            "returncode": out["returncode"],
            "tail": out.get("output", "")[-1200:],
        }
        result["fails_as_expected"] = out["returncode"] != 0
        return result
    finally:
        try:
            env.cleanup()
        except Exception:  # noqa: BLE001
            pass


def run_gold(instance: dict[str, Any], image: str) -> dict[str, Any]:
    result: dict[str, Any] = {"patches_apply_ok": False, "runs": []}
    env = make_env(image)
    try:
        put_text(env, instance["patch"], "/tmp/gold.patch")
        put_text(env, instance["test_patch"], "/tmp/gold_tests.patch")
        applied = env.execute({
            "command": (
                "cd /testbed && git apply --whitespace=nowarn /tmp/gold.patch "
                "&& git apply --whitespace=nowarn /tmp/gold_tests.patch"
            )
        })
        result["patches_apply_ok"] = applied["returncode"] == 0
        if not result["patches_apply_ok"]:
            result["apply_tail"] = applied.get("output", "")[-1200:]
            return result
        tests = list(instance["FAIL_TO_PASS"]) + list(instance["PASS_TO_PASS"])[:P2P_CAP]
        command = test_command(tests)
        for attempt in (1, 2):
            out = env.execute({"command": command})
            result["runs"].append({
                "attempt": attempt,
                "returncode": out["returncode"],
                "tail": out.get("output", "")[-1200:],
            })
        result["passes_twice"] = all(run["returncode"] == 0 for run in result["runs"])
        return result
    finally:
        try:
            env.cleanup()
        except Exception:  # noqa: BLE001
            pass


def load_queue() -> tuple[list[dict[str, Any]], str]:
    for path in (QUEUE_FILE, GATE_FILE):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    queue_hash = sha256(QUEUE_FILE)
    gate = json.loads(GATE_FILE.read_text())
    if gate.get("verdict") != "CLEAN" or gate.get("task_list_sha256") != queue_hash:
        raise SystemExit("smoke queue decontamination gate is absent, stale, or not CLEAN")
    queue = json.loads(QUEUE_FILE.read_text())
    if any(row.get("teacher_spend_approved") is not False for row in queue):
        raise SystemExit("queue is missing its no-teacher-spend marker")
    return queue, queue_hash


def run(args: argparse.Namespace) -> None:
    queue, queue_hash = load_queue()
    requested = set(args.only or [])
    if requested:
        unknown = requested - {row["instance_id"] for row in queue}
        if unknown:
            raise SystemExit(f"--only ids absent from queue: {sorted(unknown)}")
        queue = [row for row in queue if row["instance_id"] in requested]
    if args.limit is not None:
        queue = queue[: args.limit]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for instance in queue:
        iid = instance["instance_id"]
        ledger = RUNS_DIR / f"{iid}.json"
        if ledger.exists():
            existing = json.loads(ledger.read_text())
            if existing.get("queue_sha256") != queue_hash:
                raise SystemExit(f"stale smoke ledger for {iid}: queue hash changed")
            print(f"SKIP {iid} (ledger exists)")
            continue

        image = image_name(iid)
        print(f"\n=== {iid} [{instance['tier']}] ===")
        started = time.time()
        record: dict[str, Any] = {
            "instance_id": iid,
            "repo": instance["repo"],
            "tier": instance["tier"],
            "heuristic_difficulty": instance["heuristic_difficulty"],
            "queue_sha256": queue_hash,
            "image": image,
            "teacher_spend_usd": 0.0,
        }
        try:
            docker_pull(image)
            record["base"] = run_base(instance, image)
            record["gold"] = run_gold(instance, image)
            record["smoke_pass"] = bool(
                record["base"].get("fails_as_expected")
                and record["gold"].get("passes_twice")
            )
        except Exception as exc:  # noqa: BLE001
            record["smoke_pass"] = False
            record["driver_error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_s"] = round(time.time() - started, 2)
        tmp = ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        tmp.replace(ledger)
        print(f"SMOKE {iid}: {'PASS' if record['smoke_pass'] else 'FAIL'}")

        if not args.keep_images:
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
    report()


def report() -> None:
    queue, queue_hash = load_queue()
    by_id = {row["instance_id"]: row for row in queue}
    records = []
    stale = []
    for path in sorted(RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []:
        record = json.loads(path.read_text())
        if record.get("queue_sha256") != queue_hash:
            stale.append(record.get("instance_id", path.stem))
            continue
        if record.get("instance_id") in by_id:
            records.append(record)
    passed = [record for record in records if record.get("smoke_pass")]
    failed = [record for record in records if not record.get("smoke_pass")]
    done_ids = {record["instance_id"] for record in records}
    output = {
        "schema_version": "expansion_gold_smoke_report_v1",
        "queue": str(QUEUE_FILE.relative_to(REPO_DIR)),
        "queue_sha256": queue_hash,
        "teacher_spend_usd": 0.0,
        "queue_tasks": len(queue),
        "completed": len(records),
        "passed": len(passed),
        "failed": len(failed),
        "pending": len(queue) - len(records),
        "stale_ledgers": stale,
        "passed_ids": [record["instance_id"] for record in passed],
        "failed_ids": [record["instance_id"] for record in failed],
        "pending_ids": [row["instance_id"] for row in queue if row["instance_id"] not in done_ids],
        "by_runner_tier": {},
        "by_repo": {},
    }
    for key, field in (("by_runner_tier", "tier"), ("by_repo", "repo")):
        groups: dict[str, dict[str, int]] = {}
        for row in queue:
            group = groups.setdefault(row[field], {"queue": 0, "completed": 0, "passed": 0})
            group["queue"] += 1
        for record in records:
            group = groups[record[field]]
            group["completed"] += 1
            group["passed"] += int(bool(record.get("smoke_pass")))
        output[key] = dict(sorted(groups.items()))
    REPORT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps({
        key: output[key]
        for key in ("queue_tasks", "completed", "passed", "failed", "pending")
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--only", action="append")
    run_parser.add_argument("--keep-images", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        report()


if __name__ == "__main__":
    main()
