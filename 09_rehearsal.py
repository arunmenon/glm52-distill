#!/usr/bin/env python3
"""Trajectory-leg rehearsal (trajectory_task_spec.md section 6, REHEARSAL).

10 SWE-Gym-Lite tasks stratified by difficulty tier (gold-patch stats) across
>=6 repos, N=2 rollouts each: the teacher (MODEL_NAME) via pinned OpenRouter
providers driving
mini-swe-agent (native bash-tool format) against local Docker (x86 emulation
on Apple silicon). Environment verification per agentic_trajectory_design.md
section 4: anti-tamper (reject test-file diffs), gold test re-apply, double
test run. Resumable: per-task result JSON is the ledger.

Outputs:
  runs/rehearsal/<instance_id>/rollout<N>.traj.json   full message trajectories
  runs/rehearsal/<instance_id>/result.json            per-task ledger entry
  rehearsal_report.json                               aggregate, by tier

Usage:
  venv/bin/python 09_rehearsal.py select   # pick + preflight the 10 tasks
  venv/bin/python 09_rehearsal.py run      # rollouts + verification (resumable)
  venv/bin/python 09_rehearsal.py report   # aggregate report only
"""

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

REPO_DIR = Path(__file__).parent
RUNS_DIR = REPO_DIR / "runs" / "rehearsal"
SCRATCH = Path("/private/tmp/claude-501/-Users-arunmenon-projects-glm52-distill/"
               "ad896283-5c39-4a76-bdab-80daf2f812f4/scratchpad")
PARQUET = SCRATCH / "swegym_lite.parquet"
TASKS_FILE = RUNS_DIR / "rehearsal_tasks.json"
REPORT_FILE = REPO_DIR / "rehearsal_report.json"

SEED = 42
TIER_TARGETS = {"easy": 3, "medium": 4, "hard": 3}
PER_REPO_CAP = 2
N_ROLLOUTS = 2
STEP_LIMIT = 50
COST_LIMIT_USD = 0.80          # per rollout
WALL_LIMIT_S = 40 * 60         # per rollout
CMD_TIMEOUT_S = 120            # per agent command (emulation is slow)
VERIFY_TIMEOUT_S = 1800        # per verification pytest run
P2P_CAP = 15                   # max pass-to-pass tests run (time)

# Teacher swap 2026-08-21: Qwen3.8-27B (dense 27B, apache-2.0). Tokenizer gate
# vs Qwen3-8B student = none (248k vs 151k vocab) -> SFT-only leg, unchanged.
# Allowlist from endpoint_smoke 2026-08-21: all pin, reasoning field OK,
# fp8+, ctx>=262k; Parasail first (only provider with prompt caching).
MODEL_NAME = "qwen/qwen3.8-27b"
PROVIDER_PIN = {"order": ["Parasail", "Reka", "AkashML"],
                "allow_fallbacks": False}

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$")


def load_env_key():
    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        env_file = REPO_DIR / ".env.openrouter"
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()


def image_name(instance_id: str) -> str:
    return f"xingyaoww/sweb.eval.x86_64.{instance_id.replace('__', '_s_')}"


def image_exists_on_hub(instance_id: str) -> bool:
    url = (f"https://hub.docker.com/v2/repositories/"
           f"{image_name(instance_id)}/tags?page_size=1")
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.status == 200
    except Exception:
        return False


def as_list(value):
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def patch_stats(patch_text: str) -> tuple[int, int]:
    files = len(re.findall(r"^diff --git ", patch_text, re.M))
    changed = sum(1 for line in patch_text.splitlines()
                  if (line.startswith("+") and not line.startswith("+++"))
                  or (line.startswith("-") and not line.startswith("---")))
    return files, changed


def tier_of(row) -> str:
    files, changed = patch_stats(row["patch"])
    n_f2p = len(as_list(row["FAIL_TO_PASS"]))
    if files >= 3 or changed > 60 or n_f2p > 5:
        return "hard"
    if files == 1 and changed < 15:
        return "easy"
    return "medium"


def select_tasks():
    """Stratified 3/4/3 by tier, per-repo cap, unseen-repo preference,
    Docker Hub image preflight. Deterministic (SEED)."""
    import random
    rng = random.Random(SEED)
    df = pd.read_parquet(PARQUET)
    df["tier"] = df.apply(tier_of, axis=1)
    print("tier distribution in SWE-Gym-Lite:",
          df.tier.value_counts().to_dict())

    chosen, repo_counts = [], {}
    for tier, want in TIER_TARGETS.items():
        pool = df[df.tier == tier].to_dict("records")
        rng.shuffle(pool)
        # prefer repos we haven't used yet, then repos under the cap
        pool.sort(key=lambda r: repo_counts.get(r["repo"], 0))
        got = 0
        for row in pool:
            if got == want:
                break
            if repo_counts.get(row["repo"], 0) >= PER_REPO_CAP:
                continue
            if not image_exists_on_hub(row["instance_id"]):
                print(f"  preflight MISS (no image): {row['instance_id']}")
                continue
            chosen.append(row)
            repo_counts[row["repo"]] = repo_counts.get(row["repo"], 0) + 1
            got += 1
            print(f"  {tier:6s} {row['instance_id']:40s} {row['repo']}")
        if got < want:
            sys.exit(f"could not fill tier {tier}: got {got}/{want}")

    print(f"repos used: {len(repo_counts)} {sorted(repo_counts)}")
    if len(repo_counts) < 6:
        sys.exit("fewer than 6 distinct repos; adjust PER_REPO_CAP")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = [{k: (as_list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS")
                         else v)
                     for k, v in row.items()} for row in chosen]
    TASKS_FILE.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"wrote {TASKS_FILE}")


def docker_pull(image: str):
    """Pull with retries; Docker Hub blips must not kill a multi-hour run."""
    for attempt in range(3):
        proc = subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image],
            capture_output=True, timeout=3600, text=True)
        if proc.returncode == 0:
            return
        print(f"pull attempt {attempt + 1} failed: {proc.stderr[-300:]}")
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"image pull failed 3x: {image}")


def make_env(image: str, timeout: int):
    from minisweagent.environments.docker import DockerEnvironment
    return DockerEnvironment(
        image=image, cwd="/testbed", timeout=timeout,
        run_args=["--rm", "--platform=linux/amd64", "--network=none"],
        env={"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
             "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
             "BASH_ENV": "/root/.bashrc"},
        container_timeout="3h")


# SWE-Gym images ship the repo's FULL git history, including the upstream
# fix the task asks for; 37/45 pilot trajectories retrieved it via git
# log/show (external review 2026-08). Delete every ref except the checked-out
# base commit and prune, so new trajectories are independent diagnoses. The
# packer's history_assisted flag doubles as the leak detector: any True row
# generated after this landed means the strip failed.
STRIP_CMD = (
    "cd /testbed && git checkout -q --detach HEAD 2>/dev/null; "
    "git for-each-ref --format='%(refname)' refs/heads refs/tags "
    "refs/remotes | xargs -r -n 1 git update-ref -d; "
    "git reflog expire --expire=now --all 2>/dev/null; "
    "git gc --prune=now -q 2>/dev/null; "
    "git log --oneline -1")


def strip_future_history(env):
    if os.environ.get("KEEP_GIT_HISTORY") == "1":
        return
    out = env.execute(STRIP_CMD)
    if out.get("returncode") not in (0, None):
        raise RuntimeError(f"git history strip failed: {out}")


API_TIMEOUT_S = 300  # library hardcodes 60s; long GLM thinking steps exceed it


def _patch_api_timeout():
    """mini-swe-agent's OpenRouterModel passes timeout=60 to requests.post;
    long thinking steps time out and the retry double-bills the generation.
    Shim the module's `requests` so every post uses API_TIMEOUT_S.

    Self-hosted teacher: when TEACHER_BASE_URL is set (an OpenAI-compatible
    /v1/chat/completions endpoint, e.g. vLLM serving the teacher), the shim
    redirects every request there, swaps auth to TEACHER_API_KEY, and strips
    OpenRouter-only body fields (provider pin). The vLLM server must use
    --served-model-name matching MODEL_NAME so no rename is needed."""
    import os
    import types

    import requests as real_requests

    from minisweagent.models import openrouter_model as orm

    teacher_url = os.environ.get("TEACHER_BASE_URL")
    teacher_key = os.environ.get("TEACHER_API_KEY")

    def post_with_timeout(url, *args, **kwargs):
        kwargs["timeout"] = API_TIMEOUT_S
        if teacher_url:
            url = teacher_url
            body = kwargs.get("json")
            if isinstance(body, dict):
                body.pop("provider", None)
            if teacher_key:
                headers = dict(kwargs.get("headers") or {})
                headers["Authorization"] = f"Bearer {teacher_key}"
                kwargs["headers"] = headers
        return real_requests.post(url, *args, **kwargs)

    orm.requests = types.SimpleNamespace(
        post=post_with_timeout, exceptions=real_requests.exceptions)


def rollout(instance: dict, idx: int, out_dir: Path) -> dict:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.openrouter_model import OpenRouterModel

    _patch_api_timeout()

    import minisweagent
    stock = yaml.safe_load((Path(minisweagent.__file__).parent / "config" /
                            "benchmarks" / "swebench.yaml").read_text())
    model = OpenRouterModel(
        model_name=MODEL_NAME,
        model_kwargs={"temperature": 0.7, "max_tokens": 8000,
                      "provider": PROVIDER_PIN,
                      "reasoning": {"enabled": True}},
        cost_tracking="ignore_errors")
    env = make_env(image_name(instance["instance_id"]), CMD_TIMEOUT_S)
    strip_future_history(env)
    started = time.time()
    try:
        agent = DefaultAgent(
            model, env,
            system_template=stock["agent"]["system_template"],
            instance_template=stock["agent"]["instance_template"],
            step_limit=STEP_LIMIT, cost_limit=COST_LIMIT_USD,
            wall_time_limit_seconds=WALL_LIMIT_S)
        info = agent.run(instance["problem_statement"])
        exit_status = info.get("exit_status")
        submission = info.get("submission") or ""
    except Exception as exc:  # noqa: BLE001 - one rollout must not kill the run
        exit_status, submission = f"driver:{type(exc).__name__}", ""
        agent = None
    finally:
        try:
            env.cleanup()
        except Exception:  # noqa: BLE001
            pass

    record = {
        "instance_id": instance["instance_id"], "rollout": idx,
        "exit_status": exit_status,
        "submitted": exit_status == "Submitted",
        "n_steps": getattr(agent, "n_calls", None),
        "cost_usd": round(getattr(agent, "cost", 0.0), 4),
        "wall_s": round(time.time() - started, 1),
        "patch_chars": len(submission),
    }
    (out_dir / f"rollout{idx}.traj.json").write_text(json.dumps({
        **record,
        "provider_pin": PROVIDER_PIN, "model": MODEL_NAME,
        "messages": getattr(agent, "messages", []),
    }, indent=2, default=str))
    record["submission"] = submission
    return record


def sh_b64(env, b64: str, dest: str):
    return env.execute({"command": f"echo {b64} | base64 -d > {dest}"})


def verify(instance: dict, patch_text: str) -> dict:
    """Fresh container: agent patch (non-test only) + gold tests, pytest x2."""
    result = {"tampered": False, "apply_ok": False, "runs": []}
    if TEST_PATH_RE.search("\n".join(
            re.findall(r"^diff --git a/(\S+)", patch_text, re.M))):
        result["tampered"] = True
        return result
    env = make_env(image_name(instance["instance_id"]), VERIFY_TIMEOUT_S)
    try:
        sh_b64(env, base64.b64encode(patch_text.encode()).decode(),
               "/tmp/agent.patch")
        out = env.execute({"command":
                           "cd /testbed && git apply --whitespace=nowarn "
                           "/tmp/agent.patch"})
        if out["returncode"] != 0:
            result["apply_error"] = out["output"][-800:]
            return result
        result["apply_ok"] = True
        # gold tests re-applied FRESH (anti-tamper: agent cannot have
        # weakened them; its own test edits were rejected above)
        sh_b64(env, base64.b64encode(instance["test_patch"].encode()).decode(),
               "/tmp/gold_tests.patch")
        out = env.execute({"command":
                           "cd /testbed && git checkout -- . 2>/dev/null; "
                           "git apply --whitespace=nowarn /tmp/agent.patch "
                           "&& git apply --whitespace=nowarn "
                           "/tmp/gold_tests.patch"})
        if out["returncode"] != 0:
            result["apply_error"] = "gold test apply: " + out["output"][-800:]
            return result
        f2p = as_list(instance["FAIL_TO_PASS"])
        p2p = as_list(instance["PASS_TO_PASS"])[:P2P_CAP]
        test_ids = " ".join(f"'{t}'" for t in f2p + p2p)
        for attempt in (1, 2):   # design: double run, flaky successes die
            out = env.execute({"command":
                               f"cd /testbed && python -m pytest -q "
                               f"--no-header {test_ids}"})
            result["runs"].append({"attempt": attempt,
                                   "returncode": out["returncode"],
                                   "tail": out["output"][-500:]})
        result["verified"] = all(r["returncode"] == 0 for r in result["runs"])
    finally:
        try:
            env.cleanup()
        except Exception:  # noqa: BLE001
            pass
    result.setdefault("verified", False)
    return result


def run():
    load_env_key()
    tasks = json.loads(TASKS_FILE.read_text())
    for instance in tasks:
        iid = instance["instance_id"]
        out_dir = RUNS_DIR / iid
        out_dir.mkdir(parents=True, exist_ok=True)
        ledger = out_dir / "result.json"
        if ledger.exists():
            print(f"SKIP {iid} (ledger exists)")
            continue
        print(f"\n=== {iid} [{instance['tier']}] {instance['repo']} ===")
        print("pulling image ...")
        try:
            docker_pull(image_name(iid))
        except Exception as exc:  # noqa: BLE001 - skip task, never kill the run
            print(f"TASK {iid}: SKIPPED (pull failed: {exc})")
            ledger.write_text(json.dumps({
                "instance_id": iid, "tier": instance["tier"],
                "repo": instance["repo"], "rollouts": [],
                "task_verified": False, "error": f"pull failed: {exc}"}))
            continue
        task_result = {"instance_id": iid, "tier": instance["tier"],
                       "repo": instance["repo"], "rollouts": []}
        for idx in range(1, N_ROLLOUTS + 1):
            print(f"rollout {idx}/{N_ROLLOUTS} ...")
            rec = rollout(instance, idx, out_dir)
            print(f"  exit={rec['exit_status']} steps={rec['n_steps']} "
                  f"cost=${rec['cost_usd']} patch={rec['patch_chars']}ch")
            if rec["submitted"] and rec["patch_chars"] > 0:
                print("  verifying ...")
                rec["verify"] = verify(instance, rec.pop("submission"))
                print(f"  verified={rec['verify'].get('verified')} "
                      f"tampered={rec['verify']['tampered']}")
            else:
                rec.pop("submission", None)
                rec["verify"] = {"verified": False,
                                 "reason": "no submission"}
            task_result["rollouts"].append(rec)
        task_result["task_verified"] = any(
            r["verify"].get("verified") for r in task_result["rollouts"])
        ledger.write_text(json.dumps(task_result, indent=2))
        print(f"TASK {iid}: verified={task_result['task_verified']}")
        # image GC after seal: Docker Desktop's VM disk is the binding
        # constraint (two pull-failure incidents, 2026-07-29); images are
        # re-pullable, sealed tasks never need theirs again
        subprocess.run(["docker", "rmi", image_name(iid)],
                       capture_output=True)
    report()


def report():
    rows = [json.loads(p.read_text())
            for p in sorted(RUNS_DIR.glob("*/result.json"))]
    if not rows:
        print("no results yet")
        return
    by_tier = {}
    for row in rows:
        t = by_tier.setdefault(row["tier"], {"tasks": 0, "verified": 0,
                                             "rollouts": 0, "successes": 0,
                                             "cost": 0.0, "steps": []})
        t["tasks"] += 1
        t["verified"] += int(row["task_verified"])
        for r in row["rollouts"]:
            t["rollouts"] += 1
            t["successes"] += int(r["verify"].get("verified", False))
            t["cost"] += r.get("cost_usd", 0)
            if r.get("n_steps"):
                t["steps"].append(r["n_steps"])
    for t in by_tier.values():
        t["cost"] = round(t["cost"], 3)
        t["mean_steps"] = (round(sum(t["steps"]) / len(t["steps"]), 1)
                           if t["steps"] else None)
        del t["steps"]
    total_cost = round(sum(t["cost"] for t in by_tier.values()), 3)
    out = {"date": time.strftime("%Y-%m-%d"), "model": MODEL_NAME,
           "provider_pin": PROVIDER_PIN, "n_tasks": len(rows),
           "by_tier": by_tier, "total_rollout_cost_usd": total_cost,
           "note": ("local Mac x86-emulated Docker; network-isolated "
                    "containers; N=2, step<=50, $0.80/rollout cap")}
    REPORT_FILE.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"select": select_tasks, "run": run, "report": report}[
        sys.argv[1] if len(sys.argv) > 1 else "run"]()
