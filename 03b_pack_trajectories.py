#!/usr/bin/env python3
"""Pack VERIFIED agent trajectories into the canonical multi-turn format.

This is the MULTITURN_READY layer (agentic_trajectory_design.md section 5):
student-agnostic message lists with a role-based loss-mask convention;
student-specific tokenization (chat template, think tags) happens at train
time in 04, not here.

Loss-mask convention (documented contract, applied by the trainer):
  TRAINABLE : assistant turns — reasoning, content, and tool_calls
  CONTEXT   : system, user, and tool (observation) turns — never in the loss

Input : runs/rehearsal/<iid>/result.json + rollout<N>.traj.json
        (only rollouts whose verify.verified == True are packed)
Output: packed/trajectories/trajectories_v0.parquet   (gitignored, data)
        trajectory_pack_report.json                   (committable, funnel)

Format id: fc_bash_v1 — mini-swe-agent 2.x native tool-calling, single bash
tool. Never mix format ids in one training run (design section 2).
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).parent
# All run directories holding <iid>/result.json ledgers to pack together.
RUNS_DIRS = [REPO_DIR / "runs" / "rehearsal",
             REPO_DIR / "runs" / "walk_regen" / "bugfix"]
OUT_DIR = REPO_DIR / "packed" / "trajectories"
REPORT = REPO_DIR / "trajectory_pack_report.json"
TOKENIZER_JSON = Path("/private/tmp/claude-501/-Users-arunmenon-projects-"
                      "glm52-distill/ad896283-5c39-4a76-bdab-80daf2f812f4/"
                      "scratchpad/tokenizer_glm52.json")
FORMAT_ID = "fc_bash_v1"
TOKEN_CAP = 32768          # design section 5: report, and flag over-cap rows

KEEP_KEYS = {"assistant": ("role", "content", "reasoning", "tool_calls"),
             "system": ("role", "content"),
             "user": ("role", "content"),
             "tool": ("role", "content", "tool_call_id")}


def clean_messages(raw_messages: list[dict]) -> list[dict]:
    """Strip per-message API debris (extra/refusal/annotations); keep only
    the fields the training template needs. Validates tool-call linkage."""
    cleaned, open_call_ids = [], set()
    for msg in raw_messages:
        role = msg.get("role")
        if role == "exit":
            continue   # mini-swe-agent bookkeeping (exit status), not a turn
        if role not in KEEP_KEYS:
            raise ValueError(f"unexpected role: {role}")
        kept = {k: msg[k] for k in KEEP_KEYS[role]
                if msg.get(k) not in (None, [], "")}
        kept["role"] = role
        if role == "assistant":
            open_call_ids = {tc["id"] for tc in msg.get("tool_calls") or []}
            # normalize tool_calls to (id, name, arguments) only
            if "tool_calls" in kept:
                kept["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"],
                                  "arguments": tc["function"]["arguments"]}}
                    for tc in kept["tool_calls"]]
        elif role == "tool":
            if msg.get("tool_call_id") not in open_call_ids:
                raise ValueError("tool message without matching tool_call")
        cleaned.append(kept)
    if cleaned and cleaned[-1]["role"] != "assistant":
        # trajectory must END on the model's turn (the submission)
        while cleaned and cleaned[-1]["role"] != "assistant":
            cleaned.pop()
    return cleaned


def main():
    tokenizer = None
    if TOKENIZER_JSON.exists():
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
        except Exception:  # noqa: BLE001 - fall back to estimate
            pass

    def count_tokens(text: str) -> int:
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        return int(len(text) / 3.5)

    rows, funnel = [], {"tasks": 0, "rollouts": 0, "verified": 0,
                        "packed": 0, "over_cap": 0, "clean_errors": 0}
    for result_file in sorted(
            f for d in RUNS_DIRS for f in d.glob("*/result.json")):
        result = json.loads(result_file.read_text())
        funnel["tasks"] += 1
        for rollout_rec in result.get("rollouts", []):
            funnel["rollouts"] += 1
            if not rollout_rec.get("verify", {}).get("verified"):
                continue
            funnel["verified"] += 1
            traj_file = (result_file.parent /
                         f"rollout{rollout_rec['rollout']}.traj.json")
            traj = json.loads(traj_file.read_text())
            try:
                messages = clean_messages(traj["messages"])
            except ValueError as exc:
                print(f"CLEAN ERROR {traj['instance_id']} "
                      f"r{rollout_rec['rollout']}: {exc}")
                funnel["clean_errors"] += 1
                continue
            text_view = "\n".join(
                str(m.get("content", "")) + str(m.get("reasoning", "")) +
                json.dumps(m.get("tool_calls", "")) for m in messages)
            est_tokens = count_tokens(text_view)
            over_cap = est_tokens > TOKEN_CAP
            funnel["over_cap"] += int(over_cap)
            rows.append({
                "instance_id": traj["instance_id"],
                "repo": result["repo"], "tier": result["tier"],
                "rollout": rollout_rec["rollout"],
                "source": "swegym_lite_rehearsal",
                "format": FORMAT_ID,
                "teacher_model": traj["model"],
                "n_steps": rollout_rec.get("n_steps"),
                "cost_usd": rollout_rec.get("cost_usd"),
                "n_messages": len(messages),
                "n_assistant_turns": sum(
                    1 for m in messages if m["role"] == "assistant"),
                "est_tokens": est_tokens,
                "over_token_cap": over_cap,
                "messages_json": json.dumps(messages, ensure_ascii=False),
            })
            funnel["packed"] += 1

    if not rows:
        sys.exit("no verified rollouts found to pack")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_file = OUT_DIR / "trajectories_v0.parquet"
    df.to_parquet(out_file, index=False)

    report = {
        "date": time.strftime("%Y-%m-%d"), "format": FORMAT_ID,
        "loss_mask_convention": "trainable=assistant turns only",
        "funnel": funnel,
        "by_tier": df.groupby("tier").size().to_dict(),
        "token_stats": {"min": int(df.est_tokens.min()),
                        "mean": int(df.est_tokens.mean()),
                        "max": int(df.est_tokens.max()),
                        "cap": TOKEN_CAP,
                        "tokenizer": "glm52" if tokenizer else "char-estimate"},
        "output": str(out_file.relative_to(REPO_DIR)),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
