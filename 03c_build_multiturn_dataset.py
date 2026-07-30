#!/usr/bin/env python3
"""Tokenize fc_bash_v1 trajectories into 04_train_kd.py's dataset contract.

Bridges 03b (canonical messages) -> 04 (input_ids + labels, -100 masked):
renders each trajectory in the student's chat markup with EXPLICIT templates
rather than tokenizer.apply_chat_template, because inference templates strip
reasoning from historical turns; the teacher ran with interleaved thinking
preserved across the whole trajectory, so the student must train that way
too (deployment note: the student's harness should likewise resend prior
reasoning).

Loss mask: trainable = assistant turn bodies (think block + content + tool
calls + closing tag). Turn headers, system/user/tool-observation text: -100.

Families supported: qwen3 (ChatML + <think> + <tool_call>). GLM-family
rendering lands at WALK with the flagship student.

Usage:
  python3 03c_build_multiturn_dataset.py \
      --data packed/trajectories/trajectories_v0.parquet \
      --tokenizer Qwen/Qwen3-8B --out packed/mt_qwen3
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

IGNORE = -100
MAX_LEN_DEFAULT = 32768


def render_qwen3_segments(messages: list[dict]) -> list[tuple[str, bool]]:
    """-> [(text, trainable)] in Qwen3 ChatML. Consecutive tool observations
    merge into one user turn of <tool_response> blocks (matches Qwen3's own
    template behavior)."""
    segments = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        if role in ("system", "user"):
            segments.append(
                (f"<|im_start|>{role}\n{msg.get('content', '')}<|im_end|>\n",
                 False))
            i += 1
        elif role == "tool":
            blocks = []
            while i < len(messages) and messages[i]["role"] == "tool":
                blocks.append(f"<tool_response>\n"
                              f"{messages[i].get('content', '')}\n"
                              f"</tool_response>")
                i += 1
            segments.append(
                ("<|im_start|>user\n" + "\n".join(blocks) + "<|im_end|>\n",
                 False))
        elif role == "assistant":
            segments.append(("<|im_start|>assistant\n", False))  # header
            body = ""
            reasoning = msg.get("reasoning", "")
            body += f"<think>\n{reasoning}\n</think>\n\n"
            if msg.get("content"):
                body += msg["content"]
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                try:
                    arguments = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = fn["arguments"]
                body += (f"\n<tool_call>\n"
                         f"{json.dumps({'name': fn['name'], 'arguments': arguments}, ensure_ascii=False)}"
                         f"\n</tool_call>")
            body += "<|im_end|>"
            segments.append((body, True))                        # trainable
            segments.append(("\n", False))
            i += 1
        else:
            raise ValueError(f"unexpected role {role}")
    return segments


def tokenize_with_mask(tokenizer, segments, max_len: int):
    input_ids, labels = [], []
    for text, trainable in segments:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids if trainable else [IGNORE] * len(ids))
    if len(input_ids) > max_len:
        return None, None
    return input_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="packed/trajectories/"
                                      "trajectories_v0.parquet")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="packed/mt_qwen3")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # boundary-safety: segment joins happen at special/added tokens, which
    # tokenize atomically; assert that here so a family with non-atomic tags
    # fails loudly instead of silently mis-masking
    for tag in ("<|im_start|>", "<|im_end|>", "<think>"):
        ids = tokenizer(tag, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"{tag} not atomic in {args.tokenizer}: {ids}"

    df = pd.read_parquet(args.data)
    assert (df["format"] == "fc_bash_v1").all(), "mixed formats refused"

    rows, dropped = [], 0
    for _, row in df.iterrows():
        segments = render_qwen3_segments(json.loads(row.messages_json))
        input_ids, labels = tokenize_with_mask(tokenizer, segments,
                                               args.max_len)
        if input_ids is None:
            dropped += 1
            print(f"DROP over max-len: {row.instance_id} r{row.rollout}")
            continue
        rows.append({"input_ids": input_ids, "labels": labels,
                     "instance_id": row.instance_id, "tier": row.tier,
                     "n_tokens": len(input_ids),
                     "n_trainable": sum(1 for l in labels if l != IGNORE)})

    # task-level split: rollouts of one instance never straddle splits
    instance_ids = sorted({r["instance_id"] for r in rows})
    random.Random(args.seed).shuffle(instance_ids)
    n_val_tasks = max(1, len(instance_ids) // 5)
    val_ids = set(instance_ids[:n_val_tasks])
    train = [r for r in rows if r["instance_id"] not in val_ids]
    val = [r for r in rows if r["instance_id"] in val_ids]

    ds = DatasetDict({"train": Dataset.from_list(train),
                      "validation": Dataset.from_list(val)})
    ds.save_to_disk(args.out)

    trainable_frac = (sum(r["n_trainable"] for r in rows) /
                      max(1, sum(r["n_tokens"] for r in rows)))
    print(json.dumps({
        "packed": len(rows), "dropped_over_len": dropped,
        "train": len(train), "validation": len(val),
        "val_tasks": sorted(val_ids),
        "token_len_min_max": [min(r["n_tokens"] for r in rows),
                              max(r["n_tokens"] for r in rows)],
        "trainable_token_fraction": round(trainable_frac, 3),
        "out": args.out}, indent=2))

    # spot-check: decode the first trainable span of row 0 — must read as a
    # think block, never as tool output or user text
    sample = rows[0]
    span = [t for t, l in zip(sample["input_ids"], sample["labels"])
            if l != IGNORE][:60]
    print("\nfirst trainable span (row 0):")
    print(tokenizer.decode(span)[:400])


if __name__ == "__main__":
    main()
