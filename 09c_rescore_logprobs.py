#!/usr/bin/env python3
"""09c_rescore_logprobs.py — teacher-forced top-K logprob rescore of verified
trajectories through a self-hosted teacher (vLLM offline API).

WHY: public API logprobs are CONTENT-ONLY (reasoning tokens uncovered, see
endpoint_smoke_report.json) and the OpenRouter caps vary by provider. Full-trace
logit KD needs top-K at EVERY assistant token, reasoning included. A single
prefill pass per trajectory on a self-hosted teacher provides exactly that.

DESIGN (alignment-free by construction):
  1. Render the whole conversation ONCE with the TEACHER chat template,
     reasoning kept in history (Qwen3.8 `preserve_thinking`).
  2. Loss mask via incremental prefix tokenization: tokens added by each
     assistant turn (minus its generation prompt) are scoreable.
  3. One vLLM prefill with prompt_logprobs=TOP_K -> top-K (id, logprob) per
     position.
  4. Shard = token_ids + mask + top-K arrays. The student consumes the SAME
     token ids (teacher/student share the 248k vocab; gate=remap with only 7
     audio-only specials missing — unreachable in text trajectories).

Run ON the GPU box:
  python3 09c_rescore_logprobs.py --traj-dir runs/rehearsal --out-dir shards \
      [--model Qwen/Qwen3.8-27B-FP8] [--only-verified] [--limit N]

Resumable: one .npz shard per rollout; existing shards are skipped.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Mirror of minisweagent.models.openrouter_model.BASH_TOOL (v2.4.6): the tool
# schema every generation request carried; the chat template needs the same
# `tools` input to reproduce the system-prompt tool block (and some templates
# reject the "tool" role entirely when tools are absent).
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string",
                    "description": "The bash command to execute"}},
        "required": ["command"]}}}

TOP_K = 20
MAX_LEN = 131072          # teacher context guard; longer trajectories are skipped
NEG_INF = -1e30           # fill for positions with fewer than TOP_K entries


def teacher_messages(raw_messages: list) -> list:
    """Map OpenRouter-shaped messages to the teacher chat-template shape.

    OpenRouter returns assistant reasoning in `reasoning`; Qwen templates
    expect `reasoning_content`. Non-serializable extras are dropped.
    """
    out = []
    for m in raw_messages:
        # mini-swe-agent appends harness pseudo-messages (role "exit") that
        # were never in the model's context; only real chat roles render.
        if m["role"] not in ("system", "user", "assistant", "tool"):
            continue
        clean = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant":
            if m.get("reasoning"):
                clean["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                # OpenRouter stores function.arguments as a JSON string; Qwen
                # chat templates iterate it as a mapping.
                calls = []
                for tc in m["tool_calls"]:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except json.JSONDecodeError:
                            pass
                    tc["function"] = fn
                    calls.append(tc)
                clean["tool_calls"] = calls
        if m["role"] == "tool" and m.get("tool_call_id"):
            clean["tool_call_id"] = m["tool_call_id"]
        out.append(clean)
    return out


def render(tokenizer, messages: list, upto: int, add_generation_prompt: bool):
    kwargs = {}
    # Qwen3.8: keep reasoning of PAST assistant turns in the rendering so the
    # scored sequence matches what the teacher actually emitted step by step.
    if "preserve_thinking" in (tokenizer.chat_template or ""):
        kwargs["preserve_thinking"] = True
    out = tokenizer.apply_chat_template(
        messages[:upto], tools=[BASH_TOOL], tokenize=True,
        add_generation_prompt=add_generation_prompt, **kwargs)
    # transformers may return list[int], BatchEncoding (not a dict subclass
    # in v5), or tokenizers.Encoding
    if hasattr(out, "ids"):
        return list(out.ids)
    try:
        return list(out["input_ids"])
    except (TypeError, KeyError, IndexError):
        return list(out)


def tokens_and_mask(tokenizer, messages: list):
    """Full token ids + bool mask marking tokens each assistant turn ADDED.

    For assistant turn at index i: mask covers tokens between
    render(messages[:i], gen_prompt=True) and render(messages[:i+1], no prompt).
    The generation prompt itself (role header) is context, not prediction.
    """
    full = render(tokenizer, messages, len(messages), False)
    mask = np.zeros(len(full), dtype=bool)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(render(tokenizer, messages, i, True))
        end = len(render(tokenizer, messages, i + 1, False))
        prefix = render(tokenizer, messages, i + 1, False)[:start]
        if prefix != full[:start][:len(prefix)]:
            raise ValueError(f"prefix mismatch at assistant turn {i}")
        mask[start:end] = True
    return full, mask


def rescore_one(llm, sampling, tokenizer, traj_path: Path, out_path: Path) -> dict:
    from vllm import TokensPrompt

    raw = json.loads(traj_path.read_text())
    messages = teacher_messages(raw["messages"])
    ids, mask = tokens_and_mask(tokenizer, messages)
    if len(ids) > MAX_LEN:
        return {"skipped": f"too long ({len(ids)} > {MAX_LEN})"}

    out = llm.generate(
        [TokensPrompt(prompt_token_ids=list(ids))], sampling)[0]
    plp = out.prompt_logprobs  # list, one entry per prompt position (0th=None)

    n = len(ids)
    topk_ids = np.full((n, TOP_K), -1, dtype=np.int64)
    topk_lps = np.full((n, TOP_K), NEG_INF, dtype=np.float32)
    actual_lp = np.full(n, NEG_INF, dtype=np.float32)
    for pos, entry in enumerate(plp or []):
        if entry is None:
            continue
        ranked = sorted(entry.items(), key=lambda kv: kv[1].rank or 1 << 30)
        for j, (tok_id, lp) in enumerate(ranked[:TOP_K]):
            topk_ids[pos, j] = tok_id
            topk_lps[pos, j] = lp.logprob
        if ids[pos] in entry:
            actual_lp[pos] = entry[ids[pos]].logprob

    np.savez_compressed(
        out_path, token_ids=np.asarray(ids, dtype=np.int64), loss_mask=mask,
        topk_ids=topk_ids, topk_logprobs=topk_lps, actual_logprob=actual_lp,
        meta=json.dumps({
            "instance_id": raw.get("instance_id"), "rollout": raw.get("rollout"),
            "gen_model": raw.get("model"), "top_k": TOP_K,
            "n_tokens": n, "n_scored": int(mask.sum())}))
    covered = actual_lp[mask] > NEG_INF / 2
    return {"n_tokens": n, "n_scored": int(mask.sum()),
            "scored_coverage": round(float(covered.mean()), 4) if mask.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--only-verified", action="store_true",
                    help="rescore only rollouts whose task ledger verified them")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    traj_dir, out_dir = Path(args.traj_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for traj in sorted(traj_dir.glob("*/rollout*.traj.json")):
        ledger_path = traj.parent / "result.json"
        if args.only_verified:
            if not ledger_path.exists():
                continue
            ledger = json.loads(ledger_path.read_text())
            idx = int(traj.stem.replace("rollout", "").split(".")[0]) - 1
            rollouts = ledger.get("rollouts", [])
            if idx >= len(rollouts) or not rollouts[idx].get(
                    "verify", {}).get("verified"):
                continue
        jobs.append(traj)
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} trajectories to rescore")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 0.75 default: Vast vLLM-template boxes can strand GPU memory in an
    # unreapable zombie (parent is the container's PID-1 bash); leave headroom.
    # max_num_seqs low: rescore is one-prompt-at-a-time prefill; Qwen3.8's
    # hybrid Mamba layers need one cache block per seq slot, so the vLLM
    # default (1024) cannot fit under a capped gpu_memory_utilization.
    # max_num_batched_tokens caps the prefill chunk: prompt_logprobs
    # materializes chunk_len x vocab(248k) logits, so 4096 tokens ~= 1.9 GiB
    # transient instead of 15+ GiB for a whole 30k+-token trajectory.
    llm = LLM(model=args.model, max_model_len=MAX_LEN, enforce_eager=False,
              max_num_seqs=int(os.environ.get("RESCORE_MAX_SEQS", "8")),
              max_num_batched_tokens=int(os.environ.get(
                  "RESCORE_CHUNK_TOKENS", "4096")),
              gpu_memory_utilization=float(os.environ.get("RESCORE_GPU_UTIL",
                                                          "0.75")))
    sampling = SamplingParams(max_tokens=1, prompt_logprobs=TOP_K,
                              temperature=0.0)

    results = {}
    for traj in jobs:
        shard = out_dir / f"{traj.parent.name}__{traj.stem}.npz"
        if shard.exists():
            print(f"SKIP {shard.name} (exists)")
            continue
        try:
            res = rescore_one(llm, sampling, tokenizer, traj, shard)
        except Exception as exc:  # noqa: BLE001 - one bad trajectory must not kill the pass
            res = {"error": f"{type(exc).__name__}: {exc}"}
        results[shard.name] = res
        print(shard.name, res)

    (out_dir / "rescore_report.json").write_text(json.dumps(results, indent=2))
    print(f"report -> {out_dir / 'rescore_report.json'}")


if __name__ == "__main__":
    main()
