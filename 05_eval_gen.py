#!/usr/bin/env python3
"""
05_eval_gen.py — Generate answers on the held-out set.

Two modes:
  local model (default): offline vLLM engine on available GPUs
      python 05_eval_gen.py --model ckpts/qwen3_sft/final \
          --heldout data/prompts_heldout.jsonl --out evals/qwen3_sft.jsonl
  served model (teacher reference):
      python 05_eval_gen.py --served --base-url http://localhost:8000/v1 \
          --model glm-5.2 --heldout data/prompts_heldout.jsonl --out evals/teacher_ref.jsonl

Fixed decoding for comparability: temperature 0.6, top_p 0.95, seed 42, max 4096.
"""

import argparse
import json
from pathlib import Path

TEMP, TOP_P, MAX_TOK, SEED = 0.6, 0.95, 4096, 42


def load_heldout(path):
    return [json.loads(l) for l in open(path)]


def run_local(args, prompts):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=16384,
        seed=SEED,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            add_generation_prompt=True, tokenize=False,
        )
        for p in prompts
    ]
    sp = SamplingParams(temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOK, seed=SEED)
    outs = llm.generate(texts, sp)
    return [o.outputs[0].text for o in outs]


def run_served(args, prompts):
    import asyncio
    import aiohttp

    async def one(session, sem, p):
        async with sem:
            async with session.post(
                f"{args.base_url}/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": p["prompt"]}],
                    "temperature": TEMP, "top_p": TOP_P,
                    "max_tokens": MAX_TOK, "seed": SEED + p["sample_id"],
                },
                timeout=aiohttp.ClientTimeout(total=1200),
            ) as r:
                d = await r.json()
                return d["choices"][0]["message"].get("content") or ""

    async def all_():
        sem = asyncio.Semaphore(64)
        async with aiohttp.ClientSession() as s:
            return await asyncio.gather(*[one(s, sem, p) for p in prompts])

    return asyncio.run(all_())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--served", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()

    prompts = load_heldout(args.heldout)
    answers = run_served(args, prompts) if args.served else run_local(args, prompts)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p, a in zip(prompts, answers):
            f.write(json.dumps(
                {"sample_id": p["sample_id"], "slice": p["slice"],
                 "prompt": p["prompt"], "answer": a}, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} answers to {args.out}")


if __name__ == "__main__":
    main()
