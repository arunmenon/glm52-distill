#!/usr/bin/env python3
"""
06_judge.py — Pairwise judging: candidate (student) vs reference (teacher) on the
held-out set, judged by the served teacher (or any served judge model).

Controls: position randomized per vote (seeded), 3 votes per pair, majority wins.
Outputs overall + per-slice win/tie/loss and a win-rate summary JSON.

Usage (teacher server must be up):
  python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/air_kd.jsonl \
      --out evals/judge_air_kd.json --base-url http://localhost:8000/v1 --judge glm-5.2
"""

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict

import aiohttp

SEED = 42
VOTES = 3
# The judge server runs without a reasoning parser: its thinking arrives inline
# in content, so it needs token room before the verdict appears.
JUDGE_MAX_TOKENS = 4096


def strip_think(text: str) -> str:
    """Drop <think>...</think> blocks (students think inline; so does the judge)."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()

JUDGE_PROMPT = """You are an impartial judge. Compare two answers to the same question.
Judge on correctness first, then completeness, then clarity. Ignore length and style.

[Question]
{q}

[Answer A]
{a}

[Answer B]
{b}

Reply with exactly one token: "A" if A is better, "B" if B is better, "TIE" if equal."""


async def vote(session, sem, judge, base_url, q, a, b, seed):
    async with sem:
        try:
            async with session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": judge,
                    "messages": [{"role": "user",
                                  "content": JUDGE_PROMPT.format(
                                      q=q[:6000],
                                      a=strip_think(a)[:8000],
                                      b=strip_think(b)[:8000])}],
                    "temperature": 0.0, "max_tokens": JUDGE_MAX_TOKENS, "seed": seed,
                },
                timeout=aiohttp.ClientTimeout(total=600),
            ) as r:
                d = await r.json()
                msg = d["choices"][0]["message"]
                txt = strip_think((msg.get("content") or "")
                                  or (msg.get("reasoning_content") or "")).upper()
                if txt.startswith("TIE"):
                    return "TIE"
                if txt.startswith("A"):
                    return "A"
                if txt.startswith("B"):
                    return "B"
                return "TIE"
        except Exception:
            return "TIE"


async def judge_pair(session, sem, args, rng, item):
    q, cand, ref = item["prompt"], item["cand"], item["ref"]
    results = []
    for v in range(VOTES):
        cand_is_a = rng.random() < 0.5
        a, b = (cand, ref) if cand_is_a else (ref, cand)
        r = await vote(session, sem, args.judge, args.base_url, q, a, b,
                       seed=SEED + item["sample_id"] * 10 + v)
        if r == "TIE":
            results.append("tie")
        elif (r == "A") == cand_is_a:
            results.append("win")
        else:
            results.append("loss")
    c = Counter(results)
    verdict = max(("win", "loss", "tie"), key=lambda k: c[k])
    return item["sample_id"], item["slice"], verdict


async def main_async(args):
    ref = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.ref)}
    cand = [json.loads(l) for l in open(args.cand)]
    items = [
        {"sample_id": c["sample_id"], "slice": c["slice"], "prompt": c["prompt"],
         "cand": c["answer"], "ref": ref[c["sample_id"]]["answer"]}
        for c in cand if c["sample_id"] in ref
    ]
    rng = random.Random(SEED)
    sem = asyncio.Semaphore(48)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[judge_pair(session, sem, args, rng, it) for it in items]
        )

    overall = Counter(v for _, _, v in results)
    per_slice = defaultdict(Counter)
    for _, sl, v in results:
        per_slice[sl][v] += 1

    n = sum(overall.values())
    summary = {
        "n": n,
        "overall": dict(overall),
        "win_rate_vs_teacher": round((overall["win"] + 0.5 * overall["tie"]) / max(n, 1), 4),
        "per_slice": {
            sl: {**dict(c),
                 "win_rate": round((c["win"] + 0.5 * c["tie"]) / max(sum(c.values()), 1), 4)}
            for sl, c in per_slice.items()
        },
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="teacher reference answers jsonl")
    ap.add_argument("--cand", required=True, help="student answers jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge", default="glm-5.2")
    asyncio.run(main_async(ap.parse_args()))
