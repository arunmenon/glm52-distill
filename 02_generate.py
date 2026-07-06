#!/usr/bin/env python3
"""
02_generate.py — Distillation corpus generation against a vLLM OpenAI-compatible server,
with optional best-of-N rejection sampling.

Per generated token position it captures:
  - chosen token id
  - top-K (token_id, logprob) pairs   (K=20)

Best-of-N (--best-of N, default 1 = off):
  1. One request with n=N returns N candidates (each with its own logprobs).
  2. Candidates that truncated (finish_reason != "stop") are discarded first.
  3. If >1 survive, a single judge call to the SAME server picks the best
     (temperature 0, deterministic seed). The winner's tokens + logprobs are kept.
  Cost: ~N x generation tokens + 1 cheap judge call per prompt. With N=4 either
  cut --prompts to ~75-100k or budget ~2 extra days of generation.

IMPORTANT: launch the vLLM server with `--return-tokens-as-token-ids` so token ids
can be parsed exactly (tokens come back as "token_id:12345"). 10_serve_teacher.sh
already does this and smoke-tests it.

Resumable: one parquet shard per SHARD_SIZE samples; complete shards are skipped
on restart. Deterministic: per-sample seeds derived from GLOBAL_SEED; judge seed
is also per-sample.

Usage:
  python 02_generate.py --prompts prompts.jsonl --out corpus/ \
      --base-url http://localhost:8000/v1 --model glm-5.2 \
      --concurrency 112 --shard-size 5000 --best-of 4
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

GLOBAL_SEED = 42
TOP_K = 20
MAX_TOKENS = 8192
TEMPERATURE = 0.7
TOP_P = 0.95
REQUEST_TIMEOUT_S = 2400
MAX_RETRIES = 4
JUDGE_MAX_CAND_CHARS = 8000
# The server runs WITHOUT a reasoning parser, so the judge's own thinking arrives
# inline in content. It needs room to think before the verdict token appears.
JUDGE_MAX_TOKENS = 4096

SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.int64()),
        pa.field("slice", pa.string()),
        pa.field("prompt", pa.string()),
        pa.field("response_text", pa.string()),
        pa.field("reasoning_text", pa.string()),
        pa.field("response_token_ids", pa.list_(pa.int32())),
        pa.field("topk_ids", pa.list_(pa.list_(pa.int32()))),
        pa.field("topk_logprobs", pa.list_(pa.list_(pa.float32()))),
        pa.field("finish_reason", pa.string()),
        pa.field("gen_seed", pa.int64()),
        pa.field("bon_n", pa.int32()),        # candidates generated
        pa.field("bon_pick", pa.int32()),     # index of chosen candidate
        pa.field("bon_judged", pa.bool_()),   # False if judge fell back to 0
    ]
)

JUDGE_PROMPT = """You are grading candidate answers to a question. Pick the single best
answer judged on correctness first, then completeness, then clarity. Ignore length.

[Question]
{q}

{cands}

Reply with ONLY the number of the best candidate (e.g. "2"). Nothing else."""


def strip_think(text: str) -> str:
    """Drop <think>...</think> blocks (no reasoning parser on the server)."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_token_id(tok: str) -> int:
    if tok.startswith("token_id:"):
        return int(tok.split(":", 1)[1])
    raise ValueError(
        f"Token '{tok[:40]}' is not in token_id form. "
        "Relaunch vLLM with --return-tokens-as-token-ids."
    )


def extract_logprobs(choice: dict):
    content = (choice.get("logprobs") or {}).get("content") or []
    token_ids, topk_ids, topk_lps = [], [], []
    for pos in content:
        token_ids.append(parse_token_id(pos["token"]))
        ids_k, lps_k = [], []
        for t in pos.get("top_logprobs", [])[:TOP_K]:
            ids_k.append(parse_token_id(t["token"]))
            lps_k.append(float(t["logprob"]))
        while len(ids_k) < TOP_K:
            ids_k.append(-1)
            lps_k.append(-1e9)
        topk_ids.append(ids_k)
        topk_lps.append(lps_k)
    return token_ids, topk_ids, topk_lps


async def post_chat(session, base_url, payload):
    async with session.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def judge_pick(session, base_url, model, prompt, candidates, seed):
    """Return (index_into_candidates, judged_ok). judged_ok is False when the
    verdict was unparseable and index falls back to 0.

    Robustness fixes (journal Open item 3):
      - candidate order is randomized per judgment (seeded) so the judge's
        positional bias doesn't systematically favor slot 1; the pick is
        mapped back to the original index.
      - the verdict is accepted ONLY if the judge's thinking actually CLOSED
        (</think> present) — an unclosed think block that hit max_tokens has
        no verdict, and scanning it for a digit would grab a number from the
        reasoning text. Take the LAST number after </think> (the final
        answer), not the first.
    """
    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    shown = [candidates[i] for i in order]
    cands = "\n\n".join(
        f"[Candidate {i + 1}]\n{strip_think(c['response_text'])[:JUDGE_MAX_CAND_CHARS]}"
        for i, c in enumerate(shown)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": JUDGE_PROMPT.format(q=prompt[:6000], cands=cands)}
        ],
        "temperature": 0.0,
        "max_tokens": JUDGE_MAX_TOKENS,
        "seed": seed,
    }
    try:
        data = await post_chat(session, base_url, payload)
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or "") or (msg.get("reasoning_content") or "")
        # Only trust a verdict whose thinking closed; else it's truncated.
        if "</think>" in raw or "<think>" not in raw:
            verdict = strip_think(raw)
            nums = re.findall(r"\d+", verdict)
            if nums:
                shown_idx = int(nums[-1]) - 1   # last number = final verdict
                if 0 <= shown_idx < len(shown):
                    return order[shown_idx], True
        print(f"[judge-fallback] unparseable/truncated verdict: {raw[:80]!r}",
              file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[judge-fallback] {e}", file=sys.stderr)
    return 0, False


async def generate_one(session, sem, base_url, model, sample, best_of: int):
    gen_seed = GLOBAL_SEED + sample["sample_id"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": sample["prompt"]}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "n": best_of,
        "logprobs": True,
        "top_logprobs": TOP_K,
        "seed": gen_seed,
    }
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                data = await post_chat(session, base_url, payload)
                # Parse every candidate.
                cands = []
                for ch in data["choices"]:
                    msg = ch["message"]
                    token_ids, topk_ids, topk_lps = extract_logprobs(ch)
                    cands.append(
                        {
                            "text": (msg.get("reasoning_content") or "")
                            + (msg.get("content") or ""),
                            "response_text": msg.get("content") or "",
                            "reasoning_text": msg.get("reasoning_content") or "",
                            "response_token_ids": token_ids,
                            "topk_ids": topk_ids,
                            "topk_logprobs": topk_lps,
                            "finish_reason": ch.get("finish_reason", "unknown"),
                        }
                    )
                # Rejection step 1: drop truncated/empty candidates.
                clean = [
                    c for c in cands
                    if c["finish_reason"] == "stop" and c["response_text"].strip()
                ]
                if not clean:
                    return None  # all candidates bad -> drop sample
                # Rejection step 2: judge picks among survivors.
                if len(clean) > 1:
                    pick, judged = await judge_pick(
                        session, base_url, model, sample["prompt"], clean,
                        seed=gen_seed + 1_000_000,
                    )
                else:
                    pick, judged = 0, True   # single survivor: nothing to judge
                best = clean[pick]
                return {
                    "sample_id": sample["sample_id"],
                    "slice": sample.get("slice", "general"),
                    "prompt": sample["prompt"],
                    "response_text": best["response_text"],
                    "reasoning_text": best["reasoning_text"],
                    "response_token_ids": best["response_token_ids"],
                    "topk_ids": best["topk_ids"],
                    "topk_logprobs": best["topk_logprobs"],
                    "finish_reason": best["finish_reason"],
                    "gen_seed": gen_seed,
                    "bon_n": best_of,
                    "bon_pick": pick,
                    "bon_judged": judged,   # False = judge fell back (audit)
                }
            except Exception as e:  # noqa: BLE001
                if attempt == MAX_RETRIES - 1:
                    print(f"[FAIL] sample {sample['sample_id']}: {e}", file=sys.stderr)
                    return None
                await asyncio.sleep(2**attempt)


def write_shard(rows, shard_ids, out_dir: Path, shard_idx: int, manifest: dict):
    kept = [r for r in rows if r is not None]
    kept_ids = {r["sample_id"] for r in kept}
    failed_ids = sorted(i for i in shard_ids if i not in kept_ids)
    path = out_dir / f"shard_{shard_idx:04d}.parquet"
    tmp = out_dir / f".shard_{shard_idx:04d}.parquet.tmp"
    table = pa.Table.from_pylist(kept, schema=SCHEMA)
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(path)  # atomic: no half-written shard ever bears the final name
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest[path.name] = {"sha256": sha, "rows": len(kept),
                           "dropped_sample_ids": failed_ids}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] {path.name}: {len(kept)} rows ({len(failed_ids)} dropped)  "
          f"sha256={sha[:12]}...")
    # Per-seal durability (audit fix): push each shard the moment it seals.
    repo = os.environ.get("SHARD_PUSH_REPO")
    if repo:
        for f in (path, out_dir / "MANIFEST.json"):
            subprocess.Popen(
                ["hf", "upload", repo, str(f), f"corpus/{f.name}",
                 "--repo-type", "dataset"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[store] shard push spawned -> {repo}")


async def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    prompts_sha = hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()
    if "_prompts_sha256" in manifest and manifest["_prompts_sha256"] != prompts_sha:
        raise SystemExit(
            "REFUSING RESUME: prompts file differs from the one this corpus "
            "was generated from — sealed shards would mismatch sample_ids.")
    manifest.setdefault("_prompts_sha256", prompts_sha)

    prompts = [json.loads(l) for l in open(args.prompts)]
    for i, p in enumerate(prompts):
        p.setdefault("sample_id", i)

    shards = [
        prompts[i : i + args.shard_size] for i in range(0, len(prompts), args.shard_size)
    ]
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 16)
    async with aiohttp.ClientSession(connector=connector) as session:
        for shard_idx, shard in enumerate(shards):
            if f"shard_{shard_idx:04d}.parquet" in manifest:
                print(f"[SKIP] shard {shard_idx} already complete")
                continue
            tasks = [
                generate_one(session, sem, args.base_url, args.model, s, args.best_of)
                for s in shard
            ]
            rows = await asyncio.gather(*tasks)
            write_shard(list(rows), [s["sample_id"] for s in shard],
                        out_dir, shard_idx, manifest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--concurrency", type=int, default=112)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--best-of", type=int, default=1,
                    help="N candidates per prompt; judge keeps best. 1 = off. "
                         "N=4 costs ~4x generation tokens.")
    asyncio.run(main(ap.parse_args()))
