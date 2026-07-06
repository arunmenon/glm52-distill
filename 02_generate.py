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
# R1: a thinking teacher's CoT routinely exceeds 8192 on hard prompts; the old
# cap silently dropped exactly the hardest traces. Raise, and escalate once
# before giving up. Keep pack --max-len >= MAX_TOKENS (corpus_spec.md).
MAX_TOKENS = 16384
MAX_TOKENS_ESCALATED = 28672
TOP_P = 0.95
TEMPERATURE = 0.7   # default; per-slice overrides below (D6)
# D6: match sampling to the slice — tight for verifiable reasoning, warm for
# open-ended. `slice` was previously read only for labeling.
SLICE_TEMPS = {
    "coding": 0.4, "math": 0.3, "tool_calling": 0.4, "structured_science": 0.5,
    "general": 0.8, "long_context": 0.7, "multiturn": 0.8,
}
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
        pa.field("topk_coverage", pa.list_(pa.float32())),  # top-20 mass per pos
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
    import math
    content = (choice.get("logprobs") or {}).get("content") or []
    token_ids, topk_ids, topk_lps, coverage = [], [], [], []
    for pos in content:
        token_ids.append(parse_token_id(pos["token"]))
        ids_k, lps_k = [], []
        real_mass = 0.0
        for t in pos.get("top_logprobs", [])[:TOP_K]:
            ids_k.append(parse_token_id(t["token"]))
            lp = float(t["logprob"])
            lps_k.append(lp)
            real_mass += math.exp(lp)
        while len(ids_k) < TOP_K:
            ids_k.append(-1)
            lps_k.append(-1e9)
        topk_ids.append(ids_k)
        topk_lps.append(lps_k)
        # residual-mass telemetry: fraction of prob mass the top-20 captures.
        # <1 means the true distribution is wider than 20 here (high entropy).
        coverage.append(min(1.0, real_mass))
    return token_ids, topk_ids, topk_lps, coverage


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
        # R5: scan the ANSWER channel only. Falling back to reasoning_content
        # lets a verdict be read out of the judge's (possibly truncated)
        # thinking, recording a spurious pick as trusted.
        content = msg.get("content") or ""
        # Accept only if thinking closed (or none) AND there is a real answer.
        if content.strip() and ("</think>" in content or "<think>" not in content):
            verdict = strip_think(content)
            # R7: prefer a trailing bare number ("...\n2"); else last number.
            m = re.search(r"(\d+)\s*$", verdict.strip())
            nums = [m.group(1)] if m else re.findall(r"\d+", verdict)
            if nums:
                shown_idx = int(nums[-1]) - 1
                if 0 <= shown_idx < len(shown):
                    return order[shown_idx], True
        print(f"[judge-fallback] unparseable/truncated verdict: {content[:80]!r}",
              file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[judge-fallback] {e}", file=sys.stderr)
    return 0, False


def _dropped(sample, reason):
    """Sentinel telling write_shard why a sample was dropped, with slice for
    per-slice drop-reason telemetry (R2)."""
    return {"_dropped": reason, "sample_id": sample["sample_id"],
            "slice": sample.get("slice", "general")}


async def generate_one(session, sem, base_url, model, sample, best_of: int):
    gen_seed = GLOBAL_SEED + sample["sample_id"]
    slice_name = sample.get("slice", "general")
    temp = SLICE_TEMPS.get(slice_name, TEMPERATURE)   # D6

    def payload_for(max_tokens):
        return {
            "model": model,
            "messages": [{"role": "user", "content": sample["prompt"]}],
            "temperature": temp,
            "top_p": TOP_P,
            "max_tokens": max_tokens,
            "n": best_of,
            "logprobs": True,
            "top_logprobs": TOP_K,
            "seed": gen_seed,
        }

    async with sem:
        # R1: escalate the token budget once if EVERY candidate truncated,
        # before giving up on the (hardest) prompt.
        for max_tokens in (MAX_TOKENS, MAX_TOKENS_ESCALATED):
            for attempt in range(MAX_RETRIES):
                try:
                    data = await post_chat(session, base_url, payload_for(max_tokens))
                    cands = []
                    for ch in data["choices"]:
                        msg = ch["message"]
                        token_ids, topk_ids, topk_lps, cov = extract_logprobs(ch)
                        cands.append({
                            "response_text": msg.get("content") or "",
                            "reasoning_text": msg.get("reasoning_content") or "",
                            "response_token_ids": token_ids,
                            "topk_ids": topk_ids,
                            "topk_logprobs": topk_lps,
                            "topk_coverage": cov,
                            "finish_reason": ch.get("finish_reason", "unknown"),
                        })
                    any_truncated = any(c["finish_reason"] == "length" for c in cands)
                    # Rejection step 1: keep only complete candidates whose
                    # ANSWER (post-</think>) is non-empty (R6: a trace that
                    # reasoned but never answered must not be trained on).
                    clean = [
                        c for c in cands
                        if c["finish_reason"] == "stop"
                        and strip_think(c["response_text"]).strip()
                    ]
                    if not clean:
                        # all truncated -> escalate; all empty/other -> drop now
                        if any_truncated and max_tokens == MAX_TOKENS:
                            break  # retry outer loop with escalated budget
                        return _dropped(sample, "truncated" if any_truncated else "empty")
                    if len(clean) > 1:
                        pick, judged = await judge_pick(
                            session, base_url, model, sample["prompt"], clean,
                            seed=gen_seed + 1_000_000)
                    else:
                        pick, judged = 0, True
                    best = clean[pick]
                    return {
                        "sample_id": sample["sample_id"],
                        "slice": slice_name,
                        "prompt": sample["prompt"],
                        "response_text": best["response_text"],
                        "reasoning_text": best["reasoning_text"],
                        "response_token_ids": best["response_token_ids"],
                        "topk_ids": best["topk_ids"],
                        "topk_logprobs": best["topk_logprobs"],
                        "topk_coverage": best["topk_coverage"],
                        "finish_reason": best["finish_reason"],
                        "gen_seed": gen_seed,
                        "bon_n": best_of,
                        "bon_pick": pick,
                        "bon_judged": judged,
                    }
                except Exception as e:  # noqa: BLE001
                    if attempt == MAX_RETRIES - 1:
                        print(f"[FAIL] sample {sample['sample_id']}: {e}", file=sys.stderr)
                        return _dropped(sample, "error")
                    await asyncio.sleep(2**attempt)
        return _dropped(sample, "truncated")


SHARD_KEEP_FLOOR = float(os.environ.get("SHARD_KEEP_FLOOR", "0.5"))


def _atomic_write_json(path: Path, obj):
    """R4: write to .tmp then os.replace so a crash never leaves a torn file
    (the manifest is the resume gate — a torn manifest is un-resumable)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _update_funnel(out_dir: Path, kept, drops):
    """R2: persist a per-slice funnel (kept + drop-reason counts) so a narrowed
    corpus is detectable AFTER the run — not just an ephemeral stdout line."""
    fp = out_dir / "funnel.json"
    funnel = json.loads(fp.read_text()) if fp.exists() else {}
    for r in kept:
        s = funnel.setdefault(r["slice"], {"kept": 0})
        s["kept"] += 1
    for d in drops:
        s = funnel.setdefault(d["slice"], {"kept": 0})
        key = f"dropped_{d['_dropped']}"
        s[key] = s.get(key, 0) + 1
    _atomic_write_json(fp, funnel)


def write_shard(results, shard_ids, out_dir: Path, shard_idx: int, manifest: dict):
    kept = [r for r in results if r is not None and "_dropped" not in r]
    drops = [r for r in results if r is not None and "_dropped" in r]
    kept_ids = {r["sample_id"] for r in kept}
    failed = {}   # sample_id -> reason
    for d in drops:
        failed[d["sample_id"]] = d["_dropped"]
    for i in shard_ids:
        if i not in kept_ids and i not in failed:
            failed[i] = "missing"   # returned None (crash before sentinel)

    # R2: refuse to SEAL a shard that lost too much — a sustained outage would
    # otherwise permanently delete a shard's worth of prompts (sealed = skipped
    # on resume forever). Leave it unsealed for regeneration.
    total = max(len(shard_ids), 1)
    if len(kept) / total < SHARD_KEEP_FLOOR:
        print(f"[HOLD] shard {shard_idx}: kept {len(kept)}/{total} < floor "
              f"{SHARD_KEEP_FLOOR} — NOT sealing; will regenerate on resume.",
              file=sys.stderr)
        _update_funnel(out_dir, kept, drops)
        return False

    path = out_dir / f"shard_{shard_idx:04d}.parquet"
    tmp = out_dir / f".shard_{shard_idx:04d}.parquet.tmp"
    table = pa.Table.from_pylist(kept, schema=SCHEMA)
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(path)  # atomic: no half-written shard ever bears the final name
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    from collections import Counter
    manifest[path.name] = {
        "sha256": sha, "rows": len(kept),
        "dropped_sample_ids": sorted(failed),
        "drop_reasons": dict(Counter(failed.values())),
    }
    _atomic_write_json(out_dir / "MANIFEST.json", manifest)
    _update_funnel(out_dir, kept, drops)
    print(f"[OK] {path.name}: {len(kept)} rows ({len(failed)} dropped: "
          f"{dict(Counter(failed.values()))})  sha256={sha[:12]}...")
    # Per-seal durability (audit fix): push each shard the moment it seals.
    repo = os.environ.get("SHARD_PUSH_REPO")
    if repo:
        for f in (path, out_dir / "MANIFEST.json", out_dir / "funnel.json"):
            subprocess.Popen(
                ["hf", "upload", repo, str(f), f"corpus/{f.name}",
                 "--repo-type", "dataset"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[store] shard push spawned -> {repo}")
    return True


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
