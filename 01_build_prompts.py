#!/usr/bin/env python3
"""
01_build_prompts.py — build the prompt corpus + stratified held-out set.

v2 (corpus_spec.md): multi-source slices with per-source caps, buffered
stream shuffling (never head-of-stream), benchmark decontamination in the
BASE build, round-robin cross-slice dedup, and a per-slice funnel report.

Slices (see corpus_spec.md for rationale):
  coding 40% | math 20% | general 20% | tool_calling 5% | long_context 5%
  structured_science 5% | multiturn 5% (loader pending 02/03 message support;
  its share is redistributed to coding until then — see MULTITURN_READY)

Steps: shuffled-stream load -> length filter -> benchmark screen ->
round-robin MinHash dedup -> trim -> stratified held-out -> JSONL + MANIFEST.

Add your own domain prompts via --extra my_prompts.jsonl (fields: prompt, slice).
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

SEED = 42
MAX_PROMPT_CHARS = 12000   # ~<= 4k tokens for most tokenizers
LONGCTX_MAX_CHARS = 120000  # long-context slice ceiling (~32k tokens)
MIN_PROMPT_CHARS = 20
SHORT_TOKENS_EXACT_ONLY = 15   # below this, MinHash unreliable -> exact dedup only
SHUFFLE_BUFFER = 100_000

MULTITURN_READY = False  # flips when 02/03 support message lists

# benchmarks scored in 07_eval_benchmarks.sh + classic coding sets — keep in
# sync with 01b_extend_prompts.BENCHMARKS when editing
BENCHMARKS = [
    ("gsm8k", "openai/gsm8k", "main", "test", lambda r: r.get("question", "")),
    ("math500", "HuggingFaceH4/MATH-500", None, "test", lambda r: r.get("problem", "")),
    ("ifeval", "google/IFEval", None, "train", lambda r: r.get("prompt", "")),
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", lambda r: r.get("question", "")),
    ("humaneval", "openai/openai_humaneval", None, "test", lambda r: r.get("prompt", "")),
    ("mbpp", "google-research-datasets/mbpp", None, "test", lambda r: r.get("text", "")),
    # LiveCodeBench (review D3): coding is the largest slice and LCB is a scored
    # eval; without this screen, train/test leakage inflates the coding delta.
    ("livecodebench", "livecodebench/code_generation_lite", "release_v5", "test",
     lambda r: _first_of(r, "question_content", "question", "prompt")),
]


def _first_of(row, *fields):
    for f in fields:
        v = row.get(f)
        if v:
            return v
    return ""


def _hermes_first_user(row):
    """First human/user turn from hermes-function-calling conversations."""
    for m in row.get("conversations") or []:
        if m.get("from") in ("human", "user"):
            return m.get("value", "")
    return ""


def _json_mode_user(row):
    """User turn from json-mode-eval (D2: prompt is a message list [system-schema,
    user-request]; take the USER request, not the system schema boilerplate)."""
    p = row.get("prompt")
    if isinstance(p, list):
        for m in p:
            if m.get("role") == "user":
                return m.get("content", "")
        return p[-1].get("content", "") if p else ""
    return _first_of(row, "prompt", "question")


def _tulu_general(row):
    """Tulu first user turn, EXCLUDING math/code/safety subsets (mix distortion
    + unaudited jailbreak prompts; corpus_spec.md)."""
    rid = (row.get("id") or "").lower()
    banned = ("persona_math", "numinamath", "math_grade", "evol_codealpaca",
              "persona_python", "code_alpaca", "wildjailbreak", "wildguard",
              "coconot")
    if any(b in rid for b in banned):
        return ""
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# slice -> {share, sources: [(name, repo, config, split, extract, cap_frac)]}
# cap_frac limits any one source's contribution to its slice (corpus_spec.md).
SLICES = {
    "coding": {"share": 0.40, "sources": [
        ("opc_sft", "OpenCoder-LLM/opc-sft-stage2", "educational_instruct", "train",
         lambda r: _first_of(r, "instruction", "question", "prompt"), 0.20),
        ("codefeedback", "m-a-p/CodeFeedback-Filtered-Instruction", None, "train",
         lambda r: _first_of(r, "query", "instruction"), 0.20),
        ("magicoder_oss", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train",
         lambda r: r.get("problem", ""), 0.20),
        ("opencodeinstruct", "nvidia/OpenCodeInstruct", None, "train",
         lambda r: _first_of(r, "input", "instruction"), 0.20),
        ("codefeedback2", "m-a-p/Code-Feedback", None, "train",
         lambda r: (r.get("messages") or [{}])[0].get("content", ""), 0.20),
    ]},
    "math": {"share": 0.20, "sources": [
        ("numina", "AI-MO/NuminaMath-CoT", None, "train",
         lambda r: r.get("problem", ""), 0.50),
        ("openmath2", "nvidia/OpenMathInstruct-2", None, "train",
         lambda r: r.get("problem", ""), 0.50),
    ]},
    "general": {"share": 0.20, "sources": [
        ("tulu", "allenai/tulu-3-sft-mixture", None, "train", _tulu_general, 1.0),
    ]},
    "tool_calling": {"share": 0.05, "sources": [
        ("hermes_fc", "NousResearch/hermes-function-calling-v1", None, "train",
         lambda r: _hermes_first_user(r), 0.60),
        ("glaive", "glaiveai/glaive-function-calling-v2", None, "train",
         lambda r: _first_of(r, "question", "prompt", "system"), 0.60),
    ]},
    "long_context": {"share": 0.05, "max_chars": LONGCTX_MAX_CHARS, "sources": [
        ("longalign", "THUDM/LongAlign-10k", None, "train",
         lambda r: (r.get("messages") or [{}])[0].get("content", ""), 0.60),
        ("longwriter", "THUDM/LongWriter-6k", None, "train",
         lambda r: (r.get("messages") or [{}])[0].get("content", ""), 0.60),
    ]},
    "structured_science": {"share": 0.05, "sources": [
        # D2: a real structured-OUTPUT source (JSON/schema/extraction), extractor
        # fixed to pull the USER turn (was grabbing the system schema boilerplate).
        # json-mode-eval is a small (~hundreds) real source. IMPORTANT (review D2):
        # the pre-Tier-3 smoke test MUST inspect realized_source_mix in
        # funnel.json — if json_mode is <~25% of this slice at scale, swap in a
        # higher-volume structured-output source (verify it loads first).
        ("json_mode", "NousResearch/json-mode-eval", None, "train",
         _json_mode_user, 0.30),
        ("webinstruct", "TIGER-Lab/WebInstruct-verified", None, "train",
         lambda r: _first_of(r, "question", "instruction", "prompt"), 0.45),
        ("sciq_style", "allenai/sciq", None, "train",
         lambda r: r.get("question", ""), 0.25),   # cap the MC-factoid source
    ]},
    # multiturn 5%: pending MULTITURN_READY; share redistributed in slice_targets
}


def slice_targets(n_total: int) -> dict:
    shares = {k: v["share"] for k, v in SLICES.items()}
    if not MULTITURN_READY:
        # D1: spread the parked multi-turn 5% across the NON-coding capability
        # slices (not dumped on coding, which was silently pushing it to 45%).
        spread = {"general": 0.02, "tool_calling": 0.01,
                  "long_context": 0.01, "structured_science": 0.01}
        for k, add in spread.items():
            shares[k] += add
    total_share = sum(shares.values())
    targets = {k: int(n_total * s / total_share) for k, s in shares.items()}
    # remainder to the largest slice (audit fix: no silent flooring loss)
    targets[max(targets, key=targets.get)] += n_total - sum(targets.values())
    return targets


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def norm_nonum(text: str) -> str:
    return re.sub(r"\d+", "N", norm(text))


def minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=SEED)
    toks = norm(text).split()
    for i in range(max(1, len(toks) - 2)):
        m.update(" ".join(toks[i : i + 3]).encode())
    return m


def first_user_turn(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def load_benchmark_index():
    exact, nonum = set(), set()
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    n = 0
    for name, repo, config, split, extract in BENCHMARKS:
        try:
            ds = load_dataset(repo, config, split=split, trust_remote_code=True) \
                if config else load_dataset(repo, split=split, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"benchmark decontam REQUIRES {name} ({repo}): {e}\n"
                "If this benchmark genuinely cannot load, remove it from "
                "BENCHMARKS AND from 07_eval_benchmarks.sh TASKS together — "
                "never score a benchmark you don't decontaminate against.")
        texts = [extract(r) for r in ds if extract(r)]
        for t in texts:
            exact.add(hashlib.md5(norm(t).encode()).hexdigest())
            nonum.add(hashlib.md5(norm_nonum(t).encode()).hexdigest())
            lsh.insert(f"bench{n}", minhash(t))
            n += 1
        print(f"benchmark decontam: {name} -> {len(texts)} items")
    return exact, nonum, lsh


def load_slices(over: dict, consumed_out: dict | None = None) -> dict:
    """Collect `over[slice]` clean candidates per slice from SHUFFLED streams,
    respecting per-source caps. Records raw rows consumed per source."""
    pools = {}
    for slice_name, cfg in SLICES.items():
        if slice_name not in over:
            continue
        want = over[slice_name]
        max_chars = cfg.get("max_chars", MAX_PROMPT_CHARS)
        pool = []
        for (name, repo, config, split, extract, cap_frac) in cfg["sources"]:
            if len(pool) >= want:
                break
            budget = min(max(1, int(want * cap_frac)), want - len(pool))
            got, seen, consumed = [], set(), 0
            try:
                ds = (load_dataset(repo, config, split=split, streaming=True)
                      if config else load_dataset(repo, split=split, streaming=True))
                ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
                for row in ds:
                    consumed += 1
                    p = extract(row)
                    if not p or not isinstance(p, str):
                        continue
                    if len(p) > max_chars or len(p) < MIN_PROMPT_CHARS:
                        continue
                    h = hashlib.md5(norm(p).encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    got.append({"prompt": p, "slice": slice_name, "source": name})
                    if len(got) >= budget:
                        break
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: source {name} ({repo}) failed: {e} — continuing")
            if consumed_out is not None:
                consumed_out[name] = consumed
            print(f"  {slice_name}/{name}: {len(got)} collected "
                  f"(consumed {consumed} rows)")
            pool.extend(got)
        pools[slice_name] = pool
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/")
    ap.add_argument("--n-total", type=int, default=150000)
    ap.add_argument("--n-heldout", type=int, default=1000)
    ap.add_argument("--extra", default=None,
                    help="jsonl with {prompt, slice}; slice field REQUIRED")
    ap.add_argument("--skip-benchmark-decontam", action="store_true",
                    help="debugging ONLY — never for production")
    args = ap.parse_args()

    random.seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = slice_targets(args.n_total)
    over = {k: int(v * 1.35) for k, v in targets.items()}

    funnel = {k: {} for k in targets}
    consumed = {}
    pools = load_slices(over, consumed_out=consumed)
    for k in targets:
        funnel[k]["collected"] = len(pools.get(k, []))

    extra = []
    if args.extra:
        extra = [json.loads(l) for l in open(args.extra)]
        for e in extra:
            assert "slice" in e, "--extra rows must carry an explicit slice field"
            e.setdefault("source", "extra")

    if args.skip_benchmark_decontam:
        print("WARNING: benchmark decontamination SKIPPED — not production-safe")
        bench_exact, bench_nonum, bench_lsh = set(), set(), None
    else:
        bench_exact, bench_nonum, bench_lsh = load_benchmark_index()

    # Per-source cap on the POST-DEDUP kept set (D7): collection-time caps are
    # upper bounds BEFORE dedup, so the realized source mix can skew. Enforce
    # the cap on what actually survives, per slice.
    src_cap = {}
    for k in targets:
        srcs = SLICES.get(k, {}).get("sources", [])
        for (name, *_rest, cap_frac) in srcs:
            src_cap[(k, name)] = max(1, int(targets[k] * cap_frac))

    # Round-robin cross-slice dedup (audit fix: no fixed-order slice bias).
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    exact_seen = set()
    kept = {k: [] for k in targets}
    src_kept = {k: {} for k in targets}       # realized per-source counts
    n_bench = {k: 0 for k in targets}
    n_dup = {k: 0 for k in targets}
    n_capped = {k: 0 for k in targets}
    queues = {k: [e for e in extra if e["slice"] == k] + pools.get(k, [])
              for k in targets}
    idx = {k: 0 for k in targets}
    active = True
    while active:
        active = False
        for k in targets:
            q = queues[k]
            while idx[k] < len(q) and len(kept[k]) < targets[k]:
                item = q[idx[k]]
                idx[k] += 1
                active = True
                p = item["prompt"]
                src = item.get("source", "extra")
                # post-dedup per-source cap (skip, don't drop — a later source
                # may still fill the slice)
                cap = src_cap.get((k, src))
                if cap is not None and src_kept[k].get(src, 0) >= cap:
                    n_capped[k] += 1
                    continue
                h = hashlib.md5(norm(p).encode()).hexdigest()
                if (h in bench_exact
                        or hashlib.md5(norm_nonum(p).encode()).hexdigest() in bench_nonum):
                    n_bench[k] += 1
                    continue
                if h in exact_seen:
                    n_dup[k] += 1
                    continue
                short = len(norm(p).split()) < SHORT_TOKENS_EXACT_ONLY
                m = None
                if not short:
                    m = minhash(p)
                    hits = lsh.query(m)
                    if hits:
                        n_dup[k] += 1
                        continue
                    if bench_lsh is not None and bench_lsh.query(m):
                        n_bench[k] += 1
                        continue
                exact_seen.add(h)
                if m is not None:
                    lsh.insert(f"{k}{len(kept[k])}", m)
                kept[k].append(item)
                src_kept[k][src] = src_kept[k].get(src, 0) + 1
                break   # round-robin: next slice after each keep

    # D7 backfill: caps are anti-monoculture ceilings, not hard floors. If a
    # slice finished SHORT of target (a source under-delivered after dedup and
    # no other source could compensate because each was pinned at its cap),
    # fill the remainder from that slice's leftover, cap-ignoring but still
    # dedup'd — coverage beats a perfectly-balanced-but-starved slice.
    n_backfill = {k: 0 for k in targets}
    for k in targets:
        while len(kept[k]) < targets[k] and idx[k] < len(queues[k]):
            item = queues[k][idx[k]]
            idx[k] += 1
            p = item["prompt"]
            h = hashlib.md5(norm(p).encode()).hexdigest()
            if (h in bench_exact
                    or hashlib.md5(norm_nonum(p).encode()).hexdigest() in bench_nonum
                    or h in exact_seen):
                continue
            short = len(norm(p).split()) < SHORT_TOKENS_EXACT_ONLY
            m = None
            if not short:
                m = minhash(p)
                if lsh.query(m) or (bench_lsh is not None and bench_lsh.query(m)):
                    continue
            exact_seen.add(h)
            if m is not None:
                lsh.insert(f"{k}bf{len(kept[k])}", m)
            kept[k].append(item)
            src_kept[k][item.get("source", "extra")] = \
                src_kept[k].get(item.get("source", "extra"), 0) + 1
            n_backfill[k] += 1

    for k in targets:
        nk = max(len(kept[k]), 1)
        funnel[k].update({
            "kept": len(kept[k]), "target": targets[k],
            "dropped_dup": n_dup[k], "dropped_benchmark": n_bench[k],
            "dropped_source_cap": n_capped[k],
            "backfilled": n_backfill[k],
            "realized_source_mix": {s: round(c / nk, 3)
                                    for s, c in src_kept[k].items()},
        })
        if len(kept[k]) < 0.98 * targets[k]:
            print(f"WARNING: slice {k} short: {len(kept[k])}/{targets[k]}")

    final = [x for sl in kept.values() for x in sl]
    random.shuffle(final)

    # stratified held-out from ACTUAL kept counts (audit fix)
    total_kept = len(final)
    per_slice_ho = {k: int(args.n_heldout * len(kept[k]) / max(total_kept, 1))
                    for k in targets}
    per_slice_ho[max(per_slice_ho, key=per_slice_ho.get)] += \
        args.n_heldout - sum(per_slice_ho.values())
    heldout, train = [], []
    counts = {k: 0 for k in targets}
    for x in final:
        if counts[x["slice"]] < per_slice_ho[x["slice"]]:
            counts[x["slice"]] += 1
            heldout.append(x)
        else:
            train.append(x)

    for i, x in enumerate(train):
        x["sample_id"] = i
    for i, x in enumerate(heldout):
        x["sample_id"] = i

    manifest = {"funnel": funnel, "source_rows_consumed": consumed,
                "benchmark_decontam": not args.skip_benchmark_decontam}
    for name, rows in [("prompts.jsonl", train), ("prompts_heldout.jsonl", heldout)]:
        p = out_dir / name
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "rows": len(rows),
        }
        print(f"wrote {p} ({len(rows)} rows)")

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("FUNNEL:", json.dumps(funnel, indent=2))


if __name__ == "__main__":
    main()
