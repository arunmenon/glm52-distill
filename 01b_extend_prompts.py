#!/usr/bin/env python3
"""
01b_extend_prompts.py — build ADDITIONAL prompts guaranteed disjoint from an
existing prompt set and its held-out set (the scale-up step).

Contamination guarantees:
  1. exact disjointness: any candidate whose normalized-text hash appears in
     the base train OR held-out set is dropped (same normalization as the
     hard gate in 03_pack_dataset.py)
  2. near-duplicate disjointness: MinHash LSH at the same 0.8 threshold used
     inside 01_build_prompts.py, seeded with every base prompt — candidates
     that are near-copies of base prompts (or of each other) are dropped
  3. id/seed disjointness: sample_ids start at --id-offset (default 100000),
     so per-sample generation seeds and corpus rows never collide with the
     base run
  4. the base held-out set remains THE eval set; this script never creates a
     second one (comparability across runs), and 03_pack_dataset.py's hard
     contamination gate remains the final backstop at pack time
  5. benchmark decontamination: candidates matching (exactly or near, same
     0.8 MinHash threshold) any test item of the benchmarks scored in
     07_eval_benchmarks.sh (GSM8K, IFEval, MMLU-Pro) are dropped — otherwise
     benchmark deltas measured on the distilled student are untrustworthy.
     Loading the benchmark sets is REQUIRED; use --skip-benchmark-decontam
     only for offline debugging, never for a production build

Usage:
  python 01b_extend_prompts.py --base pilot_teacher/data --n-total 25000
  # -> pilot_teacher/data/prompts_ext.jsonl
  python 02_generate.py --prompts pilot_teacher/data/prompts_ext.jsonl \
      --out pilot_teacher/corpus_ext ...
  python 03_pack_dataset.py --corpus pilot_teacher/corpus pilot_teacher/corpus_ext ...
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bp", Path(__file__).parent / "01_build_prompts.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

from datasketch import MinHashLSH  # noqa: E402


def norm_hash(text: str) -> str:
    return hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()


# single source of truth for the benchmark list: the base builder
BENCHMARKS = bp.BENCHMARKS


def load_benchmark_prompts():
    from datasets import load_dataset
    texts = []
    for name, repo, config, split, extract in BENCHMARKS:
        ds = load_dataset(repo, config, split=split) if config \
            else load_dataset(repo, split=split)
        t = [extract(r) for r in ds if extract(r)]
        print(f"benchmark decontam: {name} -> {len(t)} test items loaded")
        texts.extend(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="dir with the existing prompts.jsonl + prompts_heldout.jsonl")
    ap.add_argument("--out", default=None, help="output dir (default: --base)")
    ap.add_argument("--n-total", type=int, required=True,
                    help="how many NEW prompts to produce")
    ap.add_argument("--id-offset", type=int, default=100000)
    ap.add_argument("--skip-benchmark-decontam", action="store_true",
                    help="offline debugging ONLY — never for a production build")
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out) if args.out else base
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = []
    for name in ("prompts.jsonl", "prompts_heldout.jsonl"):
        base_rows.extend(json.loads(l) for l in open(base / name))
    print(f"base set: {len(base_rows)} prompts (train + held-out)")

    base_hashes = {norm_hash(r["prompt"]) for r in base_rows}
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    for i, r in enumerate(base_rows):
        lsh.insert(f"base{i}", bp.minhash(r["prompt"]))

    bench_hashes = set()
    if args.skip_benchmark_decontam:
        print("WARNING: benchmark decontamination SKIPPED — do not train a "
              "production model on this set")
    else:
        bench_texts = load_benchmark_prompts()
        bench_hashes = {norm_hash(t) for t in bench_texts}
        for i, t in enumerate(bench_texts):
            lsh.insert(f"bench{i}", bp.minhash(t))

    targets = bp.slice_targets(args.n_total)
    # extra headroom: base-overlap and near-dups will eat some candidates
    over = {k: int(v * 1.4) for k, v in targets.items()}
    # D4: shift the shuffle seed so the extension draws a DIFFERENT stream
    # window than the base run (else every candidate is a base duplicate).
    _orig_seed = bp.SEED
    bp.SEED = _orig_seed + args.id_offset
    pools = bp.load_slices(over)
    bp.SEED = _orig_seed

    kept = {k: [] for k in targets}
    n_exact, n_near, n_bench = 0, 0, 0
    for slice_name, pool in pools.items():
        for item in pool:
            if len(kept[slice_name]) >= targets[slice_name]:
                break
            h = norm_hash(item["prompt"])
            if h in bench_hashes:
                n_bench += 1
                continue
            if h in base_hashes:
                n_exact += 1
                continue
            m = bp.minhash(item["prompt"])
            hits = lsh.query(m)
            if hits:
                if any(k.startswith("bench") for k in hits):
                    n_bench += 1
                else:
                    n_near += 1
                continue
            lsh.insert(f"ext{slice_name}{len(kept[slice_name])}", m)
            kept[slice_name].append(item)

    final = [x for sl in kept.values() for x in sl]
    random.Random(bp.SEED + 1).shuffle(final)
    for i, x in enumerate(final):
        x["sample_id"] = args.id_offset + i

    dest = out_dir / "prompts_ext.jsonl"
    with open(dest, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["prompts_ext.jsonl"] = {
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "rows": len(final),
        "id_offset": args.id_offset,
        "excluded_exact_dups_of_base": n_exact,
        "excluded_near_dups": n_near,
        "excluded_benchmark_contamination": n_bench,
        "benchmark_decontam": not args.skip_benchmark_decontam,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {dest}: {len(final)} rows "
          f"({ {k: len(v) for k, v in kept.items()} })")
    print(f"excluded: {n_exact} exact dups of base, {n_near} near-dups, "
          f"{n_bench} benchmark-contaminated")
    for k, v in kept.items():
        if len(v) < targets[k]:
            print(f"WARNING: {k} short of target ({len(v)}/{targets[k]}) — "
                  f"raise the oversample factor or source more data")


if __name__ == "__main__":
    main()
