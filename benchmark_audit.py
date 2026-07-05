#!/usr/bin/env python3
"""
benchmark_audit.py — retroactive benchmark-contamination scan of an existing
corpus (or prompt file) against the benchmark test sets in 01b's BENCHMARKS.

Three detectors per prompt:
  exact  — normalized-text hash equality
  nonum  — equality after stripping digits (catches GSM8K template
           re-instantiations: "Janet has 5 apples" vs "Maria has 7 apples")
  near   — MinHash LSH at the same 0.8 threshold as the builders

Output: JSON report with per-row verdicts; feed it to 03_pack_dataset.py
--exclude-report to drop contaminated rows at pack time.

  python benchmark_audit.py --corpus pilot_teacher/corpus --out contamination_report.json
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasketch import MinHashLSH

_spec = importlib.util.spec_from_file_location(
    "b1", Path(__file__).parent / "01b_extend_prompts.py")
b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b1)
bp = b1.bp  # 01_build_prompts module (minhash, SEED)


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def nonum(text: str) -> str:
    return re.sub(r"\d+", "N", norm(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*", default=[],
                    help="corpus dirs with shard_*.parquet (scans the prompt column)")
    ap.add_argument("--prompts", nargs="*", default=[],
                    help="prompt jsonl files to scan")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench = b1.load_benchmark_prompts()
    exact = {hashlib.md5(norm(t).encode()).hexdigest() for t in bench}
    nodig = {hashlib.md5(nonum(t).encode()).hexdigest() for t in bench}
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    for i, t in enumerate(bench):
        lsh.insert(f"b{i}", bp.minhash(t))
    print(f"benchmark index: {len(bench)} test items")

    items = []
    for d in args.corpus:
        for shard in sorted(glob.glob(str(Path(d) / "shard_*.parquet"))):
            t = pq.read_table(shard, columns=["sample_id", "prompt", "slice"])
            items.extend(t.to_pylist())
    for f in args.prompts:
        items.extend(json.loads(l) for l in open(f))
    print(f"scanning {len(items)} prompts")

    contaminated = []
    for r in items:
        p = r["prompt"]
        kinds = []
        if hashlib.md5(norm(p).encode()).hexdigest() in exact:
            kinds.append("exact")
        if hashlib.md5(nonum(p).encode()).hexdigest() in nodig:
            kinds.append("nonum")
        if not kinds and lsh.query(bp.minhash(p)):
            kinds.append("near")
        if kinds:
            contaminated.append({"sample_id": r.get("sample_id"),
                                 "slice": r.get("slice"), "kinds": kinds,
                                 "prompt_head": p[:120]})

    report = {
        "scanned": len(items),
        "contaminated_count": len(contaminated),
        "contaminated_frac": round(len(contaminated) / max(len(items), 1), 4),
        "by_kind": {k: sum(1 for c in contaminated if k in c["kinds"])
                    for k in ("exact", "nonum", "near")},
        "contaminated": contaminated,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "contaminated"},
                     indent=2))


if __name__ == "__main__":
    main()
