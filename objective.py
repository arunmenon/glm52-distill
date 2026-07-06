#!/usr/bin/env python3
"""
objective.py — turn a trial's eval outputs into a single comparable scalar,
versus the base anchor. score_fn_version keeps scores comparable across a sweep
(determinism contract). Reads lm-eval results + the IFEval think-stripped rescore.
"""
import argparse
import glob
import json
from pathlib import Path

SCORE_FN_VERSION = "v1"


def read_bench(outdir: str) -> dict:
    """Extract the metrics we score from an lm-eval OUTDIR tree."""
    m = {}
    for path in glob.glob(f"{outdir}/*/*/results*.json") + glob.glob(f"{outdir}/*/results*.json"):
        d = json.load(open(path))
        for task, res in d.get("results", {}).items():
            if "gsm8k" in task:
                m["gsm8k_flexible"] = res.get("exact_match,flexible-extract")
                m["gsm8k_strict"] = res.get("exact_match,strict-match")
            if "ifeval" in task:
                m["ifeval_strict_raw"] = res.get("prompt_level_strict_acc,none")
    # think-stripped IFEval (the trustworthy number, Discovery #6)
    for f in glob.glob(f"{outdir}/*_ifeval_thinkstripped.json"):
        try:
            j = json.load(open(f))
            m["ifeval_strict_stripped"] = j.get("prompt_level_strict_acc")
        except Exception:
            pass
    return {k: v for k, v in m.items() if v is not None}


def score(metrics: dict, anchor: dict | None) -> float:
    """Scalar objective: mean of (metric deltas vs anchor) over the screen set.
    Absolute if no anchor. Higher = better."""
    keys = ["gsm8k_flexible", "gsm8k_strict", "ifeval_strict_stripped"]
    vals = []
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if anchor and k in anchor:
                v = v - anchor[k]        # delta vs base
            vals.append(v)
    return round(sum(vals) / len(vals), 4) if vals else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--anchor", default=None, help="anchor metrics json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    metrics = read_bench(args.bench_dir)
    anchor = json.load(open(args.anchor)) if args.anchor and Path(args.anchor).exists() else None
    s = score(metrics, anchor)
    Path(args.out).write_text(json.dumps({
        "score_fn_version": SCORE_FN_VERSION, "metrics": metrics,
        "anchor": anchor, "score": s}, indent=2))
    print(json.dumps({"score": s, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
