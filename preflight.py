#!/usr/bin/env python3
"""
preflight.py — validate experiments.yaml and expand the grid into a deterministic
trial list BEFORE any spend. Guardrail #7 + the alpha>0 => GLM-only contract.

Emits runs/<sweep>/trials.json (ordered, content-addressed experiment_ids).
Fails LOUD on any validation error — the conductor refuses to proceed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import yaml


def experiment_id(cfg: dict) -> str:
    key = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def est_cost(cfg, plan):
    # crude, conservative: student pod $8.36/hr; ~ (rows/1000 * epochs * rank/64)
    # * 0.04 hr/unit for train + 0.3 hr eval. Calibrate in cost.py later.
    rows = plan["rung"]["proxy_rows"] if cfg["rung"] == 0 else cfg["_full_rows"]
    train_hr = (rows / 1000) * cfg["epochs"] * (cfg["lora_rank"] / 64) * 0.04
    return round((train_hr + 0.3) * 8.36, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="experiments.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--full-rows", type=int, default=1140,
                    help="row count of the full corpus (for cost + promotion)")
    args = ap.parse_args()

    plan = yaml.safe_load(open(args.plan))
    errors = []

    # --- schema / value checks ---
    for k in ("sweep", "budget", "student", "corpus", "eval", "anchor", "rung", "grid"):
        if k not in plan:
            errors.append(f"missing top-level key: {k}")
    fam = plan.get("student", {}).get("family", "")
    grid = plan.get("grid", {})

    # the load-bearing contract: logit KD (alpha>0) needs a GLM-family student
    # (topk columns only exist for the token-exact/remap path in 03_pack_dataset).
    if fam != "glm" and any(a != 0.0 for a in grid.get("alpha", [0.0])):
        errors.append(f"student family '{fam}' is non-GLM but grid has alpha>0 "
                      "(logit KD requires a GLM-family student; use alpha=0)")

    b = plan.get("budget", {})
    if b.get("reserve_usd", 0) >= b.get("hard_cap_usd", 0):
        errors.append("reserve_usd must be < hard_cap_usd")

    if errors:
        raise SystemExit("PREFLIGHT FAILED:\n  - " + "\n  - ".join(errors))

    # --- expand grid -> deterministic ordered trials ---
    keys = ["alpha", "lr", "lora_rank", "epochs"]
    combos = list(itertools.product(*[grid[k] for k in keys]))
    trials = []
    total = 0.0
    for combo in combos:
        cfg = dict(zip(keys, combo))
        cfg["rung"] = 0
        cfg["_full_rows"] = args.full_rows
        cfg["experiment_id"] = experiment_id({k: cfg[k] for k in keys})
        cfg["est_cost"] = est_cost(cfg, plan)
        total += cfg["est_cost"]
        cap = b.get("per_trial_cap_usd", 1e9)
        if cfg["est_cost"] > cap:
            raise SystemExit(f"trial {cfg['experiment_id']} est ${cfg['est_cost']} "
                             f"> per_trial_cap ${cap}")
        trials.append(cfg)
    # sort deterministically by experiment_id so scheduling is reproducible
    trials.sort(key=lambda c: c["experiment_id"])

    out = Path(args.out or f"runs/{plan['sweep']}/trials.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sweep": plan["sweep"], "trials": trials,
                               "projected_rung0_cost": round(total, 2)}, indent=2))
    print(f"PREFLIGHT OK: {len(trials)} rung-0 trials, projected ${total:.2f} "
          f"(promote top {plan['rung']['keep_top']} to full). -> {out}")
    for c in trials:
        print(f"  {c['experiment_id']}  alpha={c['alpha']} lr={c['lr']} "
              f"rank={c['lora_rank']} ep={c['epochs']}  ~${c['est_cost']}")


if __name__ == "__main__":
    main()
