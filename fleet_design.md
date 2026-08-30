# Fleet layer — required before any multi-box sweep campaign

Status: SPEC ONLY. Campaign 1 runs on ONE box. This layer gates campaign 2+.
Source: external review of the horizontal-scaling proposal (2026-08-30).

## Why per-plan safety does NOT compose across boxes

- Three conductors with $40 caps = a $120 fleet exposure; per-plan spend
  ledgers know nothing about each other.
- All conductors can pass the account-reserve check simultaneously against
  the same visible balance (shared-read race).
- Cross-box results are not comparable by concatenation: each box has its
  own baseline, environment, and floors.
- The conductor supports one trial per box, period. "More GPUs" means more
  independent single-GPU instances, never several trials inside one
  multi-GPU box.
- Real added cost of 3-way: ~2 extra baselines (~$2.40) + provisioning and
  data transfer per box; mix30 is the slowest partition, so balanced-case
  wall-time estimates are optimistic.

## Required components (all on the conductor VM)

1. **Campaign manifest**: parent record listing child plan digests and the
   exact expected union of trial IDs (12, disjoint).
2. **Fleet budget ledger** (flock-protected): each conductor RESERVES its
   pessimistic next-attempt cost before launch and SETTLES actual cost
   after. Fleet cap enforced on reserved+settled, not per-plan counters.
3. **Fleet preflight**: available credit >= fleet cap + reserve before any
   conductor starts.
4. **Merger** (replaces per-plan report for campaigns):
   - reject duplicate or missing configs vs the campaign manifest;
   - verify identical code/model/dataset/evaluator/gate identities across
     child manifests;
   - apply each box's own baseline-relative eligibility gate;
   - rank eligible trials by (metric - own box baseline);
   - REFUSE automatic ranking when baseline spread across boxes exceeds
     the gray band — human review instead;
   - aggregate actual fleet spend;
   - select the global top-2 ORIGINAL retained checkpoints (finals/ on
     their boxes), never retrained substitutes.
5. **Fleet HALT** (stops all conductors) plus per-plan HALTs.

## Sequencing

campaign 1 (12 trials, one box) -> smoke + real run clean ->
build fleet layer -> certify (review + smoke incl. two-box scenario) ->
campaign 2 (KD-alpha grid) on 2-4 boxes.

Note: the r3 conductor revision (namespace/lock/lease/spend) is implemented
but NOT yet adversarially certified; certification is the standing
precondition for any paid run, single-box included.
