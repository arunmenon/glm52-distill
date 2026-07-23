# Trajectory Task Specification

Decided 2026-07-22. The task-pool counterpart of corpus_spec.md, for the
agentic trajectory leg (agentic_trajectory_design.md). Governs WHICH tasks
are rolled out, at what depth, per phase. Grounded in measured source
composition (HF datasets-server, 2026-07-22): SWE-Gym is 2,438 instances
over 11 repos with pandas=30%, MONAI=15%, moto=14% (top-3 ~ 60%); SWE-smith
encodes generation strategy in instance_id; R2E-Gym carries difficulty
proxies (num_non_test_files / num_non_test_lines / modified_entity_summaries).

## 1. The two axes this spec controls

- **BREADTH**: repo, task type, statement style, interaction horizon.
- **DEPTH**: difficulty tiers with enforced quotas + adaptive rollout
  budgets, so success-filtering (design review T4) cannot silently strip
  the hard tail.

Language stays Python-only in v1 (design review T2, deliberate; multilingual
is a WALK+ decision).

## 2. Difficulty tiers (v1 proxy, computable at build time)

Tier by GOLD patch stats (SWE-bench schema: patch + FAIL_TO_PASS):

| Tier | Definition (any triggers the higher tier) |
|---|---|
| easy | 1 file AND <15 changed lines |
| medium | <=2 files AND 15-60 changed lines |
| hard | >=3 files OR >60 lines OR >5 F2P tests |

R2E-Gym: map from its native num_non_test_files/lines fields. The proxy is
for STRATIFIED SAMPLING only; the true difficulty signal is measured
teacher success rate, which the rehearsal calibrates and every phase's
funnel re-measures. Recalibrate tier boundaries once per phase, never
mid-phase (determinism).

## 3. Adaptive rollout policy (replaces flat N=4)

| Tier | N (max) | Early stop | Rollout-budget floor |
|---|---|---|---|
| easy | 2 | at first verified success | - |
| medium | 4 | at second verified success | - |
| hard | 6 | never early-stop | >=30% of phase rollout budget |

Corpus quota: **>=25% of VERIFIED trajectories must be hard-tier**. If the
hard-tier verified rate falls below 10% over a 50-rollout window, the leg
PAUSES for a human mix decision instead of silently backfilling with easy
(this is the T4 guard, enforced in code like the funnel drift halt).

Long-horizon depth quota: >=15% of the verified corpus with >=20 steps or
>=3 files touched. 32k-pack drop rate is reported BY TIER; a hard-tier
drop rate >25% triggers the windowing decision (design section 5), not a
silent loss.

## 4. Task-type mix (breadth)

| Type | Source / construction | CRAWL | WALK |
|---|---|---|---|
| bugfix (issue -> patch) | SWE-Gym(-Lite), R2E-Gym | 100% | 50% |
| mutation repair | SWE-smith procedural strategies (func_basic, combine_file, combine_module), parsed from instance_id; per-strategy cap 40% of the slice | - | 20% |
| PR-mirror (incl. feature-flavored) | SWE-smith pr_* strategies | - | 15% |
| test-writing (inverted task) | constructed from SWE-Gym instances: given issue + repo, WRITE a reproducing test. Verifier: agent's test FAILS on base, PASSES on gold-patched repo; agent diff may touch ONLY test paths (inverse of the anti-tamper rule) | - | 10% |
| degraded-statement overlay | programmatic statement degradation (strip repro steps, truncate to 1-2 sentences) applied across types; ONE variant per task, never both | - | ~15% of tasks (overlay, not a slice) |

RUN adds: R2E-Gym full, SWE-rebench freshness slice (post-teacher-cutoff
tasks, contamination-free by construction), the multilingual entry
decision, and verifier R&D for refactor/greenfield types (T1).

SWE-smith onboarding gate (WALK): mutation tasks can be degenerate;
spot-audit 20 instances PER STRATEGY (statement coherent? tests
meaningful?) before bulk inclusion, and record the audit in the store.

## 5. Repo caps (breadth, T3)

Per-repo cap: **<=12% of any phase's task set** (pandas at 30% and
top-3-at-60% get downsampled). Realized per-repo mix is a first-class
funnel.json field. Repo-level disjointness from SWE-bench Verified is
asserted by the decontam gate (SWE-Gym's 11 repos are disjoint from
Verified's 12 today; the gate makes it a check, not an assumption).

## 6. Phase task plans

**REHEARSAL (~10 tasks, ~$5)**: SWE-Gym-Lite, stratified 3-4 per tier
across >=6 repos, N=2 each. Purpose: measure success rate BY TIER (the
unmeasured prior from review round 2), calibrate adaptive-N and tier
boundaries, reprice CRAWL. Output: rehearsal_report.json.

**CRAWL (50 tasks -> ~200 adaptive rollouts, $150 cap)**: SWE-Gym-Lite,
pure bugfix. Stratified: tier mix 40/40/20 (easy/medium/hard by count;
hard gets its 30% rollout-budget floor), per-repo <=12%, seed 42.
Stretch goal ONLY if budget remains: 5-task test-writing pilot to de-risk
the WALK verifier variant.

**WALK (600-900 tasks -> 1-2k verified)**: full mix table above; SWE-Gym
full + SWE-smith (audited strategies); repriced from CRAWL's measured
funnel before the budget freezes into experiments.yaml.

**RUN**: production mix per the design doc, plus the additions above.

## 7. Sampling hygiene (inherited from corpus_spec.md, adapted)

1. Deterministic stratified sampling, seed 42, per phase; task lists are
   frozen artifacts pushed to the store before generation starts.
2. Decontam gate (instance + text + repo level) runs on the frozen task
   list, not on the fly.
3. Funnel per (source, repo, tier, task_type): tasks -> images pulled ->
   rollouts -> submitted -> verified -> reproduced -> packed, with drop
   reasons. Shortfall vs quota is a hard warning at phase end.
4. A task appears in at most ONE phase (no CRAWL tasks reused in WALK;
   cheap to enforce, keeps phase metrics independent).
