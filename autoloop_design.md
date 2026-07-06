# Autonomous Distillation Loop — Design (2026-07-06)

Design workflow: 3 architectures (deterministic-grid / llm-orchestrator / bandit-optimizer) x 3 adversarial judge lenses -> synthesis. Ranking: deterministic-grid 8.0, llm-orchestrator 7.7, bandit 7.7.

# FINAL DESIGN — Autonomous, Guardrailed, Cost-Capped Distillation Loop for `glm52-distill`

Grounded in the actual repo: `common.sh` (`mark`/`done_p`, `launch_vllm`, `wait_ready_and_smoke`, `ensure_glm_env`/`ensure_pod_env`, `store_push`/`store_pull`, `PILOT_STORE`), `watchdog.sh` (on-pod `podStop` dead-man), `deploy_netvol.sh` (netvol-first, `allowedCudaVersions:["13.0"]`, `runpod/pytorch:...cu1281`), `02_generate.py` (`SHARD_KEEP_FLOOR=0.5`, `_atomic_write_json`, `MANIFEST._prompts_sha256` resume guard, `funnel.json`), `03_pack_dataset.py` (held-out hard-fail, Discovery #6 packer), `04_train_kd.py` (α=0 SFT / α>0 logit-KD, `set_seed`), `05/06/07` evals, and journal Discoveries #4 (0/1,140 contamination), #5 (Air needs `ds_zero3_4gpu_offload.yaml`), #6 (broken eval config, not data), #7 (trustworthy +48pt GSM8K-strict at ~$19, anchor-gated staged spend).

---

## 1. RECOMMENDED ARCHITECTURE

**Primary philosophy: the `deterministic-grid` conductor is the SPINE** (judge-ranked 8.0, and correctly so for an unmanned + determinism + cost mandate). Onto that spine graft exactly two things and demote a third:

- **GRAFT from `bandit-optimizer`: multi-fidelity successive-halving (ASHA-style rungs), but WITHOUT the TPE/Bayesian surrogate.** The valuable, budget-saving part of BOHB is the *scheduler* (train cheap-proxy rungs, kill losers before full runs), which is fully deterministic. The surrogate is not worth it: by that design's own admission (weakness #5), TPE over a conditional mixed space at the few-dozen top-rung observations this budget affords is "barely better than random," and it injects a stateful RNG trajectory that complicates the reproducibility story. So: **keep the rungs, drop the Bayesian search.** Rung-0 candidates come from the declared grid (seeded Sobol/coarse cross-product), promotion is deterministic argmax.
- **GRAFT from `llm-orchestrator`: content-addressed artifact resolution as the sample-efficiency multiplier** (corpus = hash(prompts-SHA, best_of_N, slice_temps, teacher-ver); packed = hash(corpus, student, max_len); ckpt = hash(packed, α, T, lr, epochs, rank, seed)). Identical config → cache hit → zero spend, bit-identical output. This is already latent in the repo (prompts-SHA guard, `mark`/`done_p`, per-seal push) and just needs to be made the addressing scheme.
- **DEMOTE the LLM from `llm-orchestrator` to an offline, human-reviewed proposer — OFF by default.** An LLM in the autonomous spend loop violates the determinism contract (proposals change across model versions — its own weakness #3), can chase per-slice noise down dead alleys (weakness #1), and adds an always-on control-host single point of failure (weakness #6). Instead, `propose.py` is a **batch tool a human runs between sweeps**: it reads the ledger and emits *candidate rows appended to `experiments.yaml`*, which a human approves and the same validator gates. Creativity in proposal, determinism in execution — but the creativity happens at plan-authoring time, not at 3am while $8–35/hr is burning.

**Why this hybrid beats any single design against the three goals:**

| Goal | How the hybrid satisfies it |
|---|---|
| **Unmanned** | FSM tick is stateless-restartable; ledger + MANIFEST + checkpoints ARE the state; `watchdog.sh` is an independent per-pod dead-man; no live SSH held; no always-on LLM host required. |
| **Cost** | Two-tier split (teacher generated ONCE, judge paid ONCE in a batched window) + successive-halving kills losers at 8k-row rungs before any 120k full run + content-addressed cache never pays twice + `balance − projected > RESERVE` pre-deploy check (the journal's "don't start what you can't finish" rule that saved ~$16). |
| **Determinism** | Seed-42 threaded everywhere already; the only adaptive component (halving) is a pure function of the persisted ledger; no TPE RNG, no LLM in-loop. Everything below `experiments.yaml` replays bit/number-identically. |

Net: a **deterministic state-machine spine for safety/reproducibility, with a bounded, seeded multi-fidelity pruner as the only adaptive element, entirely inside the guardrails.**

---

## 2. THE SOP / STATE MACHINE

One `08_conductor.py` tick does **exactly one unit of work, persists, exits** (re-invoked by cron or a `while` loop). A killed conductor loses nothing. Single-writer to `ledger.jsonl`; concurrent GCP student pods write per-trial `trials/<trial_id>/result.json` that the conductor merges (never co-writes the ledger).

**Persisted state (all in `PILOT_STORE` HF dataset repo via `store_push`/`store_pull`, atomic `_atomic_write_json` tmp+os.replace):**
```
runs/<sweep>/experiments.yaml      # frozen, human-approved plan (the contract)
runs/<sweep>/ledger.jsonl          # append-only: (experiment_id, config, rung, state, score, cost, ts)
runs/<sweep>/state.json            # FSM cursor + phase + running-pod registry
runs/<sweep>/leaderboard.md        # rendered each SCORED tick
runs/<sweep>/funnel_baseline.json  # for drift detection
trials/<trial_id>/{config.json,result.json,logs}
anchors/<student_family>.json      # base-model anchor scores (Discovery #6/#7 gate)
HALT / PAUSE  sentinels            # kill switch + human-pause
approvals/<exp_id>.ok              # sign-off sentinels
```

**States and transitions:**

| State | Action (reuses) | On success → | On fail → |
|---|---|---|---|
| `BOOTSTRAP` | `store_pull` ledger/state; check `HALT`/`PAUSE`; poll RunPod balance+pods via GraphQL | `PREFLIGHT` (first run) or resume at cursor | `HALTED_GUARDRAIL` |
| `PREFLIGHT` | `preflight.py`: schema-check `experiments.yaml`, enforce **α>0 ⇒ GLM-family only** (matches `04_train_kd.py` topk-column contract), dry-run `01_build_prompts.py` source loaders (catches oss_instruct-gone / xlam-gated), compute `experiment_id`s, sum projected cost, write initial ledger | `ANCHOR` | `PAUSED_HUMAN` (validation error) |
| `ANCHOR` | For each student family: `07_eval_benchmarks.sh` + `rescore_ifeval.py` on the **untrained base** under the fixed eval config → `anchors/<family>.json` (**mandatory per Discovery #6/#7**) | `TEACHER_GEN` | `PAUSED_HUMAN` (anchor below published range ⇒ methodology bug, not model) |
| `TEACHER_GEN` | If any declared corpus variant absent from store: `deploy_netvol.sh` → `ensure_glm_env` → `launch_vllm` (`--seed 42`, `VLLM_DEEP_GEMM_WARMUP=skip`, `--return-tokens-as-token-ids`) → `wait_ready_and_smoke` → `02_generate.py` (`SHARD_PUSH_REPO` per-seal push) for **ALL variants in one teacher window** → `watchdog.sh` armed → `podStop` | `PACK` | mark corpus FAILED; if teacher deploy fails N× → `PAUSED_HUMAN` (no CUDA-13 capacity is a decision) |
| `PACK` | `03_pack_dataset.py` ONCE per (corpus_sha × student), `--exclude-report`, held-out hard-fail assert; cache full-size, trials `.select(range(r))` for rungs | `SCHEDULE` | mark FAILED |
| `SCHEDULE` | Deterministic halving: ask for next `(trial_id, config, rung)` — either PROMOTE a rung leader or emit next rung-0 grid point in fixed `priority`/`experiment_id` order | `TRAIN` | — |
| `TRAIN` | Provision student pod (`deploy_student.sh`, `ensure_pod_env` cu128) → `run_experiment.sh <trial_id>` under `nohup` + `watchdog.sh` + `heartbeat.sh`; on-pod chains `04_train_kd.py` (accelerate + `ds_zero3_4gpu.yaml`, or `_offload.yaml` for Air per Discovery #5, `resume_from_checkpoint` if partial) → `merge_lora.py` → `07_eval_benchmarks.sh` (+ `05` on proxy held-out) with `mark`/`done_p` | `SCORE` | divergence/NaN → mark FAILED, no merge; advance FSM |
| `SCORE` | `objective.py`: fold `07` deltas + (later) `06` win-rate → scalar vs anchor, apply floor + margin gates; append ledger; re-render `leaderboard.md`; `store_push`; notify; `podStop` idle | `SCHEDULE` (more rungs) or `JUDGE_WINDOW` | — |
| `JUDGE_WINDOW` | Batch ALL screen-survivors into ONE teacher-up window: `deploy_netvol.sh` → `05_eval_gen.py --served` teacher refs → `06_judge.py` (seeded) each survivor → `podStop` | `FINALIZE` | `PAUSED_HUMAN` |
| `FINALIZE` | argmax objective over SCORED+margin-passed; write winner; `PAUSED_HUMAN` to confirm incumbent | `DONE` | — |
| `HALTED_*` / `PAUSED_*` / `DRAINING` | drain in-flight, `podStop` all registered pods, exit | — | — |

Progress within any long step is judged by **byte-delta liveness** (`liveness.sh` — the exact signal that caught the silent hf-downloader wedge and the DeepGEMM/flashinfer "hang" false-positive), never a wall-clock timer, so the storage-speed lottery (6→40 s/shard) is not misread as a hang.

---

## 3. SEARCH POLICY

**Search space (exactly the pipeline's knobs):**
- **Teacher-side (expensive, generated once, coarse):** corpus variant = best-of-N ∈ {1,4}, `SLICE_TEMPS` profile, size/mix from `corpus_spec.md` (40/20/20 + capability buckets). best-of-N is a *teacher* knob, not searched per-trial.
- **Student-side (cheap, massively parallel, the real search):** student ∈ {GLM-4.5-Air remap-logit-KD, Qwen3-8B/14B/32B/30B-A3B seq-KD, GLM-Z1-32B after a ~$1 `00_check_tokenizer.py` gate}; α ∈ {0, 0.3, 0.5, 0.7} (**pinned to 0 for non-GLM** — the cross-constraint that prunes most of the naive cross-product); KD-T ∈ {1,2,4}; LoRA rank ∈ {64,128,256}; lr ∈ {5e-6,1e-5,2e-5}; epochs ∈ {1,2,3}.

**Multi-fidelity pruning (the budget-respecting core):** rungs by training-samples-consumed `r ∈ {8k, 32k, full~120k}`, **eta=3, carry ≥2 to the top rung**, deterministic `.select(range(r))` on the seed-42-shuffled pack so rung-0 is always the same 8k rows. Rank at each rung by the composite; keep top-1/eta; re-enqueue survivors. This spends full-fidelity dollars **only on configs showing signal at 8k rows** — directly answering the deterministic-grid's core weakness (sample-inefficiency) without importing TPE's nondeterminism.

**Cost-normalized halving (from bandit guardrail):** budget measured in normalized-$ per student, so Air's ZeRO-3-offload ~4× cost (Discovery #5) doesn't unfairly favor cheap Qwen. A reserved seed guarantees each student family gets ≥1 rung-0 trial before pruning collapses onto the cheapest model.

**Objective (`objective.py`, `score_fn_version`-stamped, deterministic):**
```
score = 0.50 * judge_winrate_vs_teacher      # 06, only measured at JUDGE_WINDOW / top rung
      + 0.20 * gsm8k_strict                   # the strongest observed transfer signal (+48pt, Disc #7)
      + 0.15 * ifeval_thinkstripped           # rescore_ifeval.py; the axis the pilot dented
      + 0.10 * mmlu_pro
      + 0.05 * livecodebench
```
each normalized as a **delta over the base anchor**. Lower rungs use only the cheap benchmark subset (no teacher); judge win-rate enters at the top rung / batched window. **Noise-band gate:** a "win" must clear a bootstrap CI on judge win-rate (journal saw ±14pp at n=49, ~±3pp at n=1000) and the ±3pp benchmark band, else logged `INCONCLUSIVE`, not `WIN` — so halving cannot chase noise and `FINALIZE` argmax cannot crown a lucky config.

**Known limitation, mitigated:** multi-fidelity assumes rung-0 rank predicts full-scale rank; KD can rank-invert (high α / low rank wins small, loses at scale). `carry≥2` + a "protected promotion" of the top-1 of each student family hedges this; it is an accepted, bounded risk documented in the plan.

---

## 4. GUARDRAILS SPEC (load-bearing for unmanned operation)

Every guardrail is enforced by **code, never by a model**. Two independent layers: the conductor (soft, can pause/prune) and `watchdog.sh` (hard, self-stops the pod even if the conductor is dead).

1. **Spend — three nested backstops.** (a) `watchdog.sh` on EVERY pod: after `MAX_MINUTES` (derived from the *pessimistic* slow-storage cost estimate) it `pkill`s generation, waits 180s for per-seal pushes to flush, then `podStop` via GraphQL — proven when sshd died at the $12 floor. (b) Conductor refuses to deploy any step unless `polled_balance − projected_step_cost > RESERVE` (the ~$16-saving rule), and enters `STOPPED_BUDGET` when the ceiling is within one step. (c) Every tick reconciles cumulative spend against the **true RunPod balance** (ground truth, not an estimate). Soft cap 90% → stop sampling new configs, finish in-flight; hard cap 100% → `DRAINING`.
2. **Cost estimator** (`cost.py`) calibrated on journal rates ($0.16/best-of-4 answer, $35/hr teacher, $8/hr student, measured train throughput) gates every config **pre-spend**; a config whose per-experiment estimate exceeds its cap is rejected before a pod spins.
3. **Contamination.** `03_pack_dataset.py` held-out hard-fail (asserts 0 overlaps, already 0/1,140 per Discovery #4) + `benchmark_audit.py --exclude-report` run ONCE per corpus_sha. Conductor **refuses to promote or judge any checkpoint whose corpus_sha didn't pass the audit**; the heldout-hash is pinned so no proposed corpus can reuse eval data.
4. **Quality floor / base-anchor** (Discovery #6/#7 made mandatory). The loop won't enter `SCHEDULE` until each base student recovers to its published range under the fixed `07`+`rescore_ifeval` config; any trial not beating its anchor on the screen benchmark is pruned and its children skipped. An anchor *regression* that looks like methodology (the Disc #6 trap: base 4× below known ability) → `PAUSED_HUMAN`, not a silent bad sweep.
5. **Corpus-quality drift.** `02`'s `SHARD_KEEP_FLOOR=0.5` won't seal a decimated shard (regenerates on resume). Additionally a `funnel.json` narrowing detector compares each slice's kept-rate to `funnel_baseline.json`; a collapse **halts teacher spend** before a bad generation window burns $35/hr.
6. **Divergence / NaN.** `liveness.sh` tails `04`'s `loss_ce`/`loss_kl`; NaN/Inf or sustained loss climb kills the run, marks FAILED, **does NOT merge**. The KL≈0 self-distill invariant (journal) is a sanity assert.
7. **Source availability & safety.** PREFLIGHT dry-runs all `01_build_prompts.py` loaders with pinned revisions and refuses teacher $ until every source resolves (catches oss_instruct-gone / xlam-gated at $0). Validator rejects any source outside the `corpus_spec.md` allow-list (safety subsets excluded, English-only) — an unapproved corpus can never be generated.
8. **Env pin (not a search axis).** Only `ensure_glm_env` (cu130, vLLM 0.24) and `ensure_pod_env` (cu128, torch 2.8, vLLM 0.11), both PIN_ENV-guarded; teacher locked to `imageName ...cu1281` + `allowedCudaVersions:["13.0"]` + `VLLM_DEEP_GEMM_WARMUP=skip`. A failed `wait_ready_and_smoke` blocks generation instead of burning balance on a broken server. Venvs on local container disk only (netvol partial-copy corruption fix).
9. **Wall-clock per-rung cap** so a storage-lottery-slow trial is reaped, not run forever; readiness proven by the actual token-id smoke test (240×30s budget), never a timer.
10. **Dedup / margin.** No config within epsilon of a prior `experiment_id` (cache hit → free); `FINALIZE` requires a margin gate so a noise-level win isn't crowned.

---

## 5. DETERMINISM CONTRACT

**Bit/number-reproducible (the guarantee):**
- All artifacts *below* `experiments.yaml`. Seed 42 is already threaded: `GLOBAL_SEED` + per-sample `gen_seed` in `02`, `03` shuffle/split seed, `04` `set_seed` + `TrainingArguments.seed`, `05/06` fixed decode (T=0.6/top_p=0.95/seed=42), judge seeds, `vllm --seed 42`.
- Content-addressed DAG: corpus keyed by prompts-SHA (`02` REFUSES resume on `MANIFEST._prompts_sha256` mismatch), packed/ckpt keyed by config hash; identical config → cache hit → identical output; ledger records completion so re-execution is a no-op.
- `objective.py` carries `score_fn_version` so scores stay comparable across the whole sweep.

**Reproducible-as-a-function-of-history (weaker but sufficient):**
- The **search trajectory**. Because successive-halving is a *pure function of the persisted ledger* (no TPE RNG state, no LLM), replaying the ledger from any checkpoint reproduces the identical promotion/prune decisions. This is strictly stronger than bandit-optimizer (whose TPE stream must be seeded and re-fit) and than llm-orchestrator (reproducible only within one pinned Claude regime).

**NOT reproducible (accepted, and why it's fine):**
- **GLM-5.2 teacher generation** is not bit-identical across differing pod concurrency/storage. **Mitigation = never re-derive:** each corpus variant is generated ONCE in the single teacher window and content-addressed; the sweep consumes the cached corpus_sha forever. A forced regen may not match — so the design's rule is *don't force regen*. This is a correctness discipline, not a hardware guarantee, and it is acceptable because the corpus is an immutable input to everything downstream.
- The **optional offline LLM proposer** is explicitly outside the determinism boundary — it only suggests rows a human approves, so its nondeterminism never touches an autonomous spend decision.

---

## 6. HUMAN-IN-THE-LOOP

**Minimal sign-off set (only these four ever block spend):**
1. **`experiments.yaml` approval** — up front, before any dollar. PREFLIGHT prints projected total cost + source-availability report + the anchor plan; refuses on validation error.
2. **Teacher corpus generation** — the single largest spend (~$75 at production scale). Explicit approval sentinel before `deploy_netvol.sh`, and before any teacher *redeploy*.
3. **Anchor regression that looks like methodology** (Disc #6 trap) or a contamination-audit regression → `PAUSED_HUMAN`.
4. **`FINALIZE` incumbent confirmation** before declaring the winner.

Everything cheaper (student rungs, screens, promotions) runs unmanned. Configs above `HUMAN_APPROVAL_USD` queue as `PENDING_APPROVAL` via a one-line sentinel `approvals/<exp_id>.ok` while cheaper queued work keeps flowing.

**Reporting cadence:** after every SCORED trial → append `ledger.jsonl`, re-render `leaderboard.md` (trials ranked by composite + per-slice judge win-rates + spend/ETA), `store_push`, fire a `PushNotification`/webhook one-liner (experiment, objective, spend, budget-left). **Nightly digest** rolls up deltas + spend-burn-vs-budget + Pareto front (score vs serve-cost) so an overnight run is auditable at a glance. **Auto-pause-and-ping** on unproductive search: 3 consecutive `INCONCLUSIVE`, or spend-per-win above threshold.

**Kill switch (two independent layers):**
- Conductor checks a `HALT` sentinel object at every transition (and traps SIGTERM) → drains, `podStop`s every registered pod via GraphQL, marks `HALTED_GUARDRAIL`, exits.
- `drain_all.sh` / `watchdog.sh` remain independent per-pod dead-men even if the conductor process is gone — the pod stops itself. Fail-safe (no burn), never fail-dangerous.

---

## 7. BUILD ROADMAP (crawl → walk → run)

**Reused UNCHANGED throughout:** `01/01b/02/03/04/05/06/07`, `merge_lora.py`, `benchmark_audit.py`, `rescore_ifeval.py`, `deploy_netvol.sh`, `watchdog.sh`, `common.sh` (`mark`/`done_p`/`launch_vllm`/`wait_ready_and_smoke`/`ensure_*_env`/`store_*`), all `configs/*.yaml`.

**CRAWL — thin deterministic orchestrator we can trust (est. 2–3 days):**
- `08_conductor.py` — the FSM (compute-next-action / guard / deploy / poll / score / persist / exit), single-writer ledger. Start with **no rungs** (full-fidelity only) executing a small frozen grid.
- `experiments.yaml` + `preflight.py` (schema, α>0⇒GLM constraint, source dry-run, cost sum, `experiment_id`s).
- `runpod.py` — GraphQL wrappers (deploy/resume/stop/podReset/balance/list) generalizing the snippets already in `deploy_netvol.sh` + `watchdog.sh`.
- `run_experiment.sh` — on-pod idempotent recipe (pack→train→merge→screen) via `mark`/`done_p`; **generalize the already-correct shape of `pilot_student.sh`/`pilot_teacher.sh`.**
- `objective.py` + `leaderboard.py`, `liveness.sh`/`heartbeat.sh`, `deploy_student.sh`.
- Guardrails 1–9. **Goal: prove an unmanned full-grid run on cheap students before adding any adaptivity.**

**WALK — bounded adaptive search (est. 2–3 days):**
- Add multi-fidelity rungs + deterministic successive-halving to `SCHEDULE` (the `.select(range(r))` slices, eta=3, carry≥2, cost-normalized budget).
- Add the mandatory `ANCHOR` state + noise-band CI gating in `objective.py`.
- Add the batched `JUDGE_WINDOW` (one teacher-up window for all survivors).
- Add `cost.py` calibration + balance reconcile + `PENDING_APPROVAL` flow.

**RUN — scale + optional intelligence (est. 2–3 days):**
- `fleet.py` — swap RunPod GraphQL for a gcloud backend on the 16×RTX PRO 6000; conductor runs up to 4 concurrent independent 4-GPU student jobs (per-trial result files merged by the single-writer conductor). Scripts unchanged.
- `propose.py` — the **offline, human-reviewed** LLM candidate generator (pinned model, temp 0, reads ledger, emits `experiments.yaml` rows). Never in the spend loop.

---

## 8. FIRST CONCRETE MILESTONE

**The smallest end-to-end unmanned run — ships next, zero teacher spend:**

**Reuse the existing pilot corpus** (1,140 GLM-5.2 traces already in `PILOT_STORE`, contamination-clean per Discovery #4). This eliminates the entire `TEACHER_GEN` risk surface and the $35/hr tier for v1 — we prove the *spine*, not the teacher.

**Scope:** on the 4×RTX PRO 6000 (~$8/hr):
- `preflight.py` validates a tiny `experiments.yaml`: student = Qwen3-8B (seq-KD, α=0 forced), grid over lr ∈ {5e-6,1e-5}, rank ∈ {64,128}, epochs ∈ {1,2} = 8 configs.
- `ANCHOR`: base Qwen3-8B under the fixed `07`+`rescore_ifeval` config → must recover to published range (the Disc #7 gate) before any training.
- **One halving round:** all 8 at rung-0 (8k rows), keep top-2, promote to full 1,140 (small enough to be the "full" rung here).
- `SCORE` on **screen benchmarks only** (GSM8K-strict, IFEval-thinkstripped, MMLU-Pro) — **no `JUDGE_WINDOW`** (no teacher needed). We already have the Disc #7 baseline (+48pt GSM8K-strict) as a known-good target to sanity-check the objective against.
- Unmanned end to end: cron-driven `08_conductor.py`, `watchdog.sh` armed, `HALT` sentinel wired, `PushNotification` per trial + nightly digest.

**Guardrail budget:** hard cap **$40**, `RESERVE $10`, per-trial cap ~$3, per-pod `watchdog.sh` at ~4h wall-clock. Expected actual: ~1–2 pod-hours of student compute (8 rung-0 + 2 full ≈ well under the cap), so the budget is a genuine backstop, not the operating point.

**Exit criteria (what "trusted" means):** the run completes unmanned, the ledger + leaderboard reproduce Discovery #7's GSM8K-strict delta on the winning config, a mid-run `kill -9` of the conductor followed by re-invocation resumes with zero lost/duplicated trials, and the `watchdog.sh` self-`podStop` fires correctly on a deliberately low cap. Passing this greenlights WALK (rungs + anchor CI + judge window) and the first real teacher-generation sweep.

---

**New components, named precisely:** `08_conductor.py` (FSM spine) · `experiments.yaml` (frozen plan/contract) · `preflight.py` (validator) · `objective.py` (`score_fn_version`-stamped scorer) · `cost.py` (estimator + balance reconciler) · `runpod.py` / later `fleet.py` (GraphQL/gcloud fleet driver) · `run_experiment.sh` (on-pod recipe) · `deploy_student.sh` · `liveness.sh` + `heartbeat.sh` · `leaderboard.py` · `propose.py` (offline, off-by-default). State lives in the existing `PILOT_STORE` HF dataset repo — **no new infra service.**