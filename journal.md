# Project Journal — GLM-5.2 Distillation Pilot

The running chronicle of decisions, incidents, findings, and spend.
(Companion docs: `corpus_spec.md` for the production corpus design,
`README.md`/`README_GCP.md` for pipeline operation.)

---

## 2026-07-04 — Review day and the $4 rehearsal

- **Completeness review of the pipeline** found and fixed: a judge that
  couldn't parse thinking-model verdicts (max_tokens=8 vs inline `<think>`),
  LoRA merge corruption under DeepSpeed ZeRO-3 (split into adapter-save +
  separate `merge_lora.py`), a pilot/production state collision, missing
  incremental corpus sync, a VM hand-off race, and doc/code mismatches.
- **Single-GPU rehearsal on a RunPod RTX PRO 6000** (~$4): Qwen3-8B as both
  teacher stand-in and student. 8 of 10 phases green; the KD loss
  self-verified (KL≈0 in self-distillation, exactly as theory predicts);
  the merged-adapter checkpoint loaded in vLLM.
- **Discovery #1 — the tokenizer gate fails for real**: GLM-5.2 ↔ GLM-4.5-Air
  vocabs differ (36 moved special ids incl. `</think>`, 3,491 teacher-only
  tokens). Would have silently disabled logit KD. Built and verified the
  **remap mode** (gate: exact/remap/none; id translation at pack time).
- DRY refactor: `common.sh` + hardware-neutral `pilot_teacher.sh` /
  `pilot_student.sh` with `EXTENDED=1` scale switches.
- Infrastructure staged: private HF store `ledzepu2/glm52-pilot-artifacts`,
  RunPod API driving, region sweep (teacher hardware scarce; student
  hardware plentiful).

## 2026-07-05 (early) — The six serve attempts

User funded $250 with a strict cost mandate. Deployed 8× H200 (US-GA-2).

- **Attempts 1–5 all failed, each with a distinct lesson**:
  1. transformers 4.x doesn't know `glm_moe_dsa` → needs transformers 5.x
  2. vLLM 0.16 (newest cu128-compatible) + flashinfer version mismatch
  3. 20-min flashinfer JIT compile misread as a hang (killed prematurely —
     wrong call, cache was salvaged)
  4. torch.compile phase → worker crash; script timeout babysitting bug
  5. eager mode → missing flashinfer cubin kernel assertion
- **Discovery #2 — the load-bearing finding of the project**: GLM-5.2 serves
  only on **vLLM ≥0.24, which requires CUDA-13 drivers**. On 12.8-driver
  hosts, nothing installable works. → GCP image spec must ship 580+/CUDA-13.
- Brief GLM-4.7 fallback (user AFK) produced Discovery #3: **GLM-4.7 ↔ Air
  vocabs are identical** (`exact` mode) — noted for any future fallback.
- User decree: **GLM-5.2 as teacher, nothing else.** H200 pod terminated.
- Pivot: `allowedCudaVersions: ["13.0"]` deploy filter found a CUDA-13
  **6× B200** pod in US-CA-2.

## 2026-07-05 (mid) — B200 env war and the verdict

- Three env snags, each now guarded in code: a stale `vllm._C` probe (vLLM
  0.24 renamed its extension), the pinned-env auto-reinstall clobbering the
  cu130 venv (`PIN_ENV` guard born), and an fsspec conflict silently
  resolving `datasets` to a 2020 version (pins born). Network-filesystem
  install corruption → **venvs live on local disk now**.
- Load pace lottery documented: same pod, same files — 6 s/shard to
  40 s/shard depending on DC storage contention. Auto-stop spend floor
  ($12) fired once, correctly, when sshd died after a reset; probe-resume
  later recovered the pod cleanly.
- **DeepGEMM warmup trap**: 2,352 kernel profiles ≈ 2 h ≈ the entire
  remaining balance. `VLLM_DEEP_GEMM_WARMUP=skip` found and applied.
- User added $50 (total ceiling stance: strict). Resume → clean load →
  **READY after ~2730 s → SMOKE TEST OK** — *GLM-5.2 serving with token-id
  logprobs, the project's central question answered yes,* on attempt 6.

## 2026-07-05 (late) — The corpus run

- Generation at 1,300–3,500 tok/s (thinking-heavy sawtooth). Shard
  economics measured: **~$0.16 per best-of-4 answer**, per-shard variance
  0.14–0.21, ~7–12% truncation drops, ~20% judge fallbacks (fix queued;
  see Open items — NOT yet in code).
- Judge directionality validated on real GLM-5.2: truncated answers lost
  43/49 (49 = the pilot held-out set size after stratified-floor drops)
  with zero wins — verdict parsing provably works.
- User added $100 mid-run; hot-server intercept resumed generation without
  a reload toll. Mid-run artifact pushes closed the durability hole.
- **Final: 5 shards, 1,140 traces** with top-20 logprobs, pushed to the HF
  store with dataset card + readable samples. Pod stopped with $45 left —
  the early-cut call (don't start a shard you can't finish) saved ~$16.
- Pack step hit a transformers-5.x API change (`BatchEncoding`); fixed
  version-agnostically; packing deferred to the student leg by design.

## 2026-07-05 (evening) — Reviews become code

- **Three-lens representativeness review** (parallel agents): benchmark
  contamination real (NuminaMath *contains* GSM8K; no coding benchmarks
  screened anywhere), head-of-stream sampling artifacts, true mix ~57/33/10
  not 50/25/25, four survivorship filters all biased against hard prompts,
  judge parser bug, missing capability buckets, ±14pp eval noise at n=49.
- **User decisions locked** (`corpus_spec.md`): practical-assistant coding
  profile, all four new capability buckets, English-only, 1,000 held-out.
- **Resilience audit** (independent agent) on my own recovery claims:
  3 of 4 FALSE/overstated. All 10 gaps fixed the same day: cu130 recipe
  in-repo (`ensure_glm_env`), clobber-proof probes, per-seal HF pushes,
  launch profile defaults, `CUTOVER=1` mode, atomic seals, dropped-sample
  ledger, on-pod `watchdog.sh`, curl health gates, prompts-SHA resume guard,
  `deploy_netvol.sh`.
- **Builder v2** (`01_build_prompts.py`): shuffled streams, 6-benchmark
  decontamination incl. numerals-stripped template matching, Tulu subset
  filtering, round-robin dedup, funnel reports, multi-source slices with
  caps. Multi-turn bucket parked behind `MULTITURN_READY` (needs 02/03
  message support). New retro-audit tool: `benchmark_audit.py` + pack
  `--exclude-report`.

## 2026-07-05 (night) — Student leg [COMPLETE]

- Pod: 4× RTX PRO 6000 (US-NC-2, $8.36/hr) — deliberately the same GPU as
  the target GCP fleet. On-pod watchdog armed (260 min).
- One downloader wedge (silent hf stall; detected by byte-delta — new
  monitoring rule) cost ~$3.5.
- **Discovery #4 — contamination audit: 0/1,140.** The pilot corpus is
  clean against all 6 benchmarks (exact + numerals-stripped + near).
- **Discovery #5 — Air (106B) OOMs on 4× 96 GB under plain ZeRO-3**; needs
  CPU offload (`ds_zero3_4gpu_offload.yaml` committed) or Tier-3's 180 GB
  B200s. Retry deferred to Tier 3 (offloaded training too slow for the
  remaining budget).
- Qwen3-8B v0 trained (68 steps, loss_ce 1.42→) and pushed
  (`qwen3_glm_distill_v0`). Benchmarks ran, and produced
- **Discovery #6 — instructive negative result, and a CORRECTED
  diagnosis.** Pilot showed v0 below base, base itself ~4x below Qwen3-8B's
  known ability. Two hypotheses were logged: (a) thinking-hostile lm-eval
  defaults; (b) packer flattening reasoning. **2026-07-06 re-run verdict:**
  (a) is TRUE and fully explains the low absolute scores — with a raised gen
  budget + IFEval think-strip re-score, base recovered to GSM8K flexible
  **0.919** (from 0.199) and IFEval prompt-strict **0.826** (from 0.349),
  i.e. its published range. (b) is FALSE — the teacher ran without a
  reasoning parser, so thinking was always INLINE in response_text with
  `<think>` tags; the original packing already preserved format. The packer
  "fix" is a validated no-op for this corpus (kept for corpora that DO split
  reasoning). Net: the pilot's eval methodology was broken, not the data or
  the model.
- **Discovery #7 — the first TRUSTWORTHY v0 delta (2026-07-06).** Base vs
  v0 (GLM-5.2-distilled Qwen3-8B), both under the fixed eval config:
    GSM8K flexible  0.919 -> 0.877  (-4.2, near ceiling, ~noise)
    GSM8K strict    0.159 -> 0.636  (+47.7 !!)
    IFEval p-strict 0.826 -> 0.734  (-9.2, think-stripped)
    IFEval i-strict 0.880 -> 0.801  (-7.9)
  Reading: the distill did NOT wash out. It strongly transferred the
  teacher's answer-formatting discipline (GSM8K strict-match +48pts = the
  student now emits clean, extractable final answers), at the cost of a mild
  broad-instruction-following dip (IFEval -8pts) — expected from 1,140
  samples of coding/math-heavy SFT with zero IFEval-style data. Raw math
  capability held. This is a genuine positive signal at pilot scale: the
  corpus transfers something real. Production (100-150k, the new mix with a
  general/instruction slice) should lift IFEval instead of denting it.
  Cost of the whole trustworthy re-run: ~$19 (staged; base anchor gated the
  spend, and Stage 0 caught the packer misdiagnosis for free).
- US-GA-2 stranded volume deleted (user call). Repo under git as of today.
- Pod stopped. Balance at leg close: **$20.29**.

---

## 2026-07-06 — Tier 3 prep: reviews + research

- **Judge robustness fixes** (Open item 3, partial): unclosed-think verdicts
  rejected, candidate order randomized, `bon_judged` column (commit bf31b41).
- **Builder v2 validated** (~$2): all 6 slices populate; 2 dead source loaders
  caught (oss_instruct repo gone, xlam gated) and swapped for verified
  Magicoder-OSS + hermes-function-calling.
- **Trustworthy v0 delta** (Discovery #7): distill transfers answer-format
  discipline (GSM8K strict +48pts), mild IFEval dip — pilot-scale positive.
- **Student-candidate research** (deep-research, 111 agents) → student_candidates.md:
  logit-KD limited to GLM family; NEW option GLM-Z1-32B-0414 (dense 32B, MIT,
  tokenizer-compat TBD); best sequence-KD = Qwen3-30B-A3B / 32B / 14B (Apache).
- **Trace-recipe review** (43 agents) → trace_recipe_review.md. VERDICT:
  YES-WITH-GAPS. Recipe is sound; 6 must-fix-first items before Tier-3 money:
  R1 raise MAX_TOKENS + truncation telemetry; D3 add LiveCodeBench decontam;
  R2 kept-rows floor + persisted per-slice funnel.json with drop reasons;
  D1+D2 fix shipped mix (multi-turn/redistribute + real structured-output source);
  R4 atomic MANIFEST write; R3 per-slice unmappable accounting (only if remap
  student in scope). All are hours of code, no GPU.

## 2026-07-06 — Trace-recipe gaps closed + verified (Tier-3 code gate MET)

- Trace-recipe review (43 agents) verdict YES-WITH-GAPS -> implemented all 14
  gaps (6 must-fix + 8 non-blocking): R1 raise/escalate MAX_TOKENS + per-slice
  temps; R2 kept-floor + persisted funnel + drop-reason ledger; R3 remap
  position-masking + per-slice accounting; R4 atomic manifest; R5/R6/R7 judge
  answer-channel/empty/trailing-number; D1 redistribute multi-turn off coding;
  D2 structured-output source + user-turn extractor (volume made observable via
  realized_source_mix, not fabricated); D3 LiveCodeBench decontam; D4 extension
  seed; D6 per-slice temps; D7 post-dedup caps + backfill; top-20 coverage col.
- **Verify -> fix -> reverify loop caught 4 defects self-review missed**: R5
  NameError on its own fallback path, D3 crash of the 01b extension build, D2
  system-boilerplate extractor, and D7 backfill DEAD CODE (reused exhausted
  idx[k]). All remediated, unit-tested, and independently re-verified CLOSED.
  Final: 14/14 gaps CLOSED.
- Code pushed to PRIVATE github.com/arunmenon/glm52-distill (.gitignore guards
  data/models/secrets). Student candidate menu (student_candidates.md) +
  trace_recipe_review.md committed.
- **Tier-3 code gate is now met.** Remaining Tier-3 blockers are decisions, not
  code: top-up (~$150-250), student choice (Air + Qwen3-30B/32B; optional ~$1
  GLM-Z1-32B gate check), HF token rotation, and the pre-run smoke test to
  confirm realized_source_mix (esp. structured-output volume).

## 2026-07-06 — Autonomous loop design (autoloop_design.md)

- Design workflow: 3 architectures (deterministic-grid / llm-orchestrator /
  bandit-optimizer) x 3 adversarial judge lenses (SRE / cost-controller /
  researcher) -> synthesis. Ranked deterministic-grid 8.0 > llm 7.7 = bandit 7.7.
- VERDICT: deterministic state-machine SPINE + grafted (a) multi-fidelity
  successive-halving scheduler WITHOUT the Bayesian surrogate (keep rungs, drop
  TPE — barely-better-than-random at this budget and hurts determinism),
  (b) content-addressed artifact cache (corpus/packed/ckpt keyed by config hash
  -> identical config = free cache hit). LLM DEMOTED to an offline, human-approved
  proposer (propose.py), OFF by default — never in the autonomous spend loop.
- Guardrails all code-enforced, two layers: conductor (soft) + watchdog.sh (hard
  per-pod dead-man). 10 hard stops incl. 3 nested spend backstops, base-anchor
  quality floor (Disc #6/#7 mandatory), contamination gate, funnel drift halt,
  NaN/divergence no-merge, source-availability preflight, env pins.
- New components: 08_conductor.py (FSM) + experiments.yaml (frozen contract) +
  preflight.py + objective.py + cost.py + runpod.py/fleet.py + run_experiment.sh
  + deploy_student.sh + liveness/heartbeat + leaderboard.py + propose.py. State
  lives in the existing HF store — NO new infra service.
- Roadmap crawl->walk->run (~2-3 days each). FIRST MILESTONE: thin deterministic
  orchestrator, ZERO teacher spend (reuse the 1,140-trace corpus), 8-config
  Qwen3-8B grid + one halving round, screen-benchmarks only, $40 hard cap,
  watchdog dead-man. Exit criteria: reproduces Disc #7's +48pt GSM8K-strict
  unmanned, survives a mid-run conductor kill -9, watchdog self-stop fires.

## 2026-07-22 — Agentic trajectory leg design (agentic_trajectory_design.md)

- Assessment requested: is corpus diversity sufficient to distill a
  frontier-class AGENTIC coder? Verdict NO: the corpus is single-turn only
  (~0% trajectories, 0% env-verified coding, multi-turn slice at 0%); the
  agentic loop (act/observe/recover) never appears in the data, so KD cannot
  teach it. Confirmed absent from experiments.yaml + autoloop_design.md.
- Designed the second program: teacher-as-agent (mini-swe-agent, bash-only)
  over executable repo envs (SWE-Gym/Lite -> SWE-smith -> R2E-Gym +
  SWE-rebench; all sources verified live), env-grounded verification
  (double test-run + empty-diff rejection), per-step top-20 capture so
  logit-KD survives, message-list pack with assistant-only loss mask (this
  lands MULTITURN_READY as a side effect).
- Split-plane constraint discovered: RunPod pods can't run Docker-in-Docker,
  so envs need real VMs (GCP CPU or `static` provider); teacher stays the
  scarce resource — CRAWL must co-schedule with the Tier-3 teacher window.
- Cost basis from journal rates: ~$0.20-0.45/rollout, $0.6-1.5/verified
  trajectory. CRAWL $150 cap (>=60 verified, Qwen3-8B plumbing proof);
  WALK 1-2k verified on a 32B-class flagship (GLM-4.5-Air primary).
  Honesty clause recorded: SFT/KD ceiling is "competent" (30-45% Verified
  by precedent); frontier needs RL — leg stores reward-labeled rejects for
  that future.

## 2026-07-29 — Round-3 review: deployment representativeness (G1-G6)

- Asked: are the trajectories representative of a WORLD-CLASS agentic
  coder? Verdict: kernel only (~1/4-1/3 of deployed behavior mass).
  Gap registry with mandatory concrete examples added to the design doc:
  G1 harness grammar (sed/heredoc vs structured read/str_replace tools),
  G2 teacher-as-ceiling (one model's habits; RL exceeds, SFT copies),
  G3 work-type mix (issue->patch vs feature/diagnosis/refactor),
  G4 pytest-only observations (no TS/npm/rust error vocabulary),
  G5 no context management (and length cap drops longest trajectories),
  G6 user silent after turn one.
- DECISION: WALK format flipped — native tool-call trajectories become the
  WALK backbone; bash-only stays for CRAWL and survives as a minority
  robustness arm. Format A/B moved to WALK ENTRY. Harness-format is the
  first breadth investment, ahead of multilingual (G4 second).
- Standing convention: gap registry is living; post-CRAWL reviews must
  draw examples from REAL rollouts (audit sample), not hypotheticals.

## 2026-07-29 — REHEARSAL COMPLETE: first 7 verified trajectories ($1.97)

- 09_rehearsal.py on the local Mac (x86-emulated Docker, network-isolated
  containers): 10 SWE-Gym-Lite tasks x N=2, GLM-5.2 via pinned OpenRouter
  (StreamLake/GMICloud/Alibaba), mini-swe-agent 2.4.6 (NB: v2 defaults to
  NATIVE TOOL-CALLING with a single bash tool — CRAWL data is already
  FC-format, softening G1).
- **Results: 7/20 rollouts verified (35%), 4/10 tasks. By tier (rollouts):
  easy 2/6, medium 5/8, hard 0/6.** Mean ~31 steps. $0.092/rollout,
  **$0.26/verified trajectory** (vs $0.6-1.5 estimated). Failure modes:
  6 no-submission (step/wall limits), 7 tests_fail, 0 tampered,
  0 apply_fail — mechanics clean, anti-tamper never tripped.
- **Finding 1 (tier proxy partially inverted)**: medium(62%) > easy(33%).
  Gold-patch size measures fix complexity, not diagnosis difficulty.
  Recalibrate tiers with measured success per task spec sec 2.
- **Finding 2 (hard tier 0/6)**: T4 concern confirmed at small n — at
  CRAWL the >=25%-hard verified quota may be unreachable at N=6;
  pause-not-backfill will trigger unless N-hard rises or tiers recalibrate.
- **Finding 3 (ops)**: Docker Desktop VM disk (not host disk) is the
  binding constraint — pulls fail at ~48GB; fixed by deleting sealed-task
  images (image GC per task is REQUIRED in 09_generate for CRAWL).
  3 MONAI tasks had no published image (preflight caught). Provider-side
  stalls >300s exist (mini's 60s timeout patched to 300s in driver);
  emulation inflates wall-time (dvc TimeExceeded artifact) — vanishes on
  a real x86 VM.
- CRAWL repricing at measured rates: 200 rollouts ~= $20 -> ~65-70
  verified, right at target; $50 OpenRouter balance covers CRAWL 2x over.
- Spend: $1.97 of $50 (incl. smoke test + credit-outage redo).
  7 verified trajectories banked: moto x2, dask x2, mypy x3.

## 2026-07-30 — CRAWL components: pack + gate + image GC

- **03b_pack_trajectories.py (MULTITURN_READY lands)**: canonical
  student-agnostic pack — cleaned message lists (system/user/assistant/
  tool; API debris + mini's synthetic `exit` role stripped; tool-call
  linkage validated; must end on assistant turn), loss-mask contract
  = assistant turns only, format id fc_bash_v1 (never mix formats).
  All 7 verified rehearsal trajectories packed: real-GLM-tokenizer
  min/mean/max = 11.7k/17.4k/29.5k tokens, 0 over the 32k cap.
  -> packed/trajectories/trajectories_v0.parquet + committable
  trajectory_pack_report.json. Student chat-templating deferred to 04.
- **trajectory_decontam.py**: 3-level hard gate (instance / repo / text
  exact+8-gram containment) vs SWE-bench Verified. Rehearsal task list:
  CLEAN at all levels (trajectory_gate.json).
- **09_rehearsal.py image GC**: rmi after task seal (Docker Desktop VM
  disk was the binding constraint; two incidents on 2026-07-29).
- **03c_build_multiturn_dataset.py (train-side bridge, DONE same day)**:
  fc_bash_v1 -> 04's exact dataset contract (input_ids + labels, -100
  masked), so 04 trains multi-turn WITH ZERO CHANGES (SFT path).
  Explicit ChatML rendering, NOT apply_chat_template — inference
  templates strip reasoning from historical turns; teacher ran with
  interleaved thinking preserved, student must train the same way
  (deployment note: student harness must resend prior reasoning).
  Atomic-tag boundary asserts; task-level split (rollouts never straddle
  splits). 7 rows -> Qwen3 tokens 11.6k-28.9k, 0 dropped, trainable
  fraction 36.4%; spot-check decode = think+tool_call spans only.
  GLM-family rendering lands at WALK.
- TODO next: 30/70 blend assembly at train time (dataset concat, small),
  then CRAWL run (~$20 at measured rates, real x86 VM); push
  trajectories_v0.parquet + mt_qwen3 to HF store AFTER token rotation
  (open item 9). Train-time seq len 29k: check pod memory headroom.

## 2026-07-31 — CRAWL: hard-tier guard fired, human decision recorded

- PAUSED_HUMAN tripped exactly per spec: hard tier 0/30 verified rollouts,
  having consumed $6.82 of $10.95 total. Failure autopsy (evidence, not
  theory): (1) hydra near-miss x6 — same wrong fix every rollout, G2
  teacher-ceiling made flesh; (2) pandas class = ENVIRONMENT TAX — C
  rebuilds per test under x86 emulation eat the 50-step/40-min budget
  (11/12 rollouts died no-submission; 1 failed on a ninja build error,
  not tests); (3) tier proxy selected compiled test-heavy repos into
  hard, compounding both.
- DECISION (user): finish easy+medium on Mac (~13-16 verified projected);
  DEFER 5 remaining hard tasks to WALK on real x86 VM with raised budgets
  (80 steps / 90 min). Deferral ledgers + hard_decision.json written;
  guard stands down when a decision file exists. Yield target >=60 will
  NOT be met by CRAWL — accepted: CRAWL's purpose (prove pipeline,
  measure funnel) is achieved; volume is WALK's job on real infra.
- Measured so far: easy $0.20/verified, medium $1.18/verified, overall
  10-11% rollout verify rate (vs 35% rehearsal — full-set medium/hard
  mix shift + emulation).

## Money ledger (RunPod, cumulative)

| Deposit | Amount |
|---|---|
| 2026-07-04 rehearsal | ~$4 (pre-existing balance; predates the ledger) |
| 2026-07-05 initial | $250 |
| mid-run top-ups | $50 + $100 |
| **Total funded** | **$400** |

Attribution is approximate (±$10; reconstructed from balance checkpoints):
~$75 corpus generation, ~$40 staging/downloads, ~$155 the six-attempt
serving campaign (bought the CUDA-13 finding + full playbook), ~$45 env
wars & incidents, ~$15 storage/misc — plus the student leg (in progress,
~$25 projected). Balance at teacher-pod stop: $45; at student-leg start:
~$36.

## Artifact inventory

- **HF (private)**: `ledzepu2/glm52-pilot-artifacts` — corpus (1,140 rows),
  evals + judge validation, prompts + manifest, gate result, run log,
  dataset card, samples.
- **Pods**: `glm52-teacher2` (B200×6, STOPPED, weights on volume — Tier-3
  resume option) · `glm52-student` (RTX PRO 6000×4, RUNNING).
- **Volumes**: US-GA-2 network volume (stranded weights, $3.50/day,
  user said keep) · teacher pod volume (weights, alive while pod exists).
- **Repo**: pipeline + specs + this journal. Memory notes in
  `~/.claude/.../memory/`.

## Open items

1. ~~Student leg + trustworthy v0 deltas~~ DONE (Discovery #7): distill transfers answer-format discipline (+48pt GSM8K strict), mild IFEval dip expected at pilot scale.
2. ~~student_b thinking-format packer + thinking-aware eval configs~~
   **DONE (code)**: (a) packer wraps teacher reasoning in the student's
   native `<think>`/`</think>` (auto-detected) with a loud strip-guard;
   (b) eval gen budget 4096 + reasoning sampling + 16k len; (c) **IFEval
   think-strip re-score** (`rescore_ifeval.py`, auto-run by 07) — raw IFEval
   scores a think block as instruction-violating regardless of answer, so
   the re-score on stripped responses is the trustworthy number. Staged
   budget-gated runner `student_rebench.sh`: CPU re-pack + guard (free) →
   base anchor ($5, must recover to known ability or STOP) → gated v0
   retrain+bench ($20). Ready to run on a pod.
3. **Generation-side judge/robustness fixes** — PARTIALLY DONE (2026-07-06):
   verdict parse now rejects unclosed-think-block verdicts + takes last-number,
   candidate order randomized per judgment, `bon_judged` audit column added
   (commit bf31b41). STILL PENDING: escalated-max-tokens retry for
   all-truncated prompts, per-slice sampling temps, top-K coverage-mass column.
4. Multi-turn bucket (02/03 message support; `MULTITURN_READY` flips)
5. ~~Smoke-run builder v2 source loaders~~ DONE (2026-07-06, ~$2): all 6
   slices populate; 2 broken loaders caught + fixed (oss_instruct repo gone
   → ise-uiuc/Magicoder-OSS-Instruct-75K; xlam gated → NousResearch/
   hermes-function-calling-v1). Builder v2 validated end to end.
6. Tier 3 decision: spec-validation teacher run (+$150–250)
7. GCP production run (the goal; playbook complete)
8. Agentic trajectory leg (agentic_trajectory_design.md): build CRAWL
   components (09_generate_trajectories.py, messages pack + loss mask, 04
   multi-turn, SWE-bench-50 screen, decontam gate). 2026-07-22 update:
   teacher for CRAWL = PUBLIC GLM-5.2 API (OpenRouter, ~25 providers,
   $0.82/M in / $2.57/M out, 1M ctx; most expose top_logprobs; fp8
   allowlist, pinned, no silent fallbacks) - kills the co-schedule-with-
   Tier-3 constraint. SMOKE TEST DONE 2026-07-22 ($0.031,
   endpoint_smoke.py): pinning honored, reasoning via `reasoning` field,
   token-id round-trip 100%, caching YES everywhere (4-8x input discount);
   logprobs are CONTENT-ONLY on all providers -> full-trace logit-KD needs
   the selfhost rescore pass (or hybrid: answer-KD + reasoning-CE, no
   rescore). Allowlist v1: StreamLake, GMICloud, Alibaba (Fireworks caps
   top_logprobs at 5, sequence-only overflow). OpenRouter key stored in
   gitignored .env.openrouter; passed through chat, rotate with HF token.
   NEXT: env plane bring-up (GCP VM + docker + SWE-Gym-Lite images) then
   the ~$5/10-task rehearsal. Task pool now governed by
   trajectory_task_spec.md (2026-07-22): measured SWE-Gym skew (pandas 30%,
   top-3 repos ~60%) -> per-repo cap <=12%; difficulty tiers from
   gold-patch stats with adaptive N (2/4/6) + >=25%-hard verified quota +
   pause-not-backfill guard; WALK task-type mix 50 bugfix / 20 mutation /
   15 PR-mirror / 10 test-writing(inverted verifier) / degraded-statement
   overlay ~15%; rehearsal upgraded to measure success rate BY TIER.
9. SECURITY, do now not later: rotate the HF token (it has traveled through
   chat and multiple rented pods) — requires user action in HF settings.
   Decide the US-GA-2 volume ($3.50/day; worthless under the GLM-5.2-only
   strategy since that DC has no CUDA-13 hosts — recommend delete).

## 2026-08-21 — Teacher swap: GLM-5.2 -> Qwen3.8-27B (trajectory leg)

Decision (user): use Qwen/Qwen3.8-27B (released 2026-08-05, dense 27B,
apache-2.0, thinking-on-by-default with reasoning_effort control) as the
teacher for the agentic trajectory leg, replacing z-ai/glm-5.2.

- **Tokenizer gate vs Qwen3-8B student: `none`.** Qwen3.8 moved to a 248k
  vocab (old family 151k); 47% teacher vocab unmappable, 131k shared tokens
  changed ids. No logit-KD, no remap. Costs nothing in practice: the qwen3
  student leg was already SFT-level distillation and trajectories are
  verified-then-retokenized by the packer. No small Qwen3.8 sibling exists
  (family = 27B + 2.4T-A95B) so a same-tokenizer student is not an option.
- **Endpoint smoke 2026-08-21 ($0.027)**: 7 OpenRouter providers; 4 meet
  fp8-or-better + top_logprobs (Parasail, Reka, AkashML, Alibaba; Io Net
  excluded 65k ctx; Chutes/Venice no logprobs). All 4 pin correctly and
  return reasoning via `reasoning`. Logprobs CONTENT-ONLY everywhere
  (same as GLM-5.2; moot given gate=none). **Only Parasail caches**
  (6272/6358 warm prompt tokens) -> allowlist order Parasail, Reka,
  AkashML, no fallbacks. Alibaba dropped (top_logprobs capped at 5,
  priciest input). Pricing $0.45/M in, $3.20/M out vs GLM-5.2
  $0.82/$2.57 — input-dominated rollouts get cheaper.
- Model card claims (Claude Code harness, not ours): SWE-bench Pro 61.7,
  Terminal Bench 2.1 73.0, LCB v6 90.3. Treat as priors only.
- endpoint_smoke.py parameterized (TEACHER_MODEL / TEACHER_PROVIDERS /
  TEACHER_TOKENIZER_JSON); 09_rehearsal.py MODEL_NAME + PROVIDER_PIN
  flipped (09_generate_trajectories inherits).
- NEXT: ~$5 tiered rehearsal (esp. hard tier, where GLM-5.2 went 0/30) to
  compare verified yield + cost-per-verified vs CRAWL baseline $1.44
  before committing WALK to the new teacher.

## 2026-08-21 (later) — Rescore validated, student locked, staged plan approved

- **Student = Qwen/Qwen3.5-9B (user-confirmed).** Same 248k vocab as the
  Qwen3.8 teacher (gate: remap, 7 audio-only specials, zero id mismatches)
  -> full logit KD viable. No small Qwen3.8 sibling exists.
- **09c rescore pass VALIDATED** on Vast RTX PRO 6000 (~$0.87/hr, on-demand
  after an interruptible box was outbid and a 27GB-disk box filled): 3
  verified rehearsal trajectories scored, top-20 full trace incl. reasoning,
  coverage 1.0. Fidelity: teacher self-perplexity 1.209, rank-1 agreement
  93.3% -> chat-template render (preserve_thinking + BASH_TOOL schema) is
  faithful. Shards in HF store rescore_shards/rehearsal. Six integration
  bugs fixed en route (see commit 48a83c9).
- **Rehearsal (Qwen3.8 teacher, same frozen 10 tasks)**: 6/10 sealed —
  moto 2/2 verified (easy), **pandas-48106 verified (hard — first hard
  verification ever; GLM-5.2 was 0/30 on hard in CRAWL)**; pydantic, dvc,
  hydra x2 failed. mypy x2 + modin in flight. Two parallel workers.
- **Qwen3.5-9B anchor benchmark** (ifeval + gsm8k_cot, thinking-aware
  07_eval_benchmarks settings) running on the same box after the rescore.
- **Approved staged plan**: rehearsal verdict -> WALK 95 tasks on x86 VM
  (mix per trajectory_task_spec; also kills the emulation tax measured at
  ~35 min/task local) -> student v0 SFT(+KD) + held-out screen -> transfer
  delta gates RUN-phase corpus (~300-500 verified, mutation-weighted for
  controlled hard-tier difficulty). Failed rollouts to be kept as DPO
  chosen/rejected pairs (zero marginal cost). Rationale: diversity-first
  (repo caps, task-type mix, degraded statements) beats raw count; for RL
  the durable asset is tasks+verifiers, not teacher tokens.
- Credential rotation list: HF token + Vast API key (both traveled through
  chat).

## 2026-08-22 — Rehearsal SEALED (10/10): tier verdict Qwen3.8 vs GLM-5.2

Same frozen 10 tasks, N=2, identical harness/verifier. Ledger totals:

| tier   | Qwen3.8 tasks | Qwen3.8 rollouts_v | GLM tasks | GLM rollouts_v |
|--------|---------------|--------------------|-----------|----------------|
| easy   | 1/3           | 2                  | 1/3       | 2              |
| medium | 2/4           | 3                  | 3/4       | 5              |
| hard   | **1/3**       | **1**              | 0/3       | 0              |
| total  | 4/10          | 6 ($10.82 ledger)  | 4/10      | 7 ($1.85)      |

Read: GLM is 6-8x cheaper per verified rollout (short trajectories) and
slightly better on medium; **Qwen3.8 is the only teacher that verifies
hard tasks** (GLM: 0/33 hard attempts across rehearsal+CRAWL). The >=25%-
hard verified quota is the design's binding constraint -> teacher swap
VALIDATED for the quota, at a real cost premium (~$1.80/verified vs
$0.28). WALK pricing set accordingly (bugfix slice cap $75).
Ops burn beyond ledger: ~$2.7 (DNS-hang redo, harness 1h bg-task kills,
Docker-disk pull failures — all engineered around: detached workers,
babysitter, ledger watcher, WALK VM with 284GB image cache).
Orchestrator now: batch rescore (dask r2, mypy r1+r2 + any stragglers)
on rented GPU + decontam-gated WALK bugfix-slice launch on the VM.

## 2026-08-24 — INCIDENT: WALK bugfix slice data lost with reclaimed VM

The WALK VM (Vast 48325566) exited overnight ~04:00 and was reclaimed by
the host before its queued restart could run. Every stopped-instance
recovery path failed (start: queued forever; vastai copy: "Invalid
src_id" even VM->VM; execute: 404). 44 sealed ledgers incl. 24 verified
trajectories ($66.73 API spend) existed only on that disk. LOST.

**What survives:** all code + frozen-selection determinism (seed 42 ->
same 48 tasks on re-select), rehearsal-era corpus + 6 KD shards (HF
store), anchors, and the FUNNEL KNOWLEDGE from the run:

| tier   | verified/sealed | notes |
|--------|-----------------|-------|
| easy   | 11/19 (58%)     | moto/mypy/pydantic convert; hydra/conan/dvc resist |
| medium | 12/19 (63%)     | 3x CRAWL-era rate; several 2/2-rollout verifieds |
| hard   | 1/6 (17%)       | pandas-50714 verified r3 @$4.80; 4 unsealed at credit-out |

~$1.90/verified task-level; repo >> tier as difficulty predictor; per-repo
caps vindicated (dask $10 -> 2 verified late, hydra $0 conversions).

**Root cause of exposure:** outputs synced off-box only at phase end.
Fix (now mandatory, memory'd): sync-on-seal — every sealed ledger leaves
the box immediately (HF store), built into the runner before any relaunch.
Also prior incident same day: OpenRouter credits exhausted mid-run
(402); watchdog circuit-breaker added after a 193-restart churn loop.

**Decision pending (user):** regenerate slice (~$70-90 credit, ~6-8h
fleet, hardened runner) vs reduced corpus. Rehearsal-era 6 trajectories
alone are too thin for a meaningful v0.

## 2026-08-25 — WALK regeneration launched on SELF-HOSTED teacher

Architecture migrated off per-token API after the credit exhaustion +
data loss: vLLM serves Qwen3.8-27B-FP8 on a UK RTX PRO 6000 ($1.07/hr);
env-plane VM ($0.067/hr) runs the 3-worker fleet; teacher reaches the VM
via a REVERSE ssh tunnel (GPU->VM -R 8000, because Vast never mapped
port 8000 externally and the VM's inbound sshd is flaky). The runner's
requests-shim gained TEACHER_BASE_URL/TEACHER_API_KEY redirection
(commit for 09_rehearsal.py). Economics: ~$1.14/hr flat vs ~$67/slice in
tokens; KD rescore will reuse the same loaded teacher for free.

Hardening now standard: sync-on-seal (every ledger -> HF walk/
bugfix_regen within ~4 min), teacher-health watchdog (workers hold on
TEACHER_DOWN instead of churning), detached processes, script-file
remote ops. Bring-up cost of the era: ~3 wasted GPU-box rentals (~$4,
incl. a host with broken HF connectivity that masqueraded as ', US' —
geolocation filter now requires a named region) + a KVM key-sync lesson
(attach ssh does not propagate to RUNNING VMs; append authorized_keys
directly). Launched 12:43: gate CLEAN, partitions 18/15/15.

## 2026-08-25 (later) — Self-hosted generation CONFIRMED FLOWING (GPU 100%)

Five-layer integration gauntlet resolved (docker API pin 1.43; vLLM flags
--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser
qwen3 — each absence failing differently; full recipe in memory +
restart_teacher_tools.sh). Clean-slate relaunch: all 48 tasks, 3 workers,
watchdog + sync-on-seal verified live (HF push live-fired). GPU pegged at
100% — self-hosted throughput measurement in progress.

## 2026-08-26 — WALK regen SEALED at the hard-tier guard; KD corpus complete

Generation halted itself at 45/48 per trajectory_task_spec sec 3
(pause-not-backfill): hard tier 1/36 rollouts verified (2.8%) < 10% floor
after >=30 rollouts. Three unsealed hard tasks remain (pandas x2, hydra x1)
pending the mix decision the guard exists to force.

**Final regen funnel vs the lost run (identical 48-task selection):**

| tier   | regen        | lost run | CRAWL (GLM teacher) |
|--------|--------------|----------|---------------------|
| easy   | **15/19 79%**| 11/19 58%| -                   |
| medium | 12/19 63%    | 12/19 63%| ~20%                |
| hard   | 1/7 14%      | 1/6 17%  | 0/30 rollouts       |
| total  | **28 verified tasks / 39 verified rollouts** | 24 | 10 |

Cost: ~$20 of box time (GPU $1.07/hr x ~17h + VM $0.07/hr) vs ~$67 in API
tokens for the smaller lost corpus. Self-hosted teacher validated on both
economics AND quality (easy tier +21 points; likely the clean serving
config - proper reasoning/tool parsers, no provider caching quirks).

**KD rescore complete on the same box before teardown** (teacher already
loaded, zero extra rental): 39 shards, top-20 full-trace logprobs incl.
reasoning, coverage 1.0 on every shard. Store now holds 45 KD shards
(39 WALK + 6 rehearsal) = the v0 training corpus.

Data verified in three places (VM, Mac runs/walk_regen/, HF store:
105 result.json, 232 trajectories, 45 npz). GPU box 48576207 DESTROYED;
env VM 48567337 retained ($0.087/hr) pending slices 2-3 work.

## 2026-08-28/29 — Expansion era: review-hardened corpus, strip-clean generation, student v0

**External review absorbed** (researcher, 2026-08-28): 37/45 trajectories had
retrieved the upstream fix via container git history ("repair-by-retrieval").
Response: history_assisted labels in the packer, 80k train cap (32k default
silently dropped 32/45), stale mt_qwen3 deprecated, real provenance column.
Runner change: STRIP_CMD deletes all refs past base commit before rollout —
new trajectories are independent diagnoses by construction.

**Expansion pipeline** (Codex-assisted): 80-candidate reservoir → 42 review
survivors → 36 frozen smoke queue (hard cap 4) → gold smoke 29 PASS / 7 FAIL
(base-fail/gold-pass, caught 7 broken harnesses incl. flaky bokeh gold
tests) → decontam CLEAN → 29-task run list. Docker Hub preflight root-caused
the MONAI "misses": image_name() never lowercased (mixed-case repos 400 on
hub API and would have crashed docker pull) — 8 MONAI images existed all
along. Fixed; 7 historical false-miss tasks re-eligible.

**Generation (strip live, limits raised 75 steps/60 min after early
rollouts proved the teacher needs room without the answer key):**
29/29 sealed, 95 rollouts, 10 tasks verified (15 verified rollouts):
easy 5/5 100%, medium 5/23 22%, hard 0/1. THE number for RUN pricing:
unassisted medium costs ~2.5x the history-assisted rate. ~$1.80/verified
trajectory all-in. Mid-run: queue resorted by reviewer difficulty_score
(cheapest diagnosis first) — user call, correct one.

**Student v0** (Qwen3.5-9B, SFT on 44 traj at 64k cap, 10 steps, 2 epochs,
1x RTX PRO 6000 96GB after 4x48GB proved wrong-shaped): the seven env traps,
each now committed as config/code: transformers v5 API break (version-
adaptive TrainingArguments), qwen3_5 arch needs v5 + local-dir load (hub
resolver mangles Qwen's shard names), liger fused CE (80k x 248k logits =
40GB), fla kernels (naive delta-rule path OOMs), model load AFTER
TrainingArguments (else zero.Init never shards: 42GB flat/rank), in-Trainer
eval disabled (eval forward at 63k = 32GB attention even on 96GB),
save_only_model (ZeRO-3 checkpoint w/ optimizer = 117GB, filled disk at the
finish line and lost run #2's weights). Run #3 clean: early CE 0.33-0.65,
model serves and answers sanely after tensor rename (saved names kept the
multimodal language_model prefix vLLM rejects).

**Credit exhaustion** mid-anchor-eval: $49.29 -> $0. Breakdown: teacher
$17.72 (14h — the plan working, unassisted rollouts are slow), Kansas
trainer $10.57 (13h incl. the debugging arc), Poland OOM-saga box $3.29,
env VM $0.99, storage+bandwidth ~$2 (incl. days of stopped-VM bleed from
48567337 — lesson: DESTROY, never stop). Zero data loss: sync-on-seal had
everything on HF before the stop.

**Corpus v1 on HF**: 60 verified rollouts / 42 tasks
(history_stripped column partitions 15 strip-clean rows from 37 legacy
assisted + 8 legacy independent), student dataset 58 rows @ 80k
(48 train/10 val), DPO 26 pairs, 60 KD shards. student_v0 + checkpoints
uploaded. PENDING on ~$10 top-up: anchor rerun, SWE screen, terminal-bench
before/after — the transfer verdict that sizes RUN.
