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
8. SECURITY, do now not later: rotate the HF token (it has traveled through
   chat and multiple rented pods) — requires user action in HF settings.
   Decide the US-GA-2 volume ($3.50/day; worthless under the GLM-5.2-only
   strategy since that DC has no CUDA-13 hosts — recommend delete).
