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
- **Discovery #6 — the pilot's most instructive negative result**: v0
  REGRESSED vs base (GSM8K flexible 19.9→5.1, IFEval strict 34.9→29.4),
  but base's own scores are ~4x below Qwen3-8B's known ability. Diagnosis:
  (a) lm-eval defaults are thinking-model-hostile (gen budget + answer
  extraction) — both models' scores are artifacts; (b) student_b packing
  flattens GLM-format reasoning into plain content, teaching the student to
  break its native `<think>` convention. Fixes are code (thinking-aware
  eval config; format-mapping packer) — found for $25 instead of after a
  100k-corpus production run. → Open items.
- US-GA-2 stranded volume deleted (user call). Repo under git as of today.
- Pod stopped. Balance at leg close: **$20.29**.

---

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

1. Student leg completion → v0 benchmark deltas (the go/no-go evidence)
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
3. **Generation-side judge/robustness fixes** (corpus_spec.md §"Generation-side
   fixes"): verdict parse anchored after `</think>` only, `bon_judged` audit
   column, randomized candidate order, judge head+tail truncation,
   escalated-max-tokens retry for all-truncated prompts, per-slice sampling
   temps, top-K coverage mass column. NOT in code yet — the production run
   inherits a 20% silent judge fallback until this lands.
4. Multi-turn bucket (02/03 message support; `MULTITURN_READY` flips)
5. Smoke-run builder v2 source loaders (`--n-total 200`, CPU)
6. Tier 3 decision: spec-validation teacher run (+$150–250)
7. GCP production run (the goal; playbook complete)
8. SECURITY, do now not later: rotate the HF token (it has traveled through
   chat and multiple rented pods) — requires user action in HF settings.
   Decide the US-GA-2 volume ($3.50/day; worthless under the GLM-5.2-only
   strategy since that DC has no CUDA-13 hosts — recommend delete).
