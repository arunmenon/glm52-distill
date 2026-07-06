# Trace-Recipe Review (adversarial, 2026-07-06)

Multi-dimension review + 3-vote verification (43 agents, 34 confirmed findings).
Question: does the pipeline extract ROBUST + DIVERSE traces from GLM-5.2?

# DECISION-GRADE REVIEW: Does the pipeline yield a recipe for robust + diverse GLM-5.2 distillation traces?

## 1. VERDICT

**YES-WITH-GAPS.**

The pipeline is a genuine, working end-to-end recipe. The architecture is right: diverse multi-source prompt mining with real decontamination and dedup, best-of-N teacher generation with a correctness judge, top-20 logprob capture for logit-KD, and a two-student packing stage (logit-KD for GLM-4.5-Air, sequence/remap-KD for Qwen). None of the verified findings is a foundational design flaw. Every load-bearing issue is a bounded, well-localized fix. This is not a NO.

But it is not an unqualified YES either. There is a cluster of load-bearing gaps that (a) silently bias the corpus toward easier prompts, (b) narrow coverage below what the spec claims, and (c) make a narrowed corpus hard to detect after the fact. These must be addressed before spending Tier-3 money, because several of them corrupt the very signal a validation run exists to measure.

---

## 2. THE RECIPE (what the pipeline gets right)

1. **Diverse prompt mining (01_build_prompts.py).** Seven capability slices (coding, math, general, structured/science, long-context, tool-calling, multi-turn) drawn from ~15+ distinct HF sources, each with a per-source cap to resist single-source monoculture.
2. **Hygiene on the prompt pool.** Buffered stream shuffle (seed 42, buffer 100k) instead of head-of-stream; exact md5 dedup; global MinHash LSH (3-gram, threshold 0.8) for near-dup collapse; a short-prompt exact-only guard so MinHash does not misfire on <15-token text.
3. **Benchmark decontamination.** Exact-normalized, numerals-stripped (template-clone), and MinHash screens against GSM8K/MATH-500/IFEval/MMLU-Pro/HumanEval/MBPP so train prompts do not leak into scored evals.
4. **Best-of-N teacher generation (02_generate.py).** N candidates per prompt at temp 0.7/top_p 0.95, truncated candidates dropped, a correctness>completeness>clarity judge selects one winner. Per-sample deterministic seeds for reproducible inputs.
5. **High-fidelity signal capture.** Top-20 (token_id, logprob) pairs per position captured for logit-KD, plus the full response token stream, with thinking content preserved inline.
6. **Two-student packing (03_pack_dataset.py).** student_a consumes response_token_ids + top-20 arrays for logit-KD to GLM-4.5-Air (same-family, no remap); student_b re-tokenizes for sequence-KD to any Qwen model, with a teacher→student vocab remap path for the tokenizer-divergent case.
7. **Operational durability.** Shards written atomically (tmp+rename), a MANIFEST ledger for resume, per-seal shard push for pod-loss protection, held-out split gated by a hard assert.
8. **Scale-up path (01b_extend_prompts.py).** Reuses the base slice definitions and decontam to extend the corpus while excluding base duplicates.

That is a coherent recipe. The gaps below are about how much of the intended diversity actually survives, and how much of the hardest signal is silently lost.

---

## 3. ROBUSTNESS GAPS (per-trace quality / correctness / fidelity), ranked

**R1 (load-bearing). 8192-token truncation drop removes the hardest, longest-reasoning traces.** CONFIRMED across multiple dimensions. MAX_TOKENS=8192; any candidate with finish_reason='length' is discarded, and a prompt whose every candidate truncates is dropped entirely. GLM-5.2 is a thinking model whose CoT routinely exceeds 8192 on hard math/coding, so the corpus has a hard difficulty ceiling exactly where teacher signal is most valuable. No length-based retry/escalation exists. **Fix:** raise MAX_TOKENS to the model's practical limit, escalate on finish_reason='length' before dropping, and log dropped-by-truncation counts per slice.

**R2 (load-bearing). Empty / near-empty shards are sealed as complete and never regenerated.** CONFIRMED. write_shard seals unconditionally after asyncio.gather with no floor on kept-rows; resume skips by filename forever. A sustained vLLM outage across an in-flight shard can permanently delete up to a full shard with no exception or alarm. **Fix:** require kept/total ≥ a floor (e.g. 0.5) before adding a shard to the manifest; otherwise leave it for regeneration and exit non-zero.

**R3 (load-bearing for the Qwen/remap student). Remap drops the whole sample on any single unmappable token, non-uniformly by language/notation.** CONFIRMED. One unmappable response token discards the entire sample; unmappable tokens cluster in CJK/rare-Unicode/math/emoji, so whole samples in those domains are lost wholesale while the global rate stays <1%. This is both robustness and diversity. **Fix:** mask only the offending label position (or fall back to byte/merge-level remap) instead of dropping the sample; report drops per slice/script, gate on the worst slice not the mean.

**R4 (medium). MANIFEST.json — the resume gate — is written non-atomically.** CONFIRMED. Plain write_text; a crash mid-write torns the JSON and makes the run un-resumable (or forces a full re-run). Cheap to fix. **Fix:** write to .tmp and os.replace(); guard the resume-time load to fail loudly on a torn file.

**R5 (medium). Judge verdict can be read from truncated reasoning_content, recording a spurious pick as trusted.** CONFIRMED. When judge output truncates mid-reasoning, raw falls back to reasoning_content, the unclosed-think guard passes (no literal tags present), and the last in-range digit is taken as a trusted verdict (bon_judged=True). **Fix:** only scan the answer channel (content) for the verdict; if content is empty, treat as unparseable and fall back. Only bites at best_of>1.

**R6 (medium). Empty-answer filter inspects full think+answer, so "reasoned but never answered" traces pass into training.** CONFIRMED. The filter tests raw content (which includes the <think> block) for non-emptiness, so a trace that reasons, closes </think>, then stops with no answer is kept and trained on. **Fix:** apply the existing strip_think before the emptiness test and require a non-empty post-</think> answer.

**R7 (low). Last-number verdict heuristic can pick a wrong in-range candidate.** CONFIRMED but low-frequency (temp 0 + explicit instruction). **Fix:** anchor the verdict to a trailing bare number (re.fullmatch) rather than "last digit anywhere."

Note: the student_a "loses CoT if a reasoning parser is enabled" finding (PARTIAL) is a latent hardening gap only — the pipeline explicitly runs with no parser and preserves thinking inline; add a manifest assertion but it is not a live defect.

---

## 4. DIVERSITY GAPS (coverage narrowing / homogenization), ranked

**D1 (load-bearing). The shipped capability mix is not the spec mix.** CONFIRMED. MULTITURN_READY=False drops the entire 5% multi-turn slice and parks its share on coding, yielding 45/20/20/5/5/5/0 instead of the intended 40/20/20/5/5/5/5. Conversational capability is unrepresented; coding is over-dominant. This is a spec-sanctioned deferral (02/03 lack message-list support), not an accidental bug, but a validation run would validate the wrong mix. **Fix:** either build the WildChat message-list path before the run, or redistribute the 5% across general/long-context/tool rather than piling it on coding; at minimum surface the 45% coding share loudly in the manifest.

**D2 (load-bearing). structured_science has zero structured-output source.** CONFIRMED. The slice named "structured output + science" is filled only by WebInstruct (science QA) and sciq (MC factoids). No JSON-schema/extraction source exists anywhere, so a labeled 5% capability is entirely unbacked and reported coverage overstates actual coverage. **Fix:** add a real schema-constrained extraction / JSON-generation source and cap sciq.

**D3 (load-bearing). No LiveCodeBench decontamination, and coding is the largest slice.** CONFIRMED. BENCHMARKS omits LiveCodeBench, which the spec marks mandatory; yet 07_eval runs LiveCodeBench as a scored eval on every checkpoint. Train/test leakage into the 45% coding slice can inflate the reported delta with no screen to catch it. This directly corrupts spec validation. **Fix:** add a LiveCodeBench entry to BENCHMARKS (inherited by 01b).

**D4 (load-bearing for scale-up only). Extension builder reuses shuffle seed 42, starving the scale-up.** CONFIRMED. 01b draws the identical stream prefix in identical order as the base run, then discards it all as base duplicates, yielding a near-empty residual and firing shortfall warnings on every slice. Adds neither diversity nor volume. Not load-bearing for a single 10-20k base run, but blocks any scale-up. **Fix:** thread a distinct shuffle seed (SEED + offset) or skip past base's consumed depth per source.

**D5 (medium). Single-winner best-of-N + judge is rejection sampling that reduces retained variance.** PARTIAL. Real effect but opt-in (default best_of=1) and per-prompt; corpus diversity is dominated by prompt diversity. For a coverage corpus, keeping only argmax discards N-1 valid-but-different reasoning paths. **Fix:** for coverage data, keep all correct candidates as separate rows (use the judge as a correctness gate, not a top-1 selector); reserve strict best-of-1 for a quality-critical subset.

**D6 (medium). One fixed sampling config (temp 0.7 / top_p 0.95) for every domain.** CONFIRMED. `slice` is read only for labeling, never to modulate sampling; the same conservative setting applies to creative, math, and code. Softened by top-20 logprob capture (student sees soft tail) and best-of-N. **Fix:** make temp/top_p per-slice; optionally sample open-ended slices at higher temperature or multiple temperatures.

**D7 (medium). Per-source caps applied before dedup; realized source mix is skewed.** CONFIRMED. Caps are collection-time upper bounds enforced before MinHash and benchmark screening, and the kept set is never re-balanced by source. With source-order trim + 1.35x over-collection, early sources exceed 20% and the last source can be fully starved, so the anti-monoculture guarantee fails post-dedup. **Fix:** enforce caps on the post-dedup kept set (round-robin across sources within a slice); report realized per-source proportions.

**D8 (medium). Buffered shuffle samples only a ~100-125k head window of multi-million-row sources.** PARTIAL. Real coverage limit on the giant sources (OpenCodeInstruct ~5M, OpenMathInstruct ~14M), partly mitigated by HF shard-order randomization and per-source caps. The spec's own 100k buffer is weak, not violated. **Fix:** use a buffer comparable to a large fraction of source size, or interleave shards, and warn when budget is met inside the first buffer.

**Lower-priority diversity notes (nice-to-have, not load-bearing):** math is a competition/grade-school monoculture missing the applied/stats flavor (D-tier); long_context is half long-output not long-input; general slice is single-source Tulu; coding has no repo-level/hard tasks and no difficulty axis; coding caps sum to exactly 1.0 with two overlapping m-a-p sources. Each is a real but modest coverage observation; none blocks a validation run.

**Fidelity note (top-20 truncation, PARTIAL):** no residual-mass bucket is recorded, so high-entropy positions look identical to confident ones. The KL is correctly renormalized over the shared top-20 (the "over-sharpening" mechanism in the finding is inaccurate), so this is a milder calibration gap, not a corruption. Nice-to-have: record a per-position residual bucket.

---

## 5. TIER-3 GO / NO-GO

**GO, conditional on a small must-fix-first set.** The recipe is sound enough that a 10-20k spec-validation run is worth the money — but only after closing the items that would either silently narrow the corpus, bias it easier, or make the resulting eval untrustworthy. Spending Tier-3 dollars to validate a spec while (a) validating the wrong capability mix, (b) leaking coding benchmark data, or (c) being unable to detect a narrowed corpus afterward would waste the run.

**MUST-FIX-FIRST (blocking, small set):**

1. **R1 — Raise MAX_TOKENS + truncation telemetry.** Without this the corpus is silently biased away from the hardest prompts, which is precisely what a distillation corpus most needs. Non-negotiable.
2. **D3 — Add LiveCodeBench decontamination.** Coding is the largest slice and LiveCodeBench is a scored eval; leakage invalidates the headline coding result of the validation run itself.
3. **R2 + observability (merge with the "no reconciled funnel" and per-slice drop-reason findings) — floor on kept-rows before sealing, plus a persisted per-slice funnel.json that records drop reason (truncation vs timeout vs empty vs unmappable) by slice.** A validation run exists to measure coverage and loss; right now those numbers live only in ephemeral stdout and cannot be reconciled. This is cheap and is the single highest-leverage observability fix.
4. **D1 + D2 — Fix the shipped mix to match the spec you intend to validate.** Either restore multi-turn or explicitly redistribute its 5% (do not leave coding silently at 45%), and add a real structured-output source. Otherwise the run validates a mix the spec does not describe.
5. **R4 — Atomic MANIFEST write.** One-line fix; prevents an un-resumable run that wastes the entire Tier-3 spend on a mid-run crash.
6. **R3 (only if the Qwen/remap student is in scope for this run) — per-slice unmappable accounting + position-masking instead of whole-sample drop.** If the validation run targets only GLM-4.5-Air (same-family logit-KD, no remap), this is deferrable. If any Qwen/remap student is being validated, it is blocking, because non-ASCII slices can be near-totally lost while the global rate reads <1%.

**Explicitly NOT blocking for Tier-3** (fix after, before full-scale production): D4 (extension seed — only matters at scale-up), D5/D6/D7/D8 (sampling and cap refinements), R5/R6/R7 (judge edge cases, low frequency at best_of=1), top-20 residual mass, and all lower-priority math/long-context/general coverage notes.

Bottom line: the recipe works. Close the six must-fix items above — most are hours, not days — and the Tier-3 validation run will measure what it is supposed to measure.