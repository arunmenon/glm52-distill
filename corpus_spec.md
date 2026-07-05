# Production Corpus Specification

Decided 2026-07-05 (user decisions + tri-lens representativeness review).
Target: 100–150k prompts + 1,000 held-out. Supersedes the pilot's 50/25/25
three-source mix for all production builds.

## Mix

| Slice | Share | Sources (per-source cap 20% of slice) | Notes |
|---|---|---|---|
| coding | 40% | OpenCoder-LLM/opc-sft-stage2, m-a-p/CodeFeedback-Filtered-Instruction, bigcode/self-oss-instruct-sc2-exec-filter-500k, nvidia/OpenCodeInstruct (capped), repo-level tasks (SWE-Gym) | "Practical assistant" profile: real code, debugging, multi-language, SQL — not competition monoculture |
| math | 20% | AI-MO/NuminaMath-CoT (≤50% of slice), nvidia/OpenMathInstruct-2 | Add applied/statistics flavor; Numina capped |
| general | 20% | allenai/tulu-3-sft-mixture, FILTERED: exclude persona_math, numinamath_tir, evol_codealpaca, persona_python (mix distortion) and WildJailbreak/WildGuardMix/CoCoNot subsets (safety policy below) | True general: writing, chat, knowledge QA, summarization |
| tool/function calling | 5% | Salesforce/xlam-function-calling-60k or glaiveai/glaive-function-calling-v2 | Flagship agentic skill |
| multi-turn dialogue | 5% | allenai/WildChat-1M conversation prefixes (full message list) | Requires pack-format extension (multi-message prompts in 02/03) |
| long-context | 5% | LongAlign/LongWriter-style task prompts | Char cap raised to ~32k tokens for this slice only; generation + pack budgets must follow |
| structured output + science | 5% | JSON-schema/extraction prompts + TIGER-Lab/WebInstruct-verified science | MMLU-Pro-adjacent reasoning |

## Language
English-only — deliberate decision (documented here, not an accident of
source choice). Revisit only with a new decision.

## Held-out
1,000 prompts, stratified across ALL slices (incl. new buckets), computed
from ACTUAL post-trim counts (not targets), flooring remainder assigned to
largest slice. Frozen once generated; never rebuilt.

## Safety policy (default pending explicit revisit)
Jailbreak/safety subsets EXCLUDED from the general sample. No safety bucket
in v1 production build. If safety distillation becomes a goal: curated
bucket with response-side filtering (keep teacher refusals on harmful,
compliances on benign).

## Sampling & hygiene rules (from the review; apply to 01/01b)
1. Buffered stream shuffle (seed 42, buffer ≥100k) on every source — never
   head-of-stream. Record per-source consumption depth in MANIFEST.
2. Benchmark decontamination in BASE builder and extension: GSM8K, MATH-500,
   IFEval, MMLU-Pro, HumanEval, MBPP, LiveCodeBench; exact + MinHash 0.8 +
   numerals-stripped normalization for template clones. Mandatory.
3. Cross-slice dedup: round-robin insertion (not fixed slice order); log
   per-slice collision counts.
4. Short prompts (<15 tokens): exact-hash dedup only (MinHash unreliable).
5. Slice shortfall: hard fail below 98% of target unless --allow-short.
6. Per-slice funnel report every build: prompts → generated → survived
   truncation → judged vs fallback → packed (per student).

## Generation-side fixes required before production run (from review)
- Judge verdict parse anchored after </think> only; bon_judged column;
  randomized candidate order; head+tail truncation for judge view.
- Escalated-max-tokens retry for all-truncated prompts; per-slice drop
  logging with reason codes (truncated/empty/error) + salvage pass.
- Generation MAX_TOKENS coordinated with pack --max-len (no guaranteed
  casualties); per-slice sampling temperature.
- Store per-position top-K coverage mass (sum of exp(topk logprobs)) for
  KD tail correction.
