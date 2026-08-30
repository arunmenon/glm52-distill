# Review request: autonomous research loop, before we trust it with a recipe sweep

## Context

Public repo: https://github.com/arunmenon/glm52-distill (branch `master`, commit `684fea3`). Clone it; all referenced files are in the repo root.

We distill agentic bug-fixing from a self-hosted teacher (Qwen3.8-27B) into Qwen3.5-9B. Current state: a 60-trajectory verified corpus exists, and a first SFT student ("v0", `04_train_kd.py`, alpha=0) measured: GSM8K 77.6 vs 78.9 base (held), IFEval think-stripped 35.9/47.7 vs 49.7/58.5 base (regressed ~12 pts — format collapse from 44 same-format training rows). An 8-task agentic SWE screen (base vs v0, gold-test verified) is in flight.

We now want to run an **unattended recipe sweep** to fix the regression while keeping transfer: axes like general-data mix ratio (`04`'s new `--mix-data`/`--mix-ratio`), LR, epochs, and later KD alpha. The repo contains an autonomous sweep loop built in an earlier phase of this project (single-turn distillation on RunPod pods). **Your job: review that loop thoroughly — as-built correctness AND fitness for the new use — before we let it spend money overnight.** Assume real bugs exist; an earlier external review of another subsystem found several.

## Files to review (in this order)

- `autoloop_design.md` — the design contract (deterministic-grid FSM spine + ASHA-style successive halving, content-addressed artifacts, LLM proposer demoted to offline tool)
- `08_conductor.py` — the implementation: FSM tick, `ledger.jsonl` resume, HALT sentinel, balance-reserve check, per-trial spend caps, ssh executor
- `providers.py` — provider abstraction (RunPod-era; we now run on Vast.ai single boxes)
- `objective.py` — trial scoring
- `preflight.py` — pre-sweep validation / cap derivation
- `experiments.yaml` — the declared grid format
- `watchdog.sh` — on-pod dead-man (RunPod API `podStop`)
- Consumers it drives: `04_train_kd.py` (note the freshly added `--mix-data` block), `05_eval_gen.py`, `06_judge.py`, `07_eval_benchmarks.sh`, `rescore_ifeval.py`

## What to check (priority order)

1. **Ledger/resume correctness.** Kill -9 at any point: does re-run resume exactly, never re-spending a completed trial, never double-counting spend? Is `ledger.jsonl` append crash-safe (partial-line handling)? Are trial IDs stable across plan edits (content-addressed as the design claims, or positional — and if positional, what happens when a row is inserted mid-plan)?
2. **Spend-guard integrity.** Trace every code path that starts paid work: is the balance-reserve check actually before EACH deploy/trial (not just sweep start)? Do caps use live-queried balance or stale numbers? What happens when the provider API errors mid-check — fail-open or fail-closed? (Our account hit $0 mid-run once; Vast pauses instances at zero.)
3. **Determinism claims vs reality.** The design promises bit-identical replay below `experiments.yaml`. Verify: seeds threaded into 04/05/06/07? vLLM sampling in evals is NOT deterministic across restarts even seeded — does the promotion logic (rung argmax) tolerate that, and are ties broken deterministically? Any wall-clock/`time.time()` leaking into decisions?
4. **Successive-halving statistics — the biggest fitness question.** Rung-0 proxies were designed for single-turn evals with thousands of items. Our new promotion metrics are noisy and small: IFEval has 541 prompts, and the agentic SWE screen has **8 tasks** (binary verify each). At n=8, the difference between 1/8 and 3/8 is not statistically meaningful. Is the halving logic going to promote noise? What rung sizes / metric choices would make promotion decisions sound — and does `objective.py` support composite gates (e.g. "anchor floor as hard constraint + screen as ranking") or only scalar argmax?
5. **RunPod → Vast portability.** Enumerate every RunPod-specific assumption: `providers.py` API calls, `watchdog.sh` `podStop`, env/paths (`/workspace`, netvol), deploy flow. We intend to run trials sequentially on ONE long-lived Vast box, no deploys — which conductor code paths does that bypass, and do any guardrails (dead-man, balance check) silently vanish in that mode? What is the MINIMAL safe adaptation?
6. **The new `--mix-data` block in 04** (committed today, untested): labels train on ALL tokens of mix rows including user/system spans (deliberate simplification, flagged in-comment) — assess the risk; check the `n_mix` ratio arithmetic; check schema compatibility when mix rows carry `None` in trajectory-only columns through `KDCollator`; check `apply_chat_template` failure handling.
7. **Anchor-floor gate.** The design says "base-anchor quality floor" is code-enforced. Where? Does a trial that catastrophically regresses IFEval actually get killed/rejected, or only scored low? Given IFEval regression is the very thing we're sweeping against, this gate is load-bearing.
8. **Single-box sequential mode hazards.** GPU state leakage between trials (zombie vLLM/engine processes holding memory — we hit this twice), disk accumulation across trials (checkpoints filled a 150GB disk once — `save_only_model` now mitigates but conductor should enforce cleanup), HF cache growth.

## Known context (don't re-report)

- The loop has ONE prior successful campaign (single-turn, RunPod, ~$19, journal Discovery #7).
- We know providers.py is RunPod-specific; the question is the minimal-safe-adaptation, not "it doesn't support Vast."
- Budget scale for the intended sweep: ~$20–25 total, ~$1–2/trial, one RTX PRO 6000 96GB box at $0.80/hr.

## Output format

Ranked findings: `[severity: blocker/major/minor] — file:line — what's wrong or unfit — concrete fix`. Then a short **"minimal safe adaptation plan"** section: the smallest set of changes to run a 12-trial mix/LR/epochs sweep on one Vast box with all guardrails live. Label speculation as such — we re-verify every blocker/major before acting.
