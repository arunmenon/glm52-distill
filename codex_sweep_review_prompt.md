# Review request: rebuilt sweep conductor — verify it actually closes your 13 findings

## Context

Public repo: https://github.com/arunmenon/glm52-distill (branch `master`, commit `679b2d0`). Clone it.

You previously reviewed this repo's autonomous research loop (`08_conductor.py` era) and returned 13 ranked findings plus a "minimal safe adaptation plan" (your verdict: do not run as built). We rebuilt the loop to your spec as **new files** — the old conductor is retained untouched for history. This is the follow-up review: **adversarially verify the rebuild before it is allowed to spend money.** Treat your own prior findings as the checklist; your job is to find where the rebuild claims to close a finding but doesn't, plus any new defects the rewrite introduced.

## Files to review

- `08_sweep.py` — the new conductor (~380 lines): frozen plan digest, full-input trial ids, per-attempt atomic JSON files, live Vast credit guard, generated per-trial remote script (heartbeat, GPU/disk preflight, train → tensor rename → eval → parse → atomic result), process-group kill, two-stage eligibility gate, no-partial-ranking
- `sweep_plan.yaml` — the 12-config plan (mix_ratio × lr × epochs), eligibility floors, budget caps
- `08_sweep_smoke.sh` — the five fault-injection scenarios you prescribed
- Consumers: `04_train_kd.py` (note the `--mix-data` hardening at `c7c093c` responding to your finding 9), `07_eval_benchmarks.sh`, `rescore_ifeval.py`, `configs/ds_zero3_1gpu_optoffload.yaml`
- Your original findings for cross-reference: the repo owner has them; assess against the numbered list from your prior review

## Deliberate deviations from your plan (challenge them if wrong)

- **Full-FT, not LoRA**, for all 12 trials: the grid has no LoRA axis, and full-FT keeps results comparable to the already-measured v0 (same recipe family). ~65 min/trial on the 96GB box was measured.
- **Findings 12 (judge) and 13 (weight column) left open**: judge is not in this sweep's scoring path; weights change the trained distribution equally across recipes.
- Trial-level dead-man is conductor-side (heartbeat staleness → process-group kill) rather than in-instance Vast self-stop; the account-level backstop is Vast pausing at $0.

## What to check (priority order)

1. **Re-audit each of your blockers 1–6 against the new code.** For each: point to the exact line that closes it, or show the input/crash/kill sequence that still defeats it. Special attention to:
   - Resume semantics: conductor killed between `atomic_write(running)` and the ssh launch; killed during polling; killed after remote success but before local attempt write. Does every path reconcile correctly on rerun (kill_stale → retry) without double-spend or false success?
   - The remote launch line (`setsid bash ... &` over ssh): does the process group survive ssh exit? Does `kill_stale`'s `pkill -9 -g $(pgrep -f marker)` actually kill the group — check the argv the marker lands in, the self-match hazard, and the empty-pgrep case (`kill -9 -g ""`).
   - `guard_money`: fail-closed on requests exceptions? Is `r.raise_for_status()` reachable for auth failures that return 200-with-error-body (Vast API quirk)? Is dph re-read or stale across a long sweep?
   - Trial-id integrity: `sha12(cfg)` — does cfg actually include everything that affects results (chat template? tokenizer revision? `07`'s GEN_KWARGS? lm-eval version)? What silently changes results without changing the id?
2. **The generated remote script** (`TRIAL_SCRIPT`): quoting/escaping bugs in the `.format()` (json/bash braces are doubled — verify all of them), the heredoc-in-heredoc layering, `fail()` writing result.json non-atomically vs the conductor's read race, heartbeat subshell lifetime, whether `grep -q ^{code_rev}` can false-positive, whether eval-results glob picks the right file when reruns leave old dirs.
3. **Gate arithmetic**: floors from `sweep_plan.yaml` vs the measured numbers (base IFEval-ts 0.497/0.585, v0 0.359/0.477; GSM8K base 0.789, v0 0.776). Is `ifeval_prompt_strict_ts_min: 0.45` a sane eligibility floor for ranking *recovery*, or does it exclude configs we'd want to see ranked? Is `gsm8k_strict_min: 0.74` consistent with the "within ~5 pts" intent?
4. **The smoke script**: will each scenario actually exercise the failure it claims (scenario 2's `VAST_API_KEY=broken_key` — does the conductor read the env at the right time?), the placeholder `PORT`/`IP` in scenario 1's check, `timeout` vs the conductor's own polling cadence, scenario 3's tautological `|| true`. Rewrite any scenario that cannot fail.
5. **Budget realism**: 12 trials × (train ~65 min + eval ~90–120 min) on a $0.80/hr box against `per_trial_cap_usd: 3.50` and `trial_wall_cap_s: 14400` — do the caps bind before natural completion for the epochs=2 + mix configs (longer train, bigger dataset)? Estimated total spend vs the stated ~$25–30.
6. **New-code defects**: anything the rewrite introduced that the old conductor didn't have — races, path assumptions (`/root/repo`, `/venv/main`), the `Box` port lookup shape against Vast's actual `instances/{id}` response, `HALT` semantics between vs during trials.

## Output format

Ranked findings: `[severity: blocker/major/minor] — file:line — defect — concrete fix`. Then a **verdict line**: "cleared to run after fixes X,Y,Z" or "do not run", plus a corrected version of any smoke scenario you rewrote. Label speculation; we re-verify every blocker/major before acting.
