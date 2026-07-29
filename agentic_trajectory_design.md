# Agentic Trajectory Leg (design, 2026-07-22)

Status: DESIGN, not yet in any approved plan. Confirmed absent from
experiments.yaml / autoloop_design.md / corpus_spec.md as of this date: the
current program is 100% single-turn KD. This document specifies the second
program on top of it: teaching a student the agentic *loop* (act, observe,
recover, iterate) by distilling GLM-5.2 run as an agent in executable repo
environments.

## 0. Why a separate leg (not an extension of the corpus)

The single-turn corpus (corpus_spec.md) transfers reasoning and answer
discipline. It contains ~0% of the behavior that makes an agentic coder:
multi-turn tool use with environment observations, error recovery, repo-scale
context. That behavior cannot be re-weighted into existence; it needs a
different generator (harness + environments instead of prompt list + vLLM
batch), a different verifier (tests pass, not an LLM judge), and a different
pack format (message lists with loss masking). Three of four pipeline stages
change, hence: a leg, not a slice.

Non-goals for v1: RL (we keep the hooks: failed rollouts are stored with
rewards for later DPO/GRPO, but v1 trains SFT/KD on verified successes only);
non-coding agentic domains (browser, terminal-bench style); multi-agent.

## 1. Architecture: two planes

RunPod GPU pods are themselves containers; Docker-in-Docker is not available.
Executable repo environments REQUIRE real Docker on a real VM. So the leg is
split-plane, and providers.py already models this:

- **Inference plane**, two interchangeable backends behind one
  OpenAI-compatible base URL (config: `teacher.kind: api | selfhost`):
  - `api` (DEFAULT for CRAWL): public GLM-5.2 endpoints via OpenRouter
    (verified 2026-07-22: ~25 providers, $0.82/M in, $2.57/M out, 1M ctx).
    See 1a for the allowlist and logprob caveats.
  - `selfhost` (existing): the B200 teacher pod (weights on the stopped
    `glm52-teacher2` volume), as in 02_generate.
- **Environment plane** (new): 1-N cheap x86 CPU VMs with Docker (GCP
  e2-standard-8 ~ $0.27/hr, or any BYO host via `provider: static`). Runs the
  harness + per-task containers, calls the inference plane over HTTP.

With `teacher.kind: api`, the co-scheduling constraint DISSOLVES: no serving
campaign risk (the last one cost ~$155), no $35/hr idle burn, elastic
concurrency, pay-per-token. CRAWL no longer waits for the Tier-3 teacher
window. `selfhost` remains for the logit-KD rescore pass (1a) and as
fallback if provider quality/ToS checks fail.

### 1a. Public-endpoint teacher: verified facts + caveats (2026-07-22)

Endpoint survey (OpenRouter `/models/z-ai/glm-5.2/endpoints`):
- Most providers expose `logprobs` + `top_logprobs` (StreamLake, Alibaba,
  GMICloud, Fireworks, Cloudflare, Parasail, ...); Z.AI first-party, Novita,
  SiliconFlow, Together, Chutes do NOT.
- Quantization varies: fp8 (StreamLake, GMICloud, Baidu...), fp4 (DeepInfra,
  Chutes, Parasail, Decart...), several undeclared.

Rules:
1. **Provider allowlist, pinned**: fp8-or-better ONLY (a fp4 teacher is a
   quality haircut we never measured); pin via OpenRouter `provider.order` +
   `allow_fallbacks: false`; record provider + endpoint rev in every
   trajectory's meta (content-addressed cache key includes it). Never let
   OpenRouter route silently across quantizations mid-corpus.
2. **Logit-KD path**: `top_logprobs` caps at 20 on OpenAI-compatible APIs,
   which matches our top-20 schema, BUT two things must be smoke-tested
   (~$1) before trusting it: (a) whether logprobs cover REASONING tokens
   (providers return thinking via a separate `reasoning` field; the logprob
   stream may cover content only), and (b) token-string -> token-id mapping
   fidelity vs the GLM-5.2 tokenizer. If either fails: generate via API
   sequence-only, then recover top-20 with a **selfhost forced-decode
   rescore pass** (vLLM `prompt_logprobs` over the frozen trajectory tokens;
   prefill-only, cheap, batched into the next Tier-3 window). Sequence-KD
   students need none of this.
3. **ToS/licensing**: GLM-5.2 weights are open (we hold them); distilling
   outputs of third-party hosts of open weights is the conservative route.
   Avoid the Z.AI first-party endpoint for corpus generation until its API
   ToS is checked for a train-on-outputs clause (it also lacks logprobs, so
   nothing is lost).
4. **Prompt caching**: mini-swe-agent's linear append-only history is ideal
   for cache-read discounts; prefer allowlisted providers that advertise
   them (verify in the smoke test; without caching, input tokens dominate
   rollout cost).

### 1b. Smoke test RESULTS (2026-07-22, endpoint_smoke.py, $0.031 spent)

Providers probed: StreamLake, GMICloud, Fireworks, Alibaba. All four: pin
honored (allow_fallbacks=false respected), thinking returned in the
`reasoning` field, `bytes` present on logprob entries, and token-string ->
GLM-5.2-token-id round-trip 100% single-id on sampled positions (local
tokenizer, vocab 154,856).

- **Logprob coverage: CONTENT-ONLY, everywhere.** Logprob positions match
  the answer channel; REASONING TOKENS ARE NOT COVERED by any probed
  provider. The design's anticipated fallback is therefore the plan of
  record: **API rollouts capture sequence + answer-channel top-20; full-
  trace top-20 (thinking included) comes from the selfhost forced-decode
  rescore pass** (vLLM prompt_logprobs) batched into the Tier-3 window.
  Middle option now on the table: hybrid loss (logit-KD on answer tokens,
  CE on reasoning tokens) needs NO rescore pass and may be a strong
  cost/quality point; decide at CRAWL training time.
- **top_logprobs=20**: StreamLake, GMICloud, Alibaba yes; Fireworks caps at
  5 -> DEMOTED from the KD allowlist (fine as capacity overflow for
  sequence-only rollouts).
- **Caching: YES on all four.** Warm calls showed ~6.3k/6.3k prompt tokens
  cached with input cost dropping ~4-8x (e.g. GMICloud $0.0058 cold ->
  $0.0011 warm). Cost model moves to the cached end: ~$0.25-0.50/rollout.
- **ALLOWLIST v1 (pinned, in order): StreamLake (fp8), GMICloud (fp8),
  Alibaba (quant undeclared - confirm precision before heavy use).**
  Fireworks = sequence-only overflow.

## 2. Harness: mini-swe-agent (decision)

`SWE-agent/mini-swe-agent` (verified live). Chosen over OpenHands/SWE-agent
for v1 because:

1. **One tool (bash), zero schema.** Actions are plain text; observations
   come back as user-role messages. A trajectory IS a message list, which
   packs with the smallest possible extension to 03.
2. **Fewest harness bugs to corrupt a corpus.** ~100 lines of control flow;
   linear history; no retrieval magic whose absence at student-inference time
   would break the learned policy.
3. **Student-deployment realism.** What the student learns is exactly
   reproducible at eval/deploy time with the same harness; SWE-bench
   bash-only results are directly comparable.

**Format decision UPDATED 2026-07-29 (round-3 review, G1):** native
tool-call trajectories are the **WALK DEFAULT backbone**, not a maybe.
Bash-only remains the CRAWL format (plumbing proof; fewest harness bugs)
and survives into WALK only as a minority robustness flavor, trained as a
separate ablation arm. Rationale: bash-only drills editing idioms
(sed/heredoc rewrites) that are anti-patterns in the rich-tool harnesses a
deployed agent ships into; the student would never have emitted the token
patterns its runtime expects. WALK-entry work item: pick the FC harness
(SWE-agent tool schema vs OpenHands) against GLM-5.2's native tool-call
format, and A/B bash-only vs FC on the same task set before bulk
generation. Rule unchanged: never mix formats within one training run.

## 3. Task sources (all verified live on HF, 2026-07-22)

| Source | Size | Role | Phase |
|---|---|---|---|
| SWE-Gym/SWE-Gym-Lite | 230 tasks | crawl set, prebuilt docker images | CRAWL |
| SWE-Gym/SWE-Gym | 2,438 tasks, 11 py repos | main real-repo pool | WALK |
| SWE-bench/SWE-smith | ~50k synthetic tasks + toolkit | volume + difficulty axis, task variety (bug-mutation, PR-mirror) | WALK/RUN |
| R2E-Gym/R2E-Gym-Subset (+Lite) | ~8k | repo diversity, dedup against others | RUN |
| nebius/SWE-rebench | rolling | freshness / post-cutoff tasks, decontam-safe | RUN |
| princeton-nlp/SWE-bench_Verified | 500 | EVAL ONLY, never trained on | all |

Task selection (which instances, what depth/breadth quotas, adaptive
rollout budgets) is governed by **trajectory_task_spec.md** (2026-07-22),
the task-pool counterpart of corpus_spec.md.

Same hygiene philosophy as the base corpus, adapted:
- **Decontamination is instance-level (repo, PR/issue id) AND text-level**
  (MinHash vs SWE-bench Verified problem statements). Sources claim
  train/test disjointness by construction; we enforce it anyway, in
  preflight, with a hard gate. Log the check like the 0/1,140 audit.
- **Per-source cap** on the packed trajectory set (no single source >50% in
  WALK+), realized-mix reported in funnel.json.
- **Image preflight**: every task's docker image must pull before the teacher
  window opens (env plane is cheap, do this early); missing-image tasks
  dropped with telemetry, never discovered mid-window.

## 4. Generation: 09_generate_trajectories.py (new)

Teacher-as-agent loop, mirroring 02's operational discipline:

- **Rollout**: mini-swe-agent drives GLM-5.2 (temp 0.7, thinking ON) against
  the task container. N=4 rollouts per task (independent seeds recorded).
- **Caps (all per-rollout, all logged as drop reasons)**: max 50 steps; max
  context 96k tokens (abort, don't truncate-and-hallucinate); max output
  tokens/step 8k; wall clock 45 min; per-rollout cost cap from cost.py.
- **Signal capture**: per-step top-20 (token_id, logprob) for every assistant
  turn, same as 02, so same-family logit-KD survives into the agentic leg.
  Thinking content preserved inline per step.
- **Verification (the judge is an environment now)**: after the agent
  submits, reset the container, apply ONLY the agent's non-test diff, then
  re-apply the task's GOLD test files fresh before running fail-to-pass +
  pass-to-pass tests (anti-tamper: a diff touching test files is REJECTED
  outright - weakened asserts or hardcoded expectations must never verify).
  Run the test evaluation TWICE; a success that doesn't reproduce is flaky
  and quarantined, not kept. An empty-diff "success" is always rejected.
  Containers run network-isolated (deps are pre-baked in task images); the
  API key never enters a container.
- **Outputs**: verified successes -> training pool; failures + near-misses ->
  `trajectories_rejected/` with reward labels (free RL/DPO option later);
  everything sharded + sealed atomically with MANIFEST + kept-rows floor,
  identical to the 02 fixes (R2/R4 lessons carry over verbatim).
- **Cache**: content-addressed by (task_id, teacher rev, harness rev, seed);
  a re-run never re-generates what the store already holds (autoloop rule:
  never re-derive).

Funnel (per source, persisted): tasks -> images pulled -> rollouts ->
submitted -> tests-passed -> reproduced -> under-length -> packed.

## 5. Data format + packing (this finally forces MULTITURN_READY)

Extend the pack schema from single (prompt, response, top20) to:

```
messages: [ {role, content, token_ids?, top20?}, ... ]   # assistant turns carry signal
loss_mask: assistant tokens ONLY (system/user/observation masked)
meta: {task_id, source, n_steps, outcome, seed, teacher_rev}
```

- 03_pack grows a `--format messages` path; student_b think-wrap machinery
  reused per assistant turn. This work is the multi-turn support that
  corpus_spec deferred; landing it here also unblocks the WildChat 5% slice.
- **Length policy (CRAWL)**: pack max-len 32k; overlong trajectories DROPPED
  with per-source telemetry (expect real losses; measure before engineering
  around them). WALK option if drop rate >25%: last-K-turns windowing with a
  synthesized state summary, only for sequence-KD students (windowing breaks
  logit alignment).
- 04_train grows message-format + loss-mask support and seq-len 32k (memory
  check on RTX PRO 6000 required for >8B students; FSDP if needed).

## 6. Students and the capacity ceiling (honesty clause)

Qwen3-8B remains the *plumbing* student: cheap, proves the pipeline, but 8B
is not a frontier-agentic endpoint and no corpus fixes that. The leg's
flagship targets, in order:

1. **GLM-4.5-Air** (same family, logit-KD, thinking-native): primary.
2. **Qwen3-32B** (sequence-KD): dense alternative, comparable to the
   published trajectory-distillation results (SWE-Gym 32B, SWE-agent-LM-32B
   ~40% Verified with SFT-only).

Expectation setting: verified-trajectory SFT/KD at 32B-class scale gets to
"competent agent" (30-45% SWE-bench Verified by published precedent).
Frontier (70%+) additionally requires large-scale RL in environments; this
leg's contribution to that future is the reward-labeled rejected-rollout
store and the executable-env plumbing itself.

**Mix rule for training**: agentic trajectories are blended WITH the base
single-turn corpus (start 30% trajectory tokens / 70% base by token count;
sweepable). Trajectory-only SFT measurably regresses general ability
(SWE-Gym's own ablation); the base corpus is the regularizer.

## 7. Evaluation

- **Screen** (every checkpoint): fixed 50-instance stratified subset of
  SWE-bench Verified, run with mini-swe-agent, resolve-rate. Frozen once
  like the held-out set. Student on RTX pod ~$8/hr, est. $15-30/run.
- **Anchor** (Discovery #6 discipline): the UNTRAINED base student must
  first score its published-range resolve rate under our exact harness
  config, else PAUSED_HUMAN. No delta is trusted without it.
- **Regression guard**: existing screen benchmarks (gsm8k_cot, ifeval
  think-stripped) must not drop >2pts vs the same student trained on base
  corpus alone. Agentic gains that cannibalize general ability fail the
  objective.
- **Full 500-instance Verified**: final candidates only (cost).
- objective.py: `score_fn_version` bump; composite = resolve-rate primary,
  regression guard as hard constraint, not a weighted term.

## 8. Cost model (±50% until CRAWL measures)

**API path (CRAWL default), at OpenRouter list price $0.82/M in, $2.57/M
out:** a ~30-step rollout re-sends a growing context each step (~900k
cumulative input) + ~45k output -> **~$0.85/rollout uncached**; with
cache-read discounts (linear history caches well) **~$0.25-0.50**. Verified
trajectory at 30-50% teacher success: **~$0.7-2.8 each**. No idle burn, no
spin-up risk; concurrency is a rate-limit question, not a wall-clock one.

**Selfhost path (WALK-scale option + rescore pass):** ~$0.16 per best-of-4
answer implies roughly $5/1M output tokens at measured B200x6 throughput
($35/hr); prefix caching makes input cheap; ~$0.20-0.45/rollout at high
utilization, but only inside an amortized serving window. Break-even vs API
is a CRAWL measurement, not an assumption. Env plane is noise ($0.27/hr/VM).

| Phase | Scope | Est. teacher $ | Hard cap |
|---|---|---|---|
| CRAWL | SWE-Gym-Lite: 50 tasks x4 = 200 rollouts via API teacher -> target >=60 verified; pack; train Qwen3-8B; screen eval | $50-170 (caching-dependent) | **$150** (trim rollouts, not verification, if uncached) |
| WALK | SWE-Gym full + SWE-smith onboard: ~1-2k verified; GLM-4.5-Air or Qwen3-32B; mix sweep (2-3 ratios); anchor + screen | $400-700 | frozen in experiments.yaml at WALK approval |
| RUN | GCP production: 5-10k verified, format-flavor decision, RL-ready store | playbook item | separate approval |

Every number above is re-fit from CRAWL's measured funnel before WALK is
priced (cost.py gains a trajectory estimator; conductor's
don't-start-what-you-can't-finish guard applies per rollout batch).

## 9. Guardrails (inherit all 10 autoloop stops, plus leg-specific)

1. Per-rollout AND per-phase cost caps enforced pre-spend (cost.py).
2. Flaky-verifier gate: double test-run reproduction + empty-diff rejection.
3. Decontam hard gate in preflight (instance-level + text-level) before any
   teacher spend; gate artifact pushed to the store like gate_result.json.
4. Env-plane watchdog: containers older than wall-clock cap are killed +
   docker GC per batch (disk is the env plane's failure mode); env host
   dead-man mirrors watchdog.sh.
5. Funnel drift halt: verified-rate collapse (<10% over a 50-rollout window)
   pauses the leg (bad image set / harness regression / teacher misconfig),
   PAUSED_HUMAN, don't burn the window.
6. Teacher-window discipline: image pulls, task manifests, decontam, and env
   smoke tests all green BEFORE the pod resumes; the window runs generation
   only.
7. No format mixing within a training run (bash-text vs native-FC).
8. Anchor-first eval, always (Discovery #6).

## 10. Phasing + exit criteria

**CRAWL (goal: prove the loop end-to-end, ~$150 cap)**
- Build: 09_generate_trajectories.py, messages pack format + loss masking,
  04 multi-turn support, SWE-bench-50 screen eval config, decontam gate.
- Teacher = public API (allowlisted fp8 providers, pinned); no teacher
  window needed. Precondition: the ~$1 endpoint smoke test (logprob
  coverage of reasoning tokens, token-id mapping, cache-read discount).
  If logit-KD capture fails via API, CRAWL proceeds sequence-only and the
  rescore pass rides the next Tier-3 selfhost window.
- EXIT: >=60 verified trajectories with full funnel telemetry; measured
  $/verified within 2x of estimate; Qwen3-8B trained on 30/70 blend shows
  resolve-rate > base anchor on the 50-subset (any positive delta) with
  regression guard held; a mid-run env-host kill resumes from MANIFEST with
  zero duplicate teacher spend.

**WALK (goal: real signal at target scale)**
- SWE-Gym full + SWE-smith; 1-2k verified; flagship student; mix-ratio
  sweep under the conductor (trajectory leg becomes rows in
  experiments.yaml, same FSM, same ledger).
- ENTRY: FC-harness pick (SWE-agent schema vs OpenHands vs GLM-native)
  + bash-vs-FC A/B on a shared task set; FC is the backbone unless the
  A/B contradicts it (round-3 G1).
- EXIT: flagship student beats its base anchor by a stable, reproduced
  margin on the 50-subset; funnel + realized-source-mix healthy;
  bash-only robustness arm sized from the A/B.

**RUN (production, GCP playbook)**
- 5-10k verified, R2E-Gym + SWE-rebench diversity, full-500 eval on final
  candidates, RL-ready reward store handed to a future RL program.

## 11. Diversity + robustness review, round 2 (2026-07-22)

Adversarial self-review of this design, trace_recipe_review.md style.
Verdict: adequate for CRAWL's plumbing-proof purpose; NOT yet a
frontier-diverse agentic corpus. Ranked:

- **T1 (structural)**: task-type monoculture - the fail-to-pass verifier
  buys trust but restricts the corpus to test-verifiable bugfix work; no
  feature-from-spec, refactoring, test-writing, greenfield. WALK: exploit
  SWE-smith task variety; RUN: add verifiers beyond fail2pass.
- **T2 (structural)**: Python-only, and invisible to our own eval because
  SWE-bench Verified is Python too. WALK+: SWE-smith multilingual /
  SWE-bench-M-style sources. Named deliberately, like English-only.
- **T3 (MUST-FIX before CRAWL, cheap)**: per-REPO caps (SWE-Gym is 11
  skewed repos; source caps alone permit single-repo dominance - D7 one
  level down). Realized per-repo mix goes in funnel.json.
- **T4 (MUST-FIX before CRAWL, cheap)**: success-only filtering skews easy
  (R1 pattern reborn: keep-rate ~ 1/difficulty). Difficulty-stratified
  quotas + ADAPTIVE N (more rollouts on hard tasks, early-stop on easy);
  report pack-drop rate BY difficulty, not just overall.
- **T5 (deliberate)**: one harness grammar, one temperature, one teacher.
  D6 analog; revisit at WALK with the FC-format decision.
- **T6 (deferred, RUN/RL-era)**: no mid-trajectory user turns (interrupts,
  scope changes, clarifying questions). No v1 source provides it.

### Round 3 (2026-07-29): deployment representativeness, WITH examples

Question asked: are these trajectories representative for a WORLD-CLASS
agentic coding model? Verdict: they cover the KERNEL (edit-run-observe-
recover on real code, verified), roughly a quarter to a third of the
deployed behavior distribution. Gap registry, each with a concrete
behavioral example (convention: every gap entry MUST carry one; see note
at the end of this section):

- **G1 (fixed in section 2): harness grammar was the WRONG grammar, not
  just one grammar.** Corpus teaches `sed -i 's/old/new/' file.py` and
  cat-piping; deployed harnesses expect `[read_file lines=120-180]` +
  `[str_replace old=... new=...]`. Different token patterns AND different
  decision policy ("grep -rn everything" vs "targeted read"). ->
  Native-FC is now the WALK backbone; bash-only demoted to robustness
  flavor. First breadth investment, AHEAD of multilingual.
- **G2 (structural, RL-era): teacher is the ceiling, and one personality.**
  If GLM-5.2's habit on a failing test is "patch the traceback line" and
  it never writes a reproduction script first, the corpus contains zero
  examples of repro-first debugging, at any volume. Rejection sampling
  buys quality, not strategy diversity. SFT converges to "a smaller
  GLM-5.2"; exceeding the teacher is the reward-labeled reject store's
  job (RL/DPO).
- **G3 (= T1 sharpened): work-type mix vs reality.** Corpus shape is
  always "well-written issue -> make hidden tests pass." Deployed
  majority: "Add CSV export to the reports page" (agent defines done),
  "Why does login sometimes take 10s?" (deliverable is a diagnosis),
  "Clean up this module" (success = behavior NOT changing). Fail-to-pass
  structurally admits only the first. WALK claws back some (PR-mirror,
  test-writing, degraded statements); scope-deciding is still absent.
- **G4 (= T2 sharpened): observation monoculture.** Corpus error surface
  is pytest tracebacks only. Deployed agents equally read
  `TS2345: Argument of type 'string | undefined'...`, `npm ERR! peer dep
  missing`, `error[E0502]: cannot borrow...` - each with its own recovery
  vocabulary (the npm fix starts in package.json/lockfile, a file type
  the corpus never opens). Invisible on our own eval because Verified is
  Python too. Second breadth investment (multilingual, WALK+).
- **G5 (= long-horizon, RUN/RL-era): no context management.** 32k window
  + linear history means the teacher never summarizes ("noted:
  AuthMiddleware validates in _verify(), moving on"), never keeps a plan,
  never re-orients from notes; and the length cap preferentially DROPS
  the longest trajectories (T4 interaction). Quotas + per-tier drop
  telemetry mitigate; real compaction behavior needs purpose-built long
  tasks.
- **G6 (= T6): the user never speaks after turn one.** No mid-task
  "actually per-org limits, not per-user, and don't touch billing," no
  learned when-to-ask. Candidate mechanism for RUN: scripted user
  injecting a constraint change mid-rollout.

**Convention (standing): the gap registry is LIVING.** Every review round
appends here with concrete examples; every gap carries an ID, a status
(fixed / staged-WALK / staged-RUN / RL-era), and the example that makes it
legible. Once CRAWL produces real trajectories, examples MUST be drawn
from actual rollouts (the audit sample), not hypotheticals - reviews then
happen at every phase boundary (post-CRAWL, post-WALK) against real data.

Angles that were MISSING until round 2's review:
- **Test-tampering / reward hacking (MUST-FIX, now in section 4)**: gold
  test re-apply + reject test-file diffs + LLM spot-AUDIT of a sample of
  passing diffs (judge as auditor, not selector).
- **Teacher success-rate prior unmeasured**: upgrade the rehearsal to ~10
  tasks to estimate it BEFORE pricing CRAWL (15% vs 40% is a 3x swing on
  $/verified and worsens T4).
- **Eval power**: Verified-50 at ~30% resolve => +/-6-7% SE. CRAWL's "any
  positive delta" is a plumbing claim, not a statistical one - say so.
  WALK decisions: 3 seeds x 50 or a 100-150 instance subset.
- **Container hygiene**: network-isolated rollout containers (in section 4).

## 12. Explicitly out of scope for v1

RL training itself; browser/GUI agents; multi-repo/monorepo tasks;
non-Python task construction (SWE-smith supports it later); trajectory
compression/summarization students; safety-behavior distillation (same
pending policy as corpus_spec).
