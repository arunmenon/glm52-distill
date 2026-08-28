# Review request: expansion task selection for a teacher-trajectory distillation corpus

## Context

Public repo: https://github.com/arunmenon/glm52-distill (branch `master`, commit `f427882`). Clone it; everything referenced below is in the repo.

We are distilling agentic bug-fixing ability from a teacher model (Qwen3.8-27B, self-hosted on vLLM) into a small student (Qwen3.5-9B) via SFT + token-level logit KD. The teacher solves SWE-Gym tasks inside Docker sandboxes using mini-swe-agent; a trajectory counts only if the task's gold FAIL_TO_PASS tests flip to passing and PASS_TO_PASS tests stay passing, with an anti-tamper check that rejects test-file edits.

We have 45 verified trajectories (17 easy / 26 medium / 2 hard). We are about to spend a hard-capped $50 on an "expansion" run to add ~10-15 more, using a frozen list of 24 never-before-attempted tasks. **Your job: review that frozen task list and the selection/runner code before we spend the money.** An earlier external review already caught one major flaw (see "Known issue" below), so assume more may exist.

## Files to review

- `runs/expansion/bugfix/expansion_tasks.json` — the 24 frozen tasks (10 easy / 12 medium / 2 hard; repos: moto x6, mypy x6, dvc x4, conan x4, pandas x3, dask x1). Each entry carries instance_id, repo, problem_statement, base_commit, gold patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS, tier.
- `09e_expansion_bugfix.py` — the selector that froze the list (excludes all 108 previously attempted instance ids; monkeypatches `09d_walk_bugfix.py`).
- `09d_walk_bugfix.py` — the base selector it inherits (tier targets, per-repo cap 6, Docker Hub image preflight, SWE-Gym-Lite first with top-up from SWE-Gym full).
- `09_rehearsal.py` — the rollout runner. Note `tier_of()` (tier = gold-patch size heuristic: files/lines changed/number of F2P tests) and `STRIP_CMD` (new: deletes all git refs and prunes the container repo at the base commit before the agent starts, so the teacher cannot retrieve the upstream fix from git history).
- `trajectory_decontam.py` — the contamination gate (checks the frozen list against SWE-bench Verified by instance id, repo overlap, exact and n-gram text match). Verdict on this list: CLEAN.
- `packed/trajectories/trajectories_v0.parquet` provenance columns (if useful): the existing corpus is labeled `history_assisted` — 37/45 prior trajectories retrieved the upstream fix via git history, which motivated STRIP_CMD.

## Known issue already handled (don't re-report)

Prior trajectories could read the repo's future git history (the fix exists in-container). We now strip refs/reflog and prune before rollout, and the packer flags any trajectory that still used history commands as a leak detector.

## What to check (in priority order)

1. **Task quality of the 24 chosen instances.** Look at the actual problem statements and gold patches in the JSON. Are any of them: unsolvable from the problem statement alone (statement references the fix/PR, or is empty/vague)? Trivial (one-line patches where the statement gives away the diff)? Duplicates or near-duplicates of each other or of the same underlying bug? Tests that are flaky or environment-dependent?
2. **Contamination the gate would miss.** The decontam gate checks against SWE-bench Verified only. Are any of these 24 instances in other common eval sets (SWE-bench Lite/full test split, SWE-bench Multimodal)? Both teacher and student are Qwen models — flag any instance likely present in public training corpora in a way that matters for a distillation corpus (i.e., is memorization of the fix plausible from the statement alone?).
3. **Selection-logic bugs.** In `09e_expansion_bugfix.py` + `09d_walk_bugfix.py`: does the exclusion actually cover everything (result.json globs + frozen task lists)? Does the per-repo cap interact correctly with tier iteration order (easy fills first — can it starve medium/hard of a repo's slots)? Is the monkeypatching of `excluded_ids`/`WALK_DIR` sound, or does some code path still read the WALK values (cost cap, dirs, tier targets)?
4. **History-strip robustness.** `STRIP_CMD` in `09_rehearsal.py`: after deleting refs, expiring reflog, and `git gc --prune=now`, can a determined agent still recover the fix commit inside the container (packed-refs remnants, `.git/FETCH_HEAD`, `ORIG_HEAD`, dangling objects that gc keeps, alternates)? Does detaching HEAD break anything the SWE-Gym verify flow needs (it replays gold test_patch against the base commit in a fresh container, not the mutated one)?
5. **Yield realism.** Given tier mix 10/12/2, per-tier historical verify rates (easy ~79%, medium ~55%, hard ~15% — all measured WITH history retrieval available), and a $50 cap at roughly $1.5-2.5 per attempted rollout: is ~10-15 verified trajectories a sound expectation with history stripping now on, or should the mix shift?

## Output format

A ranked list of findings: `[severity: blocker/major/minor] — file or instance_id — what's wrong — concrete fix`. If a specific task instance should be swapped out, name it and say why. If the list is fine, say so explicitly per check. Keep speculation labeled as such — we will re-verify every blocker/major claim before acting.
