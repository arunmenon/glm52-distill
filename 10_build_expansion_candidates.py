#!/usr/bin/env python3
"""Build a diverse medium/hard SWE-Gym expansion candidate reservoir.

This is deliberately a *candidate* builder, not an approval gate.  It applies
cheap structural checks and a world-knowledge-inspired difficulty proxy, then
balances the resulting pool by repository, difficulty, and bug family.  Every
auto-selected row remains ``needs_manual_review`` until its statement, gold
patch, and tests have been inspected and base-fail/gold-pass smoke-tested.

The existing frozen expansion list is never overwritten.

Default output:
  runs/expansion/bugfix/candidate_pool.json
  runs/expansion/bugfix/candidate_pool_index.json
  runs/expansion/bugfix/candidate_pool.md
  runs/expansion/bugfix/candidate_pool_report.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_DIR = Path(__file__).parent
EXP_DIR = REPO_DIR / "runs" / "expansion" / "bugfix"
FULL_PARQUET = REPO_DIR / "walk_scratch" / "swegym_full.parquet"
LITE_PARQUET = REPO_DIR / "walk_scratch" / "swegym_lite.parquet"
FROZEN_EXPANSION = EXP_DIR / "expansion_tasks.json"
OUT_FILE = EXP_DIR / "candidate_pool.json"
INDEX_FILE = EXP_DIR / "candidate_pool_index.json"
MARKDOWN_FILE = EXP_DIR / "candidate_pool.md"
REPORT_FILE = EXP_DIR / "candidate_pool_report.json"

TARGET_TOTAL = 80
TARGET_MEDIUM = 56
TARGET_HARD = 24
MAX_PER_REPO = 10
MAX_PER_FAMILY = 18

# Human-reviewed survivors from the existing expansion list.  Scores use the
# rubric documented in score_row(): localization + reasoning + coupling +
# validation - implementation hints.  Notes are intentionally retained in the
# generated pool so the approval pass cannot forget required repairs.
SEED_META: dict[str, dict[str, Any]] = {
    "dask__dask-7305": {
        "difficulty": "medium", "score": 8,
        "bug_family": "numeric_data",
        "review_status": "retained_seed_needs_repair",
        "review_notes": ["Add test_set_index_interpolate_large_uint to FAIL_TO_PASS."],
    },
    "iterative__dvc-1681": {
        "difficulty": "medium", "score": 8,
        "bug_family": "parsing_serialization",
        "review_status": "retained_seed",
        "review_notes": [],
    },
    "iterative__dvc-9395": {
        "difficulty": "medium", "score": 8,
        "bug_family": "cache_incremental",
        "review_status": "retained_seed_needs_repair",
        "review_notes": [
            "Add missing-import and intermediate-output cases to FAIL_TO_PASS."
        ],
    },
    "getmoto__moto-5885": {
        "difficulty": "medium", "score": 8,
        "bug_family": "cloud_api_state",
        "review_status": "retained_seed",
        "review_notes": [],
    },
    "python__mypy-16154": {
        "difficulty": "medium", "score": 8,
        "bug_family": "type_system",
        "review_status": "retained_seed",
        "review_notes": [],
    },
    "conan-io__conan-10917": {
        "difficulty": "medium", "score": 8,
        "bug_family": "build_dependency",
        "review_status": "retained_seed_needs_smoke",
        "review_notes": ["Gold-smoke the CMake integration tests twice."],
    },
    "getmoto__moto-6317": {
        "difficulty": "hard", "score": 10,
        "bug_family": "cloud_api_state",
        "review_status": "retained_seed_needs_repair",
        "review_notes": [
            "Add the repeated patch_client reproduction that the gold test exercises."
        ],
    },
    "getmoto__moto-7607": {
        "difficulty": "hard", "score": 10,
        "bug_family": "async_concurrency",
        "review_status": "retained_seed_needs_smoke",
        "review_notes": ["Prove repeated gold verification cannot hang."],
    },
    "python__mypy-14835": {
        "difficulty": "hard", "score": 11,
        "bug_family": "cache_incremental",
        "review_status": "retained_seed_needs_repair",
        "review_notes": ["Add representative daemon/cache PASS_TO_PASS coverage."],
    },
    "python__mypy-14988": {
        "difficulty": "hard", "score": 11,
        "bug_family": "type_system",
        "review_status": "retained_seed",
        "review_notes": [],
    },
}


HARD_CONCEPT_GROUPS = {
    "type_system": (
        "type inference", "type checker", "narrowing", "metaclass", "generic",
        "typevar", "protocol", "variance", "namedtuple", "binder", "mypy",
    ),
    "cache_state": (
        "cache", "cached", "incremental", "daemon", "stale", "restore",
        "persistence", "state", "checksum", "run-cache",
    ),
    "concurrency": (
        "async", "await", "thread", "race", "deadlock", "callback", "lock",
        "task token", "waitfortasktoken", "state machine",
    ),
    "numerical": (
        "overflow", "precision", "dtype", "uint", "integer", "floating",
        "quantile", "interpolation", "timezone", "nan", "rounding",
    ),
    "dependency": (
        "dependency", "dependencies", "resolver", "cmake", "compiler",
        "toolchain", "transitive", "build graph", "package graph",
    ),
    "serialization": (
        "serialize", "serialization", "parser", "parsing", "yaml", "json",
        "dump", "load", "schema", "configuration", "config",
    ),
    "lifecycle": (
        "lifecycle", "event handler", "session", "transaction", "backend",
        "resource", "retry", "rollback", "invalidation",
    ),
}

TRIVIAL_CONCEPTS = (
    "empty list", "empty iterable", "rename", "re-export", "export", "typo",
    "invalid enum", "case insensitive", "lowercase", "deprecated alias",
)

HIGH_LEAK_PATTERNS = (
    r"after i remove\b", r"the fix is to\b", r"fix(?:ed)? by changing\b",
    r"change [`'\"]?[^\n]{1,60}[`'\"]? to [`'\"]?[^\n]{1,60}",
    r"i have this implemented", r"ready to (?:go|submit|open a pr)",
    r"```diff",
)

AMBIGUITY_PATTERNS = (
    r"not (?:actually )?sure (?:that )?this is a bug",
    r"would like (?:an )?opinion",
    r"is this (?:a )?bug or",
    r"which (?:approach|option|behavior) should",
)

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return list(parsed) if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return list(value)


def patch_stats(patch: str) -> tuple[list[str], int]:
    files = re.findall(r"^diff --git a/(\S+) b/\S+", patch or "", re.M)
    changed = sum(
        1
        for line in (patch or "").splitlines()
        if ((line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---")))
    )
    return files, changed


def normalized_repo(repo: str) -> str:
    return (repo or "").replace("/", "__")


def problem_title(statement: str) -> str:
    for line in (statement or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:180]
    return ""


def prior_ids() -> set[str]:
    """Every previously attempted/frozen id, excluding this expansion arm."""
    ids: set[str] = set()
    paths = glob.glob(str(REPO_DIR / "runs" / "**" / "result.json"), recursive=True)
    paths += glob.glob(str(REPO_DIR / "runs" / "**" / "*tasks*.json"), recursive=True)
    for name in paths:
        path = Path(name)
        if EXP_DIR in path.parents:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict) and row.get("instance_id"):
                ids.add(row["instance_id"])
    return ids


def malformed_test_id(test_id: str) -> bool:
    if "\n" in test_id or "\r" in test_id:
        return True
    # Parametrized pytest node ids always close their final bracket.  This
    # catches SWE-Gym extraction artifacts such as dvc-4961's ``{'level':``.
    if "[" in test_id and not test_id.endswith("]"):
        return True
    return False


def quality_rejections(row: dict[str, Any]) -> list[str]:
    statement = row.get("problem_statement") or ""
    patch = row.get("patch") or ""
    test_patch = row.get("test_patch") or ""
    f2p = as_list(row.get("FAIL_TO_PASS"))
    files, changed = patch_stats(patch)
    low = statement.lower()
    reasons: list[str] = []

    if len(statement.split()) < 35:
        reasons.append("problem_statement_too_short")
    if not patch or not files:
        reasons.append("missing_gold_patch")
    if not test_patch:
        reasons.append("missing_test_patch")
    if not f2p:
        reasons.append("missing_fail_to_pass")
    if any(malformed_test_id(t) for t in f2p):
        reasons.append("malformed_fail_to_pass")
    if files and all(TEST_PATH_RE.search(path) for path in files):
        reasons.append("gold_changes_only_tests")
    if len(files) > 8 or changed > 350:
        reasons.append("oversized_for_step_limited_rollout")
    if len(f2p) > 20:
        reasons.append("verification_set_too_large")
    if any(re.search(pattern, low) for pattern in AMBIGUITY_PATTERNS):
        reasons.append("ambiguous_expected_behavior")
    if sum(bool(re.search(pattern, low)) for pattern in HIGH_LEAK_PATTERNS) >= 2:
        reasons.append("statement_reveals_implementation")
    return reasons


def bug_family(row: dict[str, Any]) -> str:
    repo = (row.get("repo") or "").lower()
    text = f"{row.get('problem_statement', '')}\n{row.get('patch', '')}".lower()

    if any(k in text for k in HARD_CONCEPT_GROUPS["concurrency"]):
        return "async_concurrency"
    if any(k in text for k in HARD_CONCEPT_GROUPS["cache_state"]):
        return "cache_incremental"
    if "mypy" in repo or any(k in text for k in HARD_CONCEPT_GROUPS["type_system"]):
        return "type_system"
    if "conan" in repo or any(k in text for k in HARD_CONCEPT_GROUPS["dependency"]):
        return "build_dependency"
    if ("pandas" in repo or "dask" in repo or "modin" in repo
            or any(k in text for k in HARD_CONCEPT_GROUPS["numerical"])):
        return "numeric_data"
    if any(k in text for k in HARD_CONCEPT_GROUPS["serialization"]):
        return "parsing_serialization"
    if "dvc" in repo or any(k in text for k in ("filesystem", "directory", "path", "remote")):
        return "filesystem_versioning"
    if "moto" in repo or any(k in text for k in ("aws", "boto3", "cloud", "backend")):
        return "cloud_api_state"
    if any(k in text for k in ("api", "request", "response", "validation", "fallback")):
        return "protocol_api"
    return "other"


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Score diagnosis difficulty, not gold-patch size alone.

    Score = localization entropy (0..3) + semantic reasoning (0..3) +
    cross-component coupling (0..3) + validation breadth (0..2) -
    implementation-hint penalty (0..2).
    """
    statement = row.get("problem_statement") or ""
    low = statement.lower()
    files, changed = patch_stats(row.get("patch") or "")
    f2p = as_list(row.get("FAIL_TO_PASS"))
    p2p = as_list(row.get("PASS_TO_PASS"))

    leak_hits = sum(bool(re.search(pattern, low)) for pattern in HIGH_LEAK_PATTERNS)
    has_exact_location = bool(
        re.search(r"github\.com/[^\s]+/(?:blob|commit)/[^\s]+#l\d+", low)
        or re.search(r"(?:file|line) [^\n]{0,80}:\d+", low)
    )
    has_repro = any(k in low for k in ("to reproduce", "steps to reproduce", "expected behavior", "actual behavior"))
    if leak_hits >= 2:
        localization = 0
    elif has_exact_location:
        localization = 1
    elif has_repro or "traceback" in low:
        localization = 2
    else:
        localization = 3

    concept_hits = sum(
        any(keyword in low for keyword in keywords)
        for keywords in HARD_CONCEPT_GROUPS.values()
    )
    if concept_hits >= 2:
        reasoning = 3
    elif concept_hits == 1:
        reasoning = 2
    else:
        reasoning = 1
    if any(k in low for k in TRIVIAL_CONCEPTS) and concept_hits <= 1:
        reasoning = max(0, reasoning - 1)

    if len(files) >= 4 or changed >= 100:
        coupling = 3
    elif len(files) >= 2 or changed >= 35:
        coupling = 2
    elif changed >= 15:
        coupling = 1
    else:
        coupling = 0

    if len(f2p) >= 4 or (len(f2p) >= 2 and len(p2p) >= 5):
        validation = 2
    elif f2p and (len(f2p) >= 2 or p2p):
        validation = 1
    else:
        validation = 0

    hint_penalty = min(2, leak_hits)
    score = localization + reasoning + coupling + validation - hint_penalty
    difficulty = "hard" if score >= 9 else "medium" if score >= 6 else "easy"
    return {
        "score": score,
        "difficulty": difficulty,
        "score_components": {
            "localization": localization,
            "reasoning": reasoning,
            "coupling": coupling,
            "validation": validation,
            "hint_penalty": hint_penalty,
        },
        "gold_patch_stats": {"files": len(files), "changed_lines": changed},
        "test_stats": {"fail_to_pass": len(f2p), "pass_to_pass": len(p2p)},
    }


def review_flags(row: dict[str, Any], scored: dict[str, Any]) -> list[str]:
    statement = (row.get("problem_statement") or "").lower()
    title = problem_title(row.get("problem_statement") or "").lower()
    files = scored["gold_patch_stats"]["files"]
    changed = scored["gold_patch_stats"]["changed_lines"]
    flags: list[str] = []
    if not as_list(row.get("PASS_TO_PASS")):
        flags.append("no_pass_to_pass")
    if files >= 5 or changed >= 150:
        flags.append("large_gold_patch")
    if any(k in statement for k in ("windows", "cmake", "compiler", "timezone", "sleep", "timeout")):
        flags.append("environment_or_time_sensitive")
    if scored["score_components"]["hint_penalty"]:
        flags.append("statement_contains_implementation_hint")
    if (
        title.startswith(
            ("[feature]", "feature:", "feature request", "[enh]", "enh:", "enhancement:")
        )
        or re.match(r"^(add|allow|implement|support|provide|introduce|enable)\b", title)
        or "enhancements" in title
    ):
        flags.append("feature_request_not_bug")
    if title.startswith(("perf:", "[perf]", "performance:")) or "performance improvement" in title:
        flags.append("performance_task")
    if title.startswith(("how do i", "how to ", "usage:", "question:")):
        flags.append("usage_question")
    if any(marker in title for marker in ("flake8", "lint", "typing cleanup", "documentation only")):
        flags.append("maintenance_task")
    if "benchmark" in statement and "performance" in statement:
        flags.append("performance_validation_may_be_weak")
    return flags


def stable_rank(instance_id: str) -> str:
    return hashlib.sha256(f"glm52-expansion-candidates-v1:{instance_id}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decorated_row(
    row: dict[str, Any], *, source: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    f2p = as_list(row.get("FAIL_TO_PASS"))
    p2p = as_list(row.get("PASS_TO_PASS"))
    core = {
        key: row.get(key)
        for key in (
            "instance_id", "repo", "problem_statement", "base_commit", "patch",
            "test_patch", "version", "hints_text", "created_at",
        )
        if key in row
    }
    core["FAIL_TO_PASS"] = f2p
    core["PASS_TO_PASS"] = p2p
    core["candidate_source"] = source
    core["problem_title"] = problem_title(row.get("problem_statement") or "")

    if metadata is not None:
        core.update({
            "difficulty_score": metadata["score"],
            "difficulty": metadata["difficulty"],
            "difficulty_components": metadata.get("score_components", {}),
            "bug_family": metadata["bug_family"],
            "gold_patch_stats": metadata.get("gold_patch_stats", {}),
            "test_stats": metadata.get("test_stats", {
                "fail_to_pass": len(f2p), "pass_to_pass": len(p2p)
            }),
            "review_status": metadata["review_status"],
            "review_flags": metadata.get("review_flags", []),
            "review_notes": metadata.get("review_notes", []),
        })
    return core


def choose_balanced(
    candidates: list[dict[str, Any]],
    *,
    want: int,
    difficulty: str,
    repo_counts: Counter[str],
    family_counts: Counter[str],
) -> list[dict[str, Any]]:
    pool = [c for c in candidates if c["difficulty"] == difficulty]
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    while len(chosen) < want:
        eligible = [
            c for c in pool
            if c["instance_id"] not in used
            and repo_counts[c["repo"]] < MAX_PER_REPO
            and family_counts[c["bug_family"]] < MAX_PER_FAMILY
        ]
        if not eligible:
            break
        eligible.sort(key=lambda c: (
            repo_counts[c["repo"]],
            family_counts[c["bug_family"]],
            -c["difficulty_score"],
            len(c["review_flags"]),
            stable_rank(c["instance_id"]),
        ))
        item = eligible[0]
        chosen.append(item)
        used.add(item["instance_id"])
        repo_counts[item["repo"]] += 1
        family_counts[item["bug_family"]] += 1
    return chosen


def validate_output(
    output: list[dict[str, Any]],
    *,
    full_ids: set[str],
    lite_ids: set[str],
    previous: set[str],
    rejected_frozen: set[str],
) -> None:
    ids = [row["instance_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise SystemExit("candidate pool contains duplicate instance ids")
    if set(SEED_META) - set(ids):
        raise SystemExit("candidate pool dropped a retained seed")
    if set(ids) - full_ids:
        raise SystemExit("candidate pool contains an id absent from SWE-Gym Full")

    auto = [row for row in output if row["candidate_source"] == "swegym_full_excluding_lite"]
    auto_ids = {row["instance_id"] for row in auto}
    violations = {
        "lite_overlap": sorted(auto_ids & lite_ids),
        "prior_overlap": sorted(auto_ids & previous),
        "rejected_frozen_overlap": sorted(auto_ids & rejected_frozen),
    }
    if any(violations.values()):
        raise SystemExit(f"candidate provenance violation: {violations}")

    difficulty = Counter(row["difficulty"] for row in output)
    if difficulty != Counter({"medium": TARGET_MEDIUM, "hard": TARGET_HARD}):
        raise SystemExit(f"wrong difficulty mix: {dict(difficulty)}")
    if max(Counter(row["repo"] for row in output).values()) > MAX_PER_REPO:
        raise SystemExit("repository cap exceeded")
    if max(Counter(row["bug_family"] for row in output).values()) > MAX_PER_FAMILY:
        raise SystemExit("bug-family cap exceeded")
    if any(row["difficulty_score"] < 6 for row in output):
        raise SystemExit("below-medium score entered candidate pool")


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, index: list[dict[str, Any]]) -> None:
    difficulty = Counter(row["difficulty"] for row in index)
    repos = Counter(row["repo"] for row in index)
    families = Counter(row["bug_family"] for row in index)
    sources = Counter(row["candidate_source"] for row in index)
    flagged = sum(bool(row["review_flags"]) for row in index)
    lines = [
        "# SWE-Gym medium/hard expansion candidate pool",
        "",
        "> Candidate reservoir only — no rollout spend is approved by this file.",
        "",
        (
            f"**{len(index)} tasks:** {difficulty['medium']} medium / "
            f"{difficulty['hard']} hard; {len(repos)} repositories; "
            f"{len(families)} bug families."
        ),
        "",
        (
            f"Sources: {sources['retained_expansion_seed']} retained reviewed seeds + "
            f"{sources['swegym_full_excluding_lite']} new SWE-Gym Full-minus-Lite candidates."
        ),
        "",
        (
            f"{flagged} tasks carry one or more caution flags. All 70 new tasks remain "
            "`needs_manual_review` until statement/gold/test inspection and smoke testing."
        ),
        "",
        "| # | Instance | Repo | Tier | Score | Family | Source | Flags | Title |",
        "|---:|---|---|:---:|---:|---|---|---|---|",
    ]
    for row in index:
        source = "seed" if row["candidate_source"] == "retained_expansion_seed" else "new"
        flags = ", ".join(row["review_flags"]) or "—"
        cells = (
            row["position"], row["instance_id"], row["repo"], row["difficulty"],
            row["difficulty_score"], row["bug_family"], source, flags,
            row["problem_title"],
        )
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=FULL_PARQUET)
    parser.add_argument("--lite", type=Path, default=LITE_PARQUET)
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_FILE)
    parser.add_argument("--report", type=Path, default=REPORT_FILE)
    args = parser.parse_args()

    if args.target != TARGET_TOTAL:
        raise SystemExit(
            f"This version's tier quotas are calibrated for --target {TARGET_TOTAL}; "
            "change TARGET_MEDIUM/TARGET_HARD with the target."
        )
    for path in (args.full, args.lite, FROZEN_EXPANSION):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    full_rows = pq.read_table(args.full).to_pylist()
    lite_ids = set(pq.read_table(args.lite, columns=["instance_id"])["instance_id"].to_pylist())
    frozen_rows = json.loads(FROZEN_EXPANSION.read_text())
    frozen_by_id = {row["instance_id"]: row for row in frozen_rows}
    missing_seeds = sorted(set(SEED_META) - set(frozen_by_id))
    if missing_seeds:
        raise SystemExit(f"retained seeds missing from frozen expansion: {missing_seeds}")

    previous = prior_ids()
    rejected_frozen = set(frozen_by_id) - set(SEED_META)
    full_ids = {row["instance_id"] for row in full_rows}

    seeds: list[dict[str, Any]] = []
    for instance_id, human in SEED_META.items():
        row = frozen_by_id[instance_id]
        automatic = score_row(row)
        metadata = {
            **automatic,
            **human,
            "review_flags": review_flags(row, automatic),
        }
        seeds.append(decorated_row(row, source="retained_expansion_seed", metadata=metadata))

    excluded_counts: Counter[str] = Counter()
    auto_candidates: list[dict[str, Any]] = []
    for row in full_rows:
        iid = row["instance_id"]
        if iid in lite_ids:
            excluded_counts["swegym_lite_member"] += 1
            continue
        if iid in previous:
            excluded_counts["previously_attempted_or_frozen"] += 1
            continue
        if iid in rejected_frozen:
            excluded_counts["rejected_current_expansion_task"] += 1
            continue
        if iid in SEED_META:
            continue
        rejections = quality_rejections(row)
        if rejections:
            for reason in rejections:
                excluded_counts[reason] += 1
            continue
        scored = score_row(row)
        if scored["difficulty"] == "easy":
            excluded_counts["difficulty_below_medium"] += 1
            continue
        family = bug_family(row)
        flags = review_flags(row, scored)
        metadata = {
            **scored,
            "bug_family": family,
            "review_status": "needs_manual_review",
            "review_flags": flags,
            "review_notes": [],
        }
        auto_candidates.append(
            decorated_row(row, source="swegym_full_excluding_lite", metadata=metadata)
        )

    repo_counts: Counter[str] = Counter(row["repo"] for row in seeds)
    family_counts: Counter[str] = Counter(row["bug_family"] for row in seeds)
    seed_medium = sum(row["difficulty"] == "medium" for row in seeds)
    seed_hard = sum(row["difficulty"] == "hard" for row in seeds)
    medium = choose_balanced(
        auto_candidates,
        want=TARGET_MEDIUM - seed_medium,
        difficulty="medium",
        repo_counts=repo_counts,
        family_counts=family_counts,
    )
    hard = choose_balanced(
        auto_candidates,
        want=TARGET_HARD - seed_hard,
        difficulty="hard",
        repo_counts=repo_counts,
        family_counts=family_counts,
    )
    chosen_ids = {row["instance_id"] for row in medium + hard}
    if len(chosen_ids) != len(medium) + len(hard):
        raise SystemExit("duplicate auto-selected instance id")

    # Interleave by tier so a prefix of the candidate file is representative.
    by_tier = {"medium": seeds[:0] + [r for r in seeds if r["difficulty"] == "medium"] + medium,
               "hard": seeds[:0] + [r for r in seeds if r["difficulty"] == "hard"] + hard}
    output: list[dict[str, Any]] = []
    indices = {"medium": 0, "hard": 0}
    # Three medium slots then one hard slot approximates the final 70/30 policy
    # while preserving the exact configured 56/24 quota over the full pool.
    pattern = ("medium", "medium", "hard", "medium")
    while any(indices[t] < len(by_tier[t]) for t in by_tier):
        progressed = False
        for tier in pattern:
            if indices[tier] < len(by_tier[tier]):
                output.append(by_tier[tier][indices[tier]])
                indices[tier] += 1
                progressed = True
        if not progressed:
            break

    if len(output) != args.target:
        raise SystemExit(
            f"could fill only {len(output)}/{args.target}: "
            f"medium={len(by_tier['medium'])}/{TARGET_MEDIUM}, "
            f"hard={len(by_tier['hard'])}/{TARGET_HARD}"
        )

    validate_output(
        output,
        full_ids=full_ids,
        lite_ids=lite_ids,
        previous=previous,
        rejected_frozen=rejected_frozen,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    index = [
        {
            "position": position,
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "problem_title": row["problem_title"],
            "difficulty": row["difficulty"],
            "difficulty_score": row["difficulty_score"],
            "difficulty_components": row["difficulty_components"],
            "bug_family": row["bug_family"],
            "candidate_source": row["candidate_source"],
            "gold_patch_stats": row["gold_patch_stats"],
            "test_stats": row["test_stats"],
            "review_status": row["review_status"],
            "review_flags": row["review_flags"],
            "review_notes": row["review_notes"],
        }
        for position, row in enumerate(output, start=1)
    ]
    args.index.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    write_markdown(args.markdown, index)

    report = {
        "schema_version": "medium_hard_candidate_pool_v1",
        "candidate_count": len(output),
        "validation": "PASS",
        "difficulty_rubric": {
            "formula": (
                "localization(0..3) + reasoning(0..3) + coupling(0..3) + "
                "validation(0..2) - implementation_hint_penalty(0..2)"
            ),
            "medium": "score 6..8",
            "hard": "score 9..11",
            "principle": "diagnostic difficulty, not gold-patch size alone",
        },
        "quality_prefilter": [
            "non-empty statement, gold patch, test patch, and FAIL_TO_PASS",
            "well-formed pytest node ids",
            "gold does not change only tests",
            "gold <=8 files and <=350 changed lines",
            "FAIL_TO_PASS <=20",
            "no strong expected-behavior ambiguity",
            "no high-confidence implementation leak",
        ],
        "selection_policy": {
            "retained_seed_count": len(seeds),
            "new_full_excluding_lite_count": len(medium) + len(hard),
            "difficulty_targets": {"medium": TARGET_MEDIUM, "hard": TARGET_HARD},
            "max_per_repo": MAX_PER_REPO,
            "max_per_bug_family": MAX_PER_FAMILY,
            "prior_attempts_and_frozen_lists_excluded": len(previous),
            "current_expansion_nonseeds_excluded": len(rejected_frozen),
            "approval_state": "CANDIDATES_ONLY_NOT_SPEND_APPROVED",
        },
        "by_difficulty": dict(Counter(row["difficulty"] for row in output)),
        "by_repo": dict(sorted(Counter(row["repo"] for row in output).items())),
        "by_bug_family": dict(sorted(Counter(row["bug_family"] for row in output).items())),
        "by_source": dict(Counter(row["candidate_source"] for row in output)),
        "by_review_status": dict(Counter(row["review_status"] for row in output)),
        "by_review_flag": dict(sorted(Counter(
            flag for row in output for flag in row["review_flags"]
        ).items())),
        "excluded_reason_counts": dict(excluded_counts.most_common()),
        "auto_eligible_before_balancing": len(auto_candidates),
        "output": str(args.output.relative_to(REPO_DIR)),
        "review_index": str(args.index.relative_to(REPO_DIR)),
        "review_markdown": str(args.markdown.relative_to(REPO_DIR)),
        "provenance": {
            "full_parquet": {
                "path": str(args.full.relative_to(REPO_DIR)),
                "rows": len(full_rows),
                "sha256": file_sha256(args.full),
            },
            "lite_parquet": {
                "path": str(args.lite.relative_to(REPO_DIR)),
                "rows": len(lite_ids),
                "sha256": file_sha256(args.lite),
            },
            "frozen_expansion": {
                "path": str(FROZEN_EXPANSION.relative_to(REPO_DIR)),
                "rows": len(frozen_rows),
                "sha256": file_sha256(FROZEN_EXPANSION),
            },
            "candidate_pool_sha256": file_sha256(args.output),
            "candidate_pool_index_sha256": file_sha256(args.index),
        },
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
