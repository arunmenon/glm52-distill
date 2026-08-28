#!/usr/bin/env python3
"""Convert the 80-task reservoir into audited review-stage slices.

This stage is deliberately deterministic and contains explicit human review
decisions for every one of the 70 new candidates.  It does not approve model
spend and it does not overwrite the prior 24-task frozen expansion list.

Outputs under ``runs/expansion/bugfix``:

* candidate_review.json: every reservoir row with image, tier, and disposition
* candidate_survivors.json: live bug-fix candidates for decontamination/smoke
* future_slice.json: live feature/performance/maintenance candidates
* quality_rejects.json: live but unsuitable reports
* hard_excluded.json: candidates whose image is genuinely absent
* candidate_review.md / candidate_review_report.json: readable audit trail
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).parent
EXP_DIR = REPO_DIR / "runs" / "expansion" / "bugfix"
POOL_FILE = EXP_DIR / "candidate_pool.json"
PREFLIGHT_FILE = EXP_DIR / "image_preflight.json"
FROZEN_FILE = EXP_DIR / "expansion_tasks.json"

REVIEW_FILE = EXP_DIR / "candidate_review.json"
SURVIVOR_FILE = EXP_DIR / "candidate_survivors.json"
FUTURE_FILE = EXP_DIR / "future_slice.json"
QUALITY_REJECT_FILE = EXP_DIR / "quality_rejects.json"
HARD_EXCLUDE_FILE = EXP_DIR / "hard_excluded.json"
MARKDOWN_FILE = EXP_DIR / "candidate_review.md"
REPORT_FILE = EXP_DIR / "candidate_review_report.json"
DECONTAM_FILE = EXP_DIR / "candidate_survivors_decontam.json"


def decision(category: str, reason: str, *flags: str) -> dict[str, Any]:
    return {"category": category, "reason": reason, "flags": list(flags)}


# Explicit review of every non-seed row.  Categories describe task semantics;
# image availability is applied later as a separate hard gate.
MANUAL_REVIEW: dict[str, dict[str, Any]] = {
    "Project-MONAI__MONAI-578": decision(
        "future_feature", "Intentional removal of an existing API option, not diagnosis of a defect."
    ),
    "bokeh__bokeh-13641": decision(
        "future_feature", "Adds a DataTable construction convenience API."
    ),
    "pandas-dev__pandas-50672": decision(
        "bugfix", "Concrete ArrowDtype predicate failure with explicit expected behavior."
    ),
    "pydantic__pydantic-9004": decision(
        "bugfix", "Silent acceptance of a forbidden model_config field is reproducible and well specified."
    ),
    "facebookresearch__hydra-1458": decision(
        "bugfix", "Misspelled Hydra keys bypass strict CLI override validation."
    ),
    "modin-project__modin-6618": decision(
        "future_performance", "Requests lazy evaluation and explicitly lacks a performance benchmark."
    ),
    "Project-MONAI__MONAI-2465": decision(
        "future_feature", "Usage question resolved by exposing a new padding option."
    ),
    "pandas-dev__pandas-50151": decision(
        "bugfix", "DataFrame dtype display loses backend information inconsistently with Series."
    ),
    "conan-io__conan-14532": decision(
        "future_feature", "New compatibility API whose statement includes a proposed source patch.",
        "statement_contains_proposed_patch",
    ),
    "dask__dask-8590": decision(
        "future_feature", "Adds a regions keyword and a new partial-write capability."
    ),
    "bokeh__bokeh-13636": decision(
        "bugfix", "Repeated DOM ids break multi-document embedding; expected uniqueness is clear."
    ),
    "pydantic__pydantic-6100": decision(
        "bugfix", "Serialization regression has a self-contained failing assertion."
    ),
    "modin-project__modin-6759": decision(
        "future_performance", "Improves an internal partial dtype cache without a user-visible wrong result."
    ),
    "conan-io__conan-14051": decision(
        "future_feature", "Adds pkg-config metadata for a new Meson use case."
    ),
    "modin-project__modin-5949": decision(
        "quality_reject", "Statement names the exact code location and proposed implementation.",
        "implementation_leak",
    ),
    "facebookresearch__hydra-952": decision(
        "bugfix", "Assigning a dictionary to an Any-typed field fails with a concise reproduction."
    ),
    "pydantic__pydantic-9137": decision(
        "future_feature", "Requests a new per-field variant of a model-level coercion option."
    ),
    "Project-MONAI__MONAI-3493": decision(
        "future_feature", "Changes the mathematical semantics of a loss after an open design question.",
        "product_semantics_decision",
    ),
    "pandas-dev__pandas-47780": decision(
        "bugfix", "PeriodIndex handles pd.NA inconsistently with None and numpy.nan."
    ),
    "pydantic__pydantic-8947": decision(
        "bugfix", "create_model mishandles a valid Annotated type accepted by BaseModel."
    ),
    "dask__dask-10054": decision(
        "bugfix", "Nullable all-null partitions break sort_values with a reproducible exception."
    ),
    "iterative__dvc-1690": decision(
        "bugfix", "Remote status incorrectly depends on a populated local cache."
    ),
    "facebookresearch__hydra-1725": decision(
        "quality_reject", "Statement leaves the required behavior undecided: allow the list or emit an error.",
        "ambiguous_expected_behavior",
    ),
    "conan-io__conan-10960": decision(
        "bugfix", "Layout use causes package version metadata to become null."
    ),
    "facebookresearch__hydra-1422": decision(
        "future_feature", "Explicit request to relax override syntax for dictionary fields."
    ),
    "dask__dask-6992": decision(
        "future_feature", "Adds pandas dropna feature parity to Dask groupby."
    ),
    "modin-project__modin-5946": decision(
        "bugfix", "JSON-lines records with differing columns trigger an incorrect length failure.",
        "external_reproduction_not_self_contained",
    ),
    "bokeh__bokeh-13608": decision(
        "bugfix", "Multiple inline-math labels reproducibly disappear."
    ),
    "bokeh__bokeh-13800": decision(
        "future_feature", "Adds credential forwarding and configurable CORS behavior."
    ),
    "getmoto__moto-5644": decision(
        "bugfix", "A version regression empties Batch job container details.",
        "statement_links_suspected_upstream_change",
    ),
    "Project-MONAI__MONAI-2942": decision(
        "future_performance", "Adds a device parameter to avoid CPU/GPU transfer."
    ),
    "pandas-dev__pandas-55764": decision(
        "bugfix", "Patch-release regression changes DataFrame subclass copy behavior."
    ),
    "iterative__dvc-1700": decision(
        "bugfix", "Recursive add incorrectly captures Git's internal files."
    ),
    "python__mypy-11448": decision(
        "future_maintenance", "Improves diagnostic wording without correcting type-checking behavior."
    ),
    "iterative__dvc-3727": decision(
        "bugfix", "JSON and text no-change output is inconsistent across related diff commands."
    ),
    "pydantic__pydantic-6293": decision(
        "bugfix", "Pydantic dataclasses reject valid typing.Annotated fields."
    ),
    "Project-MONAI__MONAI-1765": decision(
        "future_feature", "Unifies loss-component input conventions through an API behavior change."
    ),
    "modin-project__modin-6638": decision(
        "quality_reject", "No self-contained workbook or minimal reproduction is supplied.",
        "underspecified_reproduction",
    ),
    "dask__dask-10042": decision(
        "bugfix", "Parquet filtering compares missing statistics against strings and raises TypeError."
    ),
    "facebookresearch__hydra-614": decision(
        "future_feature", "Introduces new interpolation resolvers and has no PASS_TO_PASS coverage."
    ),
    "pandas-dev__pandas-54643": decision(
        "future_feature", "Explicit enhancement adding bad-line handling to the PyArrow engine."
    ),
    "conan-io__conan-12695": decision(
        "quality_reject", "Issue asks maintainers to choose between accepting whitespace and improving an error.",
        "ambiguous_expected_behavior",
    ),
    "bokeh__bokeh-13443": decision(
        "bugfix", "Cloning a normal model fails when it contains read-only properties."
    ),
    "getmoto__moto-6743": decision(
        "bugfix", "Moto rejects S3 access-point ARNs that AWS IAM accepts."
    ),
    "getmoto__moto-6709": decision(
        "bugfix", "DynamoDB projection expressions mishandle mapped names containing dots."
    ),
    "Project-MONAI__MONAI-2104": decision(
        "quality_reject", "Statement gives the exact deepcopy implementation and analogous source location.",
        "implementation_leak",
    ),
    "iterative__dvc-5080": decision(
        "future_feature", "Adds a new config --list command-line capability."
    ),
    "pandas-dev__pandas-56321": decision(
        "bugfix", "Masked string-column creation stringifies and truncates missing values."
    ),
    "modin-project__modin-6758": decision(
        "future_performance", "Preserves internal dtype-cache metadata after setitem."
    ),
    "python__mypy-11213": decision(
        "bugfix", "Lambda analysis leaks errors from an unchecked function body."
    ),
    "pydantic__pydantic-5834": decision(
        "bugfix", "NumPy annotations incorrectly produce ValidatorIterator values."
    ),
    "facebookresearch__hydra-1540": decision(
        "future_feature", "Bundles search-path inspection and warning enhancements."
    ),
    "python__mypy-15846": decision(
        "future_feature", "Requests negative TypeVar constraints and labels itself a possible feature request."
    ),
    "conan-io__conan-13450": decision(
        "future_feature", "Adds C++20 support with unresolved compiler/Meson-version semantics."
    ),
    "dask__dask-8462": decision(
        "bugfix", "Token naming ignores chunks and dtype, causing distinct graphs to collide."
    ),
    "modin-project__modin-6951": decision(
        "future_performance", "External benchmark report mixes speed and correctness without a minimal reproduction.",
        "weak_performance_validation",
    ),
    "pydantic__pydantic-5868": decision(
        "bugfix", "allow_inf_nan is ignored for Decimal despite explicit configuration."
    ),
    "bokeh__bokeh-13757": decision(
        "bugfix", "Release-candidate regression makes VBox construction fail.",
        "reported_on_windows_but_gold_tests_are_unit_level",
    ),
    "Project-MONAI__MONAI-506": decision(
        "future_feature", "Adds torch.Tensor support to a NumPy-only transform."
    ),
    "getmoto__moto-7317": decision(
        "bugfix", "Glue create_database silently discards valid AWS resource tags."
    ),
    "conan-io__conan-11330": decision(
        "future_feature", "Redesigns copy_symlink_folders defaults and API behavior."
    ),
    "pandas-dev__pandas-57297": decision(
        "bugfix", "Descending column sort incorrectly drops the row index."
    ),
    "python__mypy-15355": decision(
        "future_feature", "Requests an opt-in stubgen mode to retain default values."
    ),
    "iterative__dvc-4309": decision(
        "bugfix", "Local remote configuration cannot modify an existing repository remote."
    ),
    "dask__dask-6779": decision(
        "future_performance", "Scheduler-ordering change targets memory use rather than a wrong result."
    ),
    "facebookresearch__hydra-1655": decision(
        "future_feature", "Requests resolved interpolation in help output."
    ),
    "bokeh__bokeh-12902": decision(
        "bugfix", "Non-unique notebook DOM ids update the wrong tab and make loading appear stuck."
    ),
    "Project-MONAI__MONAI-2780": decision(
        "future_feature", "Adds padding kwargs across several transforms."
    ),
    "pandas-dev__pandas-53809": decision(
        "future_maintenance", "Deprecation/removal campaign rather than a behavioral defect."
    ),
    "pydantic__pydantic-5874": decision(
        "future_maintenance", "Broad multi-objective V2 dataclass improvement bundle."
    ),
}


FROZEN_DROP_REVIEW: dict[str, dict[str, str]] = {
    "getmoto__moto-6308": {
        "disposition": "future_easy_bug_slice",
        "reason": "Clear but low-diagnostic one-file lifecycle-ordering fix; below the requested medium/hard focus.",
    },
    "pandas-dev__pandas-56849": {
        "disposition": "future_easy_bug_slice",
        "reason": "Narrow lowercase frequency-alias regression with a small localized fix.",
    },
    "python__mypy-15413": {
        "disposition": "quality_reject",
        "reason": "Reporter explicitly leaves intended warn-return-any behavior undecided.",
    },
    "getmoto__moto-5835": {
        "disposition": "future_easy_bug_slice",
        "reason": "Simple enum validation omission; below the requested diagnostic difficulty.",
    },
    "conan-io__conan-14177": {
        "disposition": "future_feature_slice",
        "reason": "Feature request supplies the desired signature and behavior directly.",
    },
    "pandas-dev__pandas-53958": {
        "disposition": "future_api_slice",
        "reason": "API-placement design question with a one-file export-only implementation.",
    },
    "pandas-dev__pandas-51605": {
        "disposition": "future_easy_bug_slice",
        "reason": "Valid empty-input regression, but localized and trivial relative to the target slice.",
    },
    "getmoto__moto-7361": {
        "disposition": "future_feature_slice",
        "reason": "Adds emulation coverage for a previously unsupported API Gateway patch operation.",
    },
    "python__mypy-16503": {
        "disposition": "reconsider_after_preflight",
        "reason": "Good medium bug, held only because it was outside the 10 retained seeds and mypy was already dense.",
    },
    "python__mypy-11585": {
        "disposition": "reconsider_after_preflight",
        "reason": "Good type-system bug, held to avoid further mypy concentration; not rejected on quality.",
    },
    "conan-io__conan-14378": {
        "disposition": "future_environment_slice",
        "reason": "Windows resource-compiler report carries environment/toolchain smoke risk.",
    },
    "conan-io__conan-10874": {
        "disposition": "future_feature_slice",
        "reason": "Small namespace migration feature, not a diagnostic bug-fix task.",
    },
    "iterative__dvc-4961": {
        "disposition": "hard_reject_verifier_input",
        "reason": "FAIL_TO_PASS node ids are truncated parametrized pytest ids and are unsafe for verification.",
    },
    "iterative__dvc-2017": {
        "disposition": "hard_reject_task_test_mismatch",
        "reason": "Requirements/setup maintenance statement and gold patch do not match the listed pipeline FAIL_TO_PASS tests.",
    },
}


SEED_REPAIRS: dict[str, dict[str, Any]] = {
    "dask__dask-7305": {
        "add_fail_to_pass": [
            "dask/dataframe/tests/test_shuffle.py::test_set_index_interpolate_large_uint",
        ],
        "note": "Added the direct large-uint regression test to FAIL_TO_PASS.",
    },
    "iterative__dvc-9395": {
        "add_fail_to_pass": [
            "tests/func/test_repro_multistage.py::test_repro_pulls_mising_import",
            "tests/func/test_repro_multistage.py::test_repro_pulls_intermediate_out",
        ],
        "note": "Added the missing-import and intermediate-output gold cases to FAIL_TO_PASS.",
    },
    "getmoto__moto-6317": {
        "statement_append": (
            "\n\nCorpus clarification from the regression case: with the S3 mock already "
            "active, create a boto3 S3 client and call `patch_client(client)` twice. "
            "Those calls must not register the same event handler again; a subsequent "
            "create-bucket, put-object, and get-object round trip should preserve the "
            "uploaded bytes instead of producing an empty object."
        ),
        "note": "Added the repeated-patch_client reproduction exercised by the gold test.",
    },
    "python__mypy-14835": {
        "add_pass_to_pass": [
            "mypy/test/testdaemon.py::DaemonSuite::daemon.test::testDaemonBasic",
            "mypy/test/testdaemon.py::DaemonSuite::daemon.test::testDaemonRunRestart",
            "mypy/test/testdaemon.py::DaemonSuite::daemon.test::testDaemonRecheck",
        ],
        "note": (
            "Added three existing daemon lifecycle/incremental cases from base commit "
            "e21ddbf3 as PASS_TO_PASS coverage."
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return json.loads(value)
    return list(value or [])


def patch_stats(patch: str) -> tuple[int, int]:
    files = len(re.findall(r"^diff --git ", patch or "", re.MULTILINE))
    changed = sum(
        1
        for line in (patch or "").splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return files, changed


def runner_tier(row: dict[str, Any]) -> str:
    """Exact tier definition used by 09_rehearsal.py."""
    files, changed = patch_stats(row["patch"])
    n_f2p = len(as_list(row["FAIL_TO_PASS"]))
    if files >= 3 or changed > 60 or n_f2p > 5:
        return "hard"
    if files == 1 and changed < 15:
        return "easy"
    return "medium"


def append_unique(values: list[Any], additions: list[Any]) -> list[Any]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result


def apply_seed_repair(row: dict[str, Any]) -> None:
    repair = SEED_REPAIRS.get(row["instance_id"])
    if repair is None:
        return
    row["FAIL_TO_PASS"] = append_unique(
        as_list(row.get("FAIL_TO_PASS")), repair.get("add_fail_to_pass", [])
    )
    row["PASS_TO_PASS"] = append_unique(
        as_list(row.get("PASS_TO_PASS")), repair.get("add_pass_to_pass", [])
    )
    row["problem_statement"] = row.get("problem_statement", "") + repair.get(
        "statement_append", ""
    )
    row["review_status"] = "retained_seed_repaired"
    row["review_notes"] = append_unique(
        list(row.get("review_notes", [])), [repair["note"]]
    )
    row["seed_repair"] = {
        "status": "applied",
        "added_fail_to_pass": repair.get("add_fail_to_pass", []),
        "added_pass_to_pass": repair.get("add_pass_to_pass", []),
        "statement_clarified": bool(repair.get("statement_append")),
        "note": repair["note"],
    }


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    for path in (POOL_FILE, PREFLIGHT_FILE, FROZEN_FILE):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    pool = json.loads(POOL_FILE.read_text())
    preflight = json.loads(PREFLIGHT_FILE.read_text())
    frozen = json.loads(FROZEN_FILE.read_text())
    pool_ids = [row["instance_id"] for row in pool]
    if len(pool_ids) != 80 or len(pool_ids) != len(set(pool_ids)):
        raise SystemExit("candidate pool must contain 80 unique rows")
    if sha256(POOL_FILE) != preflight["pool_sha256_expected"]:
        raise SystemExit("preflight snapshot does not match candidate pool")

    preflight_by_id = {row["instance_id"]: row for row in preflight["records"]}
    if set(preflight_by_id) != set(pool_ids):
        raise SystemExit("preflight records do not cover the candidate pool exactly")
    new_ids = {
        row["instance_id"]
        for row in pool
        if row["candidate_source"] == "swegym_full_excluding_lite"
    }
    if set(MANUAL_REVIEW) != new_ids:
        missing = sorted(new_ids - set(MANUAL_REVIEW))
        extra = sorted(set(MANUAL_REVIEW) - new_ids)
        raise SystemExit(f"manual review coverage mismatch: missing={missing}, extra={extra}")

    reviewed: list[dict[str, Any]] = []
    for position, original in enumerate(pool, start=1):
        row = dict(original)
        apply_seed_repair(row)
        heuristic = row.pop("difficulty")
        row["position"] = position
        row["heuristic_difficulty"] = heuristic
        row["tier"] = runner_tier(row)
        image = preflight_by_id[row["instance_id"]]
        row["image_preflight"] = {
            "result": image["result"],
            "image": image["image"],
            "attempts": image["attempts"],
            "snapshot_at": preflight["snapshot_at"],
        }

        if row["candidate_source"] == "retained_expansion_seed":
            manual = decision(
                "bugfix",
                "Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit.",
            )
        else:
            manual = MANUAL_REVIEW[row["instance_id"]]
        row["manual_review"] = manual

        if image["result"] == "missing":
            disposition = "hard_exclude_missing_image"
        elif image["result"] != "available":
            disposition = "hold_indeterminate_image"
        elif manual["category"] == "bugfix":
            disposition = "bugfix_survivor"
        elif manual["category"].startswith("future_"):
            disposition = "future_slice"
        else:
            disposition = "quality_reject"
        row["conversion_disposition"] = disposition
        reviewed.append(row)

    survivors = [row for row in reviewed if row["conversion_disposition"] == "bugfix_survivor"]
    future = [row for row in reviewed if row["conversion_disposition"] == "future_slice"]
    quality_rejects = [row for row in reviewed if row["conversion_disposition"] == "quality_reject"]
    hard_excluded = [row for row in reviewed if row["conversion_disposition"] == "hard_exclude_missing_image"]
    if len(reviewed) != len(survivors) + len(future) + len(quality_rejects) + len(hard_excluded):
        raise SystemExit("unaccounted conversion disposition")

    frozen_seed_ids = {
        row["instance_id"]
        for row in pool
        if row["candidate_source"] == "retained_expansion_seed"
    }
    frozen_ids = {row["instance_id"] for row in frozen}
    dropped_ids = frozen_ids - frozen_seed_ids
    if set(FROZEN_DROP_REVIEW) != dropped_ids:
        raise SystemExit("frozen-list disposition coverage mismatch")
    frozen_dispositions = []
    for row in frozen:
        iid = row["instance_id"]
        title = (row.get("problem_statement") or "").splitlines()[0]
        if iid in frozen_seed_ids:
            disposition = {
                "disposition": "retained_seed",
                "reason": "One of the 10 explicitly retained human-reviewed seeds.",
            }
        else:
            disposition = FROZEN_DROP_REVIEW[iid]
        frozen_dispositions.append({
            "instance_id": iid,
            "repo": row["repo"],
            "title": title,
            "prior_frozen_tier": row["tier"],
            **disposition,
        })

    for path, rows in (
        (REVIEW_FILE, reviewed),
        (SURVIVOR_FILE, survivors),
        (FUTURE_FILE, future),
        (QUALITY_REJECT_FILE, quality_rejects),
        (HARD_EXCLUDE_FILE, hard_excluded),
    ):
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    disposition_counts = Counter(row["conversion_disposition"] for row in reviewed)
    survivor_runner_tiers = Counter(row["tier"] for row in survivors)
    survivor_heuristic = Counter(row["heuristic_difficulty"] for row in survivors)
    decontam: dict[str, Any] = {
        "path": str(DECONTAM_FILE.relative_to(REPO_DIR)),
        "status": "NOT_RUN",
    }
    if DECONTAM_FILE.exists():
        gate = json.loads(DECONTAM_FILE.read_text())
        current_hash = sha256(SURVIVOR_FILE)
        decontam.update({
            "status": (
                gate.get("verdict", "UNKNOWN")
                if gate.get("task_list_sha256") == current_hash
                else "STALE"
            ),
            "verdict": gate.get("verdict"),
            "task_list_sha256": gate.get("task_list_sha256"),
            "current_survivor_sha256": current_hash,
            "checks": gate.get("checks", {}),
        })
    report = {
        "schema_version": "expansion_candidate_review_v1",
        "approval_state": "REVIEWED_CANDIDATES_NOT_SPEND_APPROVED",
        "input_pool": str(POOL_FILE.relative_to(REPO_DIR)),
        "input_pool_sha256": sha256(POOL_FILE),
        "image_preflight": {
            "path": str(PREFLIGHT_FILE.relative_to(REPO_DIR)),
            "snapshot_at": preflight["snapshot_at"],
            "totals": preflight["totals"],
        },
        "counts": {
            "reviewed": len(reviewed),
            **dict(sorted(disposition_counts.items())),
        },
        "survivors": {
            "count": len(survivors),
            "retained_seeds": sum(
                row["candidate_source"] == "retained_expansion_seed" for row in survivors
            ),
            "new": sum(
                row["candidate_source"] == "swegym_full_excluding_lite" for row in survivors
            ),
            "by_runner_tier": dict(sorted(survivor_runner_tiers.items())),
            "by_heuristic_difficulty": dict(sorted(survivor_heuristic.items())),
            "by_repo": dict(sorted(Counter(row["repo"] for row in survivors).items())),
            "by_bug_family": dict(sorted(Counter(row["bug_family"] for row in survivors).items())),
        },
        "tier_policy": {
            "tier": "Exact 09_rehearsal.tier_of patch-size tier, used for corpus continuity.",
            "heuristic_difficulty": "Diagnostic rubric label from the candidate builder.",
        },
        "decontamination": decontam,
        "authoritative_next_stage": str(SURVIVOR_FILE.relative_to(REPO_DIR)),
        "prior_frozen_list": {
            "path": str(FROZEN_FILE.relative_to(REPO_DIR)),
            "authority": "historical frozen snapshot; not overwritten",
            "dispositions": frozen_dispositions,
        },
        "historical_monai_preflight_bug": {
            "observed_false_misses_in_frozen_selection": 7,
            "cause": "Docker repository path was not lowercased; registry returned HTTP 400, not 404.",
            "status": "fixed in runner and preflight",
            "disposition": "Those Lite tasks are re-eligible but require normal rubric review and current preflight before insertion.",
        },
        "outputs": {
            "review": str(REVIEW_FILE.relative_to(REPO_DIR)),
            "survivors": str(SURVIVOR_FILE.relative_to(REPO_DIR)),
            "future_slice": str(FUTURE_FILE.relative_to(REPO_DIR)),
            "quality_rejects": str(QUALITY_REJECT_FILE.relative_to(REPO_DIR)),
            "hard_excluded": str(HARD_EXCLUDE_FILE.relative_to(REPO_DIR)),
            "markdown": str(MARKDOWN_FILE.relative_to(REPO_DIR)),
        },
        "output_sha256": {
            "review": sha256(REVIEW_FILE),
            "survivors": sha256(SURVIVOR_FILE),
            "future_slice": sha256(FUTURE_FILE),
            "quality_rejects": sha256(QUALITY_REJECT_FILE),
            "hard_excluded": sha256(HARD_EXCLUDE_FILE),
        },
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# Expansion candidate manual review",
        "",
        "> Review-stage classification only — no rollout spend is approved.",
        "",
        (
            f"**{len(survivors)} live bug-fix survivors**, {len(future)} live future-slice tasks, "
            f"{len(quality_rejects)} quality rejects, and {len(hard_excluded)} image exclusions."
        ),
        "",
        (
            "Runner tier is the corpus-compatible patch-size label. Heuristic difficulty is the "
            "diagnostic score used to construct the reservoir."
        ),
        "",
        "| # | Instance | Runner tier | Heuristic | Preflight | Disposition | Review reason |",
        "|---:|---|:---:|:---:|:---:|---|---|",
    ]
    for row in reviewed:
        cells = (
            row["position"],
            row["instance_id"],
            row["tier"],
            row["heuristic_difficulty"],
            row["image_preflight"]["result"],
            row["conversion_disposition"],
            row["manual_review"]["reason"],
        )
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    lines.extend([
        "",
        "## Prior frozen-24 dispositions",
        "",
        "| Instance | Prior tier | Disposition | Reason |",
        "|---|:---:|---|---|",
    ])
    for row in frozen_dispositions:
        cells = (row["instance_id"], row["prior_frozen_tier"], row["disposition"], row["reason"])
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    MARKDOWN_FILE.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "validation": "PASS",
        "counts": report["counts"],
        "survivors": report["survivors"],
        "outputs": report["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
