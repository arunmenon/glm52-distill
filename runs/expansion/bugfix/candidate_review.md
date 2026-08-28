# Expansion candidate manual review

> Review-stage classification only — no rollout spend is approved.

**42 live bug-fix survivors**, 30 live future-slice tasks, 5 quality rejects, and 3 image exclusions.

Runner tier is the corpus-compatible patch-size label. Heuristic difficulty is the diagnostic score used to construct the reservoir.

| # | Instance | Runner tier | Heuristic | Preflight | Disposition | Review reason |
|---:|---|:---:|:---:|:---:|---|---|
| 1 | dask__dask-7305 | easy | medium | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 2 | iterative__dvc-1681 | easy | medium | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 3 | getmoto__moto-6317 | medium | hard | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 4 | iterative__dvc-9395 | easy | medium | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 5 | getmoto__moto-5885 | medium | medium | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 6 | python__mypy-16154 | medium | medium | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 7 | getmoto__moto-7607 | medium | hard | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 8 | conan-io__conan-10917 | medium | medium | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 9 | Project-MONAI__MONAI-578 | hard | medium | available | future_slice | Intentional removal of an existing API option, not diagnosis of a defect. |
| 10 | bokeh__bokeh-13641 | hard | medium | available | future_slice | Adds a DataTable construction convenience API. |
| 11 | python__mypy-14835 | medium | hard | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 12 | pandas-dev__pandas-50672 | hard | medium | available | bugfix_survivor | Concrete ArrowDtype predicate failure with explicit expected behavior. |
| 13 | pydantic__pydantic-9004 | medium | medium | available | bugfix_survivor | Silent acceptance of a forbidden model_config field is reproducible and well specified. |
| 14 | facebookresearch__hydra-1458 | medium | medium | available | bugfix_survivor | Misspelled Hydra keys bypass strict CLI override validation. |
| 15 | python__mypy-14988 | medium | hard | available | bugfix_survivor | Retained from the prior human-reviewed frozen list; required repair is applied and any smoke note remains explicit. |
| 16 | modin-project__modin-6618 | medium | medium | available | future_slice | Requests lazy evaluation and explicitly lacks a performance benchmark. |
| 17 | Project-MONAI__MONAI-2465 | hard | medium | available | future_slice | Usage question resolved by exposing a new padding option. |
| 18 | pandas-dev__pandas-50151 | hard | medium | available | bugfix_survivor | DataFrame dtype display loses backend information inconsistently with Series. |
| 19 | conan-io__conan-14532 | hard | hard | available | future_slice | New compatibility API whose statement includes a proposed source patch. |
| 20 | dask__dask-8590 | medium | medium | available | future_slice | Adds a regions keyword and a new partial-write capability. |
| 21 | bokeh__bokeh-13636 | hard | medium | available | bugfix_survivor | Repeated DOM ids break multi-document embedding; expected uniqueness is clear. |
| 22 | pydantic__pydantic-6100 | medium | medium | available | bugfix_survivor | Serialization regression has a self-contained failing assertion. |
| 23 | modin-project__modin-6759 | hard | hard | available | future_slice | Improves an internal partial dtype cache without a user-visible wrong result. |
| 24 | conan-io__conan-14051 | medium | medium | available | future_slice | Adds pkg-config metadata for a new Meson use case. |
| 25 | modin-project__modin-5949 | hard | medium | available | quality_reject | Statement names the exact code location and proposed implementation. |
| 26 | facebookresearch__hydra-952 | medium | medium | available | bugfix_survivor | Assigning a dictionary to an Any-typed field fails with a concise reproduction. |
| 27 | pydantic__pydantic-9137 | hard | hard | available | future_slice | Requests a new per-field variant of a model-level coercion option. |
| 28 | Project-MONAI__MONAI-3493 | hard | medium | available | future_slice | Changes the mathematical semantics of a loss after an open design question. |
| 29 | pandas-dev__pandas-47780 | medium | medium | available | bugfix_survivor | PeriodIndex handles pd.NA inconsistently with None and numpy.nan. |
| 30 | pydantic__pydantic-8947 | medium | medium | available | bugfix_survivor | create_model mishandles a valid Annotated type accepted by BaseModel. |
| 31 | dask__dask-10054 | hard | hard | available | bugfix_survivor | Nullable all-null partitions break sort_values with a reproducible exception. |
| 32 | iterative__dvc-1690 | medium | medium | available | bugfix_survivor | Remote status incorrectly depends on a populated local cache. |
| 33 | facebookresearch__hydra-1725 | medium | medium | available | quality_reject | Statement leaves the required behavior undecided: allow the list or emit an error. |
| 34 | conan-io__conan-10960 | medium | medium | available | bugfix_survivor | Layout use causes package version metadata to become null. |
| 35 | facebookresearch__hydra-1422 | medium | hard | available | future_slice | Explicit request to relax override syntax for dictionary fields. |
| 36 | dask__dask-6992 | medium | medium | available | future_slice | Adds pandas dropna feature parity to Dask groupby. |
| 37 | modin-project__modin-5946 | hard | medium | available | bugfix_survivor | JSON-lines records with differing columns trigger an incorrect length failure. |
| 38 | bokeh__bokeh-13608 | hard | medium | available | bugfix_survivor | Multiple inline-math labels reproducibly disappear. |
| 39 | bokeh__bokeh-13800 | medium | hard | available | future_slice | Adds credential forwarding and configurable CORS behavior. |
| 40 | getmoto__moto-5644 | medium | medium | available | bugfix_survivor | A version regression empties Batch job container details. |
| 41 | Project-MONAI__MONAI-2942 | medium | medium | available | future_slice | Adds a device parameter to avoid CPU/GPU transfer. |
| 42 | pandas-dev__pandas-55764 | hard | medium | missing | hard_exclude_missing_image | Patch-release regression changes DataFrame subclass copy behavior. |
| 43 | iterative__dvc-1700 | hard | hard | available | bugfix_survivor | Recursive add incorrectly captures Git's internal files. |
| 44 | python__mypy-11448 | medium | medium | available | future_slice | Improves diagnostic wording without correcting type-checking behavior. |
| 45 | iterative__dvc-3727 | medium | medium | available | bugfix_survivor | JSON and text no-change output is inconsistent across related diff commands. |
| 46 | pydantic__pydantic-6293 | medium | medium | available | bugfix_survivor | Pydantic dataclasses reject valid typing.Annotated fields. |
| 47 | Project-MONAI__MONAI-1765 | hard | hard | available | future_slice | Unifies loss-component input conventions through an API behavior change. |
| 48 | modin-project__modin-6638 | medium | medium | available | quality_reject | No self-contained workbook or minimal reproduction is supplied. |
| 49 | dask__dask-10042 | easy | medium | available | bugfix_survivor | Parquet filtering compares missing statistics against strings and raises TypeError. |
| 50 | facebookresearch__hydra-614 | medium | medium | available | future_slice | Introduces new interpolation resolvers and has no PASS_TO_PASS coverage. |
| 51 | pandas-dev__pandas-54643 | hard | hard | available | future_slice | Explicit enhancement adding bad-line handling to the PyArrow engine. |
| 52 | conan-io__conan-12695 | easy | medium | available | quality_reject | Issue asks maintainers to choose between accepting whitespace and improving an error. |
| 53 | bokeh__bokeh-13443 | medium | medium | available | bugfix_survivor | Cloning a normal model fails when it contains read-only properties. |
| 54 | getmoto__moto-6743 | medium | medium | available | bugfix_survivor | Moto rejects S3 access-point ARNs that AWS IAM accepts. |
| 55 | getmoto__moto-6709 | hard | hard | available | bugfix_survivor | DynamoDB projection expressions mishandle mapped names containing dots. |
| 56 | Project-MONAI__MONAI-2104 | medium | medium | available | quality_reject | Statement gives the exact deepcopy implementation and analogous source location. |
| 57 | iterative__dvc-5080 | medium | medium | available | future_slice | Adds a new config --list command-line capability. |
| 58 | pandas-dev__pandas-56321 | medium | medium | available | bugfix_survivor | Masked string-column creation stringifies and truncates missing values. |
| 59 | modin-project__modin-6758 | hard | hard | available | future_slice | Preserves internal dtype-cache metadata after setitem. |
| 60 | python__mypy-11213 | medium | medium | available | bugfix_survivor | Lambda analysis leaks errors from an unchecked function body. |
| 61 | pydantic__pydantic-5834 | medium | medium | available | bugfix_survivor | NumPy annotations incorrectly produce ValidatorIterator values. |
| 62 | facebookresearch__hydra-1540 | medium | medium | missing | hard_exclude_missing_image | Bundles search-path inspection and warning enhancements. |
| 63 | python__mypy-15846 | hard | hard | available | future_slice | Requests negative TypeVar constraints and labels itself a possible feature request. |
| 64 | conan-io__conan-13450 | hard | medium | available | future_slice | Adds C++20 support with unresolved compiler/Meson-version semantics. |
| 65 | dask__dask-8462 | easy | medium | available | bugfix_survivor | Token naming ignores chunks and dtype, causing distinct graphs to collide. |
| 66 | modin-project__modin-6951 | medium | medium | available | future_slice | External benchmark report mixes speed and correctness without a minimal reproduction. |
| 67 | pydantic__pydantic-5868 | hard | hard | available | bugfix_survivor | allow_inf_nan is ignored for Decimal despite explicit configuration. |
| 68 | bokeh__bokeh-13757 | easy | medium | available | bugfix_survivor | Release-candidate regression makes VBox construction fail. |
| 69 | Project-MONAI__MONAI-506 | hard | medium | available | future_slice | Adds torch.Tensor support to a NumPy-only transform. |
| 70 | getmoto__moto-7317 | medium | medium | available | bugfix_survivor | Glue create_database silently discards valid AWS resource tags. |
| 71 | conan-io__conan-11330 | hard | hard | available | future_slice | Redesigns copy_symlink_folders defaults and API behavior. |
| 72 | pandas-dev__pandas-57297 | medium | medium | available | bugfix_survivor | Descending column sort incorrectly drops the row index. |
| 73 | python__mypy-15355 | hard | medium | available | future_slice | Requests an opt-in stubgen mode to retain default values. |
| 74 | iterative__dvc-4309 | medium | medium | available | bugfix_survivor | Local remote configuration cannot modify an existing repository remote. |
| 75 | dask__dask-6779 | medium | hard | available | future_slice | Scheduler-ordering change targets memory use rather than a wrong result. |
| 76 | facebookresearch__hydra-1655 | hard | hard | available | future_slice | Requests resolved interpolation in help output. |
| 77 | bokeh__bokeh-12902 | hard | hard | available | bugfix_survivor | Non-unique notebook DOM ids update the wrong tab and make loading appear stuck. |
| 78 | Project-MONAI__MONAI-2780 | hard | hard | available | future_slice | Adds padding kwargs across several transforms. |
| 79 | pandas-dev__pandas-53809 | hard | hard | missing | hard_exclude_missing_image | Deprecation/removal campaign rather than a behavioral defect. |
| 80 | pydantic__pydantic-5874 | hard | hard | available | future_slice | Broad multi-objective V2 dataclass improvement bundle. |

## Prior frozen-24 dispositions

| Instance | Prior tier | Disposition | Reason |
|---|:---:|---|---|
| getmoto__moto-6308 | easy | future_easy_bug_slice | Clear but low-diagnostic one-file lifecycle-ordering fix; below the requested medium/hard focus. |
| dask__dask-7305 | easy | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| pandas-dev__pandas-56849 | easy | future_easy_bug_slice | Narrow lowercase frequency-alias regression with a small localized fix. |
| iterative__dvc-1681 | easy | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| python__mypy-15413 | easy | quality_reject | Reporter explicitly leaves intended warn-return-any behavior undecided. |
| getmoto__moto-5835 | easy | future_easy_bug_slice | Simple enum validation omission; below the requested diagnostic difficulty. |
| conan-io__conan-14177 | easy | future_feature_slice | Feature request supplies the desired signature and behavior directly. |
| pandas-dev__pandas-53958 | easy | future_api_slice | API-placement design question with a one-file export-only implementation. |
| pandas-dev__pandas-51605 | easy | future_easy_bug_slice | Valid empty-input regression, but localized and trivial relative to the target slice. |
| iterative__dvc-9395 | easy | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| getmoto__moto-7361 | medium | future_feature_slice | Adds emulation coverage for a previously unsupported API Gateway patch operation. |
| getmoto__moto-5885 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| getmoto__moto-6317 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| getmoto__moto-7607 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| python__mypy-16154 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| python__mypy-14835 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| python__mypy-16503 | medium | reconsider_after_preflight | Good medium bug, held only because it was outside the 10 retained seeds and mypy was already dense. |
| python__mypy-11585 | medium | reconsider_after_preflight | Good type-system bug, held to avoid further mypy concentration; not rejected on quality. |
| python__mypy-14988 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| conan-io__conan-14378 | medium | future_environment_slice | Windows resource-compiler report carries environment/toolchain smoke risk. |
| conan-io__conan-10874 | medium | future_feature_slice | Small namespace migration feature, not a diagnostic bug-fix task. |
| conan-io__conan-10917 | medium | retained_seed | One of the 10 explicitly retained human-reviewed seeds. |
| iterative__dvc-4961 | hard | hard_reject_verifier_input | FAIL_TO_PASS node ids are truncated parametrized pytest ids and are unsafe for verification. |
| iterative__dvc-2017 | hard | hard_reject_task_test_mismatch | Requirements/setup maintenance statement and gold patch do not match the listed pipeline FAIL_TO_PASS tests. |
