# SWE-Gym medium/hard expansion candidate pool

> Candidate reservoir only — no rollout spend is approved by this file.

**80 tasks:** 56 medium / 24 hard; 11 repositories; 10 bug families.

Sources: 10 retained reviewed seeds + 70 new SWE-Gym Full-minus-Lite candidates.

22 tasks carry one or more caution flags. All 70 new tasks remain `needs_manual_review` until statement/gold/test inspection and smoke testing.

| # | Instance | Repo | Tier | Score | Family | Source | Flags | Title |
|---:|---|---|:---:|---:|---|---|---|---|
| 1 | dask__dask-7305 | dask/dask | medium | 8 | numeric_data | seed | — | `partition_quantiles` finds incorrect minimum with large unsigned integers |
| 2 | iterative__dvc-1681 | iterative/dvc | medium | 8 | parsing_serialization | seed | — | dvc status regression after 0.30 release |
| 3 | getmoto__moto-6317 | getmoto/moto | hard | 10 | cloud_api_state | seed | — | S3 Put Object results in empty object |
| 4 | iterative__dvc-9395 | iterative/dvc | medium | 8 | cache_incremental | seed | — | Make that `dvc repro --pull` pulls all missing files. |
| 5 | getmoto__moto-5885 | getmoto/moto | medium | 8 | cloud_api_state | seed | — | EKS DescribeNodegroup not returning id or name launch template attribute in response |
| 6 | python__mypy-16154 | python/mypy | medium | 8 | type_system | seed | — | Incorrect Error with NamedTuple slices ? |
| 7 | getmoto__moto-7607 | getmoto/moto | hard | 10 | async_concurrency | seed | environment_or_time_sensitive | Cryptic error when using start_execution with Moto 5.0.5 |
| 8 | conan-io__conan-10917 | conan-io/conan | medium | 8 | build_dependency | seed | environment_or_time_sensitive | [bug] `cmake_find_mode` property is broken |
| 9 | Project-MONAI__MONAI-578 | Project-MONAI/MONAI | medium | 8 | protocol_api | new | — | Remove output_postfix in post transforms |
| 10 | bokeh__bokeh-13641 | bokeh/bokeh | medium | 8 | other | new | feature_request_not_bug | [FEATURE] DataTable from_dataframe() |
| 11 | python__mypy-14835 | python/mypy | hard | 11 | cache_incremental | seed | no_pass_to_pass | cached runs of dmypy stop reporting "unused type: ignore" warnings |
| 12 | pandas-dev__pandas-50672 | pandas-dev/pandas | medium | 8 | numeric_data | new | — | BUG: `is_integer_dtype` returns `False` for integer `ArrowDtype`s |
| 13 | pydantic__pydantic-9004 | pydantic/pydantic | medium | 8 | async_concurrency | new | — | It is possible to instantiate model with `model_config` field without exceptions |
| 14 | facebookresearch__hydra-1458 | facebookresearch/hydra | medium | 8 | parsing_serialization | new | — | CLI override of hydra/conf does not require `+` prefix |
| 15 | python__mypy-14988 | python/mypy | hard | 11 | type_system | seed | — | mypyc: `Cls.__dict__` is `mappingproxy`, not `dict` |
| 16 | modin-project__modin-6618 | modin-project/modin | medium | 8 | build_dependency | new | performance_task | PERF: `__setitem__` on multiple columns should be evaluated lazily |
| 17 | Project-MONAI__MONAI-2465 | Project-MONAI/MONAI | medium | 8 | protocol_api | new | usage_question | How do I to use a Pad transform with a constant value? |
| 18 | pandas-dev__pandas-50151 | pandas-dev/pandas | medium | 8 | type_system | new | — | BUG: `DataFrame.dtypes` doesn't include backend for `string` columns |
| 19 | conan-io__conan-14532 | conan-io/conan | hard | 10 | build_dependency | new | feature_request_not_bug | [feature][2.0.9] settings_target accessible in the compatibility method |
| 20 | dask__dask-8590 | dask/dask | medium | 8 | async_concurrency | new | — | zarr region support |
| 21 | bokeh__bokeh-13636 | bokeh/bokeh | medium | 8 | parsing_serialization | new | — | Use globally unique and CSS safe IDs in `<script type="application/json">` |
| 22 | pydantic__pydantic-6100 | pydantic/pydantic | medium | 8 | numeric_data | new | — | `to_string_ser_schema` doesn't work properly anymore |
| 23 | modin-project__modin-6759 | modin-project/modin | hard | 11 | cache_incremental | new | environment_or_time_sensitive | Merge partial dtype caches on `concat(axis=0)` |
| 24 | conan-io__conan-14051 | conan-io/conan | medium | 8 | build_dependency | new | feature_request_not_bug | [feature] add prefix var to pkg-config alias |
| 25 | modin-project__modin-5949 | modin-project/modin | medium | 8 | cache_incremental | new | large_gold_patch, environment_or_time_sensitive | Precompute new dtype cache for binary operations when  a scalar is involved |
| 26 | facebookresearch__hydra-952 | facebookresearch/hydra | medium | 8 | async_concurrency | new | — | [Bug] Assigning a dictionary to untyped field fails |
| 27 | pydantic__pydantic-9137 | pydantic/pydantic | hard | 10 | type_system | new | — | coerce_numbers_to_str needs a per-field variant |
| 28 | Project-MONAI__MONAI-3493 | Project-MONAI/MONAI | medium | 7 | other | new | — | Bending energy loss is not scale invariant |
| 29 | pandas-dev__pandas-47780 | pandas-dev/pandas | medium | 8 | numeric_data | new | — | BUG: Incosistent handling of null types by PeriodIndex |
| 30 | pydantic__pydantic-8947 | pydantic/pydantic | medium | 8 | type_system | new | — | create_model fails to use type annotation of typing.Annotated |
| 31 | dask__dask-10054 | dask/dask | hard | 10 | numeric_data | new | — | `sort_values` fails to sort by nullable numeric columns when a partition is entirely null |
| 32 | iterative__dvc-1690 | iterative/dvc | medium | 8 | cache_incremental | new | — | status: dependency on local cache |
| 33 | facebookresearch__hydra-1725 | facebookresearch/hydra | medium | 8 | parsing_serialization | new | — | Hydra 1.1 when composing a config that is a list at the top level. |
| 34 | conan-io__conan-10960 | conan-io/conan | medium | 8 | build_dependency | new | environment_or_time_sensitive | [bug] version is not set correctly when using layout |
| 35 | facebookresearch__hydra-1422 | facebookresearch/hydra | hard | 9 | parsing_serialization | new | — | [Feature Request] : Support adding fields to Dict without + |
| 36 | dask__dask-6992 | dask/dask | medium | 8 | numeric_data | new | — | dask groupby could support dropna argument |
| 37 | modin-project__modin-5946 | modin-project/modin | medium | 8 | async_concurrency | new | — | Reading large json lines dataset fails because of rows having different columns |
| 38 | bokeh__bokeh-13608 | bokeh/bokeh | medium | 7 | build_dependency | new | — | [BUG] Multiple inline math elements in different axes causing axis labels to disappear |
| 39 | bokeh__bokeh-13800 | bokeh/bokeh | hard | 10 | async_concurrency | new | feature_request_not_bug | [FEATURE] Allow bokeh server embed script to forward credentials |
| 40 | getmoto__moto-5644 | getmoto/moto | medium | 7 | filesystem_versioning | new | — | Regression when using mock_batch_simple since 4.0.7 |
| 41 | Project-MONAI__MONAI-2942 | Project-MONAI/MONAI | medium | 8 | cloud_api_state | new | — | ToTensor Device |
| 42 | pandas-dev__pandas-55764 | pandas-dev/pandas | medium | 8 | cache_incremental | new | — | BUG: pandas 2.1.2 changes how `copy` works |
| 43 | iterative__dvc-1700 | iterative/dvc | hard | 9 | filesystem_versioning | new | — | Prevent dvc from tracking git. |
| 44 | python__mypy-11448 | python/mypy | medium | 8 | type_system | new | — | Better error messages for some `TypeVar` init cases |
| 45 | iterative__dvc-3727 | iterative/dvc | medium | 8 | parsing_serialization | new | — | dvc metrics diff vs dvc diff output consistency |
| 46 | pydantic__pydantic-6293 | pydantic/pydantic | medium | 8 | async_concurrency | new | — | [PYD-142] pydantic.dataclass does not work with typing.Annotated |
| 47 | Project-MONAI__MONAI-1765 | Project-MONAI/MONAI | hard | 9 | other | new | — | Unify the input of loss components |
| 48 | modin-project__modin-6638 | modin-project/modin | medium | 8 | numeric_data | new | — | Length Mismatch when using `pd.read_excel()` |
| 49 | dask__dask-10042 | dask/dask | medium | 8 | cache_incremental | new | — | test_parquet.py::test_select_filtered_column[fastparquet]: pandas backend fails to filter NoneType |
| 50 | facebookresearch__hydra-614 | facebookresearch/hydra | medium | 8 | build_dependency | new | no_pass_to_pass | Option to access hydra configs from interpolation |
| 51 | pandas-dev__pandas-54643 | pandas-dev/pandas | hard | 10 | build_dependency | new | feature_request_not_bug | ENH: Pandas 2.0 with pyarrow engine add the argument like 'skip_bad_lines=True' |
| 52 | conan-io__conan-12695 | conan-io/conan | medium | 7 | type_system | new | — | [bug] behaviour when there are spaces in required_conan_version |
| 53 | bokeh__bokeh-13443 | bokeh/bokeh | medium | 7 | async_concurrency | new | — | Unable to clone models with readonly properties |
| 54 | getmoto__moto-6743 | getmoto/moto | medium | 6 | filesystem_versioning | new | feature_request_not_bug | Allow using s3 accesspoint arns in iam policies |
| 55 | getmoto__moto-6709 | getmoto/moto | hard | 11 | parsing_serialization | new | — | DynamoDB: special characters in get_item() projection expression not handled correctly |
| 56 | Project-MONAI__MONAI-2104 | Project-MONAI/MONAI | medium | 7 | other | new | — | Need to deepcopy data in RandCropByPosNegLabeld transform |
| 57 | iterative__dvc-5080 | iterative/dvc | medium | 8 | parsing_serialization | new | — | config: add --list to list all variables and their values |
| 58 | pandas-dev__pandas-56321 | pandas-dev/pandas | medium | 8 | numeric_data | new | — | BUG: Creating a string column on a mask results in NaN being stringyfied and potentially truncated based on the 'maxchar' value of the column |
| 59 | modin-project__modin-6758 | modin-project/modin | hard | 11 | cache_incremental | new | environment_or_time_sensitive | Preserve dtypes cache on `df[existing_col] = scalar` |
| 60 | python__mypy-11213 | python/mypy | medium | 8 | cache_incremental | new | — | Untyped functions content show up in mypy output |
| 61 | pydantic__pydantic-5834 | pydantic/pydantic | medium | 8 | type_system | new | — | Numpy type annotations result in ValidatorIterator fields |
| 62 | facebookresearch__hydra-1540 | facebookresearch/hydra | medium | 8 | parsing_serialization | new | feature_request_not_bug | enhancements to hydra.searchpath |
| 63 | python__mypy-15846 | python/mypy | hard | 10 | type_system | new | — | Constraining a TypeVar to be not None seems impossible. |
| 64 | conan-io__conan-13450 | conan-io/conan | medium | 8 | build_dependency | new | environment_or_time_sensitive, feature_request_not_bug | [feature] can't specify c++20 with meson generator |
| 65 | dask__dask-8462 | dask/dask | medium | 8 | async_concurrency | new | — | Chunks/dtype not considered with map_blocks result name |
| 66 | modin-project__modin-6951 | modin-project/modin | medium | 8 | numeric_data | new | performance_validation_may_be_weak | Poor Performance on TPC-H Queries |
| 67 | pydantic__pydantic-5868 | pydantic/pydantic | hard | 10 | numeric_data | new | — | Config option `allow_inf_nan` doesn't work for `Decimal` |
| 68 | bokeh__bokeh-13757 | bokeh/bokeh | medium | 7 | async_concurrency | new | environment_or_time_sensitive | VBox is not working in 3.4.0rc1 |
| 69 | Project-MONAI__MONAI-506 | Project-MONAI/MONAI | medium | 6 | protocol_api | new | — | tensor support for squeezedim |
| 70 | getmoto__moto-7317 | getmoto/moto | medium | 8 | cloud_api_state | new | — | glue.create_database does not create tags |
| 71 | conan-io__conan-11330 | conan-io/conan | hard | 10 | build_dependency | new | feature_request_not_bug | [feature] Remove or change `copy_symlink_folders` in `conan.tools.files.copy` |
| 72 | pandas-dev__pandas-57297 | pandas-dev/pandas | medium | 8 | type_system | new | — | BUG: `sort_index` not preserving `index` when `ascending=False` |
| 73 | python__mypy-15355 | python/mypy | medium | 8 | cache_incremental | new | — | stubgen drops default values for function arguments |
| 74 | iterative__dvc-4309 | iterative/dvc | medium | 8 | parsing_serialization | new | — | remote local config does not see existing remote |
| 75 | dask__dask-6779 | dask/dask | hard | 10 | async_concurrency | new | — | Concatenating then rechunking zarr files uses lots of memory |
| 76 | facebookresearch__hydra-1655 | facebookresearch/hydra | hard | 10 | numeric_data | new | — | possible to print `$CONFIG` in hydra help with interpolation resolved? |
| 77 | bokeh__bokeh-12902 | bokeh/bokeh | hard | 9 | parsing_serialization | new | — | `load_notebook()` uses non-unique DOM element IDs |
| 78 | Project-MONAI__MONAI-2780 | Project-MONAI/MONAI | hard | 9 | protocol_api | new | feature_request_not_bug | Add pad args to all padding transforms |
| 79 | pandas-dev__pandas-53809 | pandas-dev/pandas | hard | 11 | cache_incremental | new | — | [DEPR]: Remove literal string/bytes input from `read_excel`, `read_html`, and `read_xml` |
| 80 | pydantic__pydantic-5874 | pydantic/pydantic | hard | 10 | type_system | new | — | V2 dataclass improvements |
