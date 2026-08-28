# Expansion gold-smoke queue

> Base-fail/gold-pass validation only — no teacher rollout spend is approved.

**36 tasks:** 6 runner-easy / 26 runner-medium / 4 runner-hard. All are medium or hard under the diagnostic heuristic.

| # | Instance | Repo | Runner tier | Heuristic | Family | Smoke status |
|---:|---|---|:---:|:---:|---|:---:|
| 1 | getmoto__moto-6317 | getmoto/moto | medium | hard | cloud_api_state | pending |
| 2 | dask__dask-7305 | dask/dask | easy | medium | numeric_data | pending |
| 3 | getmoto__moto-5885 | getmoto/moto | medium | medium | cloud_api_state | pending |
| 4 | pandas-dev__pandas-50672 | pandas-dev/pandas | hard | medium | numeric_data | pending |
| 5 | python__mypy-16154 | python/mypy | medium | medium | type_system | pending |
| 6 | iterative__dvc-1681 | iterative/dvc | easy | medium | parsing_serialization | pending |
| 7 | getmoto__moto-7607 | getmoto/moto | medium | hard | async_concurrency | pending |
| 8 | modin-project__modin-5946 | modin-project/modin | hard | medium | async_concurrency | pending |
| 9 | conan-io__conan-10917 | conan-io/conan | medium | medium | build_dependency | pending |
| 10 | iterative__dvc-9395 | iterative/dvc | easy | medium | cache_incremental | pending |
| 11 | python__mypy-14835 | python/mypy | medium | hard | cache_incremental | pending |
| 12 | bokeh__bokeh-13608 | bokeh/bokeh | hard | medium | build_dependency | pending |
| 13 | pydantic__pydantic-9004 | pydantic/pydantic | medium | medium | async_concurrency | pending |
| 14 | dask__dask-10042 | dask/dask | easy | medium | cache_incremental | pending |
| 15 | facebookresearch__hydra-1458 | facebookresearch/hydra | medium | medium | parsing_serialization | pending |
| 16 | iterative__dvc-1700 | iterative/dvc | hard | hard | filesystem_versioning | pending |
| 17 | python__mypy-14988 | python/mypy | medium | hard | type_system | pending |
| 18 | dask__dask-8462 | dask/dask | easy | medium | async_concurrency | pending |
| 19 | pydantic__pydantic-6100 | pydantic/pydantic | medium | medium | numeric_data | pending |
| 20 | facebookresearch__hydra-952 | facebookresearch/hydra | medium | medium | async_concurrency | pending |
| 21 | bokeh__bokeh-13757 | bokeh/bokeh | easy | medium | async_concurrency | pending |
| 22 | pandas-dev__pandas-47780 | pandas-dev/pandas | medium | medium | numeric_data | pending |
| 23 | pydantic__pydantic-8947 | pydantic/pydantic | medium | medium | type_system | pending |
| 24 | iterative__dvc-1690 | iterative/dvc | medium | medium | cache_incremental | pending |
| 25 | conan-io__conan-10960 | conan-io/conan | medium | medium | build_dependency | pending |
| 26 | getmoto__moto-5644 | getmoto/moto | medium | medium | filesystem_versioning | pending |
| 27 | iterative__dvc-3727 | iterative/dvc | medium | medium | parsing_serialization | pending |
| 28 | pydantic__pydantic-6293 | pydantic/pydantic | medium | medium | async_concurrency | pending |
| 29 | bokeh__bokeh-13443 | bokeh/bokeh | medium | medium | async_concurrency | pending |
| 30 | getmoto__moto-6743 | getmoto/moto | medium | medium | filesystem_versioning | pending |
| 31 | pandas-dev__pandas-56321 | pandas-dev/pandas | medium | medium | numeric_data | pending |
| 32 | python__mypy-11213 | python/mypy | medium | medium | cache_incremental | pending |
| 33 | pydantic__pydantic-5834 | pydantic/pydantic | medium | medium | type_system | pending |
| 34 | getmoto__moto-7317 | getmoto/moto | medium | medium | cloud_api_state | pending |
| 35 | pandas-dev__pandas-57297 | pandas-dev/pandas | medium | medium | type_system | pending |
| 36 | iterative__dvc-4309 | iterative/dvc | medium | medium | parsing_serialization | pending |

## Runner-hard holds

These remain valid survivors; they are not silently dropped.

| Instance | Repo | Heuristic | Reason |
|---|---|:---:|---|
| pandas-dev__pandas-50151 | pandas-dev/pandas | medium | Held by the four-task runner-hard cap. |
| bokeh__bokeh-13636 | bokeh/bokeh | medium | Held by the four-task runner-hard cap. |
| dask__dask-10054 | dask/dask | hard | Held by the four-task runner-hard cap. |
| getmoto__moto-6709 | getmoto/moto | hard | Held by the four-task runner-hard cap. |
| pydantic__pydantic-5868 | pydantic/pydantic | hard | Held by the four-task runner-hard cap. |
| bokeh__bokeh-12902 | bokeh/bokeh | hard | Held by the four-task runner-hard cap. |
