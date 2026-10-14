# Upgraded GRC independent comparison

## Decision

The upgraded deterministic search is now frozen as `REGION_FAST_HEURISTIC`; `REGION_FAST_GAT_V05` is retained as the frozen learning-ranking comparison. The new independent matrix is complete, but it does **not** establish stable superiority over the traditional baselines. REGION+FAST is reliably better than the original heuristic ALNS, while GAT V05 adds only a small and geometry-dependent benefit. WAG+VNS remains the more robust method on several MEDIUM and the hardest LARGE geometry.

No checkpoint, C2/C4 rule, neighborhood, SA rule, direction DP, B32 policy, certifier, task model, or evaluation budget was changed after these instances were generated.

## Frozen methods and common problem

All methods use the same 20 m x 12 m platform, four robots on two rails, WHOLE/Y_SPLIT/X_SPLIT catalog, `delta_x=delta_y=0.50 m`, `min_child_length=0.20 m`, theoretical `interference_dx=interference_dy=0.50 m`, and the same independent certification semantics. Workbook parsing is outside the 60-second solve interval for every method; all initialization and optimization are inside it.

| Method | Frozen configuration |
|---|---|
| `BASELINE_HEURISTIC` | BASELINE initialization, legacy equivalent LNS computation, heuristic C2/C4 |
| `REGION_FAST_HEURISTIC` | REGION cooperative initialization, incremental-equivalent LNS, heuristic C2/C4 |
| `REGION_FAST_GAT_V05` | Same REGION+FAST skeleton, frozen `models/gat_v05.pt` for C2/C4 |
| `ADAPTED_HGA_V2` | Existing `run_adapted_hga_v2()` |
| `ADAPTED_WAG_VNS_V2` | Existing `run_adapted_wag_vns_v2()` |

The three ALNS methods retain `M192=144 atomic+48 LNS`, `Kdp=8`, `Kref=2`, `Kref_total=4`, original SA/direction/B32/certifier, `TWO_OPT_STAR=OFF`, `CRRR=OFF`, and `CRRR_ACCESS_C2=OFF`. HGA/WAG keep their frozen native search structures but share the same problem and 60-second certified result definition.

The two enhanced method configurations are created by the existing `SearchConfig`/`HybridRanker` path. A native fixed-iteration GAT smoke on consumed-development N35-S1 verified that legacy and incremental LNS produce identical candidate order, C2 Top8, C4 Top2, B32 statuses, accepted solution and certified Cmax. The GAT checkpoint was not modified.

## Independent data provenance

The original generator was found and executed rather than replaced:

- generator: `D:\pybullet_test\MRTA_GA\src\generate_welds.py`;
- source workbook: `D:\新建文件夹\小组立qcs\C_lines-change.xlsx`;
- geometry seeds fixed before solving: `20262001` through `20262015`;
- solver seeds fixed before solving: `20261901` through `20261905`.

The source workbook is the same historical PPO source (its content digest matches the recorded source), and the generated files pass the existing workbook reader and `to_parent_welds()` validation. The 15 generation seeds do not occur in the historical role records. Exact static geometry identities are mutually unique and have zero overlap with the 2,942 geometries indexed from historical TRAIN, development, validation, and old ID_TEST records. Independence here therefore means newly generated geometries that were not used for model fitting, algorithm development, or the old 375-run test; it does not claim a new physical data-generating distribution.

The fixed static selector used only actual weld count and geometry, never solver output. It selected five SMALL N35, five MEDIUM N55, and LARGE counts 85, 83, 77, 85, 85. The N77/N83 cases are the nearest legal counts available under the predeclared group-count search; no weld was deleted and no seed was replaced. The 15 necessary workbooks are stored under `data/independent/upgraded_grc/`.

## Execution completeness and certification

The matrix is complete: **15 workbooks x 5 solver seeds x 5 methods = 375 real 60-second runs**. Raw records and the computed summary are in the existing `data/development/graph_ranker_dataset_v05.sqlite` under `sci_upgraded_independent_runs` and `sci_upgraded_independent_summary`.

| Method | Completed | Certified | Failure retained |
|---|---:|---:|---:|
| BASELINE_HEURISTIC | 75 | 75 | 0 |
| REGION_FAST_HEURISTIC | 75 | 75 | 0 |
| REGION_FAST_GAT_V05 | 75 | 75 | 0 |
| ADAPTED_HGA_V2 | 75 | 73 | 2 |
| ADAPTED_WAG_VNS_V2 | 75 | 75 | 0 |

The two HGA failures are LARGE_04 seeds `20261901` and `20261903`, both `TIME_LIMIT_NO_FEASIBLE`. Their Cmax is null; they were neither retried nor replaced. Performance comparisons against HGA below are conditional on the 14 workbooks with all five HGA seeds certified, while success rate is reported separately.

## Raw certified Cmax@60

Each cell lists solver seeds `20261901` to `20261905` in order. `FAIL` is an observed missing certified solution, not a penalty value.

| Workbook (N) | BASELINE | REGION_FAST_H | REGION_FAST_GAT | HGA | WAG+VNS |
|---|---|---|---|---|---|
| SMALL_01 (35) | 3193.2 / 4013.8 / 4013.8 / 4013.8 / 4013.8 | 3153.9 / 4013.8 / 4013.8 / 4013.8 / 4013.8 | 2930.7 / 2975.2 / 3265.4 / 3533.2 / 3015.2 | 2904.3 / 3087.0 / 3134.4 / 3149.2 / 2907.4 | 3379.6 / 3303.8 / 3321.2 / 2940.7 / 2894.8 |
| SMALL_02 (35) | 1972.4 / 2004.3 / 1995.8 / 2018.2 / 2064.6 | 1850.1 / 1879.0 / 1883.4 / 1872.8 / 1876.3 | 1883.4 / 1883.4 / 1883.4 / 1883.4 / 1883.4 | 1965.3 / 1965.1 / 1931.8 / 1972.0 / 1939.0 | 1966.7 / 1883.4 / 1999.3 / 1890.7 / 1918.7 |
| SMALL_03 (35) | 2173.7 / 2165.5 / 2524.7 / 2339.8 / 2794.0 | 1889.6 / 1889.6 / 1889.6 / 1889.6 / 1889.6 | 1973.6 / 1935.7 / 1919.0 / 1891.3 / 1909.3 | 2040.6 / 1964.9 / 1965.3 / 1964.5 / 1991.1 | 1889.6 / 1889.6 / 1933.9 / 1973.7 / 1929.9 |
| SMALL_04 (35) | 2017.4 / 2017.4 / 2017.4 / 2017.4 / 2017.4 | 2006.6 / 2006.6 / 2006.6 / 2006.6 / 2006.2 | 2006.6 / 2006.6 / 2006.6 / 2006.6 / 2006.6 | 2011.0 / 2013.2 / 2006.6 / 2013.2 / 2006.6 | 1962.1 / 1962.1 / 1962.1 / 1962.1 / 1962.1 |
| SMALL_05 (35) | 2491.7 / 2572.5 / 2592.4 / 2571.1 / 2825.4 | 1862.5 / 1878.9 / 1878.9 / 1863.5 / 1878.9 | 1878.9 / 1878.9 / 1878.9 / 1878.9 / 1876.9 | 1941.4 / 1954.0 / 1906.2 / 1954.0 / 1954.0 | 1923.6 / 1923.6 / 1898.5 / 1957.9 / 1923.6 |
| MEDIUM_01 (55) | 4996.7 / 5403.5 / 4153.7 / 5647.0 / 5602.4 | 4996.7 / 5305.9 / 3875.6 / 5619.3 / 5477.9 | 3682.2 / 4527.7 / 4508.7 / 4721.7 / 4518.6 | 4949.3 / 4095.2 / 4136.9 / 3898.6 / 4011.5 | 4070.5 / 4069.6 / 3710.7 / 4069.6 / 4892.0 |
| MEDIUM_02 (55) | 6883.0 / 7121.3 / 5288.4 / 6595.6 / 7298.0 | 6810.7 / 7121.3 / 5288.4 / 6462.1 / 7298.0 | 5670.8 / 6278.9 / 5925.5 / 5108.2 / 6206.7 | 4364.1 / 4151.1 / 4455.0 / 4151.1 / 4412.7 | 4109.1 / 4109.1 / 4109.1 / 4109.1 / 4109.1 |
| MEDIUM_03 (55) | 4624.3 / 4624.3 / 4624.3 / 4624.3 / 4624.3 | 4093.6 / 4092.1 / 4093.6 / 4080.2 / 4098.4 | 4117.4 / 4093.6 / 4108.6 / 4163.3 / 4117.4 | 4170.7 / 4205.2 / 4232.2 / 4232.2 / 4232.2 | 4183.5 / 4168.0 / 4200.6 / 4200.6 / 4134.6 |
| MEDIUM_04 (55) | 2942.0 / 2942.0 / 2942.0 / 2942.0 / 2942.0 | 2942.0 / 2942.0 / 2942.0 / 2942.0 / 2942.0 | 2921.5 / 2919.9 / 2948.2 / 2937.2 / 2935.3 | 2948.2 / 2939.8 / 2948.2 / 2948.2 / 2948.2 | 2948.2 / 2939.8 / 2948.2 / 2948.2 / 2890.8 |
| MEDIUM_05 (55) | 4724.0 / 3594.9 / 3932.7 / 4423.5 / 4214.9 | 4440.4 / 3569.9 / 3932.7 / 4171.0 / 4109.1 | 3628.9 / 3478.6 / 3292.9 / 3814.1 / 3753.1 | 3413.2 / 3762.2 / 3314.1 / 3141.0 / 2955.2 | 2873.1 / 3056.3 / 2904.7 / 3052.7 / 3052.7 |
| LARGE_01 (85) | 3647.4 / 3548.3 / 3474.6 / 3474.6 / 3616.5 | 3438.6 / 3423.3 / 3468.6 / 3500.0 / 3472.1 | 3431.4 / 3446.0 / 3475.2 / 3454.9 / 3460.1 | 3629.1 / 3639.6 / 3454.9 / 3639.6 / 3586.1 | 3392.0 / 3465.0 / 3464.5 / 3577.8 / 3465.0 |
| LARGE_02 (83) | 5153.8 / 5153.8 / 5005.6 / 5153.8 / 5153.8 | 4105.1 / 4091.2 / 4091.2 / 4094.6 / 4111.8 | 4117.5 / 4119.0 / 4147.8 / 4117.5 / 4102.8 | 4501.1 / 4501.1 / 4501.1 / 4501.1 / 4493.1 | 4141.9 / 4141.9 / 4141.9 / 4141.9 / 4141.9 |
| LARGE_03 (77) | 3392.9 / 3393.7 / 3392.9 / 3468.4 / 3468.4 | 3392.9 / 3393.7 / 3392.9 / 3400.9 / 3468.4 | 3339.0 / 3368.9 / 3331.7 / 3333.5 / 3359.7 | 3400.9 / 3400.9 / 3400.9 / 3400.9 / 3400.9 | 3284.7 / 3284.7 / 3284.2 / 3284.7 / 3284.2 |
| LARGE_04 (85) | 10026.7 / 10562.3 / 10645.0 / 9108.9 / 12954.1 | 10026.7 / 10562.3 / 10645.0 / 9108.9 / 11793.0 | 9075.8 / 10377.6 / 10355.4 / 11568.7 / 9700.0 | FAIL / 12168.7 / FAIL / 11582.2 / 9099.3 | 6528.3 / 6523.0 / 6743.3 / 6830.7 / 6663.3 |
| LARGE_05 (85) | 6163.2 / 7381.8 / 7525.9 / 7676.6 / 7138.2 | 4414.7 / 4414.7 / 4472.4 / 4405.5 / 4414.7 | 4465.1 / 4470.6 / 4472.6 / 4472.6 / 4453.5 | 4511.4 / 4511.4 / 4511.4 / 4511.4 / 4511.4 | 4450.8 / 4464.1 / 4435.7 / 4464.1 / 4464.1 |

## Workbook-level paired results

For each workbook, the five-seed median is computed first. Delta is `(candidate / reference - 1) * 100%`, so negative is better. Equal means equal at stored precision.

| Comparison | Tier | W/T/L | Median delta | Mean delta | Worst regression |
|---|---|---:|---:|---:|---:|
| REGION_FAST_H vs BASELINE | SMALL | 4/1/0 | -6.39% | -10.63% | 0.00% |
|  | MEDIUM | 4/1/0 | -1.81% | -3.37% | 0.00% |
|  | LARGE | 3/2/0 | -2.25% | -12.60% | 0.00% |
|  | ALL | **11/4/0** | **-2.25%** | **-8.86%** | **0.00%** |
| REGION_FAST_GAT vs REGION_FAST_H | SMALL | 1/2/2 | 0.00% | -4.59% | +1.55% |
|  | MEDIUM | 4/0/1 | -11.69% | -7.83% | +0.58% |
|  | LARGE | 3/0/2 | -0.39% | -0.43% | +1.27% |
|  | ALL | **8/2/5** | **-0.23%** | **-4.28%** | **+1.55%** |
| REGION_FAST_H vs HGA | SMALL | 4/0/1 | -3.84% | +3.52% | +30.02% |
|  | MEDIUM | 2/0/3 | +23.99% | +21.23% | +56.06% |
|  | LARGE | 4/0/0 (4 complete) | -3.28% | -3.95% | -0.91% |
|  | ALL | **10/0/4 (14 complete)** | **-1.18%** | **+7.71%** | **+56.06%** |
| REGION_FAST_GAT vs HGA | SMALL | 5/0/0 | -2.36% | -4.60% | -0.23% |
|  | MEDIUM | 2/0/3 | +9.50% | +10.49% | +35.78% |
|  | LARGE | 4/0/0 (4 complete) | -3.31% | -3.46% | -1.13% |
|  | ALL | **11/0/3 (14 complete)** | **-2.07%** | **+1.68%** | **+35.78%** |
| REGION_FAST_H vs WAG | SMALL | 3/0/2 | -2.09% | +3.43% | +21.49% |
|  | MEDIUM | 2/0/3 | +30.38% | +25.68% | +65.75% |
|  | LARGE | 2/0/3 | +0.10% | +11.94% | +58.51% |
|  | ALL | **7/0/8** | **+0.10%** | **+13.68%** | **+65.75%** |
| REGION_FAST_GAT vs WAG | SMALL | 4/0/1 | -1.84% | -2.24% | +2.27% |
|  | MEDIUM | 2/0/3 | +11.03% | +14.42% | +44.20% |
|  | LARGE | 2/0/3 | +0.15% | +11.27% | +55.41% |
|  | ALL | **8/0/7** | **-0.29%** | **+7.82%** | **+55.41%** |

Workbook bootstrap 95% intervals for the paired median are `[-19.24%, -0.53%]` for REGION_FAST_H versus BASELINE, `[-11.69%, +0.56%]` for GAT versus REGION_FAST_H, `[-3.85%, +14.68%]` for REGION_FAST_H versus HGA, and `[-2.09%, +21.49%]` versus WAG. Thus only the deterministic upgrade over the original heuristic has a clearly non-zero workbook-level interval. The GAT interval crosses zero.

## Runtime and search effort

| Method | Median runtime | Max overshoot | Median iterations | Median references | Total strict decreases | Median candidate generation | Median LNS repair | Median B32 | Median GAT ranking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE_HEURISTIC | 62.855 s | 11.012 s | 10 | 39 | 226 | 47.717 s | 28.805 s | 12.997 s | 0 |
| REGION_FAST_HEURISTIC | 62.030 s | 9.354 s | 15 | 59 | 207 | 41.025 s | 11.284 s | 18.865 s | 0 |
| REGION_FAST_GAT_V05 | 61.803 s | 10.189 s | 15 | 66 | 235 | 39.047 s | 11.196 s | 15.140 s | 4.347 s |
| ADAPTED_HGA_V2 | 60.176 s | 3.227 s | 130 | 69 | 85 | n/a | n/a | recorded native reference time | 0 |
| ADAPTED_WAG_VNS_V2 | 60.219 s | 2.248 s | 4 | 68 | 123 | n/a | n/a | recorded native reference time | 0 |

The existing deadline semantics permit an in-flight call to return after 60 seconds, explaining ALNS overshoot. Only independently certified best events completed before the deadline are used for Cmax@60; no late improvement is backfilled. Deadline behavior was intentionally not changed during this comparison.

Incremental LNS gives the enhanced ALNS more iterations and reference opportunities, but strict-decrease counts do not grow proportionally. GAT inference costs a median 4.347 seconds per run (317.748 seconds over 75 runs). That expense is justified on several MEDIUM workbooks but not by a stable all-geometry advantage.

## Interpretation

1. **Deterministic enhancement works.** REGION+FAST beats/ties the original heuristic on all 15 workbooks (11/4/0), with a -2.25% median and -8.86% mean workbook change. The result supports freezing this as the strongest deterministic GRC-ALNS configuration, while attributing the combined independent-test gain to REGION+FAST jointly; development ablations separately established their individual roles.
2. **GAT is not a stable primary source of performance.** It wins eight workbooks and loses five against the identical deterministic skeleton. The overall median is only -0.23%, its interval crosses zero, SMALL is not improved at the median, and LARGE gains are tiny. Its strong MEDIUM median (-11.69%) is real but heterogeneous.
3. **The upgraded method has not robustly overtaken WAG/HGA.** Median comparisons look competitive overall, but means and worst cases expose large failures. REGION_FAST_GAT remains +14.42% worse than WAG on average in MEDIUM and +11.27% in LARGE. REGION_FAST_HEURISTIC has +65.75% and +58.51% worst regressions against WAG.
4. **LARGE still has a structural bottleneck.** LARGE_04 has 85 welds and about 191.95 m total weld length, far above the other LARGE instances (about 94.27--121.94 m). WAG median Cmax is 6663.3 s versus 10562.3 s for REGION_FAST_H and 10355.4 s for GAT. HGA also fails twice here. This is not a universal N85 failure--the upgraded methods are competitive on the other large workbooks--but it is a serious high-work-content geometry failure that more iterations and candidate ranking do not solve.
5. **Paper claim.** The evidence supports a contribution in certified four-robot task modelling, region-cooperative initialization, and an efficient GRC-ALNS implementation that materially improves its original baseline. It does not support claiming that graph learning is the dominant improvement or that GRC-ALNS uniformly beats strong HGA/WAG baselines. The honest statement is that deterministic enhancement is robust over the original ALNS, GAT supplies selective instance-dependent gains, and traditional region/route joint optimization remains stronger on several medium and extreme-large geometries.

## Final freeze decision

Freeze `REGION_FAST_HEURISTIC` as the principal enhanced deterministic GRC-ALNS method and `REGION_FAST_GAT_V05` as the fixed learning ablation/comparison. Keep CRRR and CRRR_ACCESS_C2 disabled. Do not reopen GAT training, CRRR tuning, or LNS micro-optimization based on this test. If a future project phase requires closing the remaining Cmax gap, it must study a different joint region-allocation/route-order search framework and use another untouched independent dataset; the 15 workbooks in this report are now consumed.

## Verification

- DRL regression: `335 passed`.
- Project-root plus DRL discovery: `707 passed, 1 skipped, 3 subtests passed`.
- All 375 records passed configuration/matrix validation; old `sci_formal_runs` records were not changed.

