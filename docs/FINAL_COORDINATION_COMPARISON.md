# Final coordination comparison

## 1. Scope and frozen methods

This is the final independent comparison used to choose the deterministic paper method. No search rule was changed after inspecting these results. The scientific model remains the 20 m × 12 m, two-rail/four-robot system with WHOLE/Y_SPLIT/X_SPLIT, `delta_x = delta_y = 0.50 m`, minimum child length 0.20 m, and 0.50 m interference distances.

The five frozen methods are:

| Code | Method | Only distinguishing setting |
|---|---|---|
| A | `REGION_FAST_HEURISTIC` | REGION initialization and incremental LNS; no initialization enhancement |
| B | `REGION_FAST_ROUTE_REORDER` | A plus `initialization_coordination_route_reorder=True` |
| C | `REGION_FAST_SAME_RAIL` | A plus `initialization_same_rail_boundary_rebalance=True` |
| HGA | `ADAPTED_HGA_V2` | Existing frozen HGA |
| WAG | `ADAPTED_WAG_VNS_V2` | Existing frozen WAG+VNS |

B and C are mutually exclusive and replace the same initialization slot. B preserves every robot task set and changes only the four route orders. C preserves rail assignment and split legality but may move legally eligible work between the two robots on the same rail. A/B/C all retain construction budget 5, `Kinit_ref=5`, M192 = 144 atomic + 48 LNS, Kdp8, Kref2, Kref_total4, the same SA, direction DP, B32(32), independent certifier, and HEURISTIC C2/C4. CRRR, JRRB, and TWO_OPT_STAR are off. GAT V05 remains in the repository and prior ablations but is not part of this frozen comparison.

## 2. New independent geometry set

The verified original PPO workbook generator (`generate_welds.py` and `ppo/src/laces_weld_slim/dataset.py`) produced the workbooks. Seeds 20262501–20262515 were assigned by the already-used count-only rule to five SMALL, five MEDIUM, and five LARGE instances. No weld was trimmed. The total distance from the three requested counts (35/55/85) is 11 welds.

HIGH_WORKLOAD is a separate static stress tier, not an i.i.d. continuation of the three size tiers. Before any solver was run, seeds 20262516–20262535 were generated with the fixed 45-group setting; the five legal geometries having the largest total weld length were selected. No Cmax or solver output entered this selection.

| Instance | Geometry seed | Actual N | Total weld length (m) | Sheet | Source workbook |
|---|---:|---:|---:|---|---|
| SMALL_01 | 20262501 | 35 | 54.298 | g19_w035 | seed_0020262501.xlsx |
| SMALL_02 | 20262504 | 35 | 51.221 | g23_w035 | seed_0020262504.xlsx |
| SMALL_03 | 20262507 | 35 | 63.739 | g27_w035 | seed_0020262507.xlsx |
| SMALL_04 | 20262508 | 35 | 65.211 | g20_w035 | seed_0020262508.xlsx |
| SMALL_05 | 20262513 | 35 | 70.758 | g26_w035 | seed_0020262513.xlsx |
| MEDIUM_01 | 20262502 | 55 | 89.678 | g32_w055 | seed_0020262502.xlsx |
| MEDIUM_02 | 20262503 | 55 | 98.433 | g30_w055 | seed_0020262503.xlsx |
| MEDIUM_03 | 20262510 | 55 | 72.721 | g31_w055 | seed_0020262510.xlsx |
| MEDIUM_04 | 20262511 | 55 | 75.621 | g38_w055 | seed_0020262511.xlsx |
| MEDIUM_05 | 20262512 | 55 | 66.446 | g45_w055 | seed_0020262512.xlsx |
| LARGE_01 | 20262505 | 83 | 121.634 | g45_w083 | seed_0020262505.xlsx |
| LARGE_02 | 20262506 | 85 | 104.749 | g38_w085 | seed_0020262506.xlsx |
| LARGE_03 | 20262509 | 85 | 127.797 | g43_w085 | seed_0020262509.xlsx |
| LARGE_04 | 20262514 | 76 | 115.975 | g45_w076 | seed_0020262514.xlsx |
| LARGE_05 | 20262515 | 85 | 90.340 | g37_w085 | seed_0020262515.xlsx |
| HIGH_WORKLOAD_01 | 20262531 | 90 | 161.927 | g45_w090 | seed_0020262531.xlsx |
| HIGH_WORKLOAD_02 | 20262520 | 87 | 140.633 | g45_w087 | seed_0020262520.xlsx |
| HIGH_WORKLOAD_03 | 20262518 | 83 | 136.322 | g45_w083 | seed_0020262518.xlsx |
| HIGH_WORKLOAD_04 | 20262525 | 83 | 132.602 | g45_w083 | seed_0020262525.xlsx |
| HIGH_WORKLOAD_05 | 20262533 | 87 | 132.087 | g45_w087 | seed_0020262533.xlsx |

All 20 workbooks are in `data/independent/final_coordination/`. They were round-trip loaded and all 40 sheets (META and data) were rendered and visually inspected. The 20 × 3 × 5 matrix has 300 unique keys. The earlier 375 independent runs remain intact.

## 3. Execution and deadline semantics

Solver seeds are 20262581, 20262582, and 20262583. Method order is cyclically rotated by workbook and seed. Four spawned workers were pinned to distinct physical cores, with one computational thread per worker. Input loading precedes timing; all algorithm initialization, model work, reference scheduling, and certification are charged to the 60 s run budget.

The primary metric is the best independently certified Cmax whose certification completed by 60 s. A late result is never backfilled. A/B/C use `HardDeadlineReferenceEvaluator`. The frozen HGA/WAG implementations retain their soft-return behavior, while their stored Cmax@60 still excludes late certifications. Therefore the checkpoint metric is uniform, but this is not a claim of identical hard real-time termination. Actual maximum return time was 65.163 s; no multi-minute B32 overrun occurred.

| Method | Completed | Certified by 60 s | Success | Runtime median / max (s) | Max overshoot (s) |
|---|---:|---:|---:|---:|---:|
| A | 60 | 60 | 100.00% | 60.859 / 63.419 | 3.419 |
| B | 60 | 60 | 100.00% | 61.177 / 65.163 | 5.163 |
| C | 60 | 60 | 100.00% | 61.098 / 64.379 | 4.379 |
| HGA | 60 | 58 | 96.67% | 60.291 / 64.756 | 4.756 |
| WAG | 60 | 57 | 95.00% | 60.343 / 63.042 | 3.042 |

The five HGA/WAG failures are all `TIME_LIMIT_NO_FEASIBLE` on HIGH_WORKLOAD_01: HGA seeds 20262582/20262583 and WAG all three seeds. They are not assigned artificial Cmax values. Comparisons against those baselines therefore use 19 complete workbook pairs and separately report success.

## 4. Raw Cmax@60 results

`NA` means no independently certified solution by the checkpoint. Full per-run initialization, loads, completion times, empty travel, WAIT, split counts, reference states, runtime, and termination data are stored under `sci_coordination_final_comparison` in `data/development/graph_ranker_dataset_v05.sqlite`.

| Instance | Seed | A | B | C | HGA | WAG |
|---|---:|---:|---:|---:|---:|---:|
| SMALL_01 | 20262581 | 1885.476 | 1885.476 | 1885.476 | 2062.525 | 1984.494 |
| SMALL_01 | 20262582 | 1864.536 | 1864.536 | 1864.536 | 2121.582 | 2030.317 |
| SMALL_01 | 20262583 | 1907.572 | 1907.572 | 1907.572 | 1996.419 | 1947.240 |
| SMALL_02 | 20262581 | 1819.137 | 1819.137 | 1819.137 | 1935.577 | 1899.833 |
| SMALL_02 | 20262582 | 1851.673 | 1851.673 | 1851.673 | 1936.568 | 1965.080 |
| SMALL_02 | 20262583 | 1791.058 | 1791.058 | 1791.058 | 1907.019 | 1936.568 |
| SMALL_03 | 20262581 | 2530.128 | 2530.128 | 2530.128 | 2705.180 | 2699.809 |
| SMALL_03 | 20262582 | 2580.261 | 2580.261 | 2580.261 | 2637.343 | 2626.876 |
| SMALL_03 | 20262583 | 2536.140 | 2536.140 | 2536.140 | 2881.253 | 2643.684 |
| SMALL_04 | 20262581 | 2158.178 | 2158.178 | 2158.178 | 2256.278 | 2123.138 |
| SMALL_04 | 20262582 | 2165.881 | 2165.881 | 2165.881 | 2193.852 | 2123.138 |
| SMALL_04 | 20262583 | 2370.028 | 2370.028 | 2370.028 | 2329.226 | 2173.693 |
| SMALL_05 | 20262581 | 2192.046 | 2192.046 | 2192.046 | 2247.604 | 2206.767 |
| SMALL_05 | 20262582 | 2192.046 | 2192.046 | 2192.046 | 2238.356 | 2206.767 |
| SMALL_05 | 20262583 | 2192.046 | 2192.046 | 2192.046 | 2247.604 | 2247.604 |
| MEDIUM_01 | 20262581 | 3539.360 | 3539.360 | 3539.360 | 3576.207 | 3539.360 |
| MEDIUM_01 | 20262582 | 3539.360 | 3539.360 | 3539.360 | 3576.207 | 3539.360 |
| MEDIUM_01 | 20262583 | 3539.360 | 3539.360 | 3539.360 | 3576.207 | 3576.207 |
| MEDIUM_02 | 20262581 | 3433.458 | 3433.458 | 3433.458 | 3432.764 | 3432.764 |
| MEDIUM_02 | 20262582 | 3433.458 | 3433.458 | 3433.458 | 3433.239 | 3432.764 |
| MEDIUM_02 | 20262583 | 3433.458 | 3433.458 | 3433.458 | 3433.636 | 3432.764 |
| MEDIUM_03 | 20262581 | 2509.773 | 2518.352 | 2509.773 | 2573.892 | 2509.837 |
| MEDIUM_03 | 20262582 | 2509.521 | 2500.295 | 2509.521 | 2564.697 | 2562.170 |
| MEDIUM_03 | 20262583 | 2526.695 | 2500.295 | 2526.695 | 2602.448 | 2514.531 |
| MEDIUM_04 | 20262581 | 2924.516 | 2924.516 | 2924.516 | 2952.192 | 2871.659 |
| MEDIUM_04 | 20262582 | 2971.878 | 2971.878 | 2971.878 | 2953.233 | 2909.595 |
| MEDIUM_04 | 20262583 | 2815.000 | 2815.000 | 2815.000 | 2990.571 | 2913.091 |
| MEDIUM_05 | 20262581 | 2848.465 | 2848.465 | 2848.465 | 2907.360 | 2885.499 |
| MEDIUM_05 | 20262582 | 2928.705 | 2928.705 | 2928.705 | 2948.190 | 2895.580 |
| MEDIUM_05 | 20262583 | 2813.035 | 2813.035 | 2813.035 | 2939.444 | 2917.279 |
| LARGE_01 | 20262581 | 4694.298 | 4694.298 | 4694.298 | 4845.106 | 4790.505 |
| LARGE_01 | 20262582 | 4721.907 | 4721.907 | 4721.907 | 4955.969 | 4790.505 |
| LARGE_01 | 20262583 | 4802.097 | 4802.097 | 4802.097 | 4860.094 | 4788.088 |
| LARGE_02 | 20262581 | 3662.096 | 3662.096 | 3662.096 | 3817.608 | 3692.126 |
| LARGE_02 | 20262582 | 3757.574 | 3757.574 | 3757.574 | 3868.095 | 3692.126 |
| LARGE_02 | 20262583 | 3790.117 | 3790.117 | 3790.117 | 3868.095 | 3692.126 |
| LARGE_03 | 20262581 | 4401.659 | 4401.659 | 4401.659 | 4628.180 | 4499.573 |
| LARGE_03 | 20262582 | 4433.839 | 4433.839 | 4433.839 | 4628.180 | 4499.573 |
| LARGE_03 | 20262583 | 4368.777 | 4368.777 | 4368.777 | 4628.180 | 4499.573 |
| LARGE_04 | 20262581 | 3840.660 | 3840.660 | 3840.660 | 4047.914 | 3965.461 |
| LARGE_04 | 20262582 | 3967.888 | 3967.888 | 3967.888 | 4086.613 | 3965.461 |
| LARGE_04 | 20262583 | 3958.718 | 3958.718 | 3958.718 | 4086.613 | 3965.461 |
| LARGE_05 | 20262581 | 3547.895 | 3547.895 | 3547.895 | 3707.851 | 3618.829 |
| LARGE_05 | 20262582 | 3570.904 | 3570.904 | 3570.904 | 3707.851 | 3618.829 |
| LARGE_05 | 20262583 | 3550.404 | 3550.404 | 3550.404 | 3707.851 | 3618.829 |
| HIGH_WORKLOAD_01 | 20262581 | 7489.798 | 7489.798 | 8469.475 | 9658.111 | NA |
| HIGH_WORKLOAD_01 | 20262582 | 9174.437 | 9174.437 | 7760.588 | NA | NA |
| HIGH_WORKLOAD_01 | 20262583 | 7892.271 | 7892.271 | 8136.441 | NA | NA |
| HIGH_WORKLOAD_02 | 20262581 | 7794.528 | 7794.528 | 7794.528 | 6338.871 | 4923.528 |
| HIGH_WORKLOAD_02 | 20262582 | 7179.418 | 7179.418 | 7179.418 | 6578.583 | 4983.274 |
| HIGH_WORKLOAD_02 | 20262583 | 7684.515 | 7684.515 | 7684.515 | 6578.583 | 5633.374 |
| HIGH_WORKLOAD_03 | 20262581 | 7483.333 | 7483.333 | 5332.744 | 4878.468 | 4852.190 |
| HIGH_WORKLOAD_03 | 20262582 | 6855.104 | 6855.104 | 5066.724 | 4878.468 | 4878.468 |
| HIGH_WORKLOAD_03 | 20262583 | 6334.595 | 6334.595 | 5158.461 | 4878.468 | 4869.423 |
| HIGH_WORKLOAD_04 | 20262581 | 9875.557 | 5101.993 | 9875.557 | 5323.373 | 4942.522 |
| HIGH_WORKLOAD_04 | 20262582 | 9634.593 | 5049.374 | 9634.593 | 5346.740 | 5371.023 |
| HIGH_WORKLOAD_04 | 20262583 | 6514.829 | 5102.573 | 6514.829 | 5346.740 | 5371.023 |
| HIGH_WORKLOAD_05 | 20262581 | 7500.770 | 7500.770 | 4421.602 | 4407.075 | 4407.075 |
| HIGH_WORKLOAD_05 | 20262582 | 7640.490 | 7640.490 | 4588.009 | 4407.075 | 4407.075 |
| HIGH_WORKLOAD_05 | 20262583 | 7934.512 | 7934.512 | 4343.277 | 4407.075 | 4355.888 |

## 5. Primary independent comparisons

Each workbook is one statistical unit; its three-seed median is compared. Negative relative change is better.

| Comparison | Tier | W/T/L | Median change | Mean change | Worst regression |
|---|---|---:|---:|---:|---:|
| B vs A | SMALL | 0/5/0 | 0.000% | 0.000% | 0.000% |
| B vs A | MEDIUM | 1/4/0 | 0.000% | -0.076% | 0.000% |
| B vs A | LARGE | 0/5/0 | 0.000% | 0.000% | 0.000% |
| B vs A | HIGH_WORKLOAD | 1/4/0 | 0.000% | -9.409% | 0.000% |
| **B vs A** | **ALL** | **2/18/0** | **0.000%** | **-2.371%** | **0.000%** |
| C vs A | SMALL | 0/5/0 | 0.000% | 0.000% | 0.000% |
| C vs A | MEDIUM | 0/5/0 | 0.000% | 0.000% | 0.000% |
| C vs A | LARGE | 0/5/0 | 0.000% | 0.000% | 0.000% |
| C vs A | HIGH_WORKLOAD | 2/2/1 | 0.000% | -12.757% | +3.094% |
| **C vs A** | **ALL** | **2/17/1** | **0.000%** | **-3.189%** | **+3.094%** |

B's two workbook wins are MEDIUM_03 (2509.773→2500.295 s, -0.378%) and HIGH_WORKLOAD_04 (9634.593→5101.993 s, -47.045%). The latter improvement occurs in all three solver seeds. C wins HIGH_WORKLOAD_03 (6855.104→5158.461 s, -24.750%) and HIGH_WORKLOAD_05 (7640.490→4421.602 s, -42.129%), but loses HIGH_WORKLOAD_01 (7892.271→8136.441 s, +3.094%). Thus both enhancements are sparse rather than broad; B is the only one with no workbook-level regression.

## 6. Traditional-baseline comparisons

Only workbooks with certified results for both methods are included (19 for every HGA/WAG comparison). The omitted HIGH_WORKLOAD_01 baseline failure is visible in the success table and is not converted to a penalty.

| Method vs baseline | W/T/L | Median change | Mean change | Worst regression |
|---|---:|---:|---:|---:|
| A vs WAG | 10/1/8 | -0.170% | +12.041% | +79.381% |
| B vs WAG | 11/1/7 | -0.566% | +7.580% | +73.369% |
| C vs WAG | 10/1/8 | -0.170% | +6.363% | +79.381% |
| A vs HGA | 14/0/5 | -2.843% | +8.316% | +80.196% |
| B vs HGA | 15/0/4 | -2.859% | +3.835% | +73.369% |
| C vs HGA | 14/0/5 | -2.843% | +2.642% | +80.196% |

Against WAG, B changes the complete-pair result from A's 10/1/8 to 11/1/7 and reduces the mean gap from +12.041% to +7.580%. This is a real reduction, not closure. On the four complete HIGH_WORKLOAD pairs, B remains 1/0/3 with median +47.492%, mean +40.836%, and worst +73.369%; A is 0/0/4 with median +63.787% and mean +61.934%. C remains 0/0/4 against WAG, with median +30.071%, mean +34.963%, and worst +79.381%. SMALL/MEDIUM/LARGE generally match or beat HGA/WAG, while selected high-workload geometries remain the decisive limitation.

## 7. Initialization feasibility and schedule components

| Metric | A | B | C |
|---|---:|---:|---:|
| RAIL_SERIAL winning/fallback runs | 12/60 (20%) | 9/60 (15%) | 3/60 (5%) |
| Initial DEADLOCK / evaluated attempts | 138/225 | 90/207 | 81/198 |
| Median ALNS iterations | 15.5 | 16.0 | 16.0 |
| Median reference calls | 58.5 | 58.5 | 58.5 |
| Strict search Cmax decreases (sum) | 192 | 185 | 184 |
| Mean max process load (s) | 3971.893 | 3800.484 | 3747.339 |
| Mean empty travel (s) | 496.590 | 478.560 | 454.484 |
| Mean WAIT (s) | 201.338 | 206.647 | 247.765 |

A's selected initial strategies are REGION 42, RAIL_SERIAL 12, and RAIL_MONOTONE 6. B selects REGION 30, ROUTE_REORDER 15, RAIL_SERIAL 9, and RAIL_MONOTONE 6. Its new candidate is attempted 60 times: 45 B32 FEASIBLE, 15 DEADLOCK, all 45 feasible results independently certified, and 15 selected. C selects SAME_RAIL 27, REGION 24, RAIL_SERIAL 3, and RAIL_MONOTONE 6. SAME_RAIL is attempted 60 times: 57 FEASIBLE, 3 DEADLOCK, all 57 feasible results independently certified, and 27 selected.

The route repair therefore directly improves initialization coordination access and reduces serial fallback without changing assignment. SAME_RAIL reduces fallback further and lowers mean maximum process load, but its mean WAIT rises. This distinction matters: load balance alone is not the final objective.

The high-workload means make the residual coupling visible:

| Method | Mean max load (s) | Mean empty travel (s) | Mean WAIT (s) |
|---|---:|---:|---:|
| A | 6932.392 | 825.020 | 574.275 |
| B | 6248.326 | 758.069 | 595.514 |
| C | 6034.178 | 656.593 | 759.983 |
| HGA (13 certified runs) | 5107.568 | 673.571 | 577.587 |
| WAG (12 certified runs) | 4737.603 | 527.643 | 352.760 |

B and C reduce load/travel on the stress tier, but coordination WAIT does not fall with them; C's WAIT rises most. The remaining WAG gap is therefore not explained by task load alone.

## 8. Decision and scientific interpretation

The independent evidence supports retaining `REGION_FAST_ROUTE_REORDER` as the deterministic paper enhancement on top of the already validated REGION+FAST base. It has two independent workbook improvements, no workbook-level loss, reduces serial fallback from 20% to 15%, and converts existing balanced assignments into certified schedules without changing assignment or reference budget. Its strongest gain is a high-workload case where all three seeds improve.

The claim must remain narrow. The median gain is 0%, 18/20 workbook medians tie, and the method still loses badly to WAG on three of four complete high-workload stress workbooks. Therefore the result supports a coordination-feasibility route-construction contribution, not universal superiority or broad average dominance.

`REGION_FAST_SAME_RAIL` remains an exploratory ablation rather than the final main method. It has larger mean improvement and the lowest fallback rate, but all benefits occur in the stress tier and it has one independent regression; its higher WAIT also shows that its assignment/load effect does not reliably become a better coordinated schedule. B and C are not combined after testing.

The paper should present: (1) REGION cooperative initialization and equivalent incremental LNS as the validated base; (2) coordination-aware route reordering as the final deterministic enhancement; (3) SAME_RAIL and frozen GAT V05 as ablations; and (4) CRRR/JRRB as negative exploratory evidence. The principal limitation is high-workload coordination: even after route repair, WAG retains a large conditional-performance advantage there. No further operator, ranker, direction-DP, or B32 change is justified from this test set, which is now consumed independent evidence.

## 9. Reproducibility checks

- `pytest -q -p no:cacheprovider`: **351 passed** in 39.71 s (final rerun; the preceding full run also passed).
- Final matrix validation: 20 distinct workbooks, three seeds, five methods, **300/300** unique completed combinations.
- Status totals: 295 `COMPLETED`, 5 `TIME_LIMIT_NO_FEASIBLE`; certification totals are reported above.
- All A/B/C profiles are mutually exclusive and keep the frozen construction/search/reference budgets.
- Original 375-run experiment count remains 375; the new experiment uses `sci_coordination_final_comparison`.
- Resume semantics skip completed keys and preserve failure/error rows.
- The SQLite summary is stored as `sci_coordination_final_summary`.

**Final stop decision:** freeze `REGION_FAST_ROUTE_REORDER` as the final deterministic GRC-ALNS paper method, keep `REGION_FAST_HEURISTIC` as its independently validated base, retain SAME_RAIL/GAT only as ablations, and stop algorithm expansion on these data.
