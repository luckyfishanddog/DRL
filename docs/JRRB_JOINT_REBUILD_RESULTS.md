# JRRB joint rail-region rebuilding results

## Decision

The first Joint Rail-Region Rebuilding (JRRB) implementation is complete and remains available behind a default-off switch. It generates legal structural changes and can produce a large certified improvement on one development geometry, but it fails the predeclared development quality gate: only **1 of 6** fixed consumed-development geometries contains any offline certified improvement, and only that same geometry improves under the online Top2 reference budget.

Per the experiment protocol, the 24-run 60-second paired matrix was therefore **not executed**. No JRRB performance claim, win/tie/loss result, WAG-gap claim, or independent-test claim is made. The retained primary deterministic method remains `REGION_FAST_HEURISTIC`; JRRB is not retained as a supported paper contribution and no JRRB V2, parameter search, GAT experiment, or new formal test was started.

Raw diagnostic records are stored in the existing `data/development/graph_ranker_dataset_v05.sqlite` database under `jrrb_development_diagnostic` and `jrrb_development_summary`. The earlier independent records remain unchanged under the logical `experiments.experiment` partitions `sci_upgraded_independent_runs` and `sci_upgraded_independent_summary`.

## Previously observed failure decomposition

This section is a post-hoc explanation of the already consumed 375-run independent comparison. It was used to characterize the old method, not to tune or select JRRB. Values below are five-seed medians; vector entries are component-wise medians and therefore need not describe one particular run.

`Lmax = max(process loads)` and `Gap = Cmax - Lmax`. Gap includes empty travel, waiting, and other scheduling effects; it is not labelled as WAIT.

| Workbook / method | Initial certified Cmax | Cmax@60 | Process loads R0/R1/R2/R3 | Lmax | Gap |
|---|---:|---:|---|---:|---:|
| MEDIUM_02 / REGION_FAST_HEURISTIC | 7440.5 | 6810.7 | 3979.2 / 82.0 / 6396.7 / 858.1 | 6396.7 | 414.0 |
| MEDIUM_02 / WAG | 4109.1 | 4109.1 | 1985.9 / 2019.4 / 3534.3 / 3570.6 | 3570.6 | 538.5 |
| LARGE_04 / REGION_FAST_HEURISTIC | 13212.1 | 10562.3 | 9101.0 / 92.2 / 10215.0 / 2617.2 | 10215.0 | 347.3 |
| LARGE_04 / WAG | 6830.7 | 6663.3 | 4470.1 / 4720.7 / 6288.1 / 6544.2 | 6544.2 | 119.2 |
| MEDIUM_03 / REGION_FAST_HEURISTIC | 4165.4 | 4093.6 | 4024.0 / 3965.0 / 2587.6 / 2617.4 | 4024.0 | 69.6 |
| MEDIUM_03 / WAG | 4232.2 | 4183.5 | 4074.1 / 4007.5 / 2587.6 / 2502.6 | 4074.1 | 109.4 |
| LARGE_02 / REGION_FAST_HEURISTIC | 4147.8 | 4094.6 | 3901.3 / 3844.3 / 3926.5 / 3918.9 | 3926.5 | 168.1 |
| LARGE_02 / WAG | 4141.9 | 4141.9 | 3713.5 / 3940.9 / 3926.5 / 4009.9 | 4009.9 | 132.0 |

| Workbook / method | Completion R0/R1/R2/R3 | Empty travel | WAIT | WHOLE/Y/X | Initialization | Iterations | Total/search references |
|---|---|---:|---:|---|---:|---:|---:|
| MEDIUM_02 / REGION_FAST_HEURISTIC | 4170.2 / 121.7 / 6810.7 / 921.5 | 646.1 | 45.7 | 52/2/2 | 3.947 s | 17 | 56/46 |
| MEDIUM_02 / WAG | 2046.0 / 2094.6 / 4109.1 / 3784.8 | 302.8 | 621.6 | 55/0/0 | 3.086 s | 3 | 65/61 |
| LARGE_04 / REGION_FAST_HEURISTIC | 9514.6 / 92.2 / 10562.3 / 2653.8 | 801.9 | 51.2 | 84/1/0 | 6.671 s | 10 | 40/32 |
| LARGE_04 / WAG | 4552.2 / 4825.5 / 6436.5 / 6663.3 | 374.9 | 0.0 | 85/0/0 | 4.857 s | 2 | 36/32 |
| MEDIUM_03 / REGION_FAST_HEURISTIC | 4093.6 / 4070.4 / 2671.3 / 2743.2 | 384.8 | 0.0 | 44/11/0 | 0.936 s | 14 | 59/54 |
| MEDIUM_03 / WAG | 4168.0 / 4114.9 / 2671.3 / 2593.7 | 361.1 | 0.0 | 45/10/0 | 1.368 s | 4 | 80/79 |
| LARGE_02 / REGION_FAST_HEURISTIC | 4094.6 / 3914.5 / 4010.6 / 4029.3 | 369.3 | 90.1 | 80/3/0 | 4.551 s | 10 | 41/36 |
| LARGE_02 / WAG | 4141.9 / 4006.0 / 3992.2 / 4117.5 | 315.4 | 351.3 | 80/3/0 | 3.495 s | 2 | 27/23 |

The evidence supports the following bounded interpretation:

- **MEDIUM_02: A, task-load/region problem.** REGION's Lmax is 79.2% above WAG's, whereas REGION's Gap is smaller. The dominant observed difference is allocation/region load, not WAIT.
- **LARGE_04: C, both problems.** REGION's Lmax is 56.1% above WAG's and its Gap is 228.4 seconds larger. REGION also has much more empty travel. Both allocation and route/coordination overhead matter.
- **MEDIUM_03 and LARGE_02 are successful references.** REGION's Lmax is slightly lower than WAG's and its final Cmax is also lower. The failure is therefore geometry-dependent, not a universal N55/N85 effect.

The old REGION initialization assigns dual-rail welds greedily under three fixed orders, optimizes the two rail boundaries separately, and then improves four routes independently. Later atomic/LNS moves usually change small structures. This explains why large rail-load changes are hard to reach, but it is a mechanism interpretation rather than a proof of causality.

## Implemented JRRB mechanism

JRRB is a one-shot macro step after certified REGION initialization and before normal ALNS. It is default-off and does not change the original REGION initializer, SA, C2/C4, direction DP, B32, certifier, split catalog, or scientific geometry.

The implementation performs the following bounded operation:

1. Read the certified current routes, process loads, rail loads, and WAIT-operation locations.
2. Build at most five fixed spatial windows around the full movable range, its left/right portions, the WAIT anchor, and the critical-load route.
3. For complete same-rail parent groups legally executable on both rails, create four deterministic region-assignment schemes: window flip, joint load balance, overloaded-region transfer, and X-band alternation.
4. For every assignment scheme, construct up to three upper and three lower boundary choices. The Cartesian `(b_up, b_low)` pair is scored as one state; neither boundary is committed first and there is no U-to-L-to-route decision order.
5. Rebuild all affected robot routes using the existing monotone, nearest/farthest insertion, and bounded 2-opt helpers.
6. Score candidates only with current/pre-direction quantities: projected maximum process load, four-robot imbalance, rail imbalance, route empty-travel proxy, same-rail X overlap, route-order risk, and current WAIT scale.
7. Preserve representatives from four fixed proxy views and canonicalize at most 16 complete candidates.
8. Run direction DP for at most 8 candidates and B32 for at most 2. Only an independently certified strict Cmax decrease completed before the common deadline may replace the incumbent.

The fixed composite proxy is interpretable rather than fitted: 0.45 maximum load, 0.15 four-robot imbalance, 0.12 rail imbalance, 0.18 empty travel, 0.05 same-rail overlap with current WAIT scaling, and 0.05 route-order risk. No future direction result, B32 status, certified Cmax, or offline label enters this score.

All original split patterns are frozen during rebuilding. `canonicalize()` checks complete, unique parent/child coverage; robot eligibility is checked explicitly; mandatory Y_SPLIT is retained; X_SPLIT children remain on their fixed spatial robots; and the existing finite X validator is reused. The 0.50 m split windows and 0.20 m minimum child length are unchanged.

### Difference from REGION, CRRR, and WAG

- Unlike the original REGION initializer, JRRB starts from a certified schedule, moves a spatial group across rails, selects both rail boundaries jointly, and rebuilds the four routes in one candidate.
- Unlike CRRR, it is not a 2--6-parent replacement inside the M192 pool and cannot be excluded by ordinary C2/C4 competition. It has its own visible 16/8/2 funnel and accounting before ALNS.
- Unlike WAG, it does not call the WAG solver, enumerate the complete WAG boundary-offset/VNS search, or use a WAG solution as input. It reuses only project-local route and legality primitives.

The mechanism is substantively different from WAG, but the development results below do not establish a useful empirical method contribution.

## Fixed development candidate-quality diagnostic

The fixed consumed-development states were `N35-S1`, `N35-S2`, `N55-I5`, `N55-I6`, `N85-I9`, and `N85-I12`, using solver seed `20262121`. Normal ALNS was disabled for this diagnostic so JRRB's direct candidate quality could be isolated. Online columns use only the intended Top8 direction/Top2 B32 funnel. Offline columns evaluate all generated candidates after the fact and do not feed labels back into the online score.

| Instance | Initial Cmax | Complete valid | Direction feasible/evaluated | Online B32 status | Online certified | Online strict improvement | Online JRRB Cmax | Offline certified | Offline improved | Offline best |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| N35-S1 | 4956.890 | 16 | 8/8 | 1 FEASIBLE, 1 DEADLOCK | 1 | 1 | 3484.499 | 13/16 | 7 | 2693.850 |
| N35-S2 | 1932.921 | 16 | 8/8 | 2 FEASIBLE | 2 | 0 | 1932.921 | 15/16 | 0 | — |
| N55-I5 | 4141.496 | 0 | 0/0 | — | 0 | 0 | 4141.496 | 0 | 0 | — |
| N55-I6 | 3495.628 | 0 | 0/0 | — | 0 | 0 | 3495.628 | 0 | 0 | — |
| N85-I9 | 4837.878 | 0 | 0/0 | — | 0 | 0 | 4837.878 | 0 | 0 | — |
| N85-I12 | 4929.263 | 16 | 8/8 | 2 FEASIBLE | 2 | 0 | 4929.263 | 9/16 | 0 | — |

The three zero-candidate states do not contain at least two complete same-rail parent groups that can legally move to either rail, so the bounded cross-rail reconstruction precondition is absent. JRRB does not fabricate illegal transfers in these cases.

For the 48 complete offline candidates, every stored candidate preserved pattern identity, complete block coverage, robot eligibility, and fixed X ownership. Direction DP was feasible for every online-selected candidate (24/24). Online B32 returned 5 FEASIBLE and 1 DEADLOCK; all five FEASIBLE schedules passed independent certification. Thus the independent-certification pass rate is 5/5 conditional on B32 FEASIBLE, or 5/6 over all online reference calls.

N35-S1 demonstrates real structural capability: all 16 candidates changed cross-rail ownership, both boundaries, and all four routes; online Top2 reduced Cmax by 1472.391 seconds (29.70%). Its offline best, 2693.850 seconds, is reported only as a post-hoc pool-quality label and is not an online result. The other two nonempty pools contain no certified improvement even after all 16 candidates are evaluated.

## Cost

| Instance | Construction | Online direction | Online B32 |
|---|---:|---:|---:|
| N35-S1 | 0.450 s | 0.003 s | 0.206 s |
| N35-S2 | 0.328 s | 0.003 s | 0.177 s |
| N55-I5 | <0.001 s | 0 | 0 |
| N55-I6 | <0.001 s | 0 | 0 |
| N85-I9 | <0.001 s | 0 | 0 |
| N85-I12 | 5.737 s | 0.007 s | 0.692 s |

The N85-I12 construction cost alone is 5.737 seconds and produced no improvement. This is material within a 60-second budget. Because the candidate-quality gate failed before online search, normal-ALNS iteration loss and post-JRRB ALNS improvement were not measured; inventing those values from a non-run matrix would be invalid.

## 60-second paired matrix

The planned matrix was 6 instances x 2 solver seeds x 2 methods = 24 real 60-second runs. It was not executed because only one development geometry had any offline certified improvement, exactly matching the protocol's stop condition.

Consequently, the following quantities do not exist and are intentionally not reported: 12-pair win/tie/loss, paired median/mean Cmax change, tier-level 60-second change, worst 60-second regression, post-JRRB ALNS gain, ALNS iteration loss, and paired 60-second WAG-gap change. The earlier HGA/WAG development records are only contextual because their execution pairing differs.

This absence is a negative experimental result, not missing successful runs. Running the matrix after the failed gate would spend 24 minutes on a mechanism that already lacks improvement opportunity on five of six fixed geometries and would contradict the specified protocol.

## Final assessment

JRRB is **not successful enough to retain as an active method**:

1. It provides a real missing structural move on N35-S1, but not on more than one development geometry.
2. Online Top2 success occurs on the same single geometry only.
3. N85-I12 pays substantial construction cost without improvement.
4. There is no 60-second evidence that the macro cost is recovered through better ALNS search.
5. The mechanism differs from WAG and from CRRR, but empirical breadth is insufficient for an SCI method claim.

The default-off implementation is retained only for reproducibility of this negative development result. `REGION_FAST_HEURISTIC` remains the frozen deterministic algorithm. No new independent test is warranted for this JRRB version; a future independent set would be needed only if a genuinely new method first passed a new development gate. The consumed 15-workbook independent set must not be reused for selection or described as a new JRRB test.

## Verification

- JRRB-specific and directly related tests: `64 passed`.
- DRL test discovery: `341 passed`.
- Project-root discovery: `372 passed, 1 skipped, 3 subtests passed`.
- JRRB switch default: OFF.
- Original ALNS budget: M192 = 144 atomic + 48 LNS, Kdp8, Kref2, Kref_total4.
- JRRB hard caps: 16 complete candidates, 8 direction candidates, 2 B32 candidates, one macro step.
- CRRR, CRRR_ACCESS_C2, TWO_OPT_STAR, GAT retraining, SA, B32, and the independent certifier were not changed.
