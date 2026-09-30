# Phase 3-2A Fair Timing and Backbone Diagnostic — 2026-09-30

`PHASE3_FAIRNESS_BACKBONE_STATUS = PASS`

`PHASE3_2B_AUTHORIZED = YES`

`NEXT_PHASE = Phase 3-2B VALIDATION Common-Model Comparison`

## A. Baseline SHA and tests

The user-reported GitHub DRL main was `51fe3f56a9c145c1602e28eb0937236ba4ef68e0`. The local `DRL` directory is not a standalone Git checkout: its enclosing repository is `luckyfishanddog/MRTA`, local HEAD `0935ad8a723057084be190e31c6be4733ce3cb62`. Therefore every run is explicitly `development_only=true`, `commit_verified=false`, and uses the label `USER_REPORTED_MAIN_51fe3f56a9c145c1602e28eb0937236ba4ef68e0_LOCAL_UNVERIFIED`; the enclosing HEAD is not presented as DRL provenance.

Before this round, the complete suite passed 209 tests in 34.25 s. After implementation, the final complete suite passed 217 tests in 30.90 s pytest time (33.154 s wall time) with `D:\pybullet_test\.venv\Scripts\python.exe` and `-p no:cacheprovider`.

## B. Old 100-iteration fairness bug

The Phase 3 runner inherited `SearchConfig.max_iterations=100`. Some nominal 30 s ALNS smoke runs therefore ended around 21 s with the iteration cap. This made the wall-clock comparison invalid. The global default remains 100 for historical replay and old tests.

## C. Timing fix

The Phase 3 comparison factory now explicitly constructs `SearchConfig(time_limit=budget, max_iterations=100000, checkpoints=(5,30,60))`. One 60 s run produces all checkpoints; 5 s and 30 s are not separate reruns. A completed evaluation is timestamped at completion and cannot backfill an earlier deadline.

All successful pre-fix, common-seed, and post-fix comparison runs terminated with `TIME_LIMIT`. An explicit small safety cap regression records `ITERATION_LIMIT`. Unified reasons are `TIME_LIMIT`, `ITERATION_LIMIT`, `INITIALIZATION_FAILED`, `NUMERIC_FAILURE`, and `COMPLETED_OTHER`.

## D. Six diagnostic instances

`PPO_PHASE3_DIAGNOSTIC_SET_V1` selected only frozen `DEVELOPMENT_CONSUMED` roles, using distance to N=25/55/85 and then `instance_geometry_hash`; all workbooks are distinct.

| tier | N | instance |
|---|---:|---|
| small | 25 | `data/PPO_TRAIN/seed_0760656368::g16_w025` |
| small | 25 | `data/PPO_TRAIN/seed_1064648049::g19_w025` |
| medium | 55 | `data/VALIDATION/seed_0967455456::g27_w055` |
| medium | 55 | `data/PPO_TRAIN/seed_0457685429::g31_w055` |
| large | 85 | `data/PPO_TRAIN/seed_0211781140::g32_w085` |
| large | 85 | `data/ID_TEST/seed_0401115467::g45_w085` |

The last two historical directory labels do not determine Phase 3 roles. The split manifest assigns all six workbooks `DEVELOPMENT_CONSUMED`; no frozen VALIDATION or ID_TEST workbook was evaluated.

## E. Initial-solution telemetry

Every method now records canonical initial hash/source, pattern and optional-split counts, four robot block counts and process loads, four route proxy costs plus total, route hashes/routes, directions, certification, initialization reference calls, and initialization time. The initial solution is the best certified incumbent available before iterative search; iteration-one improvements are excluded.

## F. HGA/WAG initial-hash finding

The 18 native HGA/WAG pairs produced:

- 6 canonical-hash-identical pairs, covering all three seeds on two instances;
- 9 assignment-identical but route-different pairs, covering three instances;
- 3 different-solution/different-Cmax pairs, covering one instance;
- zero different-solution/equal-Cmax pairs.

The two hash-identical instances are `seed_1064648049::g19_w025` and `seed_0967455456::g27_w055`. HGA and WAG execute separate initializer implementations and sources; no shared initializer call was found. The equality is a natural consequence of deterministic balanced constructions on those geometries. Exact hashes, patterns, block counts, process loads, route hashes, directions, and Cmax values are retained per instance and seed in the development artifact.

## G. HGA/WAG cap-hit profile

No HGA/WAG parameter or cap was changed. Across the 18 native runs, HGA recorded 2749 VND candidate-cap hits, 953 VND pass-cap hits, 326 population survival events, 481 optional-Y mutation attempts, and 426 accepted mutations. WAG recorded 2189 factorial-window-cap hits, 983 route-combination-cap hits, 18 WAG-variant-cap hits, 29 MOVE, 43 SWAP, 41 LNS calls, and 33 optional-Y toggle attempts; factorial-call-cap hits were zero. These are diagnostic counters, not tuning triggers.

## H. Native 60 s results

The native matrix contains 54 runs: six instances × three methods × three seeds. Median initial / Cmax@60 were:

| method | certified runs | median initial | median Cmax@60 |
|---|---:|---:|---:|
| SA-OI-ALNS V2 | 15/18 | 5303.928 | 3327.489 |
| Adapted HGA | 18/18 | 3459.709 | 3190.694 |
| Adapted WAG+VNS | 18/18 | 3429.016 | 3092.322 |

The three ALNS failures are the three preregistered seeds on `seed_0401115467::g45_w085`, all `INITIALIZATION_FAILED`. They have no final schedule or Cmax and were not converted to penalty values. All produced final schedules were certified.

## I. Common-seed 60 s results

The common-seed matrix contains 18 runs at seed 20260929. Each instance uses one canonical `RAIL_SERIAL_BOOTSTRAP` solution. Direction-DP candidates are tried first; on the last N85 all four DP candidates deadlocked, and a deterministic all-zero feasibility fallback produced a certified schedule on the fifth reference call. The solution hash did not change.

All three methods received the same canonical seed hash and initial Cmax on every instance, and all 18 final schedules were certified. Median Cmax@5 / @30 / @60 were:

| method | Cmax@5 | Cmax@30 | Cmax@60 |
|---|---:|---:|---:|
| SA-OI-ALNS V2 | 5177.245 | 4506.929 | 4506.929 |
| Adapted HGA | 5600.830 | 4640.983 | 3515.086 |
| Adapted WAG+VNS | 3534.877 | 3204.801 | 3137.946 |

Common seed construction time is recorded separately. The decision uses `search_from_seed_time`; total time including construction is also retained.

## J. Initialization-versus-search decomposition

Native ALNS initial Cmax had mean 5806.483 versus HGA 3427.712 and WAG 3389.299, confirming major initialization dominance. Common-seed injection repaired the unequal start but did not remove the entire search gap. Every record includes initial gap, search-improvement ratio, Cmax@60/common-seed Cmax, and absolute improvement from the common seed.

## K. CASE B decision

The decision rule was frozen before results. Common-seed median ALNS Cmax@60 divided by the best baseline median was `4506.929 / 3137.946 = 1.436267`, and ALNS was within 10% of the best baseline on only 2/6 instances. Because more than 3/6 instances remained over 10%, the result is `CASE_B_SEARCH_MECHANISM_GAP`.

## L. INIT_POLICY_V3 design

`RAIL_MONOTONE_BALANCED_BOOTSTRAP` keeps the existing deterministic WHOLE/mandatory-Y pattern policy. Flexible blocks are stably assigned to the less-loaded rail, tie upper. Within each rail, blocks are X-sorted, every boundary is evaluated by maximum process load, imbalance, then boundary index; the left robot receives the ascending prefix and the right robot the descending suffix.

The strategy joins, rather than replaces, LOAD_FIRST, RAIL_BALANCED, X_ORDER_AWARE, SPATIAL_SPREAD, and the historical RAIL_SERIAL fallback. It performs no local search, ALNS, optional-Y sweep, or route exchange. If the V3 direction-DP schedule deadlocks, at most four deterministic direction candidates are tested for the first certified feasible schedule, not the best Cmax. The complete initializer has a hard maximum of 10 reference calls.

## M. V3 result

All six V3/OFF post-fix runs were certified, including the N85 where V2 deterministically failed. V3/OFF mean initial Cmax was 3763.450 and mean Cmax@60 was 3634.651. Five of six winners used `RAIL_MONOTONE_BALANCED_BOOTSTRAP`; one retained the historical rail-serial candidate because it was the best certified portfolio member.

## N. TWO_OPT_STAR result

CASE B authorized exactly one operator. `TWO_OPT_STAR` swaps suffixes only between R0↔R1 or R2↔R3, rejects invalid targets without repair, preserves patterns, and traverses the existing cheap-screen → direction-DP → reference/certifier pipeline. The configuration defaults OFF historically and is ON only for the V3 production candidate.

Across six V3/ON confirmation runs it was attempted 2224 times, constructed 2176, cheap-valid 2089, entered C3 256 and C4 15 times, was accepted 7 times, and produced one best improvement. V3/ON mean Cmax@60 was 3655.896 versus V3/OFF 3634.651. This small diagnostic does not show aggregate superiority; per the predeclared rule no second operator was added.

## O. Reference-call efficiency

Common-seed total reference calls were 830 ALNS, 580 HGA, and 807 WAG. Mean improvement per reference call was 56.561, 56.605, and 86.206 respectively. Post-fix V3/OFF used 1038 total calls and V3/ON 972. Scheduler time, repair/local-search time, initialization time, reference count, and improvement/reference are stored separately; no conclusion is based on final Cmax alone.

## P. Certification

Common-seed runs are 18/18 certified. Post-fix production-relevant V3/OFF, V3/ON, HGA, and WAG runs are 24/24 certified. Historical V2 failures produce no final schedule. `NUMERIC_FAILURE=0`, and no certifier mismatch occurred.

## Q. Termination reasons

All successful 60 s comparison runs report `TIME_LIMIT`; the 100000 safety cap never became a normal primary terminator. The four V2 diagnostic failures report `INITIALIZATION_FAILED` (three in native pre-fix, one replay in post-fix). Unit regression separately verifies explicit small caps report `ITERATION_LIMIT`.

## R. Full regression

The final suite passed 217 tests. Added regressions cover comparison max-iteration override, explicit cap termination, deterministic hashes, common seed identity/certification, diagnostic-set isolation, V3 monotone/flexible/coverage/eligibility/pattern/certification behavior, and `TWO_OPT_STAR` suffix/same-rail/eligibility/coverage/pattern/determinism/ablation behavior.

## S. Final production algorithm definition

The production proposed method is `SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_ON`: bounded five-construction portfolio plus historical rail-serial fallback, maximum ten initialization reference calls, existing atomic/LNS/SA/direction refinement, and the single additional same-rail `TWO_OPT_STAR`. Global V2 defaults and historical replay remain unchanged. HGA/WAG paper parameters and common-model caps remain unchanged.

No formal-model, scope hash, F4/B32, WAIT, terminal, same-rail, WHOLE/Y, evaluator, certifier, dataset split, VALIDATION role, or ID_TEST role was changed.

## T. Phase 3-2B authorization

The timing bug is closed, telemetry is traceable, the six-instance set is solver-independent, common-seed injection is real and immutable, CASE B was applied mechanically, V3 is certified, exactly one operator was added, all production-relevant confirmation schedules are certified, numeric failures are zero, and full regression passes.

Phase 3-2B is authorized to select small/medium/large instances solver-independently from the 15 frozen VALIDATION workbooks, run the three algorithms with five seeds and one 60 s trajectory per run, and freeze the final comparison protocol. ID_TEST remains prohibited until that protocol is frozen.
