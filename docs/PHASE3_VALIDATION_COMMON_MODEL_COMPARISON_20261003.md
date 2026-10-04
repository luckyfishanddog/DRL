# Phase 3-2B Frozen VALIDATION Common-Model Comparison

Date: 2026-10-03

## A. Baseline source and tests

- Repository identity: `luckyfishanddog/DRL`.
- User-supplied known main: `8321c46e4640f4f6928cee3f951b3e9f8b848f95`.
- This local `DRL` directory is nested inside a different dirty outer repository; the outer HEAD was `0935ad8a723057084be190e31c6be4733ce3cb62` and was not represented as a DRL commit.
- Explicit local revision label: `KNOWN_MAIN_8321c46e4640f4f6928cee3f951b3e9f8b848f95+LOCAL_UNCOMMITTED_DRL_TREE`.
- Scientific source-tree hash: `d6e8650de62b16e986ed8d3ff63e8296f4c40fee68169fdc0809eb4ce29961b7`.
- Provenance: `development_only=true`, `commit_verified=false`.
- Baseline regression before implementation: 217 passed. Frozen-protocol regression before the batch: 229 passed. Final regression after all 180 runs: 229 passed in 30.31 s.
- No local file was uploaded or pushed.

## B. TWO_OPT_STAR OFF decision

Phase 3-2A DEVELOPMENT evidence gave mean Cmax@60 3634.651 for V3/OFF and 3655.896 for V3/ON. `TWO_OPT_STAR` attempted 2224, constructed 2176, reached C3 256 times and C4 15 times, was accepted 7 times, and produced one global-best improvement. This does not establish an aggregate advantage. Phase 3-2B therefore fixed the proposed backbone as `SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_OFF`. The implementation and tests remain; status is `ABLATION_ONLY / EXPERIMENTAL_OPERATOR`.

## C. Shared-incumbent wording correction

The Phase 3-2A common-start experiment is a `SHARED_INCUMBENT_DIAGNOSTIC`. It showed a remaining end-to-end search-process gap after installing the same certified incumbent. It is not pure neighborhood-mechanism isolation: HGA still produced its own population members, while WAG still produced its own assignments and route constructions. The historical artifact and CASE B numbers were not rewritten.

## D. Frozen validation selection rule

`PPO_PHASE3_VALIDATION_SELECTION_V1` uses only frozen geometry metadata and frozen `assigned_role=VALIDATION`. For each tier it prefers N in [20,30], [50,60], or [80,90], centered at 25, 55, or 85. If four distinct workbooks are unavailable, it expands deterministically by `|N-target|`. Descriptors are min-max normalized within the tier candidate pool: N, total weld length, bbox/x/y coverage, cross-y6 count, upper/lower imbalance, and left/right imbalance. The first point is closest to target N with geometry-hash tie-break; later points maximize minimum Euclidean distance, with `|N-target|` and geometry hash as ties. Each workbook appears at most once.

The LARGE preferred range did not contain four distinct eligible workbooks after global workbook isolation, so the rule expanded deterministically to N=75. No solver was run before selection.

## E. Selected instances

| # | Tier | N | Instance | Geometry hash prefix |
|---:|---|---:|---|---|
| 1 | SMALL | 25 | `data/PPO_TRAIN/seed_1867256633::g15_w025` | `23894bfab839` |
| 2 | SMALL | 27 | `data/PPO_TRAIN/seed_1871665270::g15_w027` | `2cb134234b95` |
| 3 | SMALL | 30 | `data/ID_TEST/seed_0533728435::g19_w030` | `2fbef409e582` |
| 4 | SMALL | 27 | `data/ID_TEST/seed_2013829876::g15_w027` | `997fb2d99d2a` |
| 5 | MEDIUM | 55 | `data/PPO_TRAIN/seed_0088690758::g26_w055` | `0d29ed9b58cc` |
| 6 | MEDIUM | 52 | `data/PPO_TRAIN/seed_1916546866::g26_w052` | `0eb509ee0fff` |
| 7 | MEDIUM | 60 | `data/PPO_TRAIN/seed_1702002874::g32_w060` | `fc536e9828a4` |
| 8 | MEDIUM | 50 | `data/PPO_TRAIN/seed_0463639632::g20_w050` | `86d33e594c6e` |
| 9 | LARGE | 85 | `data/PPO_TRAIN/seed_0817833068::g44_w085` | `e1efea909d40` |
| 10 | LARGE | 75 | `data/PPO_TRAIN/seed_2091884649::g43_w075` | `507094f652f6` |
| 11 | LARGE | 75 | `data/VALIDATION/seed_2145450783::g41_w075` | `d219d4fa8407` |
| 12 | LARGE | 75 | `data/PPO_TRAIN/seed_0600641460::g45_w075` | `86f09e97a5ac` |

Historical folder names are not Phase 3 roles. In particular, the two paths containing `data/ID_TEST` above have frozen `assigned_role=VALIDATION`. The 15 workbooks whose frozen role is `ID_TEST` are disjoint from this set and were never passed to an Excel loader or solver.

## F. Geometry-diversity summary

All 12 workbooks are distinct and tiers are 4/4/4. The exact raw and normalized eight-dimensional descriptors, candidate-pool normalization, selection ordinals, workbook paths, sheets, and full geometry hashes are frozen in `PPO_PHASE3_VALIDATION_SET_V1.json`. Selected normalized vectors span boundary values in every tier; the LARGE selection records the required deterministic range expansion rather than hiding it.

## G. Frozen identities

- Dataset manifest hash: `3a12fc59708fcd6adc20bb19ba44ba8ece585f8e0a81a93304694960f18f2a31`.
- Phase 3 split hash: `5b5e3a6d43b943c5bad03a9ecbadaefa74c6158b33036da6e0fc236e03af1c5a`.
- Validation set hash: `ed7550654accddad3e029915fe3b31881292ab4341dd275f846db1b218bb5df3`.
- Validation protocol hash: `3d797f7d9ddb0c57b751bdaff50e451c3f67b54410023bf9953b1a89229ac2d2`.
- Scope: `FORMAL_SCOPE_V1_1`, hash `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`.

## H. Methods and config hashes

| Method | Config hash |
|---|---|
| `SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_OFF` | `53a83e8b45d91732116a64ade3bfb0bd701988c2804e2b4827339fce7130f9db` |
| `ADAPTED_HGA_V1` | `a1748ac615fd93270dbeb75361c14a32ca3018cca805c33361cb90309d903847` |
| `ADAPTED_WAG_VNS_V1` | `469c79808f2cb6c2e9d6af0c5252555ea803e0c9f500f9e8652162d071e17f67` |

Seeds were exactly `PHASE3_VALIDATION_SEEDS_V1 = [20260928, 20260929, 20260930, 20261001, 20261002]`. Each method-instance-seed used one native 60 s trajectory with checkpoints 5/30/60; initialization was charged to the same timer. ALNS used `max_iterations=100000` only as a safety cap and `enable_two_opt_star=false`. HGA and WAG used unchanged default adapted configs.

## I–J. Completion and certification

- Completed runs: 180/180; 60 per method.
- Certified Cmax@60: 180/180.
- Final certified schedules: 180/180.
- Numeric failures: 0.
- Scheduler-FEASIBLE/certifier-FAIL mismatches: 0.
- Duplicate or mismatched resume keys: 0.
- Every record carries the validation-set, method-config, scope, source-tree, and protocol identities.

## K–N. Time-to-first and anytime behavior

Across-run medians (descriptive VALIDATION analysis only):

| Method | median time-to-first (s) | median Cmax@5 | median Cmax@30 | median Cmax@60 |
|---|---:|---:|---:|---:|
| ALNS V3/OFF | 2.017993 | 3324.498 | 3315.217 | 3170.122 |
| Adapted HGA | 0.084473 | 3104.469 | 2999.821 | 2976.322 |
| Adapted WAG+VNS | 0.085557 | 2970.294 | 2967.022 | 2967.022 |

At 30 s and 60 s, all 180 method runs had a certified incumbent. At 5 s, HGA and WAG had 60/60 each; ALNS had 48/60. The missing ALNS values remain JSON `null`: 0/5 seeds at N75 `seed_2091884649`, 0/5 at N75 `seed_0600641460`, and 3/5 at N50 `seed_0463639632`. No penalty Cmax was substituted and no later improvement backfilled an earlier checkpoint. Time-to-first is reported but was not a release gate.

## O–Q. Five-seed Cmax@60 medians, ratios, and tier result

| Tier | N | Instance suffix | A_i | H_i | W_i | R_i |
|---|---:|---|---:|---:|---:|---:|
| SMALL | 25 | `1867256633::g15_w025` | 1499.886 | 1520.281 | 1520.281 | 0.986585 |
| SMALL | 27 | `1871665270::g15_w027` | 2549.253 | 2577.275 | 2577.275 | 0.989127 |
| SMALL | 30 | `0533728435::g19_w030` | 2665.128 | 2404.207 | 2585.651 | 1.108527 |
| SMALL | 27 | `2013829876::g15_w027` | 2695.532 | 2291.354 | 2441.206 | 1.176393 |
| MEDIUM | 55 | `0088690758::g26_w055` | 2624.885 | 2617.006 | 2579.819 | 1.017469 |
| MEDIUM | 52 | `1916546866::g26_w052` | 2894.794 | 2880.294 | 2871.530 | 1.008101 |
| MEDIUM | 60 | `1702002874::g32_w060` | 4248.122 | 4584.062 | 4236.986 | 1.002628 |
| MEDIUM | 50 | `0463639632::g20_w050` | 3171.429 | 3074.015 | 3069.058 | 1.033356 |
| LARGE | 85 | `0817833068::g44_w085` | 5556.854 | 4295.349 | 4174.477 | 1.331150 |
| LARGE | 75 | `2091884649::g43_w075` | 4485.223 | 3456.714 | 3260.830 | 1.375485 |
| LARGE | 75 | `2145450783::g41_w075` | 5150.329 | 4540.642 | 4372.442 | 1.177907 |
| LARGE | 75 | `0600641460::g45_w075` | 6491.745 | 4657.587 | 4611.174 | 1.407829 |

- Overall median R_i: **1.070941** (threshold <=1.10: pass).
- Instances with R_i<=1.10: **6/12** (required >=8/12: fail).
- SMALL median R_i: **1.048827** (<=1.15: pass).
- MEDIUM median R_i: **1.012785** (<=1.15: pass).
- LARGE median R_i: **1.353317** (<=1.15: fail).

## R. Reference-call efficiency

| Method | Total reference calls | Median calls/run | Median runtime (s) |
|---|---:|---:|---:|
| ALNS V3/OFF | 10847 | 116.5 | 60.693 |
| Adapted HGA | 9368 | 103.0 | 60.084 |
| Adapted WAG+VNS | 10378 | 101.5 | 60.116 |

These are workload/efficiency descriptors, not formal superiority statistics.

## S. DEADLOCK and recovery

| Method | Baseline DEADLOCK | Recovered | Remaining DEADLOCK |
|---|---:|---:|---:|
| ALNS V3/OFF | 7820 | 4534 | 3286 |
| Adapted HGA | 5412 | 1757 | 3655 |
| Adapted WAG+VNS | 7559 | 4608 | 2951 |

All reported final solutions nevertheless passed the independent certifier.

## T. HGA/WAG cap telemetry and limitations

HGA totals: `vnd_candidate_cap_hits=14503`, `vnd_pass_cap_hits=3149`, `initialization_reference_limit_hits=0`, and `population_survival_events=1619`. WAG totals: `factorial_window_cap_hits=14676`, `factorial_call_cap_hits=0`, `wag_variant_cap_hits=60`, and `route_combination_cap_hits=4309`. These caps are a limitation/telemetry result only; no parameter was changed.

The selection has one N85 instance but had to expand the remaining LARGE points to N75. This is the frozen, pre-result geometry selection outcome and must not be replaced after seeing performance.

## U. Overshoot and termination

All 180 runs terminated `TIME_LIMIT`; `ITERATION_LIMIT=0`. Maximum overshoot was 6.057 s for ALNS, 1.639 s for HGA, and 1.994 s for WAG. Overshoot consists of already-started bounded work/reference completion and final certification. Improvements completed after a checkpoint were excluded from that checkpoint.

## V. Predefined release rule

| Gate | Result |
|---|---|
| 180/180 certified Cmax@60 | PASS |
| Numeric failure = 0 | PASS |
| Certifier mismatch = 0 | PASS |
| Median R_i <= 1.10 | PASS (1.070941) |
| At least 8/12 R_i <= 1.10 | **FAIL (6/12)** |
| Every tier median R_i <= 1.15 | **FAIL (LARGE=1.353317)** |
| No iteration-limit termination | PASS |

The experiment itself is valid and complete. Failure of performance gates is not an experiment failure.

## W. Deterministic backbone decision

`PHASE3_VALIDATION_STATUS = PASS`

`DETERMINISTIC_BACKBONE_STATUS = NEEDS_CANDIDATE_POOL_AUDIT`

`TWO_OPT_STAR_STATUS = ABLATION_ONLY`

`ID_TEST_STATUS = SEALED`

The 5/30/60 single-trajectory wall-clock protocol is operationally validated and can be frozen. The current deterministic backbone is not declared competitive enough under the entire predefined gate because the LARGE tier and 8/12 count failed. This does not authorize a new operator, HGA/WAG retuning, initializer change, or VALIDATION rerun.

## X. Next-phase authorization

`NEXT_PHASE = Phase 4-0 — Candidate-Pool Oracle Recall Audit`

The audit must determine whether good candidates are generated but missed by cheap ranking/Kdp, or whether candidate generation lacks good candidates. Only after that diagnosis should an MLP ranker be considered. GAT, PPO, TRAIN_POOL labels, ID_TEST, OOD, LB_LP, final exact, and further validation-driven neighborhood engineering were not performed in this phase.

Machine-readable sources: `data/manifests/PPO_PHASE3_VALIDATION_SET_V1.json`, `data/manifests/PHASE3_VALIDATION_PROTOCOL_V1.json`, and `data/development/phase3_validation_common_model_v1.json`.
