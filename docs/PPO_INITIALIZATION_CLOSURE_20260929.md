# Phase 3-0.5 — PPO Feasible Initialization Closure

`PPO_DATA_INTAKE_STATUS = PASS`

`PPO_ALNS_INITIALIZATION_STATUS = PASS`

`PHASE3_1_AUTHORIZED = YES`

`NEXT_PHASE = Phase 3-1 — Adapted HGA / Adapted WAG`

## A. Baseline SHA and tests

- User-provided known GitHub main at task start: `29659c438cb6f25b838ec3c96c5356b4154d0467`.
- Per user instruction, no Git command, push, upload, branch, commit, or remote mutation was performed.
- Baseline command: `D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- Baseline result: 187 passed in 29.21 s.
- Final result: 192 passed in 35.69 s.
- Local smoke provenance label was supplied explicitly as `LOCAL_UNCOMMITTED_20260929`; the runner no longer contains a stale commit default.

## B. PPO N distribution

`PPO_DATASET_MANIFEST_V1` contains 2942 valid unique instances. Counts for the current N=10..90 domain are:

```text
10:1 11:1 12:1 13:1 14:2 15:5 16:5 17:8 18:7 19:9
20:9 21:8 22:7 23:9 24:11 25:12 26:12 27:19 28:21 29:20
30:27 31:32 32:32 33:39 34:41 35:39 36:39 37:44 38:48 39:50
40:52 41:44 42:41 43:50 44:56 45:56 46:63 47:60 48:58 49:61
50:53 51:55 52:63 53:67 54:69 55:70 56:63 57:61 58:60 59:62
60:61 61:59 62:58 63:59 64:64 65:61 66:53 67:51 68:54 69:58
70:54 71:53 72:52 73:48 74:42 75:40 76:39 77:35 78:26 79:25
80:23 81:25 82:22 83:16 84:16 85:17 86:16 87:13 88:12 89:9 90:9
```

## C. N>90 count and percentage

- N<=90: 2893 valid unique instances.
- N>90: 49 valid unique instances, 1.6655336506% of the valid unique set.
- N>90 entries remain valid in the frozen manifest. They are neither deleted nor relabelled invalid.

## D. Main range definition

The Phase 3 main experimental operating range is `10 <= N <= 90`. N>90 is retained as `LARGE-SCALE STRESS / OUT-OF-MAIN-RANGE` and does not block the initialization gate.

## E. N55 four-strategy diagnosis

The historical N55 instance was `data/VALIDATION/seed_0967455456::g27_w055`. All four existing constructions were built and deduplicated before formal evaluation:

| Strategy | Canonical hash prefix | Blocks by robot | Direction | Formal result |
| --- | --- | --- | --- | --- |
| LOAD_FIRST | `e010999d88bc` | 17/17/12/12 | FEASIBLE | DEADLOCK, B32 exhausted |
| RAIL_BALANCED | `d6259fc5c545` | 17/17/12/12 | NO_LEGAL_FIRST_ORIENTATION | not evaluated |
| X_ORDER_AWARE | `99b74ba99f77` | 16/18/13/11 | FEASIBLE | DEADLOCK, B32 exhausted |
| SPATIAL_SPREAD | `efb172781525` | 17/17/12/12 | FEASIBLE | DEADLOCK, B32 exhausted |

The wait-for diagnostics repeatedly ended with same-rail partner blocking: R0/R1 on the upper rail and R2/R3 on the lower rail. This excluded the Kinit reference prefilter as the root cause and identified same-rail route interaction as the representative blocker.

## F. LEVEL 1 result

LEVEL 1 evaluated all four existing construction strategies. None produced a certified FEASIBLE schedule. The third and fourth constructions were not hidden by `Kinit_ref=2`; both were explicitly evaluated in the diagnostic run and remained DEADLOCK.

## G. LEVEL 2 result

For every construction with a legal initial orientation, all legal first-orientation combinations were enumerated with the existing route DP. The four cheapest distinct direction vectors per construction were formally evaluated:

- LOAD_FIRST: 16 legal combinations; top four DEADLOCK.
- RAIL_BALANCED: zero legal first-orientation combinations.
- X_ORDER_AWARE: 16 legal combinations; top four DEADLOCK.
- SPATIAL_SPREAD: 16 legal combinations; top four DEADLOCK.

No direction alternative changed F4/B32 or produced FEASIBLE.

## H. LEVEL 3 result

The N55 instance has 52 WHOLE parents, three mandatory Y splits, and zero optional legal Y-split candidates. `Y_DIVERSE_CENTER` and `Y_DIVERSE_BALANCE` therefore reduce exactly to the historical/minimum pattern set. All generated constructions were duplicates of LEVEL 1. No new split point or pattern domain was invented.

## I. LEVEL 4 result

A bounded deterministic rail-serial prototype assigned each block to an eligible rail, used only robot 0 on the upper rail and robot 2 on the lower rail, and sorted both routes by X. On N55 it produced routes of 34/0/24/0 blocks, formal FEASIBLE without recovery, Cmax 5303.928358102911, and an independent successful certificate.

## J. Minimal production fix

The production initializer now calls at most one `RAIL_SERIAL_BOOTSTRAP` after the configured normal construction portfolio produces no certified FEASIBLE result. It:

- preserves the selected WHOLE/mandatory Y_SPLIT patterns;
- assigns flexible blocks to the less-loaded eligible rail with stable ties;
- activates one robot per rail;
- orders blocks deterministically by midpoint/min/max X and block ID;
- runs the existing constrained direction optimizer, unchanged formal evaluator, and independent certifier.

The fix changes initialization only. Weld geometry, F1/F2/F3/F4, B32, scheduler semantics, search operators, and certification are unchanged.

## K. Why more complex changes were not added

LEVEL 2 direction diversity did not solve N55. LEVEL 3 had no optional legal pattern candidates. The first bounded LEVEL 4 construction solved N55 and then certified all 20 development instances. No fifth normal heuristic, new ALNS operator, optional split rule, evaluator change, or F4 retuning was necessary.

## L. Initialization evaluation budget

- Normal construction pool: four deterministic strategies.
- Formal normal candidates: `Kinit_ref=2` after deterministic cheap ranking.
- Rail-serial fallback: `B_init_bootstrap=1` and only after normal failure.
- Direction choice: existing bounded deterministic route DP.
- Pattern portfolio additions: zero in production.
- Every reference/certifier call and initialization duration remains inside the run wall-clock accounting.

## M. PPO_INIT_BOOTSTRAP_DEVSET_V1 definition

The set is solver-independent and selected before running initialization outcomes:

- strata: N=10–30, 31–50, 51–70, 71–90;
- five instances per stratum;
- stable sort by `instance_geometry_hash`, then instance ID;
- distinct workbooks globally when possible; achieved 20 distinct workbooks for 20 instances;
- no selection based on ALNS, Cmax, DEADLOCK, or certification outcome.

The machine-readable definition is `data/manifests/PPO_INIT_BOOTSTRAP_DEVSET_V1.json`.

## N. Twenty-instance results

All rows used `FORMAL_SCOPE_V1_1`, B32, and independent certification.

| Stratum | Instance | N | Status | Winner | Init refs | Init s | Certified |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 10–30 | `seed_1064648049::g18_w024` | 24 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 0.298 | yes |
| 10–30 | `seed_0469626998::g15_w030` | 30 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.218 | yes |
| 10–30 | `seed_0900775608::g16_w030` | 30 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.231 | yes |
| 10–30 | `seed_1964628001::g15_w030` | 30 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 0.500 | yes |
| 10–30 | `seed_000505::g18_w029` | 29 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 0.431 | yes |
| 31–50 | `seed_0194123089::g21_w049` | 49 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.563 | yes |
| 31–50 | `seed_0967455456::g19_w047` | 47 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.355 | yes |
| 31–50 | `seed_1419626827::g24_w048` | 48 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.347 | yes |
| 31–50 | `seed_0760656368::g25_w038` | 38 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 0.613 | yes |
| 31–50 | `seed_1727996722::g20_w048` | 48 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 1.440 | yes |
| 51–70 | `seed_0767803876::g33_w059` | 59 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 1.290 | yes |
| 51–70 | `seed_0442383381::g41_w062` | 62 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 1.900 | yes |
| 51–70 | `seed_1137713392::g34_w068` | 68 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 1.667 | yes |
| 51–70 | `seed_0471570702::g38_w058` | 58 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.378 | yes |
| 51–70 | `seed_0457685429::g42_w070` | 70 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.770 | yes |
| 71–90 | `seed_1054426823::g37_w078` | 78 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 1.105 | yes |
| 71–90 | `seed_0566744530::g42_w073` | 73 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 0.614 | yes |
| 71–90 | `seed_0676566458::g39_w071` | 71 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 1.189 | yes |
| 71–90 | `seed_0211781140::g32_w085` | 85 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 3 | 2.158 | yes |
| 71–90 | `seed_0401115467::g37_w073` | 73 | SUCCESS | RAIL_SERIAL_BOOTSTRAP | 2 | 4.058 | yes |

Summary: 20/20 SUCCESS, 20/20 certified, NUMERIC_FAILURE=0.

## O. PPO_PHASE3_SMOKESET_V2

V2 applies the same solver-independent min/closest-to-median/max selection to valid unique N<=90 entries:

- small: `data/DEV_ONLY/seed_000202::g06_w010`, N=10;
- medium: `data/VALIDATION/seed_0967455456::g27_w055`, N=55;
- large: `data/PPO_TRAIN/seed_1118337379::g38_w090`, N=90.

All three are distinct. V1 remains unchanged as historical evidence.

## P. Five-second results

| Tier | N | Init status/strategy | Initial Cmax | Best Cmax | Iterations | Nref | Runtime/overshoot s | Final cert |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| small | 10 | SUCCESS / X_ORDER_AWARE | 1283.635 | 643.344 | 19 | 72 | 5.096 / 0.096 | yes |
| medium | 55 | SUCCESS / RAIL_SERIAL_BOOTSTRAP | 5303.928 | 3759.988 | 3 | 10 | 6.916 / 1.916 | yes |
| large | 90 | SUCCESS / RAIL_SERIAL_BOOTSTRAP | 8582.814 | 7838.110 | 1 | 4 | 7.402 / 2.402 | yes |

All three entered the search stage. The N90 initialization completed just after the nominal 5 s checkpoint, so `Cmax@5` is recorded as unavailable while the bounded run still completed one full iteration and returned a certified final schedule. Non-preemptive overshoot remains reported rather than hidden.

## Q. Medium/large 30-second results

| Tier | N | Initial Cmax | Best Cmax | Cmax@30 | Iterations | Nref | Runtime/overshoot s | Final cert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| medium | 55 | 5303.928 | 3327.489 | 3327.489 | 17 | 40 | 31.457 / 1.457 | yes |
| large | 90 | 8582.814 | 6457.822 | 6457.822 | 7 | 20 | 33.322 / 3.322 | yes |

Both satisfy the required minimum of two completed iterations.

## R. N102 out-of-range replay

One initialization-only replay was performed for the historical V1 large instance `data/ID_TEST/seed_1411387107::g45_w102`. It is outside the main range and is not a gate item. The replay returned SUCCESS through `RAIL_SERIAL_BOOTSTRAP`, formal FEASIBLE, certified=true, two initialization reference calls, and 1.838 s initialization time.

## S. Certification

- Bootstrap devset: 20/20 initial schedules independently certified.
- V2 5 s: 3/3 final schedules independently certified.
- V2 medium/large 30 s: 2/2 final schedules independently certified.
- N102 optional replay: independently certified.
- No scheduler-FEASIBLE/certifier-fail mismatch and no NUMERIC_FAILURE occurred.

## T. Development-consumed workbooks

The following 23 PPO-relative workbooks are development-consumed and cannot enter untouched TEST:

```text
data/DEV_ONLY/seed_000202.xlsx
data/DEV_ONLY/seed_000505.xlsx
data/ID_TEST/seed_0401115467.xlsx
data/ID_TEST/seed_0469626998.xlsx
data/ID_TEST/seed_1411387107.xlsx
data/PPO_TRAIN/seed_0194123089.xlsx
data/PPO_TRAIN/seed_0211781140.xlsx
data/PPO_TRAIN/seed_0442383381.xlsx
data/PPO_TRAIN/seed_0457685429.xlsx
data/PPO_TRAIN/seed_0566744530.xlsx
data/PPO_TRAIN/seed_0676566458.xlsx
data/PPO_TRAIN/seed_0760656368.xlsx
data/PPO_TRAIN/seed_0767803876.xlsx
data/PPO_TRAIN/seed_0900775608.xlsx
data/PPO_TRAIN/seed_1054426823.xlsx
data/PPO_TRAIN/seed_1064648049.xlsx
data/PPO_TRAIN/seed_1118337379.xlsx
data/PPO_TRAIN/seed_1137713392.xlsx
data/PPO_TRAIN/seed_1419626827.xlsx
data/PPO_TRAIN/seed_1727996722.xlsx
data/VALIDATION/seed_0471570702.xlsx
data/VALIDATION/seed_0967455456.xlsx
data/VALIDATION/seed_1964628001.xlsx
```

This list includes all bootstrap workbooks, all V1 historical smoke workbooks, and the V2 N90 workbook.

## U. Future workbook-level split rule

Future TRAIN/VALIDATION/ID_TEST assignment must use the complete PPO workbook/generation-seed family as the isolation unit. If any sheet from a workbook was consumed by initialization development or compatibility smoke, every sheet from that workbook is excluded from untouched TEST. Historical source directory names do not define the new study split.

## V. Tests

Regression coverage now includes:

- all-four-construction diagnostics even when `Kinit_ref=2`;
- deterministic rail-serial output and bounded one-call fallback;
- bootstrap budget disable/validation;
- V2 deterministic selection and N>90 retain-but-exclude semantics;
- four bootstrap strata, global workbook separation, deterministic replay, and consumed-workbook tracking;
- explicit source label requirement and absence of the stale hardcoded SHA;
- all prior data adapter, scheduler, search, exact, formal scope, F4, and certifier tests.

Final command: `D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.

Final result: `192 passed in 35.69s`.

## W. Phase 3-1 authorization

All Phase 3-0.5 pass conditions are satisfied. Data intake remains PASS, the main N<=90 range is explicit, N>90 remains valid, bootstrap success is 20/20, V2 smoke initialization is 3/3, all required runs entered search and certified, source provenance no longer defaults to a stale SHA, and full regression passes.

`PPO_DATA_INTAKE_STATUS = PASS`

`PPO_ALNS_INITIALIZATION_STATUS = PASS`

`PHASE3_1_AUTHORIZED = YES`

`NEXT_PHASE = Phase 3-1 — Adapted HGA / Adapted WAG`
