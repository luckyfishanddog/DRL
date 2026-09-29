# Final F4 Release Calibration

Date: 2026-09-29

## Release decision

```text
ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1
FINAL_F4_RELEASE_STATUS = PASS
PHASE3_AUTHORIZED = YES
NEXT_PHASE = Phase 3 — Adapted HGA / Adapted WAG common-model
```

F4 is final for Phase 3 and later common-model comparisons. No further budget
tuning may use HGA, WAG, ALNS, VALIDATION, ID_TEST, OOD, or TEST outcomes.
B256 and larger budgets were not tested.

## A. Baseline repository and tests

The accessible DRL directory remains nested inside a dirty outer workspace, so
all evidence in this report is development-only and has
`UNVERIFIED_SOURCE_PROVENANCE`.

- Working directory: `D:/pybullet_test/MRTA_GA/DRL`
- Outer repository HEAD: `0935ad8a723057084be190e31c6be4733ce3cb62`
- Python: `D:/pybullet_test/.venv/Scripts/python.exe`
- Baseline DRL suite: 172 passed in 33.39 s
- Baseline combined suite: 544 passed, 1 skipped, 3 subtests passed in 74.20 s

No reset, checkout, clean, commit, push, or upload was performed.

## B. Why a method-independent corpus was required

The existing 31-state stress corpus came from SA-OI-ALNS development
trajectories. It remains useful for regression but could favor the state
distribution of one search method. The final common evaluator must also serve
Adapted HGA and Adapted WAG, which may produce different assignments, routes,
directions, and DEADLOCK topologies. Budget selection therefore used a new
corpus generated directly from the formal structural domain, without running a
search algorithm or selecting candidates by Cmax or recovery outcome.

## C. Direct sampler scientific definition

`METHOD_INDEPENDENT_DIRECT_SAMPLER_V1` is implemented in the existing
`scripts/profile_scheduler.py` development driver.

1. Parents are sorted by stable parent ID.
2. Existing `generate_y_split_patterns` supplies every legal Y candidate.
3. For a whole-eligible parent, WHOLE versus Y_SPLIT is sampled first with
   probability 0.5 each; only after Y_SPLIT is selected is one legal Y
   candidate sampled uniformly. Candidate multiplicity cannot bias pattern
   kind. A mandatory-Y parent samples only a legal Y candidate.
4. Each generated block is assigned uniformly to a robot from the existing
   formal eligibility predicate.
5. Each robot's block IDs are sorted and then permuted with the versioned
   seeded PRNG.
6. Existing canonicalization is the authority. Deterministic rejection
   sampling excludes non-canonical mandatory-child adjacency; optional legal
   collapse remains allowed.
7. Directions are sampled independently and uniformly from 0/1 for the final
   canonical routes. Direction DP is not called.

Every attempt uses `random.Random` with a stable seed derived from sampler
policy ID, master seed, family, N, and sample ordinal. The stored master seed is
20260929. Tests replace ALNS, initial portfolio, direction DP, and initializer
entry points with failing stubs and confirm that direct sampling does not call
them.

## D. Sampling attempts and strata

The predeclared Stage A policy was executed completely:

- families: load_skew, spatial_cluster, handover_heavy, interference_stress;
- sizes: N=20, 50, 100;
- 12 strata total;
- 256 direct canonical attempts per stratum;
- retain the first eight unique baseline-DEADLOCK states per stratum;
- baseline scheduler only, with no recovery;
- no Cmax selection and no B32/B64/B128 result available during collection.

Stage A produced at least 48 states, so Stage B was not entered.

## E. Per-stratum accepted counts

| Family | N20 | N50 | N100 |
|---|---:|---:|---:|
| load_skew | 8 | 8 | 8 |
| spatial_cluster | 8 | 8 | 8 |
| handover_heavy | 8 | 8 | 8 |
| interference_stress | 8 | 8 | 8 |

All strata executed 256 attempts. Raw baseline DEADLOCK counts were,
respectively by table order, 59/56/68, 55/47/48, 59/65/43, and 60/82/61.
Only the first eight unique states in each stratum were retained.

## F. Final corpus size

The frozen corpus contains 96 unique baseline-DEADLOCK states, exceeding the
minimum 31. Its schema is
`F4_METHOD_INDEPENDENT_CALIBRATION_CORPUS_V1`, `development_only=true`,
`future_validation_test_ood_used=false`, and
`method_source=DIRECT_FORMAL_DOMAIN_SAMPLING`.

Path:
`data/development/f4_method_independent_calibration_v1.json`

## G. Reproducibility and round-trip

Each row stores canonical solution payload and hash, explicit directions,
complete ScientificConfig and hash, family, N, master seed, per-attempt seed,
sample ordinal, baseline diagnostics, wait-for graph, and baseline canonical
schedule hash. The exact dedup key is canonical solution hash plus directions
plus scientific-config hash. SHA-256 is only the external label; exact JSON-key
membership prevents a digest collision from merging states.

The corpus was serialized, reread, and reconstructed through
`CanonicalSolution` for all 96 entries before any recovery replay. JSON
round-trip was exact. Frozen file SHA-256:

```text
ca5cffefe29764650aae90206bd2598d84378236807fa26a02ae34b2a5a2e5e3
```

## H–J. B32, B64, and B128 results

The already optimized production-equivalent prepared evaluator was used. Each
budget pass covered all 96 rows before the next budget began. Every FEASIBLE
recovery was independently certified.

| Budget | FEASIBLE | DEADLOCK | Certified FEASIBLE | Coverage of B128 |
|---:|---:|---:|---:|---:|
| 32 | 6 | 90 | 6 | 100% |
| 64 | 6 | 90 | 6 | 100% |
| 128 | 6 | 90 | 6 | 100% |

The recovered set did not change between budgets.

## K. Monotonicity

There were zero status-monotonicity violations and zero Cmax non-worsening
violations. Every B32 FEASIBLE remained FEASIBLE at B64 and B128. Every B64
FEASIBLE remained FEASIBLE at B128. Shared recovered schedules never had a
worse Cmax under a larger budget.

## L. Certification

All 18 budget-specific FEASIBLE results (6 at each budget) passed the
independent certifier. No NUMERIC_FAILURE was swallowed or relabeled. Bounded
failures remained DEADLOCK, not INFEASIBLE.

## M. Runtime

End-to-end time includes baseline preparation, baseline dispatch, and recovery.
The hard release gate uses only the 32 N=100 rows.

| Budget | all p50 | all p95 | N100 p50 | N100 p95 | N100 mean | N100 max | rollout mean/median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.479 s | 3.877 s | 2.553 s | 5.147 s | 2.643 s | 6.299 s | 25.67 / 32 |
| 64 | 0.850 s | 5.009 s | 3.295 s | 5.466 s | 3.488 s | 6.049 s | 36.10 / 34 |
| 128 | 0.889 s | 7.158 s | 4.496 s | 9.714 s | 4.745 s | 10.355 s | 45.48 / 34 |

B32 and B64 satisfy the eight-second N100 p95 gate. B128 does not, but it is
only the fixed coverage reference.

## N–O. Predeclared rule and selected budget

`R32=6`, `R64=6`, and `R128=6`. The predeclared rule selects the smallest
budget with at least 90% of B128 certified recovery coverage and N100 p95 no
greater than 8.0 s. B32 provides 100% coverage and p95=5.147 s, so the unique
selection is B32.

The selection did not use the historical ALNS corpus, search Cmax, or future
data. No B256 or larger experiment was run.

## P. Active scope and hash

B32 was retained, so no V1.2 is created.

```text
ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1
scope_hash = 5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc
reference_policy_id = FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1
deadlock_rollout_budget = 32
```

Historical hashes remain unchanged:

- V1: `8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`
- V1.1: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`

F1, F2, F3, baseline priority, discrepancy order, WAIT, continuous conflict,
same-rail order, canonicalization, certifier, and recovered schedule selection
did not change.

## Q. Historical 31-state corpus regression

The old ALNS-derived corpus was replayed only after the final budget had been
selected.

| Budget | FEASIBLE | DEADLOCK | Certified | p50 | p95 |
|---:|---:|---:|---:|---:|---:|
| 32 | 9 | 22 | 9 | 1.151 s | 2.932 s |
| 64 | 15 | 16 | 15 | 1.182 s | 4.078 s |
| 128 | 15 | 16 | 15 | 1.191 s | 4.072 s |

There were no monotonicity or certification failures. This cross-distribution
diagnostic did not alter B32.

## R. E1–E4

| Case | Status | Source | Cmax | Rollouts | Certified |
|---|---|---|---:|---:|---|
| E1 | FEASIBLE | BASELINE | 2.0 | 0 | yes |
| E2 | FEASIBLE | BASELINE | 14.16227766056838 | 0 | yes |
| E3 | FEASIBLE | LIMITED_DISCREPANCY_RECOVERY | 16.63619980225593 | 1 | yes |
| E4 | DEADLOCK | BASELINE | n/a | 2 | n/a |

E1/E2 baseline outputs remain unchanged. E4 is not relabeled INFEASIBLE.

## S. Q1–Q6

All six cases completed, produced certified final schedules, and reproduced
exactly under fixed-seed replay.

| Case | Initial Cref | Final Cref | Iterations | Nref | Recovery calls |
|---|---:|---:|---:|---:|---:|
| Q1 assignment | 12.84 | 7.1499999999999995 | 1 | 4 | 0 |
| Q2 route | 11.545657757864154 | 11.468135899374975 | 8 | 32 | 2 |
| Q3 direction | 7.105576065114842 | 6.271416626834708 | 20 | 80 | 0 |
| Q4 optional Y | 6.0 | 4.0 | 10 | 40 | 10 |
| Q5 interference/WAIT | 7.105576065114842 | 6.271416626834708 | 12 | 48 | 12 |
| Q6 LNS basin | 19.35791211412712 | 3.557181856472053 | 4 | 16 | 0 |

Development exact values remain development evidence; no formal OPT claim was
made.

## T. Development-family smoke

The final active scope was used for 4 families × 3 sizes × 3 seeds = 36 runs,
with two fixed real iterations per run.

- initialization success: 36/36;
- status COMPLETED: 36/36;
- final certified: 36/36;
- total iterations: 72;
- total Nref: 278;
- C4 FEASIBLE: 265;
- baseline DEADLOCK records: 31;
- recovered: 9;
- remaining DEADLOCK: 22;
- LNS attempts/C4: 1152/46;
- NUMERIC_FAILURE: 0.

All baseline DEADLOCK records occurred in handover-heavy. Budget selection was
already frozen and was not changed from these results.

## U. N100/30s usability gate

Configuration: handover-heavy, N=100, current SearchConfig defaults, B32,
time_limit=30 s.

| Seed | Actual / overshoot | Iterations | Nref | Initial / best Cmax | Cmax@1 / @5 / @30 | Scheduler / repair / certifier |
|---:|---:|---:|---:|---:|---|---:|
| 20260928 | 34.009 / 4.009 s | 7 | 24 | 2210.9259 / 2210.9259 | null / 2210.9259 / 2210.9259 | 19.398 / 11.853 / 1.245 s |
| 20260929 | 36.288 / 6.288 s | 5 | 20 | 2210.9259 / 2210.9259 | null / 2210.9259 / 2210.9259 | 25.770 / 8.257 / 1.155 s |
| 20260930 | 33.660 / 3.660 s | 5 | 16 | 2210.9259 / 2210.9259 | null / 2210.9259 / 2210.9259 | 23.279 / 8.477 / 0.750 s |

All three initialized successfully, completed at least two real iterations,
and ended certified. `Cmax@1` is null because initialization had not completed
and certified by that deadline. Later completion was not backfilled into the
one-second checkpoint. No ALNS operator or budget was changed because Cmax did
not improve.

## V. Full regression

Final commands:

```powershell
# cwd D:\pybullet_test\MRTA_GA\DRL
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
# 180 passed in 28.42 s

# cwd D:\pybullet_test\MRTA_GA
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
# 552 passed, 1 skipped, 3 subtests passed in 61.96 s
```

Coverage includes sampler determinism and stable order, search-free operation,
kind-before-candidate sampling, canonicalization, DEADLOCK-only admission,
stable exact dedup, JSON round-trip, B32/B64/B128 caps and monotonicity,
certification, selection-rule edge cases, runtime-gate failure, historical
scope hashes, deadline-safe checkpoints, and all prior difficult regressions.

## W. Phase 3 authorization

The method-independent corpus exceeds the minimum size; collection was frozen
before recovery replay; all three permitted budgets were actually tested;
coverage, runtime, monotonicity, and certification gates passed; B32 was chosen
by the predeclared rule; V1/V1.1 history remains replayable; E1–E4, Q1–Q6,
family smoke, N100 usability, and full suites passed. Therefore Phase 3 is
authorized.

The next and only phase is **Phase 3 — Adapted HGA / Adapted WAG common-model**.
This release did not implement HGA, WAG, learning, datasets, formal benchmark,
LB_LP, or final exact validation.
