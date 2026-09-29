# FORMAL_SCOPE_V1_1 Evaluator Performance Closure

Date: 2026-09-29

## Release decision

```text
FORMAL_V1_1_PERFORMANCE_STATUS = FAIL
ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1
PHASE3_AUTHORIZED = NO
UNIQUE_BLOCKER = B64 adds 6 certified recoveries over B32 (Rule B)
```

The evaluator implementation performance closure succeeded, but the predeclared
B32/B64 release rule failed. B32 remains unchanged in `FORMAL_SCOPE_V1_1`; this
run did not modify the scope, try B128, or authorize Phase 3.

## A. Baseline repository and tests

The accessible workspace is a nested, dirty development workspace. The outer
repository state was recorded without resetting or cleaning user work.

- Outer repository root: `D:/pybullet_test/MRTA_GA`
- DRL working directory: `D:/pybullet_test/MRTA_GA/DRL`
- Outer HEAD: `0935ad8a723057084be190e31c6be4733ce3cb62`
- Outer remote: `https://github.com/luckyfishanddog/MRTA.git`
- Provenance: `UNVERIFIED_SOURCE_PROVENANCE`, development only
- Python: `D:/pybullet_test/.venv/Scripts/python.exe`
- Baseline DRL suite: 169 passed in 46.94 s
- Baseline combined suite: 541 passed, 1 skipped, 3 subtests passed in 98.41 s

No Git reset, checkout, clean, commit, push, or upload was performed.

## B. Pre-optimization profile

The current V1.1 implementation was first retained and profiled before the
production path was changed. The profile covered the known former 2048-prefix
recoverable case and three deterministic N100 corpus cases at B32.

Aggregate `cProfile` evidence for those four cases:

- 455,253,321 calls in 194.918 s under profiling
- `earliest_safe_start_optimized`: 190.643 s cumulative
- `forbidden_start_intervals`: 175.800 s cumulative
- `_polygon_projection`: 157.240 s cumulative
- relevant-index query: 3.057 s cumulative
- state canonical JSON and SHA-256: approximately 0.28 s cumulative
- template validation: approximately 0.10 s cumulative
- commit/tuple append: approximately 0.34 s cumulative

The dominant cost was repeated exact continuous geometry projection across
scientifically identical rollout prefixes. State hashing, frontier operations,
template validation, and tuple append were not material hotspots.

## C. Identified hotspots

The measured hotspots, in descending practical importance, were:

1. repeated exact forbidden-interval projection for identical template, fixed
   operation, and ready-time inputs;
2. repeated ESS scans for identical complete relevant-operation sets;
3. repeated conflict validation for identical operation pairs;
4. rebuilding the relevant-fixed index when continuing from branch snapshots.

Full state JSON/hash, heap operations, forced-decision tuple construction, and
fixed-operation tuple append were measured and left unchanged because they did
not justify a scientific-state representation rewrite.

## D. Implemented optimizations

The implementation changes are confined to the existing scheduler, recovery,
profiler, and scheduler-equivalence test files.

- Added `PreparedDispatchProblem`, shared by the V1.1 baseline and all bounded
  continuation rollouts in one formal evaluator call.
- Shared validated templates, the remaining-processing suffix, `ScientificConfig`,
  directions, and static call-local lookup state.
- Added exact call-local caches for forbidden intervals, operation conflicts,
  and complete ESS results.
- Added cloneable `_RelevantFixedIndex` snapshots and an exact prefix-keyed
  snapshot cache, eliminating repeated full `add()` reconstruction for a branch
  prefix after the first continuation.
- Added optional profiler counters/timers for validation, ESS/cache activity,
  state construction, index rebuilds, trace/snapshot preparation, frontier and
  dedup work, rollout time, and bounded memory counts.
- Extended `scripts/profile_scheduler.py` with `profile`, `differential`,
  `n100`, and `marginal` performance-closure modes.
- Kept fixed operations as immutable tuples and kept the original canonical
  JSON/SHA-256 identity because profiling showed that changing them would add
  risk without material benefit.

No destroy/repair operator, ALNS budget, physics tolerance, dispatch priority,
scope field, or certification rule changed.

## E. Why the optimizations preserve semantics

All caches are local to one prepared formal evaluator call. Their keys contain
the complete immutable scientific inputs needed by the cached computation:

- forbidden interval: candidate template, full fixed operation, and ready time;
- conflict result: both complete operations;
- ESS result: candidate template, ready time, and the complete ordered tuple of
  relevant fixed operations;
- relevant index snapshot: the complete immutable fixed-operation prefix.

`ScientificConfig` and directions are fixed by the owning prepared problem.
Python dictionary hashes are used only for lookup; dictionary equality compares
the exact structural keys after a hash match. A digest collision therefore
cannot merge different scientific states. The formal dispatch state identity,
canonical JSON, choice order, WAIT construction, analytic conflict functions,
and final schedule selection remain unchanged.

Profiling is optional and does not participate in any scheduler decision.
Profile-on/profile-off equality is covered by regression.

## F. Slow/oracle architecture

The pre-optimization behavior remains available as explicit oracle paths:

- `_optimized_dispatch_outcome` retains the previous per-rollout preparation
  and runtime behavior;
- `_limited_discrepancy_dispatch_recovery_slow` retains the previous V1.1
  recovery behavior;
- `_prepared_dispatch_outcome` and
  `_limited_discrepancy_dispatch_recovery_optimized` implement the optimized
  production path.

`reference_schedule_from_templates_formal` uses the prepared optimized path only
for `FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1`. The historical V1 bounded
state policy remains on its existing path.

## G. 31-state B32 slow-versus-optimized differential

Command:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts/profile_scheduler.py --performance-closure differential
```

Results:

- 31/31 baseline result and trace identical
- 31/31 final status identical
- recovered set identical: 9 FEASIBLE, 22 DEADLOCK
- all nine recovered schedules have identical Cmax and complete canonical
  scientific schedule output
- recovery counters, discrepancy counts, branch counts, exhaustion semantics,
  WAIT operations, directions, robot completion, wait-for graph, and diagnostics
  are identical
- scientific differentials: 0
- certified feasible schedules: 9/9

The timer in this command conservatively includes baseline plus recovery on both
paths. Since the absolute and speedup criteria pass for the larger end-to-end
quantity, the recovery-only quantity also satisfies the stated limits.

## H. E1-E4 regression

Final optimized B32 results:

| Case | Status | Source | Cmax | Rollouts | Certified |
|---|---|---|---:|---:|---|
| E1 | FEASIBLE | BASELINE | 2.0 | 0 | yes |
| E2 | FEASIBLE | BASELINE | 14.16227766056838 | 0 | yes |
| E3 | FEASIBLE | LIMITED_DISCREPANCY_RECOVERY | 16.63619980225593 | 1 | yes |
| E4 | DEADLOCK | BASELINE | n/a | 2 | n/a |

E1 and E2 retain the baseline schedule. E4 remains DEADLOCK and is not relabeled
INFEASIBLE.

## I. Q1-Q6 formal regression

All quality fixtures completed, were certified, and reproduced exactly on a
fixed-seed replay.

| Case | Formal Cref | Iterations | Nref | Recovered calls |
|---|---:|---:|---:|---:|
| Q1 assignment | 7.1499999999999995 | 1 | 4 | 0 |
| Q2 route | 11.468135899374975 | 8 | 32 | 2 |
| Q3 direction | 6.271416626834708 | 20 | 80 | 0 |
| Q4 optional Y | 4.0 | 10 | 40 | 10 |
| Q5 interference/WAIT | 6.271416626834708 | 12 | 48 | 12 |
| Q6 LNS basin | 3.557181856472053 | 4 | 16 | 0 |

The development exact values were reported separately. No formal-optimality
claim or cross-scope exact gap was introduced.

## J. B32 performance before and after

The complete current-machine differential produced:

| Metric | Slow oracle | Optimized | Speedup |
|---|---:|---:|---:|
| p50 | 12.754 s | 1.219 s | 10.46x |
| p95 | 39.053 s | 3.074 s | 12.70x |

The median of per-case speedups was 10.08x. Both the absolute target
(`p50 <= 3.0 s`, `p95 <= 8.0 s`) and the minimum 3x p50/p95 speedup condition
were satisfied without reducing B32.

## K. Memory and cache bounds

For the N100 B32 corpus runs, observed maxima were:

- peak frontier/branch states: 27
- peak dedup entries: 27
- cached relevant-index snapshots: 15
- fixed-operation prefix nodes materialized by one state: 397
- ESS cache entries: 4,778
- forbidden-interval cache entries: 22,158
- conflict cache entries: 24,414

All caches belong to one evaluator call and are released with its prepared
problem. Their growth is bounded by the baseline plus at most B32 complete
rollouts and their encountered operations. No global scheduler-state cache was
introduced.

## L. N100 5 s and 30 s runs

Configuration: `handover_heavy`, N=100, seeds 20260928/20260929/20260930,
`FORMAL_SCOPE_V1_1`, B32, and unchanged `SearchConfig` defaults except the
requested wall-clock limit and large iteration ceiling.

Five-second runs:

| Seed | Actual s | Overshoot s | Iterations | Nref | Scheduler s | Repair s |
|---:|---:|---:|---:|---:|---:|---:|
| 20260928 | 6.857 | 1.857 | 1 | 2 | 5.274 | 1.100 |
| 20260929 | 7.026 | 2.026 | 1 | 4 | 4.150 | 2.110 |
| 20260930 | 8.671 | 3.671 | 1 | 2 | 6.163 | 2.077 |

Thirty-second runs:

| Seed | Actual s | Overshoot s | Iterations | Nref | Scheduler s | Repair s |
|---:|---:|---:|---:|---:|---:|---:|
| 20260928 | 30.275 | 0.275 | 5 | 16 | 18.584 | 9.588 |
| 20260929 | 37.710 | 7.710 | 5 | 20 | 26.854 | 8.492 |
| 20260930 | 34.509 | 4.509 | 5 | 16 | 23.910 | 8.707 |

All six final schedules were certified. Initial and best Cmax were
2210.9259259259275 for these development runs. Cmax@1 was unavailable because
initial certification completed after one second; Cmax@5 and Cmax@30 were both
2210.9259259259275. The existing anytime test confirms that an improvement
completed after a checkpoint is not backfilled into that checkpoint.

The 30 s target of multiple complete search iterations was met. The remaining
overshoot is bounded by completing an evaluator that started before the
deadline; checkpoint scientific semantics remain unchanged.

## M. B32/B64 marginal release result

This diagnostic was run only after the optimized B32 implementation and its
differential behavior were frozen.

Command:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts/profile_scheduler.py --performance-closure marginal
```

| Budget | FEASIBLE | DEADLOCK | Total rollouts | p50 s | p95 s |
|---:|---:|---:|---:|---:|---:|
| B32 | 9 | 22 | 766 | 1.164 | 2.828 |
| B64 | 15 | 16 | 898 | 1.224 | 4.272 |

B64 added six certified recoveries. The additional identities begin with:
`22d033706041`, `2f7e2fd6c5c3`, `47de0c47dfb8`, `6316b715122e`,
`879e09318be0`, and `dfe4aadbcc29`.

Every B32-feasible case remained feasible at B64 and its Cmax did not worsen.
Nevertheless, the predeclared rule is unambiguous: six is at least two, so
Rule B applies and `EVALUATOR_SCOPE_RELEASE = FAIL`.

## N. Active scope decision

The scientific contract was not changed:

- active scope: `FORMAL_SCOPE_V1_1`
- scope hash: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`
- reference policy: `FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1`
- rollout budget: B32
- baseline priority, recovery order, recovered-schedule ordering, F1, F2, F3,
  interference, same-rail, WAIT, and certification semantics: unchanged

The historical `FORMAL_SCOPE_V1` hash remains
`8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`.
Its E1-E4 16-prefix replay passed, including certified E3 recovery and E4
remaining DEADLOCK.

## O. Full regression

Final commands and results:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
# 172 passed in 29.59 s

& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
# 544 passed, 1 skipped, 3 subtests passed in 70.63 s
```

The development-family smoke covered four families, N=20/50/100, and three
fixed ALNS seeds: 36/36 runs completed and 36/36 final schedules were certified.
The handover-heavy N100 runs exercised baseline DEADLOCK, bounded recovery,
remaining DEADLOCK, direction refinement, LNS, and certification.

Tracked-file inspection reported no `__pycache__` or `.pyc` entries. Test cache
creation was disabled. Existing unrelated outer-workspace changes were neither
modified nor cleaned.

## P. Remaining bottlenecks

The evaluator now meets the performance target. Exact ESS work remains the
largest evaluator component, especially for N100 prefixes with many relevant
operations, but it no longer prevents multiple 30 s search iterations.
ALNS repair consumed approximately 8.5-9.6 s in the N100 30 s runs and was left
unchanged as required.

The release blocker is not runtime and not an implementation differential. It
is the frozen marginal rule: B64 changes the recoverable set from 9 to 15 cases.

## Q. Phase 3 authorization

Phase 3 is not authorized by this gate. The only blocker is:

```text
B64_NEW_CERTIFIED_RECOVERIES = 6/31
RULE_B_THRESHOLD = >= 2/31
EVALUATOR_SCOPE_RELEASE = FAIL
```

No HGA, WAG, MLP, GAT, PPO, LB_LP, exact optimization, ranker dataset, formal
TRAIN/VALIDATION/TEST/OOD run, new destroy/repair operator, X_SPLIT, or parking
model was implemented.
