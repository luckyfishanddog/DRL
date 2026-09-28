# Multi-Robot Weld Allocation and Sequencing

`DRL` contains the Phase 1.1 reference evaluator, Phase 2A exact micro validation backbone, and Phase 2B-1 bounded SA-OI neighborhood-search backbone for multi-robot weld allocation and sequencing. Phase 2B-1 is not a complete SA-OI-ALNS: destroy/repair and adaptive destroy/repair operator selection are intentionally deferred to Phase 2B-2. This research is isolated from the repository's legacy V9/V10/PPO experiments.

The authoritative plan is [多机器人焊缝分配与排序实验方案](docs/多机器人焊缝分配与排序实验方案.md). Phase handoffs describe the implementation at their respective dates, not competing plans.

## Current scope

Implemented:

- immutable scientific configuration and reproducibility hash;
- parent weld, welding block, WHOLE/X_SPLIT/Y_SPLIT, route, operation, schedule, candidate, and metric models;
- deterministic Y-handover candidates and frozen weighted-median `x_up`/`x_low`;
- fail-closed formal validation for retained optional X splits;
- canonicalization and deterministic solution identity;
- open-route direction dynamic programming;
- analytic continuous-time interference and same-rail non-passing checks;
- deterministic reference list scheduler with explicit WAIT operations and DEADLOCK diagnostics;
- independent schedule certifier;
- independent tiny dispatch oracle for two active robots;
- `mrta_exact`: Y-only pattern/assignment/route/full-direction micro enumeration, with exhaustive and branch-and-bound modes;
- independent coordination scheduler using dispatch interleaving enumeration and earliest-safe-start placement;
- analytical LB0, exact/reference comparison, and explicit enumeration limits;
- initial-orientation-constrained exact empty-travel direction DP;
- deterministic bounded initial construction with one fallback;
- balanced hard-budget generation for seven atomic neighborhood moves;
- makespan-first C0-C4 screening, Kdp direction rerank, Kref reference evaluation, and seeded SA acceptance;
- structured search instrumentation, anytime checkpoints, and micro gap decomposition;
- adversarial and regression tests.

Not implemented in this research: formal full-scope exact, LB_LP, complete SA-OI-ALNS destroy/repair, adaptive operator selection, adapted HGA/WAG, MLP/GAT/rankers, PPO, ranker datasets, or formal experiments. `TWO_OPT_STAR`, X_SPLIT search, direction refinement, and deadlock repair remain inactive.

## Development and formal scope

`DEVELOPMENT_SCOPE = EXACT_Y_SCOPE_CURRENT_SEMANTICS`: WHOLE plus all current legal Y_SPLIT patterns for whole-eligible parents, mandatory Y_SPLIT otherwise, no X_SPLIT. It is suitable for Phase 2B development, micro comparisons, debugging, and profiling. Current terminal/empty-route behavior and no default repair remain unchanged.

Development optimum notation is `Cmax_OPT_Y_CURRENT`, qualified by the current dispatch+ESS enumeration domain. It is not final full-problem `Cmax_OPT`. A limit-hit result is not optimal. Tests and certification of returned schedules do not establish general continuous-time optimality; the plan's `EXACT_SCHEDULER_VALIDITY_GATE` requires a dominance proof or explicitly limited independent validation before final exact claims.

`FORMAL_SCOPE_V1` is **not frozen**. `FORMAL_SCOPE_GATE` must resolve the four questions below before adapted HGA/WAG common-model work, final ranker labels, and formal VALIDATION/ID_TEST/OOD. Formal exact must use the same final feasible set. Development results cannot be silently relabelled as formal results.

Reference statuses are `FEASIBLE`, `DEADLOCK`, `INFEASIBLE`, and `NUMERIC_FAILURE`. Only FEASIBLE has a reference Cmax; DEADLOCK is a failure of the deterministic scheduling policy, not mathematical infeasibility. Numeric failures are not normal negative training examples.

## Environment and installation

The verified environment is Python 3.11. The project has no third-party runtime dependency.

From `D:\pybullet_test\MRTA_GA\DRL`:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pip install --no-build-isolation --no-deps -e .
```

Run the complete DRL suite (Phase 1.1 + Phase 2A):

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Run the repository and DRL suites together from the repository root:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
```

## Module overview

- `src/mrta_reference/model.py`: dataclasses, enums, scientific configuration, and config hash.
- `src/mrta_reference/geometry.py`: eligibility, split geometry, frozen handover centers, direction DP, and analytic conflict geometry.
- `src/mrta_reference/solution.py`: canonicalization and official metrics.
- `src/mrta_reference/candidate.py`: complete atomic candidate moves and deterministic replay/deduplication.
- `src/mrta_reference/scheduler.py`: operation templates, reference scheduling, WAIT insertion, and deadlock diagnostics/hook.
- `src/mrta_reference/certifier.py`: independent reconstruction and certification.
- `src/mrta_reference/oracle.py`: independent tiny exhaustive dispatch oracle and reference comparison.
- `src/mrta_exact/lower_bounds.py`: analytical LB0.
- `src/mrta_exact/scheduler.py`: independent dispatch+ESS coordination enumeration.
- `src/mrta_exact/solver.py`: Y-only exact micro backbone, full directions, limits, and comparison metrics.
- `src/mrta_search/direction.py`: initial-feasibility-constrained direction DP.
- `src/mrta_search/initialization.py`: deterministic initial construction and bounded fallback.
- `src/mrta_search/neighborhood.py`: balanced seven-move raw proposal and cheap screening.
- `src/mrta_search/pipeline.py`: Kdp/Kref evaluator pipeline, SA engine, and micro decomposition.
- `src/mrta_search/stats.py`: structured counters, timings, invariants, and anytime records.
- `scripts/profile_phase2b1.py`: deterministic development-only N=20/50/100 smoke driver.
- `tests/`: boundary, adversarial, oracle, and deterministic regression tests.
- `docs/`: scientific plan and detailed AI handoff report.

## Scientific ambiguities

The evaluator deliberately does not choose rules for the following unresolved scientific questions:

1. Optional X-split candidate enumeration. A retained X split requires an explicit `XSplitValidator`; there is no production default.
2. Bounded deadlock repair trigger, priority alternatives, search order, tie breaking, budgets, and acceptance rule. The default scheduler performs no repair.
3. Terminal occupancy after a robot's final POST. Currently the robot stops participating in interference after that explicit POST ends; no terminal WAIT is inserted.
4. Parking/occupancy for an empty robot route. Currently it generates no operation, has completion `0.0`, and does not participate in interference.

These current terminal/empty-route behaviors describe the implementation, including the absence of both TCP and rail-order occupancy after final POST and for empty routes. They are development scope boundaries, not final scientific conclusions. The plan assigns them to F1–F4 in FORMAL_SCOPE_GATE; Phase 2B may proceed without inventing rules for them.

## Reproducibility identity

`ScientificConfig.scientific_hash` is SHA-256 over a canonical JSON object containing only scientific parameter values. It excludes source-control state, machine identity, timestamps, and test counts. The default Phase 1.1 configuration hash is:

```text
791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e
```

See the [Phase 1.1 handoff](docs/第一阶段公共科学模型与Reference_Evaluator_AI交接报告_20260914.md) and [Phase 2A handoff](docs/第二阶段Exact_Micro与LB0交接报告_20260927.md) for implementation evidence. The existing scientific config hash covers numeric parameters only; it does not mean FORMAL_SCOPE_V1 has been frozen.

Phase 2B-1 evidence is recorded in [bounded SA-OI backbone handoff](docs/第三阶段Phase2B1_Bounded_SA_OI_Backbone交接报告_20260928.md). Development profiling can be reproduced with:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts\profile_phase2b1.py --sizes 20 50 100 --budgets 0.2 1 5 --seed 20260928
```
