# MRTA Reference Evaluator

`DRL` contains the frozen Phase 1.1 scientific core for multi-robot weld allocation and sequencing. It is intentionally isolated from the repository's legacy V9/V10/PPO experiments.

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
- adversarial and regression tests.

Not included in this phase: initial-solution heuristics, bounded candidate pools, SA-OI-ALNS, GNN/GAT/rankers, PPO, HGA, WAG+VNS, exact MRTA, lower bounds, datasets, or formal experiments.

## Environment and installation

The verified environment is Python 3.11. The project has no third-party runtime dependency.

From `D:\pybullet_test\MRTA_GA\DRL`:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pip install --no-build-isolation --no-deps -e .
```

Run the Phase 1.1 suite:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Run the repository and Phase 1.1 suites together from the repository root:

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
- `tests/`: boundary, adversarial, oracle, and deterministic regression tests.
- `docs/`: scientific plan and detailed AI handoff report.

## Scientific ambiguities

The evaluator deliberately does not choose rules for the following unresolved scientific questions:

1. Optional X-split candidate enumeration. A retained X split requires an explicit `XSplitValidator`; there is no production default.
2. Bounded deadlock repair trigger, priority alternatives, search order, tie breaking, budgets, and acceptance rule. The default scheduler performs no repair.
3. Terminal occupancy after a robot's final POST. Currently the robot stops participating in interference after that explicit POST ends; no terminal WAIT is inserted.
4. Parking/occupancy for an empty robot route. Currently it generates no operation, has completion `0.0`, and does not participate in interference.

These current terminal/empty-route behaviors describe the implementation; they are not claims that the scientific questions have been resolved.

## Reproducibility identity

`ScientificConfig.scientific_hash` is SHA-256 over a canonical JSON object containing only scientific parameter values. It excludes source-control state, machine identity, timestamps, and test counts. The default Phase 1.1 configuration hash is:

```text
791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e
```

See `docs/第一阶段公共科学模型与Reference_Evaluator_AI交接报告_20260914.md` for the closure details and test evidence.
