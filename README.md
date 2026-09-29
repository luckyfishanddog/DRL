# Multi-Robot Weld Allocation and Sequencing

`DRL` contains the Phase 1.1 reference evaluator, Phase 2A exact micro validation backbone, Phase 2B-1 bounded SA-OI neighborhood search, Phase 2B-1.5 scheduler performance closure, and the Phase 2B-2 minimal deterministic SA-OI-ALNS backbone for multi-robot weld allocation and sequencing. `FORMAL_SCOPE_V1_1 = FINAL ACTIVE SCOPE FOR PHASE 3`; V1 remains a reproducible historical scope superseded before formal experiments. The final method-independent F4 calibration retained B32, so scientific Phase 3 scope is authorized. The scientific core remains isolated from legacy PPO solver code; Phase 3 data intake now reads only the already-frozen platform Excel outputs through an optional adapter.

The authoritative plan is [多机器人焊缝分配与排序实验方案](docs/多机器人焊缝分配与排序实验方案.md). Phase handoffs describe the implementation at their respective dates, not competing plans.

## Current scope

Implemented:

- immutable scientific configuration and reproducibility hash;
- parent weld, welding block, WHOLE/X_SPLIT/Y_SPLIT, route, operation, schedule, candidate, and metric models;
- deterministic Y-handover candidates and frozen weighted-median `x_up`/`x_low`;
- historical fail-closed X validator support; formal scope explicitly excludes X_SPLIT;
- canonicalization and deterministic solution identity;
- open-route direction dynamic programming;
- analytic continuous-time interference and same-rail non-passing checks;
- deterministic reference list scheduler with explicit WAIT operations and DEADLOCK diagnostics;
- semantics-equivalent optimized reference scheduler with the retained legacy implementation as a differential oracle;
- independent schedule certifier;
- independent tiny dispatch oracle for two active robots;
- `mrta_exact`: Y-only pattern/assignment/route/full-direction micro enumeration, with exhaustive and branch-and-bound modes;
- independent coordination scheduler using dispatch interleaving enumeration and earliest-safe-start placement;
- analytical LB0, exact/reference comparison, and explicit enumeration limits;
- initial-orientation-constrained exact empty-travel direction DP;
- bounded four-strategy deterministic initial portfolio with capped reference evaluation;
- state-aware balanced hard-budget generation for seven atomic neighborhood moves;
- parent-level random/critical-load destroy, greedy/regret-2 repair, and reproducible adaptive operator-pair weights;
- one unified complete-candidate pipeline for atomic and repaired LNS solutions;
- bounded certified schedule-aware single-flip direction refinement inside the total reference budget;
- makespan-first C0-C4 screening, Kdp direction rerank, Kref reference evaluation, and seeded SA acceptance;
- split initialization/search scheduler timings, wall-clock overshoot records, deadline-safe anytime checkpoints, and micro gap decomposition;
- fail-closed, lazy-Excel intake for the two observed frozen PPO family schemas, with per-instance geometry/formal validation, hashes, duplicate analysis, and deterministic smoke selection;
- adversarial and regression tests.

Not implemented in this research: formal full-scope exact, LB_LP, adapted HGA/WAG, MLP/GAT/rankers, PPO, ranker datasets, or formal experiments. `TWO_OPT_STAR` and X_SPLIT search remain inactive. Formal evaluation enables deterministic limited-discrepancy recovery with 32 complete alternative rollouts. The implemented ALNS layer is the minimal development backbone, not a formal frozen algorithm or tuned benchmark.

## Development and formal scope

`DEVELOPMENT_SCOPE = EXACT_Y_SCOPE_CURRENT_SEMANTICS`: WHOLE plus all current legal Y_SPLIT patterns for whole-eligible parents, mandatory Y_SPLIT otherwise, no X_SPLIT. It is suitable for Phase 2B development, micro comparisons, debugging, and profiling. Current terminal/empty-route behavior and no default repair remain unchanged.

Development optimum notation is `Cmax_OPT_Y_CURRENT`, qualified by the current dispatch+ESS enumeration domain. It is not final full-problem `Cmax_OPT`. A limit-hit result is not optimal. Tests and certification of returned schedules do not establish general continuous-time optimality; the plan's `EXACT_SCHEDULER_VALIDITY_GATE` requires a dominance proof or explicitly limited independent validation before final exact claims.

`FORMAL_SCOPE_V1` is frozen historical evidence. Its F4 uses 16 popped-prefix states and is depth-censored on large schedules. Its identity and evaluator remain replayable.

`FORMAL_SCOPE_V1_1 = ACTIVE`: F1 still excludes optional X_SPLIT; F2/F3 retain `TASK_HORIZON_RELEASE_V1`; only F4 changes to `FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1`. Baseline FEASIBLE output is untouched. Baseline DEADLOCK invokes at most 32 deterministic complete alternative rollouts, each ending in FEASIBLE or DEADLOCK regardless of template depth. Bx remains a spatial/load prior, not a split domain.

V1 hash: `8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`. Active V1.1 hash: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`. See the [historical scope contract](docs/FORMAL_SCOPE_V1.md), [V1 gate evidence](docs/FORMAL_SCOPE_GATE_V1_20260928.md), and [V1.1 F4 closure](docs/FORMAL_SCOPE_V1_1_F4_CLOSURE_20260929.md).

The final release budget was selected only from the method-independent direct-sampling corpus: 12 development strata, 256 attempts per stratum, and the first eight unique baseline-DEADLOCK states per stratum produced 96 frozen states. B32/B64/B128 each recovered 6/96 certified schedules; B32 therefore provides 100% of B128 coverage and its N100 end-to-end p95 was 5.147 s, satisfying the predeclared 8 s gate. The selected budget remains 32 and no V1.2 is created. The older 31-state ALNS-derived corpus remains an external regression only: B32/B64/B128 recovered 9/15/15 certified schedules. See the [method-independent corpus](data/development/f4_method_independent_calibration_v1.json), [historical stress corpus](data/development/f4_deadlock_stress_corpus.json), and [final release handoff](docs/FINAL_F4_RELEASE_CALIBRATION_20260929.md).

F4 is final for Phase 3 and later common-model comparisons. HGA/WAG/ALNS behavior and future TEST results must not be used to retune the evaluator budget. Evaluator performance optimization and F4 calibration are closed.

This is task-level weld allocation, sequencing, direction, and theoretical coordination. Deployment/retract/parking transitions require lower-level validation; no full physical parking or 3D collision-free execution claim is made. `mrta_exact` remains development-only. Formal results must be reevaluated and certified; development results are not relabelled.

`FINAL_F4_RELEASE_STATUS = PASS` and `PHASE3_AUTHORIZED = YES` remain unchanged. Phase 3-0 found 96 frozen PPO workbooks containing 2942 valid unique platform instances and froze `PPO_DATASET_MANIFEST_V1`. Phase 3-0.5 closed the N=55 initialization failure with one deterministic rail-serial feasibility bootstrap that is called only after the normal portfolio fails. The solver-independent 20-instance development set achieved 20/20 certified initializations, and the N=10/55/90 V2 smoke achieved 3/3 certified initializations and search entry. Therefore `PPO_DATA_INTAKE_STATUS = PASS`, `PPO_ALNS_INITIALIZATION_STATUS = PASS`, and Phase 3-1 Adapted HGA/WAG is authorized. `FORMAL_SCOPE_V1_1` and B32 were not changed.

Reference statuses are `FEASIBLE`, `DEADLOCK`, `INFEASIBLE`, and `NUMERIC_FAILURE`. Only FEASIBLE has a reference Cmax; DEADLOCK is a failure of the deterministic scheduling policy, not mathematical infeasibility. Numeric failures are not normal negative training examples.

## Environment and installation

The verified environment is Python 3.11. The scientific core has no third-party runtime dependency. PPO Excel intake is optional and lazy-loads `openpyxl` only when workbook functions are called.

From `D:\pybullet_test\MRTA_GA\DRL`:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pip install --no-build-isolation --no-deps -e .
```

Run the complete DRL suite (Phase 1.1 + Phase 2A + Phase 2B):

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Run the repository and DRL suites together from the repository root:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
```

## Module overview

- `src/mrta_reference/scope.py`: unique immutable formal scope, canonical scope hash, and RunScientificIdentity.
- `src/mrta_reference/provenance.py`: fail-closed DRL root/remote/HEAD verification and deterministic scientific source-tree hash.
- `src/mrta_reference/model.py`: dataclasses, enums, scientific configuration, and config hash.
- `src/mrta_reference/geometry.py`: eligibility, split geometry, frozen handover centers, direction DP, and analytic conflict geometry.
- `src/mrta_reference/solution.py`: canonicalization and official metrics.
- `src/mrta_reference/candidate.py`: complete atomic candidate moves and deterministic replay/deduplication.
- `src/mrta_reference/scheduler.py`: legacy/optimized reference scheduling, development profiling, WAIT insertion, and deadlock diagnostics/hook.
- `src/mrta_reference/dispatch_recovery.py`: immutable dispatch snapshots, deterministic branch traces, stable state identity, and complete-rollout recovery frontier.
- `src/mrta_reference/certifier.py`: independent reconstruction and certification.
- `src/mrta_reference/oracle.py`: independent tiny exhaustive dispatch oracle and reference comparison.
- `src/mrta_exact/lower_bounds.py`: analytical LB0.
- `src/mrta_exact/scheduler.py`: independent dispatch+ESS coordination enumeration.
- `src/mrta_exact/solver.py`: Y-only exact micro backbone, full directions, limits, and comparison metrics.
- `src/mrta_search/direction.py`: initial-feasibility-constrained direction DP and bounded schedule-aware refinement.
- `src/mrta_search/initialization.py`: deterministic bounded initial portfolio plus one-call rail-serial feasibility bootstrap.
- `src/mrta_search/neighborhood.py`: balanced seven-move raw proposal and cheap screening.
- `src/mrta_search/lns.py`: complete-candidate identity, parent destroy, bounded repair, and adaptive operator state.
- `src/mrta_search/pipeline.py`: unified atomic/LNS Kdp/Kref evaluator pipeline, SA engine, and micro decomposition.
- `src/mrta_search/stats.py`: structured counters, timings, invariants, and anytime records.
- `src/mrta_data/ppo_instances.py`: optional frozen PPO workbook discovery, schema inspection, loading, validation, metrics, manifests, and smoke-set selection; it never generates, filters, clips, or moves welds.
- `scripts/audit_ppo_instances.py`: local read-only PPO inventory and manifest CLI.
- `scripts/run_ppo_smoke.py`: manifest-driven SA-OI-ALNS compatibility smoke CLI.
- `scripts/profile_phase2b1.py`: deterministic development-only N=20/50/100 smoke driver.
- `scripts/profile_scheduler.py`: formal gate regression, method-independent direct sampling, final B32/B64/B128 calibration, historical replay, evaluator profiling, N=100 usability gates, Q1–Q6, and development-family smoke driver.
- `tests/`: boundary, adversarial, oracle, and deterministic regression tests.
- `docs/`: scientific plan and detailed AI handoff report.

## Phase 3 frozen PPO data

Paper-comparison geometry for SA-OI-ALNS, Adapted HGA, Adapted WAG, MLP, and GAT must come from the same `PPO_DATASET_MANIFEST_V1`-derived frozen Excel instances. Runtime generation through `layout_packing`, `generate_weld_instance_from_source`, `get_welds`, `get_welds_from_excel`, or the candidate-generation main flow is forbidden for the paper benchmark. Existing synthetic families, E1-E4, Q1-Q6, tiny exact fixtures, and F4 corpora remain regression, mechanism, stress, exact-micro, and historical evaluator evidence only.

Raw PPO Excel files are local and are not tracked or versioned with this project. Provide their root with `--ppo-root` or `MRTA_PPO_ROOT`; manifests contain only PPO-relative paths and hashes. The observed V9 family sheets store frozen coordinates in metres (`x1..z2`, `length_m`); their `META.input_units=mm` records the pre-packing source workbook units, not the frozen sheet units.

For local smoke replay, exact byte copies of only the three deterministic smoke workbooks may be kept under the ignored directory `data/local/ppo_smokeset/`. This local cache preserves the original PPO-relative `data/...` paths and raw SHA-256 values, so the existing manifest and smoke set can be used with `--ppo-root data/local/ppo_smokeset`. The directory is excluded by `.gitignore` and must not be uploaded.

```powershell
$env:PYTHONPATH='src'
& '<PYTHON>' scripts\audit_ppo_instances.py --ppo-root '<PATH_TO_PPO>'
& '<PYTHON>' scripts\run_ppo_smoke.py --ppo-root '<PATH_TO_PPO>' --include-30 --source-commit-label '<LOCAL_REVISION_LABEL>'
& '<PYTHON>' scripts\run_ppo_smoke.py --ppo-root 'data/local/ppo_smokeset' --smokeset data/manifests/PPO_PHASE3_SMOKESET_V1.json --source-commit-label '<LOCAL_REVISION_LABEL>'
```

The frozen intake manifest is [PPO_DATASET_MANIFEST_V1](data/manifests/PPO_DATASET_MANIFEST_V1.json). [PPO_PHASE3_SMOKESET_V1](data/manifests/PPO_PHASE3_SMOKESET_V1.json) remains historical. The current initialization-development and compatibility selections are [PPO_INIT_BOOTSTRAP_DEVSET_V1](data/manifests/PPO_INIT_BOOTSTRAP_DEVSET_V1.json) and [PPO_PHASE3_SMOKESET_V2](data/manifests/PPO_PHASE3_SMOKESET_V2.json). Dataset identity is separate from `FORMAL_SCOPE_V1_1.scope_hash`.

The Phase 3 main operating range is `10 <= N <= 90`: 2893 of 2942 valid unique instances are in range. The remaining 49 (1.6655%) stay valid in the manifest as large-scale stress/out-of-main-range data and do not block Phase 3. Future TRAIN/VALIDATION/TEST assignment must isolate complete workbooks/generation-seed families; every workbook listed as development-consumed in `PPO_INIT_BOOTSTRAP_DEVSET_V1` is excluded from untouched TEST.

## Formal API

```python
from mrta_reference import FORMAL_SCOPE_V1_1, reference_schedule_formal, certify_schedule
from mrta_search import run_bounded_sa_oi

schedule = reference_schedule_formal(solution, config, orientations=directions)
certificate = certify_schedule(solution, schedule, config, scope=FORMAL_SCOPE_V1_1)
# A publishable FORMAL_RESULT rechecks a clean standalone
# luckyfishanddog/DRL worktree; an enclosing repository HEAD is rejected.
result = run_bounded_sa_oi(
    parents, config, scope=FORMAL_SCOPE_V1_1, formal_result=True
)
```

Explicit scope binds initialization, C4, direction refinement, and final certification to the same evaluator; a development callback is rejected in a formal run. Historical `reference_schedule`/slow/optimized retain `DEVELOPMENT_NO_REPAIR_V1`. `RunScientificIdentity` is available on `result.stats.scientific_identity` and contains scope/config identity plus repository id, commit, source-tree hash, dirty state, and commit verification. Nested-workspace smoke must explicitly opt into unverified provenance and remains `development_only=true`; it cannot emit a formal result.

Development gate commands (repository root; stdout JSON lines, no dataset):

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-v1-1 replay
& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-v1-1 calibration
& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-v1-1 quality
& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-v1-1 smoke
```

The fixed recovery budget does not guarantee recovery. Every attempted V1.1 alternative is a complete continuation, so template depth cannot prevent it from reaching FEASIBLE or DEADLOCK; the bounded set of alternatives can still miss a feasible dispatch. DEADLOCK remains a policy outcome, not mathematical infeasibility. Tests, E1–E4, Q1–Q6 and the family smoke cover the active policy.

## Reproducibility identity

`ScientificConfig.scientific_hash` is SHA-256 over a canonical JSON object containing only scientific parameter values. It excludes source-control state, machine identity, timestamps, and test counts. The default Phase 1.1 configuration hash is:

```text
791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e
```

See the [Phase 1.1 handoff](docs/第一阶段公共科学模型与Reference_Evaluator_AI交接报告_20260914.md) and [Phase 2A handoff](docs/第二阶段Exact_Micro与LB0交接报告_20260927.md) for implementation evidence. The scientific config hash covers numeric parameters only. The independent scope hash and evaluator policy are defined in `src/mrta_reference/scope.py`; source commit is separate metadata.

Phase 2B-1 evidence is recorded in [bounded SA-OI backbone handoff](docs/第三阶段Phase2B1_Bounded_SA_OI_Backbone交接报告_20260928.md). Development profiling can be reproduced with:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts\profile_phase2b1.py --sizes 20 50 100 --budgets 0.2 1 5 --seed 20260928
```

Phase 2B-1.5 evidence is recorded in [reference scheduler performance closure handoff](docs/Phase2B1_5_Reference_Scheduler_Performance_Closure_20260928.md). Its JSON profiler can be reproduced with:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts\profile_scheduler.py --seed 20260928
```

Phase 2B-2 evidence is recorded in [minimal real ALNS closure handoff](docs/Phase2B2_Minimal_Real_ALNS_Closure_20260928.md). The 1/5/30 s development-only run is reproduced with:

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts\profile_phase2b1.py --sizes 20 50 100 --budgets 1 5 30 --seed 20260928
```
