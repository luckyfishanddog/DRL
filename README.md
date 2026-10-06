# Multi-Robot Weld Allocation and Sequencing

`DRL` contains the reference evaluator and deterministic SA-OI-ALNS/HGA/WAG backbones for multi-robot weld allocation and sequencing. `FORMAL_SCOPE_V2 = ACTIVE`; V1 and V1.1 remain frozen historical scopes. V2 preserves the method-independent B32 evaluator and uses a shared WHOLE/Y/X legal catalog. The scientific core remains isolated from legacy PPO solver code; data intake reads the already-frozen platform Excel outputs through an optional adapter.

The authoritative plan is [多机器人焊缝分配与排序实验方案](docs/多机器人焊缝分配与排序实验方案.md). Phase handoffs describe the implementation at their respective dates, not competing plans.

Latest research decision — [Phase 3-Z V2 validation](docs/PHASE3Z_V2_COMMON_MODEL_VALIDATION_20261006.md): `PHASE3Z_EXECUTION_STATUS = FAIL`, with 180/180 native runs completed and 179/180 final schedules certified. HGA returned no certified solution on `seed_2095476608::g44_w085`, solver seed `20261015`; after initially stopping at run 165, the unchanged runner resumed the remaining 15 runs on the user's explicit instruction. All 15 resumed final schedules were certified; the failed run was neither retried nor replaced. Numeric failures and scheduler/certifier mismatches were zero; full regression passed before/after both batches (314 tests). The competition gate is `NOT_EVALUABLE`, V2 finalization is `BLOCKED`, V2_VALIDATION is `CONSUMED`, and traditional tuning is `NOT_CLOSED` but validation-driven tuning remains prohibited. Next: **resolve this execution blocker only**, not Phase 4-0 or neural training. See the [frozen protocol](data/manifests/PHASE3Z_V2_VALIDATION_PROTOCOL_V1.json) and [full 180-run evidence including the failure](data/validation/phase3z_v2_common_model_validation_v1.json). The independent clean clone retains verified source evidence; main-directory copies are result mirrors for manual upload. No GitHub upload was performed.

The [Phase 3-YR search-access closure](docs/PHASE3YR_V2_SEARCH_ACCESS_CLOSURE_20261006.md) remains PASS, and the initial [Phase 3-Y smoke](docs/PHASE3Y_FORMAL_SCOPE_V2_CORE_CLOSURE_20261005.md) remains historical FAIL. Phase 3-YR repaired generic ALNS family shortlists and WAG pattern-access scheduling, passed 54/54 certified runs and regressions before/after V2 activation. `SA_OI_ALNS_V2`, `ADAPTED_HGA_V2`, and `ADAPTED_WAG_VNS_V2` retain the shared WHOLE/Y/X catalog and unchanged active V2 scientific definition. The 31 V2_TRAIN_POOL workbooks were not used by Phase 3-Z, and the true 15 ID_TEST workbooks remain sealed. Historical Phase 3-X/X2 evidence is unchanged; partial validation does not establish statistical superiority.

## Current scope

Implemented:

- immutable scientific configuration and reproducibility hash;
- parent weld, welding block, WHOLE/X_SPLIT/Y_SPLIT, route, operation, schedule, candidate, and metric models;
- deterministic Y-handover candidates and frozen weighted-median `x_up`/`x_low`;
- historical V1/V1.1 fail-closed X exclusion and formal V2 finite-catalog X membership validation;
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
- bounded deterministic initial portfolio with the V3 four-robot rail-monotone balanced bootstrap and capped feasibility-direction evaluation;
- state-aware balanced hard-budget generation for the historical seven atomic moves plus an ablatable same-rail `TWO_OPT_STAR` route-tail exchange;
- parent-level random/critical-load destroy, greedy/regret-2 repair, and reproducible adaptive operator-pair weights;
- one unified complete-candidate pipeline for atomic and repaired LNS solutions;
- bounded certified schedule-aware single-flip direction refinement inside the total reference budget;
- makespan-first C0-C4 screening, Kdp direction rerank, Kref reference evaluation, and seeded SA acceptance;
- split initialization/search scheduler timings, wall-clock overshoot records, deadline-safe anytime checkpoints, and micro gap decomposition;
- fail-closed, lazy-Excel intake for the two observed frozen PPO family schemas, with per-instance geometry/formal validation, hashes, duplicate analysis, and deterministic smoke selection;
- workbook-level Phase 3 role freeze with DEVELOPMENT_CONSUMED/TRAIN_POOL/VALIDATION/ID_TEST isolation;
- paper-aligned `ADAPTED_HGA_V1` and `ADAPTED_WAG_VNS_V1`, each using its own native initialization/search and the shared direction/evaluator/certifier path;
- adversarial and regression tests.

Not implemented in this research: formal full-scope exact, LB_LP, MLP/GAT/rankers, PPO, ranker datasets, or formal experiments. `TWO_OPT_STAR` is implemented and tested but remains OFF in the proposed backbone and is available only for ablation. V2 enables finite-catalog optional X_SPLIT search. Formal evaluation retains deterministic limited-discrepancy recovery with 32 complete alternative rollouts. The implemented ALNS and baseline layers are development comparison backbones, not a final ID_TEST result.

## Development and formal scope

`DEVELOPMENT_SCOPE = EXACT_Y_SCOPE_CURRENT_SEMANTICS`: WHOLE plus all current legal Y_SPLIT patterns for whole-eligible parents, mandatory Y_SPLIT otherwise, no X_SPLIT. It is suitable for Phase 2B development, micro comparisons, debugging, and profiling. Current terminal/empty-route behavior and no default repair remain unchanged.

Development optimum notation is `Cmax_OPT_Y_CURRENT`, qualified by the current dispatch+ESS enumeration domain. It is not final full-problem `Cmax_OPT`. A limit-hit result is not optimal. Tests and certification of returned schedules do not establish general continuous-time optimality; the plan's `EXACT_SCHEDULER_VALIDITY_GATE` requires a dominance proof or explicitly limited independent validation before final exact claims.

`FORMAL_SCOPE_V1` is frozen historical evidence. Its F4 uses 16 popped-prefix states and is depth-censored on large schedules. Its identity and evaluator remain replayable.

`FORMAL_SCOPE_V1_1 = FROZEN HISTORICAL`: it excludes optional X_SPLIT and retains its published F1–F4 semantics. `FORMAL_SCOPE_V2 = ACTIVE`: WHOLE/Y/X catalog membership and explicit X assignment/processing/shared-point policies extend V1.1; terminal, empty-route, continuous interference, objective and B32 recovery remain unchanged. See [FORMAL_SCOPE_V2](docs/FORMAL_SCOPE_V2.md).

V1 hash: `8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`. Historical V1.1 hash: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`. Active V2 hash: `16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599`. See the [historical scope contract](docs/FORMAL_SCOPE_V1.md), [V1 gate evidence](docs/FORMAL_SCOPE_GATE_V1_20260928.md), and [V1.1 F4 closure](docs/FORMAL_SCOPE_V1_1_F4_CLOSURE_20260929.md).

The final release budget was selected only from the method-independent direct-sampling corpus: 12 development strata, 256 attempts per stratum, and the first eight unique baseline-DEADLOCK states per stratum produced 96 frozen states. B32/B64/B128 each recovered 6/96 certified schedules; B32 therefore provides 100% of B128 coverage and its N100 end-to-end p95 was 5.147 s, satisfying the predeclared 8 s gate. The selected budget remains 32 and no V1.2 is created. The older 31-state ALNS-derived corpus remains an external regression only: B32/B64/B128 recovered 9/15/15 certified schedules. See the [method-independent corpus](data/development/f4_method_independent_calibration_v1.json), [historical stress corpus](data/development/f4_deadlock_stress_corpus.json), and [final release handoff](docs/FINAL_F4_RELEASE_CALIBRATION_20260929.md).

F4 is final for Phase 3 and later common-model comparisons. HGA/WAG/ALNS behavior and future TEST results must not be used to retune the evaluator budget. Evaluator performance optimization and F4 calibration are closed.

This is task-level weld allocation, sequencing, direction, and theoretical coordination. Deployment/retract/parking transitions require lower-level validation; no full physical parking or 3D collision-free execution claim is made. `mrta_exact` remains development-only. Formal results must be reevaluated and certified; development results are not relabelled.

`FINAL_F4_RELEASE_STATUS = PASS` and `PHASE3_AUTHORIZED = YES` remain unchanged. Phase 3-0 found 96 frozen PPO workbooks containing 2942 valid unique platform instances and froze `PPO_DATASET_MANIFEST_V1`. Phase 3-1 completed the Adapted HGA/WAG implementations. Phase 3-2A then closed the comparison-timing bug, froze a six-workbook DEVELOPMENT_CONSUMED diagnostic set, completed 54 native and 18 shared-incumbent runs, applied the predeclared 10% rule as CASE B, and completed 30 post-fix runs. The shared-incumbent diagnostic showed a remaining end-to-end search-process gap; it was not pure neighborhood-mechanism isolation because HGA still created native population members and WAG still created native assignments/routes. DEVELOPMENT means were 3634.651 for V3/OFF and 3655.896 for V3/ON, so the Phase 3-2B proposed method is `SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_OFF`; `TWO_OPT_STAR` remains implemented and tested only for ablation. This is mechanism evidence, not a superiority claim. `PHASE3_FAIRNESS_BACKBONE_STATUS = PASS`, `PHASE3_2B_AUTHORIZED = YES`, and the next phase is VALIDATION common-model comparison; ID_TEST remains sealed. `FORMAL_SCOPE_V1_1` and B32 were not changed.

Phase 3-2B is complete under frozen protocol hash `3d797f7d9ddb0c57b751bdaff50e451c3f67b54410023bf9953b1a89229ac2d2`: 180/180 native 60 s trajectories completed, all had certified Cmax@60, numeric/certifier mismatches and iteration-limit terminations were zero, and all normal terminations were `TIME_LIMIT`. The predefined backbone gate did not pass: median R was 1.070941, but only 6/12 instances met R<=1.10 and the LARGE-tier median was 1.353317. Therefore `PHASE3_VALIDATION_STATUS = PASS`, `DETERMINISTIC_BACKBONE_STATUS = NEEDS_CANDIDATE_POOL_AUDIT`, `TWO_OPT_STAR_STATUS = ABLATION_ONLY`, and `ID_TEST_STATUS = SEALED`. No VALIDATION-driven operator change is authorized; next is Phase 4-0 Candidate-Pool Oracle Recall Audit.

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
- `src/mrta_search/initialization.py`: deterministic bounded V2/V3 initial portfolio, rail-monotone balancing, bounded feasibility-direction fallback, and historical rail-serial fallback.
- `src/mrta_search/neighborhood.py`: balanced atomic proposal and cheap screening, including ablatable same-rail `TWO_OPT_STAR`.
- `src/mrta_search/lns.py`: complete-candidate identity, parent destroy, bounded repair, and adaptive operator state.
- `src/mrta_search/pipeline.py`: unified atomic/LNS Kdp/Kref evaluator pipeline, SA engine, and micro decomposition.
- `src/mrta_search/stats.py`: structured counters, timings, invariants, and anytime records.
- `src/mrta_data/ppo_instances.py`: optional frozen PPO workbook discovery, schema inspection, loading, validation, metrics, manifests, and smoke-set selection; it never generates, filters, clips, or moves welds.
- `src/mrta_data/phase3_split.py`: deterministic solver-independent workbook descriptors, role assignment, validation, and fail-closed ID_TEST access check.
- `src/mrta_baselines/common.py`: shared baseline result, timing/accounting, direction DP, formal evaluator, certifier, and official-metric path.
- `src/mrta_baselines/hga.py`: Adapted HGA population, route crossover, M1--M6 VND, optional-Y mutation, diversity, and survival.
- `src/mrta_baselines/wag_vns.py`: Adapted WAG three-stage route construction plus MOVE/SWAP/LNS VNS and optional-Y toggle.
- `scripts/audit_ppo_instances.py`: local read-only PPO inventory and manifest CLI.
- `scripts/run_ppo_smoke.py`: manifest-driven SA-OI-ALNS compatibility smoke CLI.
- `scripts/run_phase3_baseline_smoke.py`: Phase 3 smoke plus resumable native/common-seed/post-fix DEVELOPMENT_CONSUMED diagnostics with 5/30/60 checkpoints and unified telemetry.
- `scripts/run_phase3_validation.py`: dedicated fail-closed Phase 3-2B VALIDATION-only selector, protocol freezer, resumable three-method runner, summary builder, and mechanical release decision.
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

`PPO_PHASE3_DATA_SPLIT_V1` freezes all 96 workbooks before baseline performance runs: 23 DEVELOPMENT_CONSUMED, 43 TRAIN_POOL, 15 VALIDATION, and 15 untouched ID_TEST. Historical directory names do not override these roles, and every sheet inherits its workbook role. The first 30-second three-method smoke used only three distinct DEVELOPMENT_CONSUMED workbooks (N=25/55/85) and produced 27/27 certified runs; it is development evidence, not a formal comparison.

```powershell
$env:PYTHONPATH='src'
& 'D:\pybullet_test\.venv\Scripts\python.exe' scripts\run_phase3_baseline_smoke.py --ppo-root 'D:\pybullet_test\MRTA_GA\ppo' --budget 30 --source-commit-label '<LOCAL_REVISION_LABEL>'
```

See [baseline source mapping](docs/PHASE3_BASELINE_SOURCE_MAPPING_20260930.md), [Phase 3-1 implementation handoff](docs/PHASE3_HGA_WAG_BASELINE_IMPLEMENTATION_20260930.md), [split manifest](data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json), and [development smoke](data/development/phase3_baseline_smoke_v1.json).

Phase 3-2A evidence is in [the fairness/backbone handoff](docs/PHASE3_FAIR_TIMING_AND_BACKBONE_DIAGNOSTIC_20260930.md), [the six-instance diagnostic manifest](data/manifests/PPO_PHASE3_DIAGNOSTIC_SET_V1.json), and [the complete development artifact](data/development/phase3_fairness_diagnostic_v1.json). The diagnostic used only frozen `DEVELOPMENT_CONSUMED` roles; historical directory names such as `data/VALIDATION` or `data/ID_TEST` did not grant VALIDATION/ID_TEST access.

## Formal API

```python
from mrta_reference import FORMAL_SCOPE_V2, reference_schedule_formal, certify_schedule
from mrta_search import run_sa_oi_alns_v2

schedule = reference_schedule_formal(solution, config, scope=FORMAL_SCOPE_V2, orientations=directions)
certificate = certify_schedule(solution, schedule, config, scope=FORMAL_SCOPE_V2)
# A publishable FORMAL_RESULT rechecks a clean standalone
# luckyfishanddog/DRL worktree; an enclosing repository HEAD is rejected.
result = run_sa_oi_alns_v2(parents, config, formal_result=True)
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
