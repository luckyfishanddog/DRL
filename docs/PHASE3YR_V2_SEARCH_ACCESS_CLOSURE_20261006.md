# Phase 3-YR — V2 Search-Access Closure — 2026-10-06

```text
PHASE3YR_EXECUTION_STATUS = PASS
FORMAL_SCOPE_V2_STATUS = CLOSED
ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V2
COMMON_DOMAIN_V2_STATUS = PASS
ALNS_V2_ACCESS_STATUS = PASS
HGA_V2_ACCESS_STATUS = PASS
WAG_V2_ACCESS_STATUS = PASS
V2_SEARCH_ACCESS_STATUS = CLOSED
V2_VALIDATION_SET_STATUS = FROZEN_UNTOUCHED
PHASE3Z_V2_VALIDATION_AUTHORIZED = YES
ID_TEST_STATUS = SEALED
NEXT_PHASE = Phase 3-Z — V2 Common-Model Validation
```

## Result and scope

The two reported Phase 3-Y blockers are repaired. SA-OI-ALNS_V2 now covers the available decision families at C2 and uses one exploitation plus one deterministic family-exploration slot at C4. ADAPTED_WAG_VNS_V2 proposes at most one legal pattern transition in every normal VNS iteration and evaluates the transitioned assignment before heavy MOVE/SWAP/LNS route work.

The original six instances, three seeds, 30 s wall-clock including initialization, checkpoints and access gate were retained. All 54 runs completed with independently certified final schedules, zero numeric failures, zero scheduler/certifier mismatches and matching catalog hashes. ALNS and WAG each had zero ZERO_X_ACCESS positive instances; HGA had one, below the unchanged allowance of two. Full regression passed before and after activation. Phase 3-Y remains historical FAIL.

These are development access-repair results, not algorithm performance rankings or V2 validation results.

## Frozen identities and evidence

- V1: `8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`
- V1.1: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`
- V2: `16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599`
- source Phase 3-Y protocol: `6efbbed56a75cc58876182f51513347bf9ab8855f00f0285d5547d5c0de593be`
- Phase 3-YR protocol: `e2206a3d20a2dddf53912920abbf4ca62844a669db322a7cce51ce3655b99876`
- V2 roles: `7a80372eb072b10da3d8044eb8bb330a29ede93fe16c806b1aa0d58bee968a5a`
- V2 validation: `2989d8fe15330883a617504d4cc51a6a6cfc547aa9c9813853b6df8fcb2245fe`
- execution scientific tree before activation: `37ec6f09f3d7c6a06e280139219db9336e23b162296805b0b4d4dd1940295d8c`
- activated scientific tree: `533372e63cef39a1ee967b8cae8afa9dac3b1a259da08f6199d6fc0943f332a7`

The protocol was written before telemetry and access implementation changes and before any YR performance run. Its protected-file evidence verifies that the historical Phase 3-Y protocol, artifact and handoff, both V2 data manifests, HGA implementation, geometry/catalog, model, scheduler, certifier, solution, direction refinement, LNS and neighborhood generation remained byte-identical. Only the ACTIVE alias changed in the scientific scope module; all three scope hashes are unchanged.

The local source label remains `e8d32f02c155eaf2cae895481917421c2fe42eda`. DRL is nested in the unrelated MRTA checkout, so the label is not asserted to be a verified standalone DRL HEAD. This is local development evidence. No GitHub write, commit or upload was performed.

## ALNS repair and unchanged budgets

The policies are `V2_C2_PATTERN_FAMILY_STRATIFIED_V1` and `V2_C4_FAMILY_EXPLORE_EXPLOIT_V1`.

Classification reads the explicit candidate transition source:

- STRUCTURAL: relocation, swap, 2-opt and repaired LNS without a pattern transition;
- TARGET_WHOLE: split deactivation;
- TARGET_Y: an explicit transition targeting Y_SPLIT;
- TARGET_X: an explicit transition targeting X_SPLIT.

A structural candidate retaining an incumbent X pattern stays STRUCTURAL. Classification does not inspect whether the complete resulting solution happens to contain X.

C2 takes the original cheap-score best candidate from each nonempty family in fixed STRUCTURAL/WHOLE/Y/X order, then fills remaining slots using the original global cheap ranking. C4 slot 1 is the original global rerank best; slot 2 rotates through the same families using `(solver_seed + zero_based_iteration) % 4`, skipping families without an unselected feasible candidate. Slots are not duplicated and empty families reserve nothing.

The original cheap/rerank scores remain unchanged. M=64, Kdp=8, Kref=2, Kref_total=4, M_lns=16, direction refinement, SA temperature/cooling and destroy/repair remain unchanged. The measured maximum Kdp was 8 and maximum total reference calls per iteration was 4. TWO_OPT_STAR remains OFF. Historical V1/V3 scope paths keep global shortlists; the historical Phase 3-Y execution explicitly retains its old access policies.

## ALNS complete decision-family funnel

Counts across 18 YR ALNS runs:

| Stage | STRUCTURAL | TARGET_WHOLE | TARGET_Y | TARGET_X |
|---|---:|---:|---:|---:|
| generated | 13,121 | 476 | 2,952 | 1,371 |
| constructed | 9,840 | 354 | 1,734 | 1,292 |
| cheap_valid | 7,723 | 354 | 1,734 | 1,292 |
| C2_selected | 1,370 | 91 | 482 | 297 |
| direction_evaluated | 1,370 | 91 | 482 | 297 |
| direction_feasible | 1,330 | 91 | 482 | 295 |
| C4_selected | 321 | 27 | 141 | 71 |
| reference_evaluated | 321 | 27 | 141 | 71 |
| certified | 117 | 19 | 104 | 40 |
| accepted | 59 | 15 | 76 | 24 |
| global_best_update | 20 | 9 | 22 | 7 |

Generated counts explicit atomic descriptors or LNS repair attempts. Constructed atomic counts are successful screened canonical candidates; constructed LNS counts successful repairs before cross-source deduplication. Cheap_valid counts the complete unique candidates entering cheap ranking. Direction/refinement reference calls remain within the existing total budget; the funnel counts base shortlist candidates rather than inventing extra candidates for subsequent direction refinement.

The old X/Y telemetry is also retained. Its semantics count candidates whose resulting solution contains X/Y, including retained incumbent patterns, so those counts must not be confused with the TARGET_X transition funnel above.

| Legacy-compatible X metric | Phase 3-Y | Phase 3-YR |
|---|---:|---:|
| proposals | 1,300 | 1,371 |
| constructed / cheap feasible | 1,299 | 2,462 |
| C2 / direction evaluated | 19 | 593 |
| reference evaluated | 2 | 150 |
| constructed → C2 | 1.46% | 24.09% |
| C2 → reference | 10.53% | 25.30% |

The old artifact did not record source-decision families. Its C2 count is recovered from direction evaluations, which occur once per C2 candidate. YR's unambiguous TARGET_X conversion is 297/1292 = 22.99% from constructed to C2, and 71/297 = 23.91% from C2 to C4/reference. These 71 explicit X transitions and access on all 18 ALNS trajectories show that the repair addresses shortlist starvation rather than relying on one favorable seed. No oracle, WAIT, length, x-span or blocking-proxy feature entered either policy.

## WAG repair and funnel

Policy: `V2_PER_ITERATION_PATTERN_FIRST_SEEDED_FAMILY_V1`.

Every normal V2 iteration proposes at most one generic catalog transition. The full family order WHOLE/Y/X begins at `(solver_seed + zero_based_iteration) % 3`; unavailable families are skipped cyclically. Parent/pattern selection remains deterministic catalog order plus seeded selection. A constructed assignment goes directly through CommonBaselineEvaluator before heavy VNS work. All construction, direction, scheduler and certifier work shares the original wall-clock; heavy operators run only while time remains. Native initialization and paper MOVE/SWAP/LNS/route machinery retain their parameters.

The V1 optional-Y trigger remains every third iteration, and its ordering is unchanged. WAG n=10, p=0.5, Imax=500 and all factorial, variant and route-combination bounds remain unchanged. HGA's source and configuration are unchanged.

| WAG metric across 18 runs | Phase 3-Y | Phase 3-YR |
|---|---:|---:|
| X proposals | 9 | 18 |
| X-containing candidates constructed | 67 | 54 |
| X-containing candidates reference evaluated | 49 | 45 |
| iterations min / median / max | 1 / 3 / 6 | 1 / 2 / 9 |
| positive instances with zero X reference | 4 | 0 |

The overall reference count need not increase: repeated evaluations of an already-X incumbent are included in the legacy count. YR records 56 transition trace entries for exactly 56 normal iterations. Its explicit direct X-transition funnel is 18 proposals → 18 constructed assignments → 18 reference evaluations. The I9 and I12 seed 20261006 runs each start from TARGET_X and receive direct reference access during their first iteration, demonstrating access even with only one iteration. Other seeds may still end without an X proposal under the shared deadline; the unchanged gate aggregates all three seeds.

## Mechanical access and certification gate

Each cell sums X reference evaluations across the same three seeds. I2 is excluded from the positive gate.

| Method | I3 | I5 | I6 | I9 | I12 | ZERO_X_ACCESS count | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| SA_OI_ALNS_V2 | 15 | 19 | 40 | 5 | 8 | 0 | PASS |
| ADAPTED_HGA_V2 | 20 | 5 | 13 | 2 | 0 | 1 | PASS |
| ADAPTED_WAG_VNS_V2 | 5 | 2 | 29 | 1 | 1 | 0 | PASS |

HGA's I12 zero access is reported explicitly; it passes the same allowance of at most two zero-access positive instances. Wall-clock runs can execute different iteration counts even when source and seed are unchanged, so the rerun is not claimed to be bitwise identical to the original time-limited run.

54/54 runs and final certifications completed. Numeric failures = 0; scheduler/certifier mismatches = 0; catalog hashes equal the Phase 3-Y per-instance hashes for all three methods. Final X retention and Cmax ranking were not gates. On harmful-X control I2, ALNS and WAG retained no X in any final solution; HGA retained one X parent in one seed, which is allowed and certified. No control result was forced.

## First-access times and wall-clock

Seconds from the same run start, including initialization; ranges below are min / median / max over non-null values.

| Method / stage | Runs observed | Times (s) |
|---|---:|---|
| ALNS first X proposal | 18 / 18 | 0.98 / 2.67 / 7.03 |
| ALNS first X C2 | 18 / 18 | 1.66 / 3.95 / 9.09 |
| ALNS first X reference | 18 / 18 | 1.95 / 6.35 / 20.46 |
| WAG first X proposal | 12 / 18 | 0.69 / 5.10 / 25.19 |
| WAG first X reference | 12 / 18 | 0.69 / 5.10 / 25.19 |
| HGA first X reference | 12 / 18 | 3.33 / 10.53 / 26.41 |

WAG has no C2 stage; its first_x_c2_time is null. HGA's proposal-time instrumentation was not added to its frozen code, so first_x_proposal_time and first_x_c2_time are null. Historical Phase 3-Y has no first-access timestamps and they were not fabricated.

Access is not exclusively at 29.x seconds. Some later transitions are late (for example an additional I3 WAG X proposal at 28.98 s), but that run's first X reference was earlier. I9/I12 WAG access remains dependent on the seed-family rotation under low iteration counts; all per-run nulls and trace entries remain visible in the artifact.

| Method | Runtime min / median / max (s) | Maximum overshoot (s) |
|---|---|---:|
| ALNS | 30.13 / 30.94 / 36.94 | 6.94 |
| HGA | 30.00 / 30.14 / 33.18 | 3.18 |
| WAG | 30.01 / 30.24 / 31.31 | 1.31 |

The requested budget remains 30 s. Overshoot is reported as the existing non-preemptive evaluation cost, not an increased budget. Checkpoints preserve the existing no-future-backfill timing semantics.

## Regression and activation

Current full regression before repair: 256 passed in 59.78 s. The 31 new access tests cover explicit four-family classification, structural retained-X/LNS cases, C2 coverage/global fill/caps, C4 exploitation/seed rotation/empty-family fallback/no duplicates, V1 trajectory equality, WAG per-iteration/pattern-first/V1 scheduling, shared-deadline consumption, frozen protocol/HGA/catalog/hash identities and rejection of validation/train/test before workbook loading.

After smoke and before activation: 287 passed in 70.70 s.

The first post-activation regression found three historical tests using implicit ACTIVE while asserting/certifying V1.1. Their calls were explicitly pinned to V1.1 without weakening the original X-exclusion, task-horizon or certification-failure assertions. This failed attempt is preserved in the artifact's regression_attempts. No scheduler, certifier or scientific change was needed.

Final post-activation regression: 287 passed in 75.70 s. ACTIVE is V2. The Phase 3 baseline/validation runners remain explicitly V1.1, Phase 3-X/X2 remain experimental-X where historically defined, and Phase 3-Y retains its original FAIL artifact, protocol and handoff.

## Data and next phase

The unchanged overlay has 38 V2_MODEL_DEVELOPMENT_CONSUMED, 12 V2_VALIDATION, 31 V2_TRAIN_POOL and 15 ID_TEST_SEALED workbooks. The YR runner checks the development role before opening a workbook; tests prove immediate rejection of the three forbidden roles. Solver runs used only I2/I3/I5/I6/I9/I12 from the allowed development set. V2 validation metadata was used only to confirm its frozen identity.

NEXT_PHASE is Phase 3-Z — V2 Common-Model Validation, using the already-frozen 12 instances, the three V2 methods, five fixed solver seeds, 60 s single trajectories and 5/30/60 s checkpoints under a separately declared validation protocol. This round stops at authorization; it does not start validation, training, ID_TEST, or further access/operator engineering.

## Files and execution

The three approved new files are the YR protocol（对应过程文件已清理，结论保留在本报告）, YR artifact（对应过程文件已清理，结论保留在本报告）, and this handoff. The existing runner and test files were reused. [FORMAL_SCOPE_V2](FORMAL_SCOPE_V2.md) was updated only for closure/activation history; README reflects the active V2 state.

Working directory: `D:\pybullet_test\MRTA_GA\DRL`. Required interpreter: `D:\pybullet_test\.venv\Scripts\python.exe`.

```powershell
D:\pybullet_test\.venv\Scripts\python.exe -B scripts\run_phase3y_v2_core.py --freeze-yr-protocol
D:\pybullet_test\.venv\Scripts\python.exe -B scripts\run_phase3y_v2_core.py --run-yr
D:\pybullet_test\.venv\Scripts\python.exe -B scripts\run_phase3y_v2_core.py --regress-yr
# ACTIVE changed only after the smoke and pre-activation regression passed.
D:\pybullet_test\.venv\Scripts\python.exe -B scripts\run_phase3y_v2_core.py --regress-yr --after-activation
D:\pybullet_test\.venv\Scripts\python.exe -B scripts\run_phase3y_v2_core.py --summarize-yr
```
