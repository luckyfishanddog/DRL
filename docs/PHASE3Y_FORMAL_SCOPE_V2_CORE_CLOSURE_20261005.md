# Phase 3-Y — FORMAL_SCOPE_V2 Core Closure — 2026-10-06

`PHASE3Y_EXECUTION_STATUS = FAIL`

`FORMAL_SCOPE_V2_STATUS = OPEN`

`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1`

`FORMAL_SCOPE_V2_HASH = 16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599`

`COMMON_DOMAIN_V2_STATUS = FAIL`

`V2_VALIDATION_SET_STATUS = FROZEN`

`V2_TRAIN_POOL_WORKBOOK_COUNT = 31`

`PHASE3Z_V2_VALIDATION_AUTHORIZED = NO`

`ID_TEST_STATUS = SEALED`

`NEXT_PHASE = fix only the reported Phase3-Y blocker`

The approved filename retains the 20261005 phase label; the completed local execution date is 2026-10-06. No files were uploaded to GitHub.

## A. Outcome

The V2 formal core, common catalog, three method adapters, data-role freeze, differential compatibility checks, telemetry, and 54-run smoke all executed. Every returned final schedule was independently certified and there were no numeric failures or scheduler/certifier mismatches. The phase nevertheless fails because two methods exceeded the frozen allowance of at most two positive mechanism instances with zero X reference evaluations.

This is a search-access blocker, not evidence against the already-authorized scientific X domain. No algorithm parameter was changed after observing the smoke.

## B. Frozen identities

- source commit label: `e8d32f02c155eaf2cae895481917421c2fe42eda`
- scientific source-tree hash at protocol freeze: `101d429de61b96b25ad288a9d4eb8d0a6bd001b65b0f5f747f811f5ff7f34671`
- V1 hash: `8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`
- V1.1 hash: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`
- V2 hash: `16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599`
- V2 data roles hash: `7a80372eb072b10da3d8044eb8bb330a29ede93fe16c806b1aa0d58bee968a5a`
- V2 validation set hash: `2989d8fe15330883a617504d4cc51a6a6cfc547aa9c9813853b6df8fcb2245fe`
- Phase 3-Y protocol hash: `6efbbed56a75cc58876182f51513347bf9ab8855f00f0285d5547d5c0de593be`

The historical V1/V1.1 hashes remained byte-identical. V2 formal X legality is determined by scope, parents, and `ScientificConfig`; arbitrary caller callbacks cannot change it.

## C. V2 data governance

The V2 role overlay and validation set were frozen before any V2 performance smoke:

| V2 role | Workbook count |
|---|---:|
| V2_MODEL_DEVELOPMENT_CONSUMED | 38 |
| V2_VALIDATION | 12 |
| V2_TRAIN_POOL | 31 |
| ID_TEST_SEALED | 15 |

The 12 V2 validation workbooks were selected from the old TRAIN_POOL using geometry-only deterministic selection. The smoke used only the six frozen development mechanism instances I2, I3, I5, I6, I9, and I12. No solver instance from V2_VALIDATION, V2_TRAIN_POOL, or ID_TEST was loaded for this smoke.

## D. Core closure evidence

- A single deterministic legal catalog is shared by SA-OI-ALNS, HGA, WAG, formal canonicalization/validation, the reference scheduler, and the independent certifier.
- `x_up` and `x_low` are computed once from the complete parent set.
- Mandatory Y cannot be bypassed by X.
- Formal V2 accepts only frozen catalog X values and rejects forged arbitrary `t`.
- Upper X fixes left/right children to R0/R1; lower X fixes them to R2/R3.
- X processing charges SETUP + WELD + POST per child; with current defaults the two children add exactly 50 s relative to WHOLE base processing.
- Shared X points receive no interference exception.
- Combined/recursive XY and more than two final blocks are excluded.
- V1/V1.1 remain fail-closed for X.
- Historical runners explicitly pin their historical scope.

The V1.1-to-V2 no-X differential corpus contained Q1–Q6 and six PPO development solutions. All 12 matched exactly on reference status, Cmax, operations, WAIT, official metrics, and independent certification. The three V2 methods used the same per-instance catalog hashes.

## E. Frozen smoke protocol

- methods: `SA_OI_ALNS_V2`, `ADAPTED_HGA_V2`, `ADAPTED_WAG_VNS_V2`
- seeds: 20261005, 20261006, 20261007
- instances: I2, I3, I5, I6, I9, I12
- budget: 30 s including initialization
- checkpoints: 5 s, 15 s, 30 s
- total: 54 runs
- X oracle seed: disabled
- TWO_OPT_STAR: disabled
- no new X-specific hand-designed operator

I2 is the harmful-X control and is excluded from the access gate. For each method, the gate aggregates three seeds on positive instances I3/I5/I6/I9/I12. A method fails when more than two of those instances have zero `x_pattern_reference_evaluated`.

## F. Smoke result

| Method | I3 | I5 | I6 | I9 | I12 | Zero instances | Access gate |
|---|---:|---:|---:|---:|---:|---|---|
| SA_OI_ALNS_V2 | 1 | 0 | 0 | 0 | 1 | I5, I6, I9 (3) | FAIL |
| ADAPTED_HGA_V2 | 26 | 5 | 13 | 4 | 3 | none (0) | PASS |
| ADAPTED_WAG_VNS_V2 | 9 | 0 | 0 | 0 | 0 | I5, I6, I9, I12 (4) | FAIL |

Other frozen gates:

| Check | Result |
|---|---:|
| completed runs | 54 / 54 |
| independently certified final schedules | 54 / 54 |
| numeric failures | 0 |
| scheduler/certifier mismatches | 0 |
| catalog equivalence | PASS |
| search access | FAIL |

Final X retention was not required and was not used as a gate. The smoke does not rank the methods and does not support any claim that X is always beneficial.

## G. Exact blocker

`SA_OI_ALNS_V2` failed to reference-evaluate a legal X candidate across all three seeds on I5, I6, and I9. This is three zero-access positive instances, one above the allowed maximum of two.

`ADAPTED_WAG_VNS_V2` failed to reference-evaluate a legal X candidate across all three seeds on I5, I6, I9, and I12. This is four zero-access positive instances, two above the allowed maximum.

Because the common-domain search-access gate failed, Phase 3-Y cannot close even though the returned schedules were valid. V2 must not be activated and V2 validation must not begin.

## H. Regression and activation decision

The final command used the required interpreter and project directory:

```powershell
D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q
```

Result: `256 passed in 70.23s`.

The failure is therefore not a unit/regression failure. Per the frozen FAIL gate:

- `ACTIVE_FORMAL_SCOPE` remains `FORMAL_SCOPE_V1_1`;
- `FORMAL_SCOPE_V2` remains implemented but open and inactive;
- README is not changed because its update was PASS-only;
- `PHASE3Z_V2_VALIDATION_AUTHORIZED = NO`;
- ID_TEST remains sealed.

## I. Next phase boundary

`NEXT_PHASE = fix only the reported Phase3-Y blocker`

The next work may address only the demonstrated X-access deficiency in SA-OI-ALNS_V2 and ADAPTED_WAG_VNS_V2 under the already-frozen common catalog and scientific X definition. It must not revisit whether X_SPLIT should exist, alter the X candidate set, tune frozen method parameters to improve this smoke, use an oracle seed, access V2_VALIDATION/V2_TRAIN_POOL/ID_TEST, or start Phase 3-Z. A new authorized run must produce its own predeclared protocol/evidence as required; historical Phase 3-Y artifacts remain unchanged.
