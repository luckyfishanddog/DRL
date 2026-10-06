# FORMAL_SCOPE_V2 — Candidate Formal Definition

`FORMAL_SCOPE_V2_STATUS = OPEN`

`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1`

`FORMAL_SCOPE_V2_HASH = 16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599`

This document freezes the machine-readable meaning of the V2 candidate scope. It does not activate V2. Activation requires the complete Phase 3-Y gate; the first frozen common-domain smoke failed only its X-search-access gate, so V1.1 remains active.

## 1. Task and pattern domain

The task remains the four-robot, two-rail welding allocation and open-route ordering problem inherited from `FORMAL_SCOPE_V1_1`. A parent weld has one final pattern selected from the deterministic legal catalog:

- `WHOLE`
- `Y_SPLIT`
- `X_SPLIT`

Each parent may be split at most once and may produce at most two welding blocks. Recursive split, X-after-Y, Y-after-X, combined XY, and three- or four-segment patterns are excluded by `combined_xy_split_policy_id = EXCLUDED_V1`.

## 2. Single deterministic legal pattern catalog

`build_legal_pattern_catalog(parents, config, scope)` is the common legality source for SA-OI-ALNS, Adapted-HGA, Adapted-WAG+VNS, canonicalization, formal validation, the reference scheduler, and the independent certifier. Catalog entries have deterministic identities and ordering. For V2, formal legality is catalog membership; a caller-provided X callback cannot expand or alter the domain associated with the scope hash.

The complete parent collection is used once to compute the frozen weighted-median `x_up` and `x_low`; algorithms and candidates do not recompute them from assignments, WAIT, or search history. Catalog hashes are therefore method-independent for a fixed `(parents, ScientificConfig, scope)`.

## 3. Mandatory Y and whole eligibility

The existing Y rule is unchanged: `BY_LOWER_CENTER_UPPER_LEGAL_MIDPOINT_T_DEDUP_V1`. A parent that cannot be legally assigned whole to one rail remains mandatory-Y and receives only its existing mandatory Y family. X cannot bypass mandatory Y.

For a whole-eligible parent, the catalog may include WHOLE, existing legal optional Y patterns, and the legal X family for each whole-eligible rail. Every child must satisfy the configured minimum length.

## 4. X candidate rule

`optional_x_split_policy = FINITE_GEOMETRIC_X_SPLIT_V1`

`x_split_rule_id = BX_PM_DELTA_AND_MIDPOINT_LMIN_FROM_CONFIG_V1`

For an eligible rail, X candidates are the segment parameters induced by:

- `BX_LOWER`: `x_g - delta_x`
- `BX_CENTER`: `x_g`
- `BX_UPPER`: `x_g + delta_x`
- `MIDPOINT`: `t = 0.5`

The frozen default `delta_x` is 0.20 m and minimum child length is 0.20 m. Invalid intersections, endpoints, duplicates, and candidates producing a child below the minimum are removed deterministically. Vertical or negligible-`dx` parents have no X pattern. There is no length threshold, WAIT/deadlock trigger, dynamic split point, dense grid, quarter point, or process-approved joint list.

## 5. X child assignment

`x_split_assignment_policy_id = SAME_RAIL_SPATIAL_LEFT_RIGHT_FIXED_PAIR_V1`

- Upper rail: spatial left child is fixed to R0; spatial right child is fixed to R1.
- Lower rail: spatial left child is fixed to R2; spatial right child is fixed to R3.

Children cannot cross rails, swap the fixed left/right robot pair, or perform rail handover. Formal solution validation checks this exact assignment in addition to ordinary robot eligibility.

## 6. Processing and direction semantics

`x_split_processing_policy_id = ZERO_SPLIT_TIME_PER_CHILD_SETUP_WELD_POST_V1`

X split and cut time are zero. Each child is an independent welding block and pays its own `SETUP + WELD + POST` time. Thus, for parent length `L` and children `L_left + L_right = L`:

`T_left + T_right = T_whole + tpre + tpost`.

With the frozen defaults `tpre = 20 s` and `tpost = 30 s`, a true X split adds 50 s of base processing relative to WHOLE. There is no free split, single shared setup/post, or artificial extra split penalty. Each child independently retains the existing forward/reverse direction choice.

## 7. Shared-point, interference, and same-rail semantics

`x_split_shared_point_policy_id = NO_INTERFERENCE_EXCEPTION_V1`

The common split point creates no collision or occupancy exception. Continuous TCP interference, same-rail nonpassing/order, SETUP/POST occupancy, WAIT, and task-horizon rules apply normally to both children, including at the shared point.

## 8. Inherited V1.1 semantics

V2 changes only the formal pattern domain and the explicit X policies above. It inherits the following V1.1 policies unchanged:

- terminal: `TASK_HORIZON_RELEASE_V1`; after final POST there is no terminal WAIT, return-home, parking, or retract time;
- empty robot: `UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1`;
- initial deployment: `FREE_FIRST_WELD_START_ALL_ACTIVE_LEGAL_AT_ZERO_V1`;
- open route: `NO_HOME_FIRST_NO_RETURN_HOME_V1`;
- scheduler/recovery: `FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1`, 32 complete alternative rollouts, recent-branch-first limited discrepancy order;
- interference: `CLOSED_CONTINUOUS_TCP_AND_SAME_RAIL_ORDER_V1`;
- objective: `CMAX_OPTIONAL_SPLITS_PROCESS_SPREAD_EMPTY_WAIT_ID_V1`;
- certifier: `INDEPENDENT_CONTINUOUS_TASK_HORIZON_V1`.

The independent certifier rebuilds the legal catalog and independently checks pattern membership, fixed X child assignment, geometry, processing, directions, WAIT, same-rail order, continuous interference, terminal semantics, completion, and Cmax. It does not trust a scheduler FEASIBLE flag.

## 9. Data-role interpretation

`PPO_V2_DATA_ROLES_V1` is an overlay and does not modify the historical Phase 3 split:

- 38 workbooks: `V2_MODEL_DEVELOPMENT_CONSUMED`;
- 12 workbooks selected solver-independently from the old TRAIN_POOL: `V2_VALIDATION`;
- remaining 31 old TRAIN_POOL workbooks: `V2_TRAIN_POOL`;
- original 15 ID_TEST workbooks: `ID_TEST_SEALED`.

Hashes:

- `v2_data_roles_hash = 7a80372eb072b10da3d8044eb8bb330a29ede93fe16c806b1aa0d58bee968a5a`
- `v2_validation_set_hash = 2989d8fe15330883a617504d4cc51a6a6cfc547aa9c9813853b6df8fcb2245fe`

The Phase 3-Y development smoke used only `V2_MODEL_DEVELOPMENT_CONSUMED`. It did not run solvers on V2_VALIDATION, V2_TRAIN_POOL, or ID_TEST.

## 10. Historical relationship and identity

V1 and V1.1 remain frozen and replayable:

- `FORMAL_SCOPE_V1_HASH = 8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`
- `FORMAL_SCOPE_V1_1_HASH = 5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`

New X fields default to `None` and are omitted from historical canonical JSON, preserving both hashes. V1/V1.1 continue to reject X fail-closed. Deterministic differential checks on Q1–Q6 and six PPO development cases established identical V1.1/V2 results for solutions containing only WHOLE/Y: status, Cmax, operations, WAIT, official metrics, and certification all match.

V2 canonical JSON:

```json
{"certifier_policy_id":"INDEPENDENT_CONTINUOUS_TASK_HORIZON_V1","combined_xy_split_policy_id":"EXCLUDED_V1","deadlock_budget_unit":"COMPLETE_ALTERNATIVE_ROLLOUTS","deadlock_policy_id":"FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1","deadlock_rollout_budget":32,"dispatch_order_id":"DFS_ESS_REMAINING_PROCESS_COMPLETION_ROBOT_V1","dispatch_recovery_order_id":"LIMITED_DISCREPANCY_RECENT_BRANCH_FIRST_V1","empty_route_policy":"UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1","initial_deployment_policy":"FREE_FIRST_WELD_START_ALL_ACTIVE_LEGAL_AT_ZERO_V1","interference_policy_id":"CLOSED_CONTINUOUS_TCP_AND_SAME_RAIL_ORDER_V1","max_split_per_parent":1,"objective_policy_id":"CMAX_OPTIONAL_SPLITS_PROCESS_SPREAD_EMPTY_WAIT_ID_V1","open_route_policy_id":"NO_HOME_FIRST_NO_RETURN_HOME_V1","optional_x_split_policy":"FINITE_GEOMETRIC_X_SPLIT_V1","pattern_domain":["WHOLE","Y_SPLIT","X_SPLIT"],"recovery_selection_id":"CMAX_THEN_CANONICAL_SCHEDULE_JSON_V1","reference_scheduler_policy_id":"FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1","scope_id":"FORMAL_SCOPE_V2","state_count_policy_id":"COMPLETE_ALTERNATIVE_ROLLOUTS_V1","terminal_policy":"TASK_HORIZON_RELEASE_V1","x_split_assignment_policy_id":"SAME_RAIL_SPATIAL_LEFT_RIGHT_FIXED_PAIR_V1","x_split_processing_policy_id":"ZERO_SPLIT_TIME_PER_CHILD_SETUP_WELD_POST_V1","x_split_rule_id":"BX_PM_DELTA_AND_MIDPOINT_LMIN_FROM_CONFIG_V1","x_split_shared_point_policy_id":"NO_INTERFERENCE_EXCEPTION_V1","y_split_rule_id":"BY_LOWER_CENTER_UPPER_LEGAL_MIDPOINT_T_DEDUP_V1"}
```

## 11. Activation status

The candidate definition and implementation are available, but the first Phase 3-Y common-domain smoke did not meet the frozen X-access threshold for SA-OI-ALNS_V2 and ADAPTED_WAG_VNS_V2. Therefore:

`FORMAL_SCOPE_V2_STATUS = OPEN`

`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1`

`PHASE3Z_V2_VALIDATION_AUTHORIZED = NO`

