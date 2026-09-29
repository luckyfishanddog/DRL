"""The single machine-readable formal task-layer scope and result identity."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import TYPE_CHECKING

from .provenance import SourceProvenance

if TYPE_CHECKING:
    from .model import ScientificConfig

DEVELOPMENT_NO_REPAIR_V1 = "DEVELOPMENT_NO_REPAIR_V1"
FORMAL_BOUNDED_DISPATCH_POLICY_V1 = "FORMAL_BOUNDED_DISPATCH_POLICY_V1"
FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1 = (
    "FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1"
)
COMPLETE_ALTERNATIVE_ROLLOUTS = "COMPLETE_ALTERNATIVE_ROLLOUTS"
LIMITED_DISCREPANCY_RECENT_BRANCH_FIRST_V1 = (
    "LIMITED_DISCREPANCY_RECENT_BRANCH_FIRST_V1"
)


@dataclass(frozen=True)
class FormalScope:
    scope_id: str = "FORMAL_SCOPE_V1"
    pattern_domain: tuple[str, ...] = ("WHOLE", "Y_SPLIT")
    optional_x_split_policy: str = "EXCLUDED"
    y_split_rule_id: str = "BY_LOWER_CENTER_UPPER_LEGAL_MIDPOINT_T_DEDUP_V1"
    max_split_per_parent: int = 1
    terminal_policy: str = "TASK_HORIZON_RELEASE_V1"
    empty_route_policy: str = "UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1"
    initial_deployment_policy: str = "FREE_FIRST_WELD_START_ALL_ACTIVE_LEGAL_AT_ZERO_V1"
    deadlock_policy_id: str = FORMAL_BOUNDED_DISPATCH_POLICY_V1
    deadlock_state_budget: int | None = 16
    deadlock_budget_unit: str | None = None
    deadlock_rollout_budget: int | None = None
    dispatch_recovery_order_id: str | None = None
    reference_scheduler_policy_id: str = FORMAL_BOUNDED_DISPATCH_POLICY_V1
    open_route_policy_id: str = "NO_HOME_FIRST_NO_RETURN_HOME_V1"
    objective_policy_id: str = "CMAX_OPTIONAL_SPLITS_PROCESS_SPREAD_EMPTY_WAIT_ID_V1"
    certifier_policy_id: str = "INDEPENDENT_CONTINUOUS_TASK_HORIZON_V1"
    interference_policy_id: str = "CLOSED_CONTINUOUS_TCP_AND_SAME_RAIL_ORDER_V1"
    dispatch_order_id: str = "DFS_ESS_REMAINING_PROCESS_COMPLETION_ROBOT_V1"
    recovery_selection_id: str = "CMAX_THEN_CANONICAL_SCHEDULE_JSON_V1"
    state_count_policy_id: str = "POPPED_PREFIX_INCLUDING_ROOT_AND_COMPLETE_V1"

    @property
    def canonical_json(self) -> str:
        # Optional V1.1 fields are absent, rather than null, in the historical
        # V1 payload.  This preserves the published FORMAL_SCOPE_V1 hash.
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def scope_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def validate_implemented(self) -> None:
        if self.canonical_json not in {
            FORMAL_SCOPE_V1.canonical_json,
            FORMAL_SCOPE_V1_1.canonical_json,
        }:
            raise ValueError("unsupported formal scope; policy changes require implementation and validation")


FORMAL_SCOPE_V1 = FormalScope()

FORMAL_SCOPE_V1_1 = replace(
    FORMAL_SCOPE_V1,
    scope_id="FORMAL_SCOPE_V1_1",
    deadlock_policy_id=FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1,
    deadlock_state_budget=None,
    deadlock_budget_unit=COMPLETE_ALTERNATIVE_ROLLOUTS,
    deadlock_rollout_budget=32,
    dispatch_recovery_order_id=LIMITED_DISCREPANCY_RECENT_BRANCH_FIRST_V1,
    reference_scheduler_policy_id=FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1,
    state_count_policy_id="COMPLETE_ALTERNATIVE_ROLLOUTS_V1",
)

ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1


@dataclass(frozen=True)
class RunScientificIdentity:
    scope_id: str
    scope_hash: str
    scientific_config_hash: str
    reference_policy_id: str
    repository_id: str
    source_commit: str
    source_tree_hash: str
    worktree_dirty: bool
    commit_verified: bool

    @classmethod
    def from_scope(
        cls, scope: FormalScope, config: ScientificConfig, provenance: SourceProvenance
    ):
        scope.validate_implemented()
        return cls(
            scope.scope_id,
            scope.scope_hash,
            config.scientific_hash,
            scope.reference_scheduler_policy_id,
            provenance.repository_id,
            provenance.source_commit,
            provenance.source_tree_hash,
            provenance.worktree_dirty,
            provenance.commit_verified,
        )

