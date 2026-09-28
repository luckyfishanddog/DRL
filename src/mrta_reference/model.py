from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Mapping, Sequence

Point = tuple[float, float]

from .scope import (
    FormalScope, FORMAL_SCOPE_V1, RunScientificIdentity,
    DEVELOPMENT_NO_REPAIR_V1, FORMAL_BOUNDED_DISPATCH_POLICY_V1,
)


class Rail(str, Enum):
    UPPER = "UPPER"
    LOWER = "LOWER"


class SplitKind(str, Enum):
    WHOLE = "WHOLE"
    X_SPLIT = "X_SPLIT"
    Y_SPLIT = "Y_SPLIT"


class OperationKind(str, Enum):
    MOVE = "MOVE"
    SETUP = "SETUP"
    WELD = "WELD"
    POST = "POST"
    WAIT = "WAIT"


class ScheduleStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    DEADLOCK = "DEADLOCK"
    INFEASIBLE = "INFEASIBLE"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


class MoveType(str, Enum):
    INTRA_RELOCATE = "INTRA_RELOCATE"
    INTER_RELOCATE = "INTER_RELOCATE"
    SWAP = "SWAP"
    TWO_OPT = "TWO_OPT"
    TWO_OPT_STAR = "TWO_OPT_STAR"
    SPLIT_ACTIVATE = "SPLIT_ACTIVATE"
    SPLIT_DEACTIVATE = "SPLIT_DEACTIVATE"
    SPLIT_POINT_SWITCH = "SPLIT_POINT_SWITCH"


class ScientificAmbiguityError(RuntimeError):
    """Raised when a caller requests behavior not frozen by the scientific spec."""


@dataclass(frozen=True)
class ScientificConfig:
    workspace_x: tuple[float, float] = (0.0, 20.0)
    workspace_y: tuple[float, float] = (0.0, 12.0)
    weld_speed: float = 0.0108
    empty_speed: float = 0.20
    t_pre: float = 20.0
    t_post: float = 30.0
    min_child_length: float = 0.20
    delta_x: float = 0.20
    delta_y: float = 0.20
    interference_dx: float = 0.50
    interference_dy: float = 0.50
    numeric_epsilon: float = 1.0e-10

    def __post_init__(self) -> None:
        positives = (
            self.weld_speed,
            self.empty_speed,
            self.min_child_length,
            self.delta_x,
            self.delta_y,
            self.interference_dx,
            self.interference_dy,
            self.numeric_epsilon,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in positives):
            raise ValueError("scientific speeds, distances, and epsilon must be positive and finite")
        if self.t_pre < 0.0 or self.t_post < 0.0:
            raise ValueError("pre/post durations must be non-negative")
        if not (
            self.workspace_x[0] < self.workspace_x[1]
            and self.workspace_y[0] < self.workspace_y[1]
        ):
            raise ValueError("workspace bounds must be increasing")

    @property
    def by(self) -> tuple[float, float]:
        return (6.0 - self.delta_y, 6.0 + self.delta_y)

    def process_time(self, length: float) -> float:
        return self.t_pre + length / self.weld_speed + self.t_post

    def scientific_mapping(self) -> dict[str, object]:
        """Return only scientific parameter values used by the evaluator."""
        return {
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "empty_speed": self.empty_speed,
            "interference_dx": self.interference_dx,
            "interference_dy": self.interference_dy,
            "min_child_length": self.min_child_length,
            "numeric_epsilon": self.numeric_epsilon,
            "t_post": self.t_post,
            "t_pre": self.t_pre,
            "weld_speed": self.weld_speed,
            "workspace_x": list(self.workspace_x),
            "workspace_y": list(self.workspace_y),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.scientific_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def scientific_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class ParentWeld:
    parent_id: str
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if not self.parent_id:
            raise ValueError("parent_id must be non-empty")
        if not all(math.isfinite(v) for point in (self.start, self.end) for v in point):
            raise ValueError("weld coordinates must be finite")
        if self.length <= 0.0:
            raise ValueError("parent weld must have positive length")

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    def point(self, t: float) -> Point:
        return (
            self.start[0] + t * (self.end[0] - self.start[0]),
            self.start[1] + t * (self.end[1] - self.start[1]),
        )


@dataclass(frozen=True, order=True)
class SplitPattern:
    parent_id: str
    kind: SplitKind
    t: float | None = None
    point_id: str | None = None
    mandatory: bool = False

    def __post_init__(self) -> None:
        if self.kind is SplitKind.WHOLE:
            if self.t is not None or self.point_id is not None:
                raise ValueError("WHOLE cannot carry a split point")
            if self.mandatory:
                raise ValueError("WHOLE cannot be marked mandatory split")
        else:
            if self.t is None or not math.isfinite(self.t) or not (0.0 < self.t < 1.0):
                raise ValueError("split pattern requires an interior finite t")
            if not self.point_id:
                raise ValueError("split pattern requires a stable point_id")

    @property
    def pattern_id(self) -> str:
        if self.kind is SplitKind.WHOLE:
            return f"{self.parent_id}:WHOLE"
        return f"{self.parent_id}:{self.kind.value}:{self.point_id}"


@dataclass(frozen=True, order=True)
class WeldingBlock:
    parent_id: str
    block_id: str
    u_start: float
    u_end: float
    start: Point
    end: Point

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True, order=True)
class Route:
    robot_id: int
    block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.robot_id not in range(4):
            raise ValueError("robot_id must be 0..3")


@dataclass(frozen=True)
class RobotRoute:
    robot_id: int
    blocks: tuple[WeldingBlock, ...]
    orientations: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.blocks) != len(self.orientations):
            raise ValueError("one orientation is required per block")
        if any(value not in (0, 1) for value in self.orientations):
            raise ValueError("orientations must be 0 (forward) or 1 (reverse)")


@dataclass(frozen=True)
class Operation:
    operation_id: str
    robot_id: int
    kind: OperationKind
    start_time: float
    end_time: float
    start: Point
    end: Point
    sequence_index: int
    block_id: str | None = None

    def __post_init__(self) -> None:
        values = (*self.start, *self.end, self.start_time, self.end_time)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("operation values must be finite")
        if self.end_time < self.start_time:
            raise ValueError("operation cannot end before it starts")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def point_at(self, time: float) -> Point:
        if self.duration == 0.0:
            return self.end
        ratio = (time - self.start_time) / self.duration
        return (
            self.start[0] + ratio * (self.end[0] - self.start[0]),
            self.start[1] + ratio * (self.end[1] - self.start[1]),
        )


@dataclass(frozen=True)
class ScheduleResult:
    status: ScheduleStatus
    operations: tuple[Operation, ...] = ()
    cmax: float | None = None
    robot_completion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wait_for_graph: tuple[tuple[int, tuple[int, ...]], ...] = ()
    diagnostics: tuple[str, ...] = ()
    directions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (),
        (),
        (),
        (),
    )
    # Evaluator provenance is separate from the canonical scientific timeline.
    source: str = "BASELINE"
    reference_policy_id: str = "DEVELOPMENT_NO_REPAIR_V1"
    scope_id: str | None = None
    scope_hash: str | None = None
    baseline_deadlock: bool = False
    expanded_states: int = 0
    state_budget: int = 0
    recovery_exhausted: bool = False
    frontier_exhausted: bool = False

    @property
    def feasible(self) -> bool:
        return self.status is ScheduleStatus.FEASIBLE

    def canonical_json(self) -> str:
        payload = {
            "status": self.status.value,
            "cmax": self.cmax,
            "completion": self.robot_completion,
            "wait_for_graph": self.wait_for_graph,
            "diagnostics": self.diagnostics,
            "directions": self.directions,
            "operations": [
                {
                    "id": op.operation_id,
                    "robot": op.robot_id,
                    "kind": op.kind.value,
                    "start_time": op.start_time,
                    "end_time": op.end_time,
                    "start": op.start,
                    "end": op.end,
                    "sequence": op.sequence_index,
                    "block": op.block_id,
                }
                for op in self.operations
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CanonicalSolution:
    parents: tuple[ParentWeld, ...]
    patterns: tuple[SplitPattern, ...]
    routes: tuple[Route, ...]
    revision: int = 0

    def __post_init__(self) -> None:
        if tuple(route.robot_id for route in self.routes) != (0, 1, 2, 3):
            raise ValueError("canonical routes must contain robots 0,1,2,3 in order")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "parents": [
                [p.parent_id, list(p.start), list(p.end)] for p in self.parents
            ],
            "patterns": [
                [p.parent_id, p.kind.value, p.t, p.point_id, p.mandatory]
                for p in self.patterns
            ],
            "routes": [[r.robot_id, list(r.block_ids)] for r in self.routes],
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class CandidateKey:
    current_solution_revision: int
    move_type: MoveType
    affected_parent_ids: tuple[str, ...]
    source_robot_id: int | None
    destination_robot_id: int | None
    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]
    split_pattern_id: str | None = None
    split_point_id: str | None = None

    def __post_init__(self) -> None:
        if self.current_solution_revision < 0:
            raise ValueError("solution revision must be non-negative")
        if not self.affected_parent_ids:
            raise ValueError("candidate must identify at least one affected parent")
        if self.affected_parent_ids != tuple(sorted(self.affected_parent_ids)):
            raise ValueError("affected_parent_ids must be sorted")
        for robot_id in (self.source_robot_id, self.destination_robot_id):
            if robot_id is not None and robot_id not in range(4):
                raise ValueError("candidate robot IDs must be 0..3")
        if any(position < 0 for position in self.source_positions + self.destination_positions):
            raise ValueError("candidate positions must be non-negative")


@dataclass(frozen=True)
class CandidateMove:
    key: CandidateKey
    block_ids: tuple[str, ...] = ()
    split_pattern: SplitPattern | None = None


@dataclass(frozen=True)
class OfficialMetrics:
    cmax: float
    optional_split_count: int
    process_imbalance: float
    total_empty_travel: float
    total_waiting: float
    deterministic_id_order: tuple[tuple[str, ...], ...]

    def compare(self, other: "OfficialMetrics") -> int:
        eps = 1.0e-9 * max(1.0, abs(self.cmax), abs(other.cmax))
        if self.cmax < other.cmax - eps:
            return -1
        if self.cmax > other.cmax + eps:
            return 1
        left = (
            self.optional_split_count,
            self.process_imbalance,
            self.total_empty_travel,
            self.total_waiting,
            self.deterministic_id_order,
        )
        right = (
            other.optional_split_count,
            other.process_imbalance,
            other.total_empty_travel,
            other.total_waiting,
            other.deterministic_id_order,
        )
        return -1 if left < right else (1 if left > right else 0)


def robot_rail(robot_id: int) -> Rail:
    if robot_id not in range(4):
        raise ValueError("robot_id must be 0..3")
    return Rail.UPPER if robot_id < 2 else Rail.LOWER


def canonical_routes(routes: Sequence[Route] | Mapping[int, Sequence[str]]) -> tuple[Route, ...]:
    if isinstance(routes, Mapping):
        return tuple(Route(robot, tuple(routes.get(robot, ()))) for robot in range(4))
    by_robot = {route.robot_id: route for route in routes}
    if len(by_robot) != len(routes):
        raise ValueError("duplicate robot route")
    return tuple(by_robot.get(robot, Route(robot)) for robot in range(4))
