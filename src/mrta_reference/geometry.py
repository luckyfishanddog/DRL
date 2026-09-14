from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Iterable, Sequence

from .model import (
    Operation,
    ParentWeld,
    Point,
    Rail,
    ScientificAmbiguityError,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
    robot_rail,
)

XSplitCandidateProvider = Callable[
    [ParentWeld, float, ScientificConfig], Iterable[tuple[float, str]]
]
XSplitValidator = Callable[[ParentWeld, SplitPattern, ScientificConfig], bool]


def whole_eligible_rails(
    start: Point, end: Point, config: ScientificConfig
) -> frozenset[Rail]:
    lower, upper = config.by
    eligible: set[Rail] = set()
    if min(start[1], end[1]) >= lower - config.numeric_epsilon:
        eligible.add(Rail.UPPER)
    if max(start[1], end[1]) <= upper + config.numeric_epsilon:
        eligible.add(Rail.LOWER)
    return frozenset(eligible)


def robot_is_eligible(block: WeldingBlock, robot_id: int, config: ScientificConfig) -> bool:
    return robot_rail(robot_id) in whole_eligible_rails(block.start, block.end, config)


def generate_y_split_patterns(
    parent: ParentWeld, config: ScientificConfig
) -> tuple[SplitPattern, ...]:
    """Return only the four scientifically specified Y candidates, deduplicated by t."""
    dy = parent.end[1] - parent.start[1]
    sources: list[tuple[float, str]] = []
    if dy != 0.0:
        for y, point_id in (
            (config.by[0], "BY_LOWER"),
            (6.0, "BY_CENTER"),
            (config.by[1], "BY_UPPER"),
        ):
            sources.append(((y - parent.start[1]) / dy, point_id))
    midpoint = parent.point(0.5)
    if config.by[0] - config.numeric_epsilon <= midpoint[1] <= config.by[1] + config.numeric_epsilon:
        sources.append((0.5, "MIDPOINT"))

    kept: list[tuple[float, str]] = []
    for t, point_id in sources:
        if not (0.0 < t < 1.0):
            continue
        point = parent.point(t)
        if not (config.by[0] - config.numeric_epsilon <= point[1] <= config.by[1] + config.numeric_epsilon):
            continue
        if t * parent.length + config.numeric_epsilon < config.min_child_length:
            continue
        if (1.0 - t) * parent.length + config.numeric_epsilon < config.min_child_length:
            continue
        if any(abs(t - old_t) <= config.numeric_epsilon for old_t, _ in kept):
            continue
        kept.append((t, point_id))
    kept.sort(key=lambda item: (item[0], item[1]))
    mandatory = not whole_eligible_rails(parent.start, parent.end, config)
    return tuple(
        SplitPattern(parent.parent_id, SplitKind.Y_SPLIT, t, point_id, mandatory)
        for t, point_id in kept
    )


def generate_x_split_patterns(
    parent: ParentWeld,
    center: float,
    config: ScientificConfig,
    provider: XSplitCandidateProvider | None = None,
) -> tuple[SplitPattern, ...]:
    """Validate caller-provided X candidates; no default enumeration is invented."""
    if provider is None:
        raise ScientificAmbiguityError(
            "optional X_SPLIT candidate enumeration is not frozen; provide an explicit provider"
        )
    candidates: list[SplitPattern] = []
    seen_t: list[float] = []
    for t, point_id in sorted(provider(parent, center, config), key=lambda item: (item[0], item[1])):
        if any(abs(t - old_t) <= config.numeric_epsilon for old_t in seen_t):
            continue
        pattern = SplitPattern(parent.parent_id, SplitKind.X_SPLIT, t, point_id)
        validate_split_pattern(parent, pattern, config)
        candidates.append(pattern)
        seen_t.append(t)
    return tuple(candidates)


def validate_split_pattern(
    parent: ParentWeld,
    pattern: SplitPattern,
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
    require_formal_x_validation: bool = False,
) -> None:
    if pattern.parent_id != parent.parent_id:
        raise ValueError("split pattern parent mismatch")
    if pattern.kind is SplitKind.WHOLE:
        if not whole_eligible_rails(parent.start, parent.end, config):
            raise ValueError(f"parent {parent.parent_id} requires a legal Y split")
        return
    assert pattern.t is not None
    left = pattern.t * parent.length
    right = (1.0 - pattern.t) * parent.length
    if left + config.numeric_epsilon < config.min_child_length or right + config.numeric_epsilon < config.min_child_length:
        raise ValueError("split child shorter than Lmin")
    if pattern.kind is SplitKind.Y_SPLIT:
        legal = generate_y_split_patterns(parent, config)
        if not any(
            abs(pattern.t - candidate.t) <= config.numeric_epsilon
            and pattern.point_id == candidate.point_id
            for candidate in legal
        ):
            raise ValueError("Y split is not one of the deterministic legal candidates")
    elif pattern.kind is SplitKind.X_SPLIT:
        if pattern.mandatory:
            raise ValueError("X split cannot satisfy mandatory upper/lower handover")
        if not whole_eligible_rails(parent.start, parent.end, config):
            raise ValueError("a parent crossing both rails requires Y_SPLIT, not X_SPLIT")
        if require_formal_x_validation and x_split_validator is None:
            raise ScientificAmbiguityError(
                "retained X_SPLIT requires an explicit formal XSplitValidator"
            )
        if x_split_validator is not None and not x_split_validator(parent, pattern, config):
            raise ValueError(
                f"X_SPLIT {pattern.pattern_id} was rejected by the explicit validator"
            )


def blocks_for_pattern(
    parent: ParentWeld,
    pattern: SplitPattern,
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
    require_formal_x_validation: bool = False,
) -> tuple[WeldingBlock, ...]:
    validate_split_pattern(
        parent,
        pattern,
        config,
        x_split_validator=x_split_validator,
        require_formal_x_validation=require_formal_x_validation,
    )
    if pattern.kind is SplitKind.WHOLE:
        return (
            WeldingBlock(parent.parent_id, f"{parent.parent_id}::whole", 0.0, 1.0, parent.start, parent.end),
        )
    assert pattern.t is not None
    split = parent.point(pattern.t)
    blocks = (
        WeldingBlock(parent.parent_id, f"{parent.parent_id}::0", 0.0, pattern.t, parent.start, split),
        WeldingBlock(parent.parent_id, f"{parent.parent_id}::1", pattern.t, 1.0, split, parent.end),
    )
    if abs(sum(block.length for block in blocks) - parent.length) > config.numeric_epsilon * max(1.0, parent.length):
        raise ValueError("split length conservation failure")
    return blocks


def frozen_handover_centers(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> tuple[float, float]:
    def center(rail: Rail) -> float:
        rows = sorted(
            (
                ((parent.start[0] + parent.end[0]) / 2.0, config.process_time(parent.length), parent.parent_id)
                for parent in parents
                if rail in whole_eligible_rails(parent.start, parent.end, config)
            ),
            key=lambda item: (item[0], item[2]),
        )
        if not rows:
            return 10.0
        half = sum(row[1] for row in rows) / 2.0
        cumulative = 0.0
        raw = rows[-1][0]
        for midpoint_x, weight, _ in rows:
            cumulative += weight
            if cumulative >= half:
                raw = midpoint_x
                break
        return min(config.workspace_x[1] - config.delta_x, max(config.delta_x, raw))

    return center(Rail.UPPER), center(Rail.LOWER)


def x_split_relation(point: Point, center: float, config: ScientificConfig) -> str:
    lower, upper = center - config.delta_x, center + config.delta_x
    if point[0] < lower - config.numeric_epsilon:
        return "LEFT_OF_BX"
    if point[0] > upper + config.numeric_epsilon:
        return "RIGHT_OF_BX"
    return "IN_BX"


@dataclass(frozen=True)
class DirectionResult:
    empty_travel_time: float
    orientations: tuple[int, ...]


def oriented_endpoints(block: WeldingBlock, orientation: int) -> tuple[Point, Point]:
    return (block.start, block.end) if orientation == 0 else (block.end, block.start)


def optimize_directions(
    blocks: Sequence[WeldingBlock], config: ScientificConfig
) -> DirectionResult:
    if not blocks:
        return DirectionResult(0.0, ())
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, (0,)), 1: (0.0, (1,))}
    for index in range(1, len(blocks)):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for current_direction in (0, 1):
            current_start, _ = oriented_endpoints(blocks[index], current_direction)
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for previous_direction in (0, 1):
                _, previous_end = oriented_endpoints(blocks[index - 1], previous_direction)
                old_cost, old_vector = states[previous_direction]
                candidates.append(
                    (
                        old_cost + math.dist(previous_end, current_start) / config.empty_speed,
                        old_vector + (current_direction,),
                    )
                )
            next_states[current_direction] = min(candidates, key=lambda item: (item[0], item[1]))
        states = next_states
    cost, vector = min(states.values(), key=lambda item: (item[0], item[1]))
    return DirectionResult(cost, vector)


def _axis_band_interval(a: float, b: float, limit: float, lo: float, hi: float) -> tuple[float, float] | None:
    if abs(a) <= 1.0e-15:
        return (lo, hi) if abs(b) <= limit else None
    first = (-limit - b) / a
    second = (limit - b) / a
    return max(lo, min(first, second)), min(hi, max(first, second))


def _global_line(operation: Operation, axis: int) -> tuple[float, float]:
    if operation.duration == 0.0:
        return 0.0, operation.start[axis]
    velocity = (operation.end[axis] - operation.start[axis]) / operation.duration
    return velocity, operation.start[axis] - velocity * operation.start_time


def continuous_interference(
    first: Operation, second: Operation, config: ScientificConfig
) -> bool:
    lo = max(first.start_time, second.start_time)
    hi = min(first.end_time, second.end_time)
    if hi < lo:
        return False
    intervals: list[tuple[float, float]] = []
    for axis, distance in ((0, config.interference_dx), (1, config.interference_dy)):
        va, ba = _global_line(first, axis)
        vb, bb = _global_line(second, axis)
        interval = _axis_band_interval(
            va - vb, ba - bb, distance + config.numeric_epsilon, lo, hi
        )
        if interval is None or interval[0] > interval[1]:
            return False
        intervals.append(interval)
    return max(item[0] for item in intervals) <= min(item[1] for item in intervals)


def same_rail_order_violation(
    first: Operation, second: Operation, config: ScientificConfig
) -> bool:
    if robot_rail(first.robot_id) is not robot_rail(second.robot_id):
        return False
    if {first.robot_id, second.robot_id} not in ({0, 1}, {2, 3}):
        return False
    left, right = (first, second) if first.robot_id in (0, 2) else (second, first)
    lo = max(left.start_time, right.start_time)
    hi = min(left.end_time, right.end_time)
    if hi < lo:
        return False
    vl, bl = _global_line(left, 0)
    vr, br = _global_line(right, 0)
    a, b = vl - vr, bl - br + config.interference_dx
    if abs(a) <= 1.0e-15:
        maximum = b
    else:
        maximum = max(a * lo + b, a * hi + b)
    return maximum > config.numeric_epsilon


def operations_conflict(first: Operation, second: Operation, config: ScientificConfig) -> bool:
    return continuous_interference(first, second, config) or same_rail_order_violation(first, second, config)


def _polygon_projection(constraints: Sequence[tuple[float, float, float]], epsilon: float) -> tuple[float, float] | None:
    points: list[tuple[float, float]] = []
    for i, (a1, b1, c1) in enumerate(constraints):
        for a2, b2, c2 in constraints[i + 1 :]:
            determinant = a1 * b2 - a2 * b1
            if abs(determinant) <= 1.0e-15:
                continue
            s = (c1 * b2 - c2 * b1) / determinant
            t = (a1 * c2 - a2 * c1) / determinant
            if all(a * s + b * t <= c + epsilon for a, b, c in constraints):
                points.append((s, t))
    if not points:
        return None
    return min(point[0] for point in points), max(point[0] for point in points)


def _proximity_constraints(
    duration: float,
    start: Point,
    end: Point,
    fixed: Operation,
    ready: float,
    config: ScientificConfig,
) -> list[tuple[float, float, float]]:
    lower_start = max(ready, fixed.start_time - duration)
    upper_start = fixed.end_time
    constraints: list[tuple[float, float, float]] = [
        (1.0, -1.0, 0.0),
        (-1.0, 1.0, duration),
        (0.0, -1.0, -fixed.start_time),
        (0.0, 1.0, fixed.end_time),
        (-1.0, 0.0, -lower_start),
        (1.0, 0.0, upper_start),
    ]
    for axis, distance in ((0, config.interference_dx), (1, config.interference_dy)):
        candidate_velocity = 0.0 if duration == 0.0 else (end[axis] - start[axis]) / duration
        fixed_velocity, fixed_intercept = _global_line(fixed, axis)
        constant = start[axis] - fixed_intercept
        limit = distance + config.numeric_epsilon
        constraints.append((-candidate_velocity, candidate_velocity - fixed_velocity, limit - constant))
        constraints.append((candidate_velocity, fixed_velocity - candidate_velocity, limit + constant))
    return constraints


def forbidden_start_intervals(
    robot_id: int,
    duration: float,
    start: Point,
    end: Point,
    fixed: Operation,
    ready: float,
    config: ScientificConfig,
) -> tuple[tuple[float, float], ...]:
    """Project continuous conflict half-planes onto the candidate start-time axis."""
    base = _proximity_constraints(duration, start, end, fixed, ready, config)
    intervals: list[tuple[float, float]] = []
    projected = _polygon_projection(base, config.numeric_epsilon)
    if projected is not None:
        intervals.append(projected)

    if robot_rail(robot_id) is robot_rail(fixed.robot_id) and robot_id != fixed.robot_id:
        order_constraints = base[:6]
        candidate_velocity = 0.0 if duration == 0.0 else (end[0] - start[0]) / duration
        fixed_velocity, fixed_intercept = _global_line(fixed, 0)
        constant = start[0] - fixed_intercept
        if robot_id in (0, 2):
            # candidate(left) + Dx >= fixed(right) is the closed unsafe set.
            order_constraints.append(
                (
                    candidate_velocity,
                    fixed_velocity - candidate_velocity,
                    constant + config.interference_dx - config.numeric_epsilon,
                )
            )
        else:
            # fixed(left) + Dx >= candidate(right).
            order_constraints.append(
                (
                    -candidate_velocity,
                    candidate_velocity - fixed_velocity,
                    config.interference_dx - constant - config.numeric_epsilon,
                )
            )
        order_projected = _polygon_projection(order_constraints, config.numeric_epsilon)
        if order_projected is not None:
            intervals.append(order_projected)
    return tuple(intervals)


def stationary_conflict_interval(
    robot_id: int,
    point: Point,
    fixed: Operation,
    config: ScientificConfig,
) -> tuple[float, float] | None:
    probe = Operation("stationary-probe", robot_id, fixed.kind, fixed.start_time, fixed.end_time, point, point, 0)
    if not operations_conflict(probe, fixed, config):
        return None
    # Each spatial condition is affine in time. Intersect its exact interval.
    lo, hi = fixed.start_time, fixed.end_time
    for axis, distance in ((0, config.interference_dx), (1, config.interference_dy)):
        velocity, intercept = _global_line(fixed, axis)
        interval = _axis_band_interval(-velocity, point[axis] - intercept, distance + config.numeric_epsilon, lo, hi)
        if interval is None:
            return None
        lo, hi = max(lo, interval[0]), min(hi, interval[1])
    return (lo, hi) if lo <= hi else None
