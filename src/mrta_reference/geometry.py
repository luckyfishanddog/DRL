from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence

from .model import (
    FormalScope,
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


def _pattern_order(pattern: SplitPattern) -> tuple[object, ...]:
    kind_order = {SplitKind.WHOLE: 0, SplitKind.Y_SPLIT: 1, SplitKind.X_SPLIT: 2}
    return (
        kind_order[pattern.kind],
        "" if pattern.rail is None else pattern.rail.value,
        -1.0 if pattern.t is None else pattern.t,
        pattern.point_id or "",
        pattern.pattern_id,
    )


def build_legal_pattern_catalog(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    scope: FormalScope,
) -> dict[str, tuple[SplitPattern, ...]]:
    """Build the sole deterministic formal pattern domain for one instance."""
    scope.validate_implemented()
    ordered = tuple(sorted(parents, key=lambda item: item.parent_id))
    if len({parent.parent_id for parent in ordered}) != len(ordered):
        raise ValueError("duplicate parent_id in legal pattern catalog")
    x_up, x_low = frozen_handover_centers(ordered, config)
    result: dict[str, tuple[SplitPattern, ...]] = {}
    for parent in ordered:
        eligible = whole_eligible_rails(parent.start, parent.end, config)
        y_patterns = generate_y_split_patterns(parent, config)
        if not eligible:
            if not y_patterns:
                raise ValueError(f"{parent.parent_id}: mandatory Y family is empty")
            options = list(y_patterns)
        else:
            options = [SplitPattern(parent.parent_id, SplitKind.WHOLE), *y_patterns]
            if (
                "X_SPLIT" in scope.pattern_domain
                and scope.optional_x_split_policy == "FINITE_GEOMETRIC_X_SPLIT_V1"
            ):
                for rail, center in ((Rail.UPPER, x_up), (Rail.LOWER, x_low)):
                    if rail in eligible:
                        options.extend(
                            generate_x_split_patterns(
                                parent, center, config, rail=rail
                            )
                        )
        deduplicated: dict[str, SplitPattern] = {}
        for pattern in sorted(options, key=_pattern_order):
            prior = deduplicated.get(pattern.pattern_id)
            if prior is not None and prior != pattern:
                raise ValueError(f"non-unique pattern identity: {pattern.pattern_id}")
            deduplicated[pattern.pattern_id] = pattern
        result[parent.parent_id] = tuple(
            sorted(deduplicated.values(), key=_pattern_order)
        )
    return result


def pattern_catalog_payload(
    parents: Sequence[ParentWeld], config: ScientificConfig, scope: FormalScope
) -> dict[str, object]:
    catalog = build_legal_pattern_catalog(parents, config, scope)
    return {
        "scope_hash": scope.scope_hash,
        "scientific_config_hash": config.scientific_hash,
        "parents": [
            [parent.parent_id, list(parent.start), list(parent.end)]
            for parent in sorted(parents, key=lambda item: item.parent_id)
        ],
        "patterns": {
            parent_id: [
                [
                    pattern.kind.value,
                    pattern.t,
                    pattern.point_id,
                    pattern.mandatory,
                    None if pattern.rail is None else pattern.rail.value,
                ]
                for pattern in patterns
            ]
            for parent_id, patterns in catalog.items()
        },
    }


def pattern_catalog_hash(
    parents: Sequence[ParentWeld], config: ScientificConfig, scope: FormalScope
) -> str:
    encoded = json.dumps(
        pattern_catalog_payload(parents, config, scope),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pattern_in_catalog(
    parent: ParentWeld,
    pattern: SplitPattern,
    catalog: Mapping[str, Sequence[SplitPattern]],
    config: ScientificConfig,
) -> bool:
    return any(
        candidate.pattern_id == pattern.pattern_id
        and candidate.kind is pattern.kind
        and candidate.mandatory == pattern.mandatory
        and candidate.rail is pattern.rail
        and (
            candidate.t is pattern.t is None
            or (
                candidate.t is not None
                and pattern.t is not None
                and abs(candidate.t - pattern.t) <= config.numeric_epsilon
            )
        )
        for candidate in catalog.get(parent.parent_id, ())
    )


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
    *,
    rail: Rail | None = None,
) -> tuple[SplitPattern, ...]:
    """Return the finite, rail-specific geometric X candidate set."""
    eligible = whole_eligible_rails(parent.start, parent.end, config)
    if rail is None:
        if provider is None:
            raise ScientificAmbiguityError("X_SPLIT enumeration requires an explicit rail")
        # Retain the historical development-provider interface for old callers.
        if len(eligible) != 1:
            raise ScientificAmbiguityError("legacy X_SPLIT provider requires one eligible rail")
        rail = next(iter(eligible))
    if rail not in eligible:
        return ()
    dx = parent.end[0] - parent.start[0]
    if abs(dx) <= config.numeric_epsilon:
        return ()
    sources = (
        tuple(provider(parent, center, config))
        if provider is not None
        else (
            ((center - config.delta_x - parent.start[0]) / dx, "BX_LOWER"),
            ((center - parent.start[0]) / dx, "BX_CENTER"),
            ((center + config.delta_x - parent.start[0]) / dx, "BX_UPPER"),
            (0.5, "MIDPOINT"),
        )
    )
    candidates: list[SplitPattern] = []
    seen_t: list[float] = []
    for t, point_id in sorted(sources, key=lambda item: (item[0], item[1])):
        if not (0.0 < t < 1.0):
            continue
        if t * parent.length + config.numeric_epsilon < config.min_child_length:
            continue
        if (1.0 - t) * parent.length + config.numeric_epsilon < config.min_child_length:
            continue
        if any(abs(t - old_t) <= config.numeric_epsilon for old_t in seen_t):
            continue
        pattern = SplitPattern(parent.parent_id, SplitKind.X_SPLIT, t, point_id, rail=rail)
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
        if pattern.rail not in whole_eligible_rails(parent.start, parent.end, config):
            raise ValueError("X_SPLIT rail is not WHOLE-eligible for the parent")
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
    first = WeldingBlock(parent.parent_id, f"{parent.parent_id}::0", 0.0, pattern.t, parent.start, split)
    second = WeldingBlock(parent.parent_id, f"{parent.parent_id}::1", pattern.t, 1.0, split, parent.end)
    if pattern.kind is SplitKind.X_SPLIT and (
        (first.start[0] + first.end[0]) / 2.0
        > (second.start[0] + second.end[0]) / 2.0
    ):
        blocks = (
            WeldingBlock(parent.parent_id, first.block_id, second.u_start, second.u_end, second.start, second.end),
            WeldingBlock(parent.parent_id, second.block_id, first.u_start, first.u_end, first.start, first.end),
        )
    else:
        blocks = (first, second)
    if abs(sum(block.length for block in blocks) - parent.length) > config.numeric_epsilon * max(1.0, parent.length):
        raise ValueError("split length conservation failure")
    return blocks


def finite_x_split_validator(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> XSplitValidator:
    """Build a solver-independent validator using centers frozen for one instance."""
    upper, lower = frozen_handover_centers(parents, config)

    def validate(parent: ParentWeld, pattern: SplitPattern, cfg: ScientificConfig) -> bool:
        if cfg != config or pattern.kind is not SplitKind.X_SPLIT or pattern.rail is None:
            return False
        center = upper if pattern.rail is Rail.UPPER else lower
        return any(
            candidate.pattern_id == pattern.pattern_id
            and candidate.t is not None
            and pattern.t is not None
            and abs(candidate.t - pattern.t) <= cfg.numeric_epsilon
            for candidate in generate_x_split_patterns(
                parent, center, cfg, rail=pattern.rail
            )
        )

    return validate


def x_split_geometry_metadata(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> dict[str, object]:
    """Compute X-domain opportunity metadata without scheduling or search."""
    upper, lower = frozen_handover_centers(parents, config)
    patterns_by_parent: dict[str, list[SplitPattern]] = {}
    source_counts: dict[str, int] = {
        "BX_LOWER": 0,
        "BX_CENTER": 0,
        "BX_UPPER": 0,
        "MIDPOINT": 0,
    }
    for parent in parents:
        for rail, center in ((Rail.UPPER, upper), (Rail.LOWER, lower)):
            patterns = generate_x_split_patterns(parent, center, config, rail=rail)
            if patterns:
                patterns_by_parent.setdefault(parent.parent_id, []).extend(patterns)
            for pattern in patterns:
                assert pattern.point_id is not None
                source_counts[pattern.point_id] += 1
    parent_by_id = {parent.parent_id: parent for parent in parents}
    splittable = [parent_by_id[parent_id] for parent_id in sorted(patterns_by_parent)]
    process_times = [config.process_time(parent.length) for parent in splittable]
    total_process = sum(config.process_time(parent.length) for parent in parents)
    splittable_process = sum(process_times)
    max_process = max(process_times, default=0.0)
    spans = [abs(parent.end[0] - parent.start[0]) for parent in splittable]

    def crosses(parent: ParentWeld, x: float) -> bool:
        return (
            (parent.start[0] - x) * (parent.end[0] - x)
            < -(config.numeric_epsilon * config.numeric_epsilon)
        )

    return {
        "x_up": upper,
        "x_low": lower,
        "x_splittable_parent_count": len(splittable),
        "x_split_pattern_count": sum(len(items) for items in patterns_by_parent.values()),
        "x_splittable_total_process_time": splittable_process,
        "x_splittable_process_share": (
            splittable_process / total_process if total_process else 0.0
        ),
        "max_x_splittable_parent_process_time": max_process,
        "max_x_splittable_parent_process_share": (
            max_process / total_process if total_process else 0.0
        ),
        "max_x_span": max(spans, default=0.0),
        "mean_x_span": (sum(spans) / len(spans) if spans else 0.0),
        "count_cross_x_up": sum(crosses(parent, upper) for parent in parents),
        "count_cross_x_low": sum(crosses(parent, lower) for parent in parents),
        "x_split_source_counts": source_counts,
    }


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
