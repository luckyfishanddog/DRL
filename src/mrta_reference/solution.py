from __future__ import annotations

import math
from typing import Mapping, Sequence

from .geometry import XSplitValidator, blocks_for_pattern, whole_eligible_rails
from .model import (
    CanonicalSolution,
    OfficialMetrics,
    OperationKind,
    ParentWeld,
    Route,
    ScheduleResult,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    canonical_routes,
)


def canonicalize(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    routes: Sequence[Route] | Mapping[int, Sequence[str]],
    config: ScientificConfig,
    *,
    revision: int = 0,
    x_split_validator: XSplitValidator | None = None,
) -> CanonicalSolution:
    ordered_parents = tuple(sorted(parents, key=lambda parent: parent.parent_id))
    if len({parent.parent_id for parent in ordered_parents}) != len(ordered_parents):
        raise ValueError("duplicate parent_id")
    parent_by_id = {parent.parent_id: parent for parent in ordered_parents}
    for parent in ordered_parents:
        if not all(
            config.workspace_x[0] - config.numeric_epsilon <= point[0] <= config.workspace_x[1] + config.numeric_epsilon
            and config.workspace_y[0] - config.numeric_epsilon <= point[1] <= config.workspace_y[1] + config.numeric_epsilon
            for point in (parent.start, parent.end)
        ):
            raise ValueError(f"parent {parent.parent_id} lies outside the configured workspace")
    pattern_by_id: dict[str, SplitPattern] = {}
    for pattern in patterns:
        if pattern.parent_id not in parent_by_id:
            raise ValueError(f"unknown parent in split pattern: {pattern.parent_id}")
        if pattern.parent_id in pattern_by_id:
            raise ValueError(f"parent {pattern.parent_id} has more than one split pattern")
        pattern_by_id[pattern.parent_id] = pattern
    missing = sorted(set(parent_by_id) - set(pattern_by_id))
    if missing:
        raise ValueError(f"parents without a WHOLE/X_SPLIT/Y_SPLIT pattern: {missing}")

    current_routes = [list(route.block_ids) for route in canonical_routes(routes)]
    # Same-robot consecutive children are a representation artifact, not two blocks.
    for parent_id in sorted(pattern_by_id):
        pattern = pattern_by_id[parent_id]
        if pattern.kind is SplitKind.WHOLE:
            continue
        child_ids = (f"{parent_id}::0", f"{parent_id}::1")
        merged = False
        for block_ids in current_routes:
            locations = [block_ids.index(child) if child in block_ids else -1 for child in child_ids]
            if min(locations) >= 0 and abs(locations[0] - locations[1]) == 1:
                insertion = min(locations)
                block_ids[:] = [item for item in block_ids if item not in child_ids]
                block_ids.insert(insertion, f"{parent_id}::whole")
                pattern_by_id[parent_id] = SplitPattern(parent_id, SplitKind.WHOLE)
                merged = True
                break
        if merged:
            continue

    normalized_patterns = []
    for parent in ordered_parents:
        pattern = pattern_by_id[parent.parent_id]
        if pattern.kind is SplitKind.Y_SPLIT:
            expected_mandatory = not whole_eligible_rails(parent.start, parent.end, config)
            pattern = SplitPattern(
                pattern.parent_id,
                pattern.kind,
                pattern.t,
                pattern.point_id,
                expected_mandatory,
            )
        normalized_patterns.append(pattern)
    ordered_patterns = tuple(normalized_patterns)
    expected: dict[str, str] = {}
    for parent, pattern in zip(ordered_parents, ordered_patterns):
        for block in blocks_for_pattern(
            parent,
            pattern,
            config,
            x_split_validator=x_split_validator,
            require_formal_x_validation=True,
        ):
            if block.block_id in expected:
                raise ValueError("non-unique canonical block id")
            expected[block.block_id] = parent.parent_id

    actual = [block_id for route in current_routes for block_id in route]
    if len(actual) != len(set(actual)):
        raise ValueError("a welding block is assigned more than once")
    missing_blocks = sorted(set(expected) - set(actual))
    extra_blocks = sorted(set(actual) - set(expected))
    if missing_blocks or extra_blocks:
        raise ValueError(f"parent coverage failure; missing={missing_blocks}, extra={extra_blocks}")

    result_routes = tuple(Route(robot, tuple(current_routes[robot])) for robot in range(4))
    return CanonicalSolution(ordered_parents, ordered_patterns, result_routes, revision)


def block_map(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
):
    parents = {parent.parent_id: parent for parent in solution.parents}
    result = {}
    for pattern in solution.patterns:
        for block in blocks_for_pattern(
            parents[pattern.parent_id],
            pattern,
            config,
            x_split_validator=x_split_validator,
            require_formal_x_validation=True,
        ):
            result[block.block_id] = block
    return result


def official_metrics(
    solution: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
) -> OfficialMetrics:
    if not schedule.feasible or schedule.cmax is None:
        raise ValueError("official metrics require a feasible schedule")
    blocks = block_map(solution, config, x_split_validator=x_split_validator)
    process_loads: list[float] = []
    total_empty = 0.0
    for route in solution.routes:
        route_blocks = [blocks[block_id] for block_id in route.block_ids]
        process_loads.append(sum(config.process_time(block.length) for block in route_blocks))
        # T_empty is defined for the final directed route. Recover directions from WELD rows.
        welds = sorted(
            (
                operation for operation in schedule.operations
                if operation.robot_id == route.robot_id and operation.kind is OperationKind.WELD
            ),
            key=lambda operation: operation.sequence_index,
        )
        for left, right in zip(welds, welds[1:]):
            total_empty += math.dist(left.end, right.start) / config.empty_speed
    total_waiting = sum(
        operation.duration for operation in schedule.operations if operation.kind is OperationKind.WAIT
    )
    parents = {parent.parent_id: parent for parent in solution.parents}
    optional_splits = sum(
        pattern.kind is not SplitKind.WHOLE
        and bool(whole_eligible_rails(parents[pattern.parent_id].start, parents[pattern.parent_id].end, config))
        for pattern in solution.patterns
    )
    return OfficialMetrics(
        cmax=schedule.cmax,
        optional_split_count=int(optional_splits),
        process_imbalance=max(process_loads) - min(process_loads),
        total_empty_travel=total_empty,
        total_waiting=total_waiting,
        deterministic_id_order=tuple(route.block_ids for route in solution.routes),
    )
