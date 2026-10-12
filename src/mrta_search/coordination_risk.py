"""Cheap, pre-direction coordination-risk proxies for C2 screening."""
from __future__ import annotations

from dataclasses import dataclass

from mrta_reference.model import CanonicalSolution, ScientificConfig
from mrta_reference.solution import block_map


@dataclass(frozen=True)
class CoordinationRisk:
    """Static geometry only; these values never imply dynamic infeasibility."""

    same_rail_overlap: float
    first_task_order_excess: float

    @property
    def total(self) -> float:
        return self.same_rail_overlap + self.first_task_order_excess


def _merged_intervals(intervals: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for lower, upper in sorted(intervals):
        if not merged or lower > merged[-1][1]:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return tuple((lower, upper) for lower, upper in merged)


def _intersection_length(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> float:
    return sum(
        max(0.0, min(first_upper, second_upper) - max(first_lower, second_lower))
        for first_lower, first_upper in first
        for second_lower, second_upper in second
    )


def coordination_risk(
    solution: CanonicalSolution,
    config: ScientificConfig,
) -> CoordinationRisk:
    """Return static geometry available before direction DP or scheduling."""
    blocks = block_map(solution, config)
    routes = {route.robot_id: route.block_ids for route in solution.routes}
    half_clearance = config.interference_dx / 2.0
    overlap = 0.0
    first_order_excess = 0.0
    for left_robot, right_robot in ((0, 1), (2, 3)):
        occupied = {}
        for robot in (left_robot, right_robot):
            occupied[robot] = _merged_intervals([
                (min(blocks[b].start[0], blocks[b].end[0]) - half_clearance,
                 max(blocks[b].start[0], blocks[b].end[0]) + half_clearance)
                for b in routes[robot]
            ])
        overlap += _intersection_length(occupied[left_robot], occupied[right_robot])
        if routes[left_robot] and routes[right_robot]:
            left = blocks[routes[left_robot][0]]
            right = blocks[routes[right_robot][0]]
            first_order_excess += max(
                0.0,
                max(left.start[0], left.end[0])
                + config.interference_dx
                - min(right.start[0], right.end[0]),
            )
    return CoordinationRisk(overlap, first_order_excess)
