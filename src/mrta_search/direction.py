from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
import math

from mrta_reference.geometry import oriented_endpoints
from mrta_reference.model import CanonicalSolution, ScientificConfig
from mrta_reference.solution import block_map


DirectionVectors = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]


class DirectionStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    NO_LEGAL_FIRST_ORIENTATION = "NO_LEGAL_FIRST_ORIENTATION"


@dataclass(frozen=True)
class ConstrainedDirectionResult:
    status: DirectionStatus
    directions: DirectionVectors
    total_empty_travel: float | None
    diagnostics: tuple[str, ...] = ()
    legal_first_combinations: int = 0
    route_dp_calls: int = 0


def _fixed_first_dp(blocks, first: int, config: ScientificConfig):
    if not blocks:
        return 0.0, ()
    states: dict[int, tuple[float, tuple[int, ...]]] = {
        first: (0.0, (first,))
    }
    for index in range(1, len(blocks)):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for direction in (0, 1):
            start, _ = oriented_endpoints(blocks[index], direction)
            options = []
            for previous, (cost, vector) in states.items():
                _, end = oriented_endpoints(blocks[index - 1], previous)
                options.append(
                    (
                        cost + math.dist(end, start) / config.empty_speed,
                        vector + (direction,),
                    )
                )
            next_states[direction] = min(options, key=lambda item: (item[0], item[1]))
        states = next_states
    return min(states.values(), key=lambda item: (item[0], item[1]))


def _initial_error(points: dict[int, tuple[float, float]], config: ScientificConfig):
    for left, right in ((0, 1), (2, 3)):
        if left in points and right in points:
            if points[left][0] + config.interference_dx > points[right][0] + config.numeric_epsilon:
                return f"illegal first-task same-rail order R{left}/R{right}"
    robots = sorted(points)
    for index, first in enumerate(robots):
        for second in robots[index + 1 :]:
            a, b = points[first], points[second]
            if (
                abs(a[0] - b[0]) <= config.interference_dx + config.numeric_epsilon
                and abs(a[1] - b[1]) <= config.interference_dy + config.numeric_epsilon
            ):
                return f"initial theoretical interference R{first}/R{second}"
    return None


def optimize_directions_with_initial_feasibility(
    solution: CanonicalSolution,
    config: ScientificConfig,
) -> ConstrainedDirectionResult:
    """Minimize directed empty travel over every legal first-orientation tuple.

    This routine does not call the reference scheduler and does not optimize WAIT/Cmax.
    """
    blocks = block_map(solution, config)
    routes = tuple(tuple(blocks[item] for item in route.block_ids) for route in solution.routes)
    active = tuple(robot for robot, route in enumerate(routes) if route)
    cached: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    for robot in active:
        for first in (0, 1):
            cached[(robot, first)] = _fixed_first_dp(routes[robot], first, config)

    best = None
    rejected: list[str] = []
    legal = 0
    for bits in itertools.product((0, 1), repeat=len(active)):
        first_by_robot = dict(zip(active, bits))
        points = {
            robot: oriented_endpoints(routes[robot][0], first_by_robot[robot])[0]
            for robot in active
        }
        error = _initial_error(points, config)
        if error is not None:
            rejected.append(error)
            continue
        legal += 1
        vectors: list[tuple[int, ...]] = []
        total = 0.0
        for robot in range(4):
            if robot not in first_by_robot:
                vectors.append(())
                continue
            cost, vector = cached[(robot, first_by_robot[robot])]
            total += cost
            vectors.append(vector)
        directions: DirectionVectors = tuple(vectors)  # type: ignore[assignment]
        candidate = (total, tuple(value for vector in directions for value in vector), directions)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    calls = 2 * len(active)
    if best is None:
        diagnostic = tuple(dict.fromkeys(rejected)) or ("no legal initial orientation",)
        zeros: DirectionVectors = tuple(
            tuple(0 for _ in route) for route in routes
        )  # type: ignore[assignment]
        return ConstrainedDirectionResult(
            DirectionStatus.NO_LEGAL_FIRST_ORIENTATION,
            zeros,
            None,
            diagnostic,
            0,
            calls,
        )
    return ConstrainedDirectionResult(
        DirectionStatus.FEASIBLE,
        best[2],
        best[0],
        (),
        legal,
        calls,
    )

