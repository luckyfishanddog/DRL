from __future__ import annotations

import itertools
import math
import random

import pytest

from mrta_exact import ExactSolveStatus, solve_exact_micro
from mrta_reference.geometry import optimize_directions, oriented_endpoints
from mrta_reference.model import ParentWeld, Route, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scheduler import reference_schedule
from mrta_reference.solution import block_map, canonicalize
from mrta_search.direction import DirectionStatus, optimize_directions_with_initial_feasibility


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _whole(parents, routes, config=FAST):
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        routes,
        config,
    )


def _initial_legal(solution, vectors, config):
    blocks = block_map(solution, config)
    points = {}
    for robot, route in enumerate(solution.routes):
        if route.block_ids:
            points[robot] = oriented_endpoints(blocks[route.block_ids[0]], vectors[robot][0])[0]
    for left, right in ((0, 1), (2, 3)):
        if left in points and right in points:
            if points[left][0] + config.interference_dx > points[right][0] + config.numeric_epsilon:
                return False
    for first, second in itertools.combinations(points, 2):
        if (
            abs(points[first][0] - points[second][0]) <= config.interference_dx + config.numeric_epsilon
            and abs(points[first][1] - points[second][1]) <= config.interference_dy + config.numeric_epsilon
        ):
            return False
    return True


def _travel(solution, vectors, config):
    blocks = block_map(solution, config)
    total = 0.0
    for robot, route in enumerate(solution.routes):
        for index in range(len(route.block_ids) - 1):
            _, end = oriented_endpoints(blocks[route.block_ids[index]], vectors[robot][index])
            start, _ = oriented_endpoints(blocks[route.block_ids[index + 1]], vectors[robot][index + 1])
            total += math.dist(end, start) / config.empty_speed
    return total


def test_free_direction_tie_can_choose_illegal_first_orientation() -> None:
    parents = (
        ParentWeld("left", (5.0, 10.0), (1.0, 10.0)),
        ParentWeld("right", (4.0, 11.0), (6.0, 11.0)),
    )
    solution = _whole(parents, {0: ("left::whole",), 1: ("right::whole",)})
    blocks = block_map(solution, FAST)
    free = tuple(
        optimize_directions(tuple(blocks[item] for item in route.block_ids), FAST).orientations
        for route in solution.routes
    )
    assert free == ((0,), (0,), (), ())
    assert not _initial_legal(solution, free, FAST)

    constrained = optimize_directions_with_initial_feasibility(solution, FAST)
    assert constrained.status is DirectionStatus.FEASIBLE
    assert constrained.directions != free
    assert _initial_legal(solution, constrained.directions, FAST)
    scheduled = reference_schedule(
        solution,
        FAST,
        orientations={robot: constrained.directions[robot] for robot in range(4)},
    )
    assert scheduled.status is ScheduleStatus.FEASIBLE


@pytest.mark.parametrize("block_count", range(1, 7))
def test_constrained_dp_matches_exhaustive_direction_optimum(block_count: int) -> None:
    rng = random.Random(10_000 + block_count)
    parents = tuple(
        ParentWeld(
            f"p{i}",
            (rng.uniform(0.0, 18.0), 2.0),
            (rng.uniform(0.0, 18.0), 2.5),
        )
        for i in range(block_count)
    )
    solution = _whole(parents, {2: tuple(f"p{i}::whole" for i in range(block_count))})
    result = optimize_directions_with_initial_feasibility(solution, FAST)
    brute = min(
        (_travel(solution, ((), (), bits, ()), FAST), bits)
        for bits in itertools.product((0, 1), repeat=block_count)
        if _initial_legal(solution, ((), (), bits, ()), FAST)
    )
    assert result.total_empty_travel == pytest.approx(brute[0])
    assert result.directions[2] == brute[1]


def test_empty_routes_four_active_robots_and_deterministic_tie() -> None:
    empty = canonicalize((), (), {}, FAST)
    empty_result = optimize_directions_with_initial_feasibility(empty, FAST)
    assert empty_result.directions == ((), (), (), ())
    assert empty_result.total_empty_travel == 0.0

    specs = (("a", 1.0, 10.0, 0), ("b", 10.0, 10.0, 1), ("c", 1.0, 2.0, 2), ("d", 10.0, 2.0, 3))
    parents = tuple(ParentWeld(name, (x, y), (x + 1.0, y)) for name, x, y, _ in specs)
    routes = {robot: (f"{name}::whole",) for name, _, _, robot in specs}
    solution = _whole(parents, routes)
    first = optimize_directions_with_initial_feasibility(solution, FAST)
    second = optimize_directions_with_initial_feasibility(solution, FAST)
    assert first == second
    assert first.directions == ((0,), (0,), (0,), (0,))
    assert first.legal_first_combinations > 0


def test_nominal_scientific_config_direction_reference_and_micro_exact_smoke() -> None:
    config = ScientificConfig()
    parents = (ParentWeld("nominal", (1.0, 2.0), (1.3, 2.0)),)
    solution = _whole(parents, {2: ("nominal::whole",)}, config)
    direction = optimize_directions_with_initial_feasibility(solution, config)
    schedule = reference_schedule(
        solution,
        config,
        orientations={robot: direction.directions[robot] for robot in range(4)},
    )
    exact = solve_exact_micro(parents, config)
    assert direction.status is DirectionStatus.FEASIBLE
    assert schedule.status is ScheduleStatus.FEASIBLE
    assert math.isfinite(schedule.cmax)
    assert exact.status is ExactSolveStatus.OPTIMAL
    assert exact.best_cmax == pytest.approx(schedule.cmax)

