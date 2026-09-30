from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random
import time
from typing import Mapping, Sequence

from mrta_reference.geometry import (
    blocks_for_pattern,
    generate_y_split_patterns,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
    FormalScope,
    ParentWeld,
    Route,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.solution import block_map, canonicalize

from .common import BaselineResult, CommonBaselineEvaluator, canonical_config_hash


METHOD_ID = "ADAPTED_WAG_VNS_V1"
PAPER_CORE_ID = "WAG_PAPER_CORE_V1"


@dataclass(frozen=True)
class AdaptedWAGConfig:
    n: int = 10
    p: float = 0.50
    imax: int = 500
    factorial_window: int = 5
    factorial_mode: str = "COMMON_MODEL_BOUNDED"
    max_windows: int = 8
    max_factorial_calls: int = 30720
    max_wag_variants: int = 10
    max_route_combinations_per_wag: int = 3

    def __post_init__(self) -> None:
        if min(
            self.n,
            self.imax,
            self.factorial_window,
            self.max_windows,
            self.max_factorial_calls,
            self.max_wag_variants,
            self.max_route_combinations_per_wag,
        ) <= 0:
            raise ValueError("WAG integer parameters must be positive")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("p must be in (0,1]")
        if self.factorial_mode not in {"PAPER_EXACT", "COMMON_MODEL_BOUNDED"}:
            raise ValueError("unknown factorial mode")


def _initial_patterns(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> tuple[SplitPattern, ...]:
    result = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        whole = SplitPattern(parent.parent_id, SplitKind.WHOLE)
        if whole_eligible_rails(parent.start, parent.end, config):
            result.append(whole)
            continue
        splits = generate_y_split_patterns(parent, config)
        if not splits:
            raise ValueError(f"{parent.parent_id}: no legal WHOLE/Y_SPLIT pattern")
        result.append(splits[0])
    return tuple(result)


def _blocks_for(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    config: ScientificConfig,
) -> dict[str, WeldingBlock]:
    by_parent = {parent.parent_id: parent for parent in parents}
    return {
        block.block_id: block
        for pattern in patterns
        for block in blocks_for_pattern(by_parent[pattern.parent_id], pattern, config)
    }


def _route_cost(
    route: Sequence[str], blocks: Mapping[str, WeldingBlock], config: ScientificConfig
) -> float:
    if not route:
        return 0.0
    states = {direction: config.process_time(blocks[route[0]].length) for direction in (0, 1)}
    for index in range(1, len(route)):
        old = blocks[route[index - 1]]
        current = blocks[route[index]]
        next_states = {}
        for direction in (0, 1):
            start = current.start if direction == 0 else current.end
            next_states[direction] = min(
                cost
                + math.dist(old.end if previous == 0 else old.start, start)
                / config.empty_speed
                + config.process_time(current.length)
                for previous, cost in states.items()
            )
        states = next_states
    return min(states.values())


def _home(robot: int, config: ScientificConfig) -> tuple[float, float]:
    x = config.workspace_x[0] if robot in (0, 2) else config.workspace_x[1]
    y = config.by[1] if robot < 2 else config.by[0]
    return (x, y)


def nearest_addition(
    block_ids: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    pending = set(block_ids)
    result = []
    point = _home(robot, config)
    while pending:
        options = []
        for block_id in pending:
            block = blocks[block_id]
            for direction, (start, end) in enumerate(
                ((block.start, block.end), (block.end, block.start))
            ):
                options.append((math.dist(point, start), block_id, direction, end))
        _, selected, _, point = min(options, key=lambda item: item[:3])
        result.append(selected)
        pending.remove(selected)
    return tuple(result)


def farthest_insertion(
    block_ids: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    if not block_ids:
        return ()
    first = max(
        block_ids,
        key=lambda item: (
            max(blocks[item].start[0], blocks[item].end[0]), item
        ),
    )
    route = [first]
    pending = set(block_ids) - {first}
    while pending:
        before = _route_cost(route, blocks, config)
        best_by_edge = []
        for block_id in sorted(pending):
            insertions = []
            for position in range(len(route) + 1):
                candidate = route[:position] + [block_id] + route[position:]
                insertions.append(
                    (_route_cost(candidate, blocks, config) - before, position)
                )
            least_delta, position = min(insertions)
            best_by_edge.append((least_delta, block_id, position))
        _, selected, position = max(best_by_edge, key=lambda item: (item[0], item[1]))
        route.insert(position, selected)
        pending.remove(selected)
    return tuple(route)


def two_opt(
    route: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    consider_final_edge: bool,
) -> tuple[str, ...]:
    current = tuple(route)
    current_cost = _route_cost(current, blocks, config)
    changed = True
    while changed:
        changed = False
        upper = len(current) if consider_final_edge else max(0, len(current) - 1)
        for left in range(max(0, upper - 1)):
            for right in range(left + 1, upper):
                candidate = current[:left] + tuple(reversed(current[left : right + 1])) + current[right + 1 :]
                cost = _route_cost(candidate, blocks, config)
                if (cost, candidate) < (current_cost - 1.0e-12, current):
                    current, current_cost, changed = candidate, cost, True
                    break
            if changed:
                break
    return current


def factorial_edge_combination(
    route: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    window_size: int,
    consider_final_edge: bool,
    exact: bool,
    max_windows: int,
    max_factorial_calls: int,
    deadline: float | None = None,
) -> tuple[tuple[str, ...], int, int]:
    """Enumerate the paper permutation/direction neighborhood for each window."""
    current = tuple(route)
    if len(current) < 2:
        return current, 0, 0
    size = min(window_size, len(current))
    last_start = len(current) - size
    if not consider_final_edge:
        last_start -= 1
    starts = list(range(max(0, last_start) + 1))
    if not exact:
        starts = starts[:max_windows]
    calls = 0
    windows = 0
    changed = True
    while changed:
        changed = False
        for start in starts:
            if deadline is not None and time.perf_counter() >= deadline:
                return current, calls, windows
            windows += 1
            stop = start + size
            original = current[start:stop]
            best = current
            best_cost = _route_cost(current, blocks, config)
            for permutation in itertools.permutations(original):
                if deadline is not None and time.perf_counter() >= deadline:
                    return current, calls, windows
                direction_combinations = 2 ** len(permutation)
                if not exact and calls + direction_combinations > max_factorial_calls:
                    return current, calls, windows
                calls += direction_combinations
                candidate = current[:start] + permutation + current[stop:]
                cost = _route_cost(candidate, blocks, config)
                if (cost, candidate) < (best_cost - 1.0e-12, best):
                    best, best_cost = candidate, cost
            if best != current:
                current = best
                changed = True
                break
        if not exact:
            break
    return current, calls, windows


def improved_route_set(
    block_ids: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    deadline: float | None = None,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], int]:
    exact = wag_config.factorial_mode == "PAPER_EXACT"
    nearest = nearest_addition(block_ids, robot, blocks, config)
    farthest = farthest_insertion(block_ids, blocks, config)

    i_r1 = two_opt(nearest, blocks, config, consider_final_edge=True)
    i_r1, calls1, _ = factorial_edge_combination(
        i_r1,
        blocks,
        config,
        window_size=wag_config.factorial_window,
        consider_final_edge=True,
        exact=exact,
        max_windows=wag_config.max_windows,
        max_factorial_calls=wag_config.max_factorial_calls,
        deadline=deadline,
    )
    i_r2 = two_opt(farthest, blocks, config, consider_final_edge=False)
    i_r2, calls2, _ = factorial_edge_combination(
        i_r2,
        blocks,
        config,
        window_size=wag_config.factorial_window,
        consider_final_edge=False,
        exact=exact,
        max_windows=wag_config.max_windows,
        max_factorial_calls=wag_config.max_factorial_calls,
        deadline=deadline,
    )
    i_r3 = two_opt(i_r2, blocks, config, consider_final_edge=True)
    i_r3, calls3, _ = factorial_edge_combination(
        i_r3,
        blocks,
        config,
        window_size=wag_config.factorial_window,
        consider_final_edge=True,
        exact=exact,
        max_windows=wag_config.max_windows,
        max_factorial_calls=wag_config.max_factorial_calls,
        deadline=deadline,
    )
    return (i_r1, i_r2, i_r3), calls1 + calls2 + calls3


def _balanced_boundary_assign(
    block_ids: Sequence[str],
    pair: tuple[int, int],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[int, tuple[str, ...]]:
    ordered = tuple(
        sorted(
            block_ids,
            key=lambda item: (
                min(blocks[item].start[0], blocks[item].end[0]), item
            ),
        )
    )
    best = None
    for boundary in range(len(ordered) + 1):
        left = sum(config.process_time(blocks[item].length) for item in ordered[:boundary])
        right = sum(config.process_time(blocks[item].length) for item in ordered[boundary:])
        candidate = (max(left, right), abs(left - right), boundary)
        if best is None or candidate < best:
            best = candidate
    return best[2], ordered  # type: ignore[index]


def generate_wag_assignments(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    config: ScientificConfig,
    *,
    max_variants: int,
) -> tuple[CanonicalSolution, ...]:
    blocks = _blocks_for(parents, patterns, config)
    rail_groups: dict[str, list[str]] = {"UPPER": [], "LOWER": []}
    flexible: list[str] = []
    for block_id, block in blocks.items():
        upper = any(robot_is_eligible(block, robot, config) for robot in (0, 1))
        lower = any(robot_is_eligible(block, robot, config) for robot in (2, 3))
        if upper and lower:
            flexible.append(block_id)
        elif upper:
            rail_groups["UPPER"].append(block_id)
        elif lower:
            rail_groups["LOWER"].append(block_id)
        else:
            raise ValueError(f"{block_id}: no eligible rail")
    rail_load = {
        key: sum(config.process_time(blocks[item].length) for item in values)
        for key, values in rail_groups.items()
    }
    for block_id in sorted(
        flexible,
        key=lambda item: (
            min(blocks[item].start[0], blocks[item].end[0]), item
        ),
    ):
        rail = min(("UPPER", "LOWER"), key=lambda key: (rail_load[key], key))
        rail_groups[rail].append(block_id)
        rail_load[rail] += config.process_time(blocks[block_id].length)

    upper_boundary, upper_order = _balanced_boundary_assign(
        rail_groups["UPPER"], (0, 1), blocks, config
    )
    lower_boundary, lower_order = _balanced_boundary_assign(
        rail_groups["LOWER"], (2, 3), blocks, config
    )
    offsets = sorted(
        itertools.product(range(-2, 3), repeat=2),
        key=lambda item: (abs(item[0]) + abs(item[1]), item),
    )
    result = []
    seen = set()
    for upper_offset, lower_offset in offsets:
        ub = min(len(upper_order), max(0, upper_boundary + upper_offset))
        lb = min(len(lower_order), max(0, lower_boundary + lower_offset))
        routes = (
            Route(0, upper_order[:ub]),
            Route(1, upper_order[ub:]),
            Route(2, lower_order[:lb]),
            Route(3, lower_order[lb:]),
        )
        solution = canonicalize(parents, patterns, routes, config)
        if solution.canonical_hash not in seen:
            seen.add(solution.canonical_hash)
            result.append(solution)
        if len(result) >= max_variants:
            break
    return tuple(result)


def _route_combinations(
    assignment: CanonicalSolution,
    config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    deadline: float,
) -> tuple[list[CanonicalSolution], int]:
    blocks = block_map(assignment, config)
    variants = []
    factorial_calls = 0
    for route in assignment.routes:
        route_set, calls = improved_route_set(
            route.block_ids,
            route.robot_id,
            blocks,
            config,
            wag_config,
            deadline=deadline,
        )
        variants.append(route_set)
        factorial_calls += calls
    combinations = []
    for indices in itertools.product(range(3), repeat=4):
        routes = tuple(Route(robot, variants[robot][indices[robot]]) for robot in range(4))
        solution = canonicalize(assignment.parents, assignment.patterns, routes, config)
        cost = max(
            (_route_cost(route.block_ids, blocks, config) for route in routes),
            default=0.0,
        )
        combinations.append((cost, indices, solution))
    combinations.sort(key=lambda item: (item[0], item[1], item[2].canonical_hash))
    return [item[2] for item in combinations], factorial_calls


def _same_rail_neighbors(robot: int) -> tuple[int, ...]:
    return ({0: (1,), 1: (0,), 2: (3,), 3: (2,)})[robot]


def wag_move_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    n: int,
    heavy_robot: int,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config)
    if not solution.routes[heavy_robot].block_ids:
        return ()
    result = []
    for _ in range(n):
        routes = [list(route.block_ids) for route in solution.routes]
        block_id = routes[heavy_robot][rng.randrange(len(routes[heavy_robot]))]
        destinations = [
            robot
            for robot in _same_rail_neighbors(heavy_robot)
            if robot_is_eligible(blocks[block_id], robot, config)
        ]
        if sum(robot_is_eligible(blocks[block_id], robot, config) for robot in range(4)) == 4:
            destinations = list(range(4))
            destinations.remove(heavy_robot)
        if not destinations:
            continue
        destination = destinations[rng.randrange(len(destinations))]
        routes[heavy_robot].remove(block_id)
        routes[destination].append(block_id)
        result.append(
            canonicalize(
                solution.parents,
                solution.patterns,
                {robot: tuple(route) for robot, route in enumerate(routes)},
                config,
            )
        )
    return tuple(result)


def wag_swap_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    n: int,
    heavy_robot: int,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config)
    result = []
    if not solution.routes[heavy_robot].block_ids:
        return ()
    for _ in range(n):
        neighbor = _same_rail_neighbors(heavy_robot)[0]
        if not solution.routes[neighbor].block_ids:
            continue
        routes = [list(route.block_ids) for route in solution.routes]
        left = routes[heavy_robot][rng.randrange(len(routes[heavy_robot]))]
        right = routes[neighbor][rng.randrange(len(routes[neighbor]))]
        if not (
            robot_is_eligible(blocks[left], neighbor, config)
            and robot_is_eligible(blocks[right], heavy_robot, config)
        ):
            continue
        li = routes[heavy_robot].index(left)
        ri = routes[neighbor].index(right)
        routes[heavy_robot][li], routes[neighbor][ri] = right, left
        result.append(
            canonicalize(
                solution.parents,
                solution.patterns,
                {robot: tuple(route) for robot, route in enumerate(routes)},
                config,
            )
        )
    return tuple(result)


def wag_lns_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    p: float,
    max_variants: int,
) -> tuple[CanonicalSolution, ...]:
    routes = [list(route.block_ids) for route in solution.routes]
    removed = []
    for robot in range(4):
        count = max(1, math.ceil(len(routes[robot]) * p)) if routes[robot] else 0
        rng.shuffle(routes[robot])
        removed.extend(routes[robot][:count])
        routes[robot] = routes[robot][count:]
    blocks = block_map(solution, config)
    removed.sort(
        key=lambda item: (
            min(blocks[item].start[0], blocks[item].end[0]), item
        )
    )
    loads = [
        sum(config.process_time(blocks[item].length) for item in route) for route in routes
    ]
    for block_id in removed:
        eligible = [
            robot for robot in range(4) if robot_is_eligible(blocks[block_id], robot, config)
        ]
        robot = min(eligible, key=lambda item: (loads[item], item))
        routes[robot].append(block_id)
        loads[robot] += config.process_time(blocks[block_id].length)
    base = canonicalize(
        solution.parents,
        solution.patterns,
        {robot: tuple(route) for robot, route in enumerate(routes)},
        config,
    )
    return generate_wag_assignments(
        solution.parents, solution.patterns, config, max_variants=max_variants
    ) + (base,)


def toggle_optional_y(
    solution: CanonicalSolution,
    config: ScientificConfig,
) -> CanonicalSolution | None:
    current = {pattern.parent_id: pattern for pattern in solution.patterns}
    for parent in solution.parents:
        splits = generate_y_split_patterns(parent, config)
        if not splits or not whole_eligible_rails(parent.start, parent.end, config):
            continue
        current[parent.parent_id] = (
            splits[0]
            if current[parent.parent_id].kind is SplitKind.WHOLE
            else SplitPattern(parent.parent_id, SplitKind.WHOLE)
        )
        patterns = tuple(current[item.parent_id] for item in solution.parents)
        return generate_wag_assignments(
            solution.parents, patterns, config, max_variants=1
        )[0]
    return None


def _evaluate_routed_assignment(
    assignment: CanonicalSolution,
    scientific_config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    evaluator: CommonBaselineEvaluator,
    *,
    source: str,
) -> None:
    construction_started = time.perf_counter()
    combinations, factorial_calls = _route_combinations(
        assignment,
        scientific_config,
        wag_config,
        deadline=evaluator.deadline,
    )
    elapsed = time.perf_counter() - construction_started
    evaluator.accounting.construction_time += elapsed
    evaluator.accounting.factorial_local_search_time += elapsed
    evaluator.accounting.cheap_candidate_operations += factorial_calls
    for index, candidate in enumerate(
        combinations[: wag_config.max_route_combinations_per_wag]
    ):
        if evaluator.expired:
            break
        evaluator.evaluate(candidate, source=f"{source}_ROUTE_COMBINATION_{index}")


def run_adapted_wag_vns(
    parents: Sequence[ParentWeld],
    scientific_config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    seed: int,
    time_limit: float,
    scope: FormalScope,
    checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
) -> BaselineResult:
    rng = random.Random(seed)
    evaluator = CommonBaselineEvaluator(
        parents,
        scientific_config,
        scope,
        time_limit=time_limit,
        checkpoints=checkpoints,
    )
    init_started = time.perf_counter()
    try:
        patterns = _initial_patterns(parents, scientific_config)
        assignments = generate_wag_assignments(
            parents,
            patterns,
            scientific_config,
            max_variants=wag_config.max_wag_variants,
        )
    except (ArithmeticError, OverflowError, ValueError) as error:
        evaluator.reject_construction(f"WAG initialization: {error}")
        assignments = ()
    for index, assignment in enumerate(assignments):
        if evaluator.expired:
            break
        evaluator.evaluate(assignment, source=f"WAG_INITIAL_{index}_NATIVE")
        if evaluator.expired:
            break
        _evaluate_routed_assignment(
            assignment,
            scientific_config,
            wag_config,
            evaluator,
            source=f"WAG_INITIAL_{index}",
        )
        if evaluator.best is not None:
            break
    initialization_time = time.perf_counter() - init_started

    iterations = 0
    while (
        evaluator.best is not None
        and not evaluator.expired
        and iterations < wag_config.imax
    ):
        iterations += 1
        best_solution = evaluator.best.solution
        if evaluator.best.schedule is not None:
            heavy = max(
                range(4),
                key=lambda robot: (
                    evaluator.best.schedule.robot_completion[robot], -robot
                ),
            )
        else:
            blocks = block_map(best_solution, scientific_config)
            heavy = max(
                range(4),
                key=lambda robot: (
                    _route_cost(best_solution.routes[robot].block_ids, blocks, scientific_config),
                    -robot,
                ),
            )
        operator = rng.randrange(3)
        if operator == 0:
            candidates = wag_move_candidates(
                best_solution,
                scientific_config,
                rng,
                n=wag_config.n,
                heavy_robot=heavy,
            )
            name = "MOVE"
        elif operator == 1:
            candidates = wag_swap_candidates(
                best_solution,
                scientific_config,
                rng,
                n=wag_config.n,
                heavy_robot=heavy,
            )
            name = "SWAP"
        else:
            candidates = wag_lns_candidates(
                best_solution,
                scientific_config,
                rng,
                p=wag_config.p,
                max_variants=wag_config.max_wag_variants,
            )
            name = "LNS"
        if iterations % 3 == 0:
            optional = toggle_optional_y(best_solution, scientific_config)
            if optional is not None:
                candidates = candidates + (optional,)
        seen = set()
        for index, assignment in enumerate(candidates):
            if evaluator.expired:
                break
            if assignment.canonical_hash in seen:
                continue
            seen.add(assignment.canonical_hash)
            _evaluate_routed_assignment(
                assignment,
                scientific_config,
                wag_config,
                evaluator,
                source=f"WAG_{name}_{iterations}_{index}",
            )

    return evaluator.finish(
        method_id=METHOD_ID,
        method_config_hash=canonical_config_hash(wag_config),
        iterations=iterations,
        initialization_time=initialization_time,
        diagnostics=(
            "paper three-robot conflict scheduler is excluded from common-model fitness",
            "ADAPTED_WAG_ASSIGNMENT_V1",
        ),
    )
