from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Mapping, Sequence

from mrta_reference.geometry import (
    blocks_for_pattern,
    generate_y_split_patterns,
    oriented_endpoints,
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

from .common import (
    BaselineResult,
    CommonBaselineEvaluator,
    EvaluatedCandidate,
    canonical_config_hash,
)


METHOD_ID = "ADAPTED_HGA_V1"
PAPER_CORE_ID = "HGA_PAPER_CORE_V1"


@dataclass(frozen=True)
class AdaptedHGAConfig:
    mu: int = 20
    lambda_size: int = 10
    alpha: int = 20
    max_iterations: int = 200000
    initialization_reference_limit: int = 20
    vnd_max_passes: int = 2
    vnd_candidate_cap: int = 256
    pattern_mutation_probability: float = 0.15

    def __post_init__(self) -> None:
        if min(self.mu, self.lambda_size, self.alpha, self.max_iterations) <= 0:
            raise ValueError("HGA integer parameters must be positive")
        if self.initialization_reference_limit <= 0:
            raise ValueError("initialization_reference_limit must be positive")
        if min(self.vnd_max_passes, self.vnd_candidate_cap) <= 0:
            raise ValueError("VND bounds must be positive")
        if not 0.0 <= self.pattern_mutation_probability <= 1.0:
            raise ValueError("pattern_mutation_probability must be in [0,1]")


@dataclass
class _Individual:
    solution: CanonicalSolution
    proxy: tuple[float, float, str]
    evaluated: EvaluatedCandidate | None = None


def legal_pattern_options(
    parent: ParentWeld, config: ScientificConfig
) -> tuple[SplitPattern, ...]:
    y = tuple(generate_y_split_patterns(parent, config))
    if whole_eligible_rails(parent.start, parent.end, config):
        return (SplitPattern(parent.parent_id, SplitKind.WHOLE),) + y
    if not y:
        raise ValueError(f"{parent.parent_id}: no legal WHOLE/Y_SPLIT pattern")
    return y


def initial_patterns(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> tuple[SplitPattern, ...]:
    result = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        options = legal_pattern_options(parent, config)
        whole = next((value for value in options if value.kind is SplitKind.WHOLE), None)
        result.append(whole if whole is not None else options[0])
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
    states = {
        direction: (config.process_time(blocks[route[0]].length), (direction,))
        for direction in (0, 1)
    }
    for index in range(1, len(route)):
        current = blocks[route[index]]
        previous = blocks[route[index - 1]]
        next_states = {}
        for direction in (0, 1):
            start, _ = oriented_endpoints(current, direction)
            options = []
            for old_direction, (cost, vector) in states.items():
                _, end = oriented_endpoints(previous, old_direction)
                options.append(
                    (
                        cost
                        + math.dist(end, start) / config.empty_speed
                        + config.process_time(current.length),
                        vector + (direction,),
                    )
                )
            next_states[direction] = min(options, key=lambda item: item)
        states = next_states
    return min(states.values(), key=lambda item: item)[0]


def hga_surrogate(
    solution: CanonicalSolution, config: ScientificConfig
) -> tuple[float, float, str]:
    blocks = block_map(solution, config)
    costs = tuple(_route_cost(route.block_ids, blocks, config) for route in solution.routes)
    return (max(costs, default=0.0), sum(costs), solution.canonical_hash)


def _best_insertion(
    block_id: str,
    routes: list[list[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    shortest_route_only: bool,
) -> tuple[int, int]:
    eligible = [
        robot
        for robot in range(4)
        if robot_is_eligible(blocks[block_id], robot, config)
    ]
    if not eligible:
        raise ValueError(f"{block_id}: no eligible robot")
    if shortest_route_only:
        loads = {robot: _route_cost(routes[robot], blocks, config) for robot in eligible}
        minimum = min(loads.values())
        eligible = [robot for robot in eligible if abs(loads[robot] - minimum) <= 1.0e-12]
    options = []
    for robot in eligible:
        before = _route_cost(routes[robot], blocks, config)
        for position in range(len(routes[robot]) + 1):
            candidate = routes[robot][:position] + [block_id] + routes[robot][position:]
            delta = _route_cost(candidate, blocks, config) - before
            options.append((delta, robot, position))
    _, robot, position = min(options)
    return robot, position


def initialize_hga_solution(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    rng: random.Random,
    *,
    patterns: Sequence[SplitPattern] | None = None,
) -> CanonicalSolution:
    selected = tuple(patterns or initial_patterns(parents, config))
    blocks = _blocks_for(parents, selected, config)
    pending = list(blocks)
    routes: list[list[str]] = [[] for _ in range(4)]
    while pending:
        index = rng.randrange(len(pending))
        block_id = pending.pop(index)
        robot, position = _best_insertion(
            block_id, routes, blocks, config, shortest_route_only=True
        )
        routes[robot].insert(position, block_id)
    return canonicalize(
        parents, selected, {robot: tuple(route) for robot, route in enumerate(routes)}, config
    )


def initialize_hga_region_seed(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    *,
    patterns: Sequence[SplitPattern] | None = None,
) -> CanonicalSolution:
    """Adapt the paper's region decomposition to two rails and four robots."""
    selected = tuple(patterns or initial_patterns(parents, config))
    blocks = _blocks_for(parents, selected, config)
    rail_groups: list[list[str]] = [[], []]
    flexible: list[str] = []
    rail_loads = [0.0, 0.0]
    for block_id, block in blocks.items():
        eligible_rails = [
            rail
            for rail, pair in enumerate(((0, 1), (2, 3)))
            if any(robot_is_eligible(block, robot, config) for robot in pair)
        ]
        if len(eligible_rails) == 2:
            flexible.append(block_id)
        elif eligible_rails:
            rail_groups[eligible_rails[0]].append(block_id)
            rail_loads[eligible_rails[0]] += config.process_time(block.length)
        else:
            raise ValueError(f"{block_id}: no eligible rail")
    for block_id in sorted(flexible):
        block = blocks[block_id]
        rail = min(range(2), key=lambda value: (rail_loads[value], value))
        rail_groups[rail].append(block_id)
        rail_loads[rail] += config.process_time(block.length)
    routes: list[list[str]] = [[] for _ in range(4)]
    for rail, block_ids in enumerate(rail_groups):
        ordered = sorted(
            block_ids,
            key=lambda item: (min(blocks[item].start[0], blocks[item].end[0]), item),
        )
        boundary = min(
            range(len(ordered) + 1),
            key=lambda position: (
                max(
                    sum(
                        config.process_time(blocks[item].length)
                        for item in ordered[:position]
                    ),
                    sum(
                        config.process_time(blocks[item].length)
                        for item in ordered[position:]
                    ),
                ),
                position,
            ),
        )
        routes[2 * rail] = ordered[:boundary]
        routes[2 * rail + 1] = ordered[boundary:]
    return canonicalize(
        parents, selected, {robot: tuple(route) for robot, route in enumerate(routes)}, config
    )


def _eligible_routes(
    routes: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> bool:
    return all(
        robot_is_eligible(blocks[block_id], robot, config)
        for robot, route in enumerate(routes)
        for block_id in route
    )


def _location(routes: Sequence[Sequence[str]], block_id: str) -> tuple[int, int]:
    for robot, route in enumerate(routes):
        if block_id in route:
            return robot, route.index(block_id)
    raise ValueError(f"unknown block {block_id}")


def apply_hga_neighborhood(
    solution: CanonicalSolution,
    move: int,
    u: str,
    v: str,
    config: ScientificConfig,
) -> CanonicalSolution | None:
    """Apply paper neighborhoods M1-M6 to the common four-route encoding."""
    routes = [list(route.block_ids) for route in solution.routes]
    blocks = block_map(solution, config)
    ru, iu = _location(routes, u)
    rv, iv = _location(routes, v)
    longest = max(range(4), key=lambda robot: (_route_cost(routes[robot], blocks, config), -robot))
    if move == 1:
        if ru == rv or ru != longest:
            return None
        routes[ru].pop(iu)
        iv = routes[rv].index(v)
        routes[rv].insert(iv + 1, u)
    elif move == 2:
        if ru == rv or longest not in (ru, rv):
            return None
        routes[ru][iu], routes[rv][iv] = routes[rv][iv], routes[ru][iu]
    elif move == 3:
        if ru == rv or longest not in (ru, rv):
            return None
        if iu + 1 >= len(routes[ru]) or iv + 1 >= len(routes[rv]):
            return None
        left = routes[ru][iu : iu + 2]
        right = routes[rv][iv : iv + 2]
        routes[ru][iu : iu + 2] = right
        routes[rv][iv : iv + 2] = left
    elif move == 4:
        if ru != rv or iu == iv:
            return None
        lo, hi = sorted((iu, iv))
        routes[ru][lo : hi + 1] = reversed(routes[ru][lo : hi + 1])
    elif move == 5:
        if ru == rv:
            return None
        left_tail = routes[ru][iu + 1 :]
        right_tail = routes[rv][iv + 1 :]
        routes[ru][iu + 1 :] = right_tail
        routes[rv][iv + 1 :] = left_tail
    elif move == 6:
        if ru == rv:
            return None
        left_tail = routes[ru][iu + 1 :]
        right_tail = routes[rv][iv + 1 :]
        routes[ru][iu + 1 :] = reversed(right_tail)
        routes[rv][iv + 1 :] = reversed(left_tail)
    else:
        raise ValueError("move must be M1..M6")
    if not _eligible_routes(routes, blocks, config):
        return None
    try:
        return canonicalize(
            solution.parents,
            solution.patterns,
            {robot: tuple(route) for robot, route in enumerate(routes)},
            config,
        )
    except ValueError:
        return None


def alpha_nearest(
    solution: CanonicalSolution, block_id: str, alpha: int, config: ScientificConfig
) -> tuple[str, ...]:
    blocks = block_map(solution, config)
    origin = blocks[block_id]
    result = []
    for other_id, other in blocks.items():
        if other_id == block_id:
            continue
        distance = min(
            math.dist(left, right)
            for left in (origin.start, origin.end)
            for right in (other.start, other.end)
        )
        result.append((distance, other_id))
    result.sort()
    return tuple(item[1] for item in result[: min(alpha, len(result))])


def hga_vnd(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    alpha: int,
    max_passes: int,
    candidate_cap: int,
    deadline: float | None = None,
    accounting=None,
) -> CanonicalSolution:
    current = solution
    current_score = hga_surrogate(current, config)
    passes = 0
    candidate_operations = 0
    while passes < max_passes and (deadline is None or time.perf_counter() < deadline):
        passes += 1
        changed = False
        all_ids = tuple(block_id for route in current.routes for block_id in route.block_ids)
        for move in range(1, 7):
            if deadline is not None and time.perf_counter() >= deadline:
                return current
            accepted = None
            for u in all_ids:
                if deadline is not None and time.perf_counter() >= deadline:
                    return current
                for v in alpha_nearest(current, u, alpha, config):
                    if deadline is not None and time.perf_counter() >= deadline:
                        return current
                    if candidate_operations >= candidate_cap:
                        if accounting is not None:
                            accounting.vnd_candidate_cap_hits += 1
                        return current
                    candidate_operations += 1
                    if accounting is not None:
                        accounting.cheap_candidate_operations += 1
                    candidate = apply_hga_neighborhood(current, move, u, v, config)
                    if candidate is None:
                        continue
                    score = hga_surrogate(candidate, config)
                    if score < current_score:
                        accepted = (candidate, score)
                        break
                if accepted is not None:
                    break
            if accepted is not None:
                current, current_score = accepted
                changed = True
                break
        if not changed:
            break
    if passes >= max_passes and changed and accounting is not None:
        accounting.vnd_pass_cap_hits += 1
    return current


def _repair_routes(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    inherited: Sequence[Sequence[str]],
    config: ScientificConfig,
    rng: random.Random,
    *,
    shuffle_missing: bool,
) -> CanonicalSolution:
    blocks = _blocks_for(parents, patterns, config)
    expected = set(blocks)
    routes: list[list[str]] = [[] for _ in range(4)]
    seen: set[str] = set()
    for robot in range(4):
        for block_id in inherited[robot]:
            if (
                block_id in expected
                and block_id not in seen
                and robot_is_eligible(blocks[block_id], robot, config)
            ):
                routes[robot].append(block_id)
                seen.add(block_id)
    missing = sorted(expected - seen)
    if shuffle_missing:
        rng.shuffle(missing)
    for block_id in missing:
        robot, position = _best_insertion(
            block_id, routes, blocks, config, shortest_route_only=False
        )
        routes[robot].insert(position, block_id)
    # The common canonicalizer intentionally collapses consecutive siblings on
    # one robot to WHOLE.  A selected split chromosome must therefore repair
    # that representation artifact before canonicalization.
    for pattern in patterns:
        if pattern.kind is SplitKind.WHOLE:
            continue
        siblings = (f"{pattern.parent_id}::0", f"{pattern.parent_id}::1")
        locations = [_location(routes, block_id) for block_id in siblings]
        if locations[0][0] != locations[1][0] or abs(locations[0][1] - locations[1][1]) != 1:
            continue
        source = locations[1][0]
        routes[source].remove(siblings[1])
        choices = []
        for robot in range(4):
            if robot == source or not robot_is_eligible(blocks[siblings[1]], robot, config):
                continue
            before = _route_cost(routes[robot], blocks, config)
            for position in range(len(routes[robot]) + 1):
                candidate = routes[robot][:position] + [siblings[1]] + routes[robot][position:]
                choices.append(
                    (_route_cost(candidate, blocks, config) - before, robot, position)
                )
        if not choices:
            raise ValueError(f"{pattern.parent_id}: cannot preserve split sibling consistency")
        _, robot, position = min(choices)
        routes[robot].insert(position, siblings[1])
    return canonicalize(
        parents,
        patterns,
        {robot: tuple(route) for robot, route in enumerate(routes)},
        config,
    )


def route_based_crossover(
    parent_a: CanonicalSolution,
    parent_b: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
) -> CanonicalSolution:
    if parent_a.parents != parent_b.parents:
        raise ValueError("parents must describe the same instance")
    pattern_a = {pattern.parent_id: pattern for pattern in parent_a.patterns}
    pattern_b = {pattern.parent_id: pattern for pattern in parent_b.patterns}
    patterns = tuple(
        pattern_b[parent.parent_id] if rng.randrange(2) else pattern_a[parent.parent_id]
        for parent in parent_a.parents
    )
    route_a = rng.randrange(4)
    route_b = rng.randrange(4)
    inherited = [list(route.block_ids) for route in parent_a.routes]
    inherited[route_a] = list(parent_b.routes[route_b].block_ids)
    replaced = set(inherited[route_a])
    for robot in range(4):
        if robot != route_a:
            inherited[robot] = [item for item in inherited[robot] if item not in replaced]
    return _repair_routes(
        parent_a.parents,
        patterns,
        inherited,
        config,
        rng,
        shuffle_missing=True,
    )


def mutate_optional_y(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
) -> CanonicalSolution:
    patterns = {pattern.parent_id: pattern for pattern in solution.patterns}
    candidates = []
    for parent in solution.parents:
        options = legal_pattern_options(parent, config)
        if any(value.kind is SplitKind.WHOLE for value in options) and any(
            value.kind is SplitKind.Y_SPLIT for value in options
        ):
            candidates.append((parent, options))
    if not candidates:
        return solution
    parent, options = candidates[rng.randrange(len(candidates))]
    current = patterns[parent.parent_id]
    if current.kind is SplitKind.WHOLE:
        patterns[parent.parent_id] = next(
            value for value in options if value.kind is SplitKind.Y_SPLIT
        )
    else:
        patterns[parent.parent_id] = next(
            value for value in options if value.kind is SplitKind.WHOLE
        )
    selected = tuple(patterns[parent.parent_id] for parent in solution.parents)
    inherited = [list(route.block_ids) for route in solution.routes]
    return _repair_routes(
        solution.parents,
        selected,
        inherited,
        config,
        rng,
        shuffle_missing=False,
    )


def route_edge_set(solution: CanonicalSolution) -> frozenset[tuple[int, str, str]]:
    edges = set()
    for route in solution.routes:
        nodes = (f"R{route.robot_id}:START",) + route.block_ids + (
            f"R{route.robot_id}:END",
        )
        edges.update((route.robot_id, left, right) for left, right in zip(nodes, nodes[1:]))
    return frozenset(edges)


def normalized_route_edge_distance(
    left: CanonicalSolution, right: CanonicalSolution
) -> float:
    a = route_edge_set(left)
    b = route_edge_set(right)
    union = a | b
    return 0.0 if not union else len(a ^ b) / len(union)


def _objective_key(individual: _Individual):
    if individual.evaluated is not None and individual.evaluated.metrics is not None:
        metrics = individual.evaluated.metrics
        return (0, metrics.cmax, metrics.optional_split_count, metrics.process_imbalance,
                metrics.total_empty_travel, metrics.total_waiting,
                metrics.deterministic_id_order, individual.solution.canonical_hash)
    return (1, *individual.proxy)


def binary_tournament(population: Sequence[_Individual], rng: random.Random) -> _Individual:
    first = population[rng.randrange(len(population))]
    second = population[rng.randrange(len(population))]
    return min((first, second), key=_objective_key)


def manage_population(population: list[_Individual], mu: int) -> list[_Individual]:
    objective_order = {
        id(item): rank for rank, item in enumerate(sorted(population, key=_objective_key))
    }
    diversity_values = {}
    for item in population:
        distances = [
            normalized_route_edge_distance(item.solution, other.solution)
            for other in population
            if other is not item
        ]
        diversity_values[id(item)] = sum(distances) / len(distances) if distances else 0.0
    diversity_order = {
        id(item): rank
        for rank, item in enumerate(
            sorted(
                population,
                key=lambda value: (
                    -diversity_values[id(value)], value.solution.canonical_hash
                ),
            )
        )
    }
    return sorted(
        population,
        key=lambda item: (
            objective_order[id(item)] + diversity_order[id(item)],
            objective_order[id(item)],
            diversity_order[id(item)],
            item.solution.canonical_hash,
        ),
    )[:mu]


def run_adapted_hga(
    parents: Sequence[ParentWeld],
    scientific_config: ScientificConfig,
    hga_config: AdaptedHGAConfig,
    *,
    seed: int,
    time_limit: float,
    scope: FormalScope,
    checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
    common_seed: EvaluatedCandidate | None = None,
) -> BaselineResult:
    rng = random.Random(seed)
    evaluator = CommonBaselineEvaluator(
        parents,
        scientific_config,
        scope,
        time_limit=time_limit,
        checkpoints=checkpoints,
    )
    population: list[_Individual] = []
    init_started = time.perf_counter()
    if common_seed is not None:
        injected = evaluator.inject_certified(
            common_seed, source="HGA_COMMON_CERTIFIED_SEED"
        )
        population.append(
            _Individual(
                injected.solution,
                hga_surrogate(injected.solution, scientific_config),
                injected,
            )
        )
    for index in range(1 if common_seed is not None else 0, hga_config.mu):
        if evaluator.expired:
            break
        construct_started = time.perf_counter()
        try:
            solution = (
                initialize_hga_region_seed(parents, scientific_config)
                if index == 0
                else initialize_hga_solution(parents, scientific_config, rng)
            )
            native_evaluated = None
            if (
                common_seed is None
                and index < hga_config.initialization_reference_limit
                and not evaluator.expired
            ):
                native_evaluated = evaluator.evaluate(
                    solution, source=f"HGA_INITIALIZATION_{index}_NATIVE"
                )
            elif common_seed is None and index >= hga_config.initialization_reference_limit:
                evaluator.accounting.initialization_reference_limit_hits += 1
            native_hash = solution.canonical_hash
            local_started = time.perf_counter()
            solution = hga_vnd(
                solution,
                scientific_config,
                alpha=hga_config.alpha,
                max_passes=hga_config.vnd_max_passes,
                candidate_cap=hga_config.vnd_candidate_cap,
                deadline=evaluator.deadline,
                accounting=evaluator.accounting,
            )
            evaluator.accounting.local_search_time += time.perf_counter() - local_started
        except (ArithmeticError, OverflowError, ValueError) as error:
            evaluator.reject_construction(f"HGA initialization {index}: {error}")
            continue
        evaluator.accounting.construction_time += time.perf_counter() - construct_started
        individual = _Individual(solution, hga_surrogate(solution, scientific_config))
        if solution.canonical_hash == native_hash:
            individual.evaluated = native_evaluated
        elif (
            common_seed is None
            and index < hga_config.initialization_reference_limit
            and not evaluator.expired
        ):
            individual.evaluated = evaluator.evaluate(
                solution, source=f"HGA_INITIALIZATION_{index}_VND"
            )
        population.append(individual)
    initialization_time = time.perf_counter() - init_started
    initial_candidate = evaluator.best
    initial_reference_calls = evaluator.accounting.reference_calls

    iterations = 0
    while (
        population
        and not evaluator.expired
        and iterations < hga_config.max_iterations
    ):
        iterations += 1
        first = binary_tournament(population, rng)
        second = binary_tournament(population, rng)
        crossover_started = time.perf_counter()
        try:
            offspring = route_based_crossover(
                first.solution, second.solution, scientific_config, rng
            )
            if rng.random() < hga_config.pattern_mutation_probability:
                evaluator.accounting.optional_y_mutations_attempted += 1
                before_patterns = offspring.patterns
                offspring = mutate_optional_y(offspring, scientific_config, rng)
                if offspring.patterns != before_patterns:
                    evaluator.accounting.optional_y_mutations_accepted += 1
        except (ArithmeticError, OverflowError, ValueError) as error:
            evaluator.reject_construction(f"HGA crossover iteration {iterations}: {error}")
            continue
        evaluator.accounting.crossover_time += time.perf_counter() - crossover_started
        local_started = time.perf_counter()
        offspring = hga_vnd(
            offspring,
            scientific_config,
            alpha=hga_config.alpha,
            max_passes=hga_config.vnd_max_passes,
            candidate_cap=hga_config.vnd_candidate_cap,
            deadline=evaluator.deadline,
            accounting=evaluator.accounting,
        )
        evaluator.accounting.local_search_time += time.perf_counter() - local_started
        evaluated = evaluator.evaluate(offspring, source=f"HGA_OFFSPRING_{iterations}")
        population.append(
            _Individual(offspring, hga_surrogate(offspring, scientific_config), evaluated)
        )
        if len(population) >= hga_config.mu + hga_config.lambda_size:
            management_started = time.perf_counter()
            population = manage_population(population, hga_config.mu)
            evaluator.accounting.population_management_time += (
                time.perf_counter() - management_started
            )
            evaluator.accounting.population_survival_events += 1

    if evaluator.best is None and evaluator.accounting.numeric_failure:
        termination_reason = "NUMERIC_FAILURE"
    elif evaluator.best is None:
        termination_reason = "INITIALIZATION_FAILED"
    elif evaluator.expired:
        termination_reason = "TIME_LIMIT"
    elif iterations >= hga_config.max_iterations:
        termination_reason = "ITERATION_LIMIT"
    else:
        termination_reason = "COMPLETED_OTHER"

    return evaluator.finish(
        method_id=METHOD_ID,
        method_config_hash=canonical_config_hash(hga_config),
        iterations=iterations,
        initialization_time=initialization_time,
        termination_reason=termination_reason,
        initial_candidate=initial_candidate,
        initial_reference_calls=initial_reference_calls,
        diagnostics=(
            "one four-quadrant HGA region seed adapts paper region division; paper start/return physics excluded",
            "ADAPTED_HGA_BIASED_FITNESS_V1",
        ),
    )
