"""Critical-route fragment DP, followed by native best improvement.

Parent modes and the four original route heads stay fixed. Exactness refers
only to distributing one selected fragment among the selected insertion gaps.
SAEAX is run afresh; its feasible NATIVE result is retained without reevaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np
from numba import njit

from mrta_baselines.hga import _HGAWorkspace
from mrta_baselines.wag_vns import _priority_arrays
from mrta_reference.model import OperationKind, ScientificConfig
from mrta_reference.scheduler import native_schedule
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_search.saeax import (Individual, _heads_legal, _route_statistics,
    complete_identity, run_saeax, validate_individual)


@dataclass(frozen=True)
class CRFDPConfig:
    max_rounds: int = 8
    max_native_evaluations: int = 40
    max_windows: int = 6
    min_fragment: int = 5
    max_fragment: int = 8
    receiver_gaps: int = 2

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not (5 <= self.min_fragment <= self.max_fragment <= 8):
            raise ValueError('fragment lengths must be between five and eight')
        if (self.max_rounds > 8 or self.max_native_evaluations > 40 or
                self.max_windows > 6 or self.receiver_gaps > 2):
            raise ValueError('CRF-DP pilot budgets cannot be enlarged')


@dataclass
class CRFDPResult:
    status: str
    cmax: float | None
    solution: object | None
    directions: tuple | None
    schedule: object | None
    native_cmax: float | None
    runtime: float
    rounds: int
    termination_reason: str
    accounting: dict
    initial_result: object
    progress: list = field(default_factory=list)
    accepted_actions: list = field(default_factory=list)
    final_candidate: dict | None = None


@njit(cache=False)
def _subset_routes(nodes, robot, route, bits, gaps, base_cost,
                   travel, processing, eligible, parent_index):
    """Held--Karp with explicit directions and both unchanged boundaries.

The workspace's eligibility table already includes fixed X_SPLIT ownership.
The unchanged first task avoids coupling the four independent robot DPs.
"""
    k, count = len(nodes), 1 << len(nodes)
    costs = np.full(count, np.inf)
    chosen_gap = np.full(count, -1, np.int64)
    ends = np.full(count, -1, np.int64)
    predecessors = np.full((len(gaps), count, k, 2), -2, np.int64)
    # Empty routes keep their empty head in this head-preserving neighborhood.
    if not len(gaps):
        costs[0] = base_cost
        return costs, chosen_gap, ends, predecessors
    for g, position in enumerate(gaps):
        left = route[position-1]
        ld = bits[position-1]
        right = route[position] if position < len(route) else -1
        rd = bits[position] if right >= 0 else 0
        bridge = travel[left, right, ld, rd] if right >= 0 else 0.
        if right < 0 or parent_index[left] != parent_index[right]:
            costs[0], chosen_gap[0] = base_cost, g
        dp = np.full((count, k, 2), np.inf)
        for i, node in enumerate(nodes):
            if not eligible[node, robot] or parent_index[left] == parent_index[node]:
                continue
            for d in range(2):
                dp[1 << i, i, d] = processing[node] + travel[left, node, ld, d]
                predecessors[g, 1 << i, i, d] = -1
        for mask in range(1, count):
            for i, node in enumerate(nodes):
                if not (mask & (1 << i)):
                    continue
                for d in range(2):
                    value = dp[mask, i, d]
                    if not math.isfinite(value):
                        continue
                    if right < 0 or parent_index[node] != parent_index[right]:
                        closed = value + (travel[node, right, d, rd] if right >= 0 else 0.)
                        total = base_cost - bridge + closed
                        if total < costs[mask]:
                            costs[mask], chosen_gap[mask], ends[mask] = total, g, 2*i+d
                    for j, other in enumerate(nodes):
                        if (mask & (1 << j) or not eligible[other, robot] or
                                parent_index[node] == parent_index[other]):
                            continue
                        enlarged = mask | (1 << j)
                        for e in range(2):
                            candidate = value + processing[other] + travel[node, other, d, e]
                            if candidate < dp[enlarged, j, e]:
                                dp[enlarged, j, e] = candidate
                                predecessors[g, enlarged, j, e] = 2*i+d
    return costs, chosen_gap, ends, predecessors


@njit(cache=False)
def _partition_subsets(costs):
    """Minimize the maximum of all four complete route periods, in O(4*3^k)."""
    count = costs.shape[1]
    values = np.full((5, count), np.inf)
    choice = np.full((5, count), -1, np.int64)
    values[0, 0] = 0.
    for r in range(1, 5):
        for mask in range(count):
            subset = mask
            while True:
                value = max(values[r-1, mask ^ subset], costs[r-1, subset])
                if value < values[r, mask]:
                    values[r, mask], choice[r, mask] = value, subset
                if subset == 0:
                    break
                subset = (subset-1) & mask
    allocation = np.zeros(4, np.int64)
    mask = count-1
    if math.isfinite(values[4, mask]):
        for r in range(4, 0, -1):
            allocation[r-1] = choice[r, mask]
            mask ^= allocation[r-1]
    return values[4, count-1], allocation


def select_windows(item, workspace, settings):
    critical = int(np.argmax(item.costs))
    route, bits = item.routes[critical], item.directions[critical]
    ranked = []
    for length in range(settings.min_fragment, settings.max_fragment+1):
        for start in range(1, len(route)-length+1):
            stop = start+length
            contribution = float(sum(workspace.processing[route[start:stop]]))
            contribution += sum(workspace.travel[route[j-1], route[j], bits[j-1], bits[j]]
                                for j in range(start, min(stop+1, len(route))))
            if stop < len(route):
                contribution -= workspace.travel[route[start-1], route[stop], bits[start-1], bits[stop]]
            ranked.append((float(contribution), start, length))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    # Keep size diversity: one best window of each size, then fill by contribution.
    chosen, sizes = [], set()
    for row in ranked:
        if row[2] not in sizes and len(chosen) < settings.max_windows:
            chosen.append(row)
            sizes.add(row[2])
    for row in ranked:
        if len(chosen) >= settings.max_windows:
            break
        if row not in chosen:
            chosen.append(row)
    return critical, chosen


def _receiver_gaps(nodes, robot, route, bits, workspace, limit):
    ranked = []
    for position in range(1, len(route)+1):
        left, ld = route[position-1], bits[position-1]
        right = route[position] if position < len(route) else -1
        rd = bits[position] if right >= 0 else 0
        bridge = workspace.travel[left, right, ld, rd] if right >= 0 else 0.
        deltas = []
        for node in nodes:
            if not workspace.eligible[node, robot]:
                continue
            for d in range(2):
                # This ranks boundaries only. The DP can insert several nodes
                # and checks sibling legality for its actual first/last nodes.
                delta = workspace.processing[node] + workspace.travel[left, node, ld, d] - bridge
                if right >= 0:
                    delta += workspace.travel[node, right, d, rd]
                deltas.append(float(delta))
        if deltas:
            ranked.append((min(deltas), position))
    return np.array([p for _, p in sorted(ranked)[:limit]], np.int64)


def reconstruct_fragment(item, workspace, critical, start, length, *, receiver_gaps=2,
                         gap_options=None):
    """Solve the complete fixed-window neighborhood; smaller k allowed in tests.

    Only one contiguous block is inserted per receiver. No head is replaced;
    empty robots remain empty. Other retained tasks and directions stay fixed.
    """
    if start < 1 or length < 1 or start+length > len(item.routes[critical]):
        raise ValueError('fragment must follow an unchanged route head')
    nodes = item.routes[critical][start:start+length].copy()
    routes = [row.copy() for row in item.routes]
    bits = [row.copy() for row in item.directions]
    routes[critical] = np.concatenate((routes[critical][:start], routes[critical][start+length:]))
    bits[critical] = np.concatenate((bits[critical][:start], bits[critical][start+length:]))
    base, _ = _route_statistics(routes, bits, workspace.travel, workspace.processing)
    tables, gaps, all_costs = [], [], []
    for r in range(4):
        positions = (np.array([start], np.int64) if r == critical else
            _receiver_gaps(nodes, r, routes[r], bits[r], workspace, receiver_gaps))
        if gap_options is not None:
            positions = np.array(gap_options[r], np.int64)
        if any(p < 1 or p > len(routes[r]) for p in positions):
            raise ValueError('insertion boundaries must preserve the route heads')
        table = _subset_routes(nodes, r, routes[r], bits[r], positions, float(base[r]),
            workspace.travel, workspace.processing, workspace.eligible, workspace.parent_index)
        tables.append(table)
        gaps.append(positions)
        all_costs.append(table[0])
    optimum, allocation = _partition_subsets(np.array(all_costs))
    if not math.isfinite(optimum):
        return None
    assignments, transfers = [], []
    for r, subset in enumerate(allocation):
        if not subset:
            continue
        _, chosen, ends, previous = tables[r]
        g, mask = int(chosen[subset]), int(subset)
        state = int(ends[subset])
        order, vector = [], []
        while mask:
            i, d = divmod(state, 2)
            order.append(int(nodes[i]))
            vector.append(d)
            state = int(previous[g, mask, i, d])
            mask ^= 1 << i
        order.reverse()
        vector.reverse()
        position = int(gaps[r][g])
        routes[r] = np.concatenate((routes[r][:position], np.array(order, np.int64), routes[r][position:]))
        bits[r] = np.concatenate((bits[r][:position], np.array(vector, np.int64), bits[r][position:]))
        assignments.append(dict(robot=r, position=position,
            children=[workspace.ids[n] for n in order], directions=vector))
        if r != critical:
            transfers.extend(dict(child=workspace.ids[n], source=critical, target=r) for n in order)
    validate_individual(workspace, routes, bits, item.selected)
    if not _heads_legal(routes, bits, workspace.points, workspace.parameters):
        raise RuntimeError('head-preserving reconstruction changed initial-point legality')
    costs, loads = _route_statistics(routes, bits, workspace.travel, workspace.processing)
    if not np.isclose(max(costs), optimum, rtol=1.e-12, atol=1.e-8):
        raise RuntimeError('DP traceback does not reproduce its complete-route objective')
    identity = complete_identity(routes, bits, item.selected)
    trace = dict(critical=critical, start=start, length=length, lower_bound_s=float(max(costs)),
        receiver_gap_options=[row.tolist() for row in gaps], assignments=assignments,
        cross_robot_transfers=transfers, route_periods_s=costs.tolist())
    return routes, bits, costs, loads, identity, trace


def solution_record(solution, directions):
    if solution is None:
        return None
    return dict(patterns=[dict(parent_id=p.parent_id, kind=p.kind.value, t=p.t,
        point_id=p.point_id, rail=p.rail.value if p.rail else None) for p in solution.patterns],
        routes=[list(r.block_ids) for r in solution.routes], directions=directions)


def run_crf_dp(parents, config=None, settings=None, *, seed=20261110,
               scope=FORMAL_SCOPE_V2, progress_callback=None):
    config, settings = config or ScientificConfig(), settings or CRFDPConfig()
    if scope != FORMAL_SCOPE_V2:
        raise ValueError('CRF-DP uses FORMAL_SCOPE_V2')
    started = time.perf_counter()
    def initial_progress(row):
        if progress_callback is not None:
            progress_callback(dict(phase='SAEAX_INITIALIZATION', **row))
    initial = run_saeax(parents, config, seed=seed, scope=scope, progress_callback=initial_progress)
    a = dict(initialization_s=time.perf_counter()-started, initial_saeax_runtime_s=initial.runtime,
        initial_native_calls=initial.accounting['native_calls'], initial_coordination_calls=initial.accounting['coordination_calls'],
        native_calls=initial.accounting['native_calls'], coordination_calls=initial.accounting['coordination_calls'],
        coordination_s=initial.accounting['coordination_s'], dp_search_s=0., dp_s=0., crf_native_s=0., crf_native_calls=0,
        final_coordination_calls=0, final_coordination_s=0., rounds=0, windows=0, constructed=0, no_legal_partition=0,
        duplicate_candidates=0, lower_bound_pruned=0, native_finite=0, native_nonfinite=0,
        accepted_improvements=0, transferred_children_generated=0, transferred_children_evaluated=0,
        transferred_children_accepted=0, initial_cmax_s=initial.cmax, final_fallback=False,
        search_complete=False)
    if initial.status != 'FEASIBLE' or initial.schedule is None:
        a['return_reason'] = 'INITIAL_SAEAX_NOT_NATIVE_FEASIBLE'
        return CRFDPResult(initial.status, initial.cmax, initial.solution, initial.directions,
            initial.schedule, initial.native_cmax, time.perf_counter()-started, 0,
            'INITIALIZATION_FAILED', a, initial)
    search_started = time.perf_counter()
    w = _HGAWorkspace(parents, config, scope)
    selected = np.array([w.patterns.index(p) for p in initial.solution.patterns], np.int64)
    routes, bits = w.encode(initial.solution), [np.array(row, np.int64) for row in initial.directions]
    validate_individual(w, routes, bits, selected)
    if not _heads_legal(routes, bits, w.points, w.parameters):
        raise RuntimeError('SAEAX returned illegal explicit route heads')
    costs, loads = _route_statistics(routes, bits, w.travel, w.processing)
    identity = complete_identity(routes, bits, selected)
    initial_native = initial.native_cmax if initial.native_cmax is not None else math.inf
    best = Individual(routes, bits, selected, initial_native, costs, loads, identity, np.empty(0, np.int64))
    a['initial_route_periods_s'] = costs.tolist()
    progress, accepted, seen = [], [], {identity}
    termination = 'ROUND_LIMIT'
    for iteration in range(1, settings.max_rounds+1):
        a['rounds'] = iteration
        critical, windows = select_windows(best, w, settings)
        round_best, round_trace, window_log = best, None, []
        for contribution, start, length in windows:
            if a['crf_native_calls'] >= settings.max_native_evaluations:
                break
            a['windows'] += 1
            t = time.perf_counter()
            candidate = reconstruct_fragment(best, w, critical, start, length,
                receiver_gaps=settings.receiver_gaps)
            a['dp_s'] += time.perf_counter()-t
            log = dict(start=start, length=length, contribution_s=contribution)
            window_log.append(log)
            if candidate is None:
                a['no_legal_partition'] += 1
                log['result'] = 'NO_LEGAL_PARTITION'
                continue
            rr, dd, cc, pp, key, trace = candidate
            a['constructed'] += 1
            moved = len(trace['cross_robot_transfers'])
            a['transferred_children_generated'] += moved
            log.update(lower_bound_s=float(max(cc)), transferred_children=moved)
            if key in seen:
                a['duplicate_candidates'] += 1
                log['result'] = 'DUPLICATE'
                continue
            seen.add(key)
            if float(max(cc)) >= round_best.native_cmax:
                a['lower_bound_pruned'] += 1
                log['result'] = 'LOWER_BOUND'
                continue
            t = time.perf_counter()
            value, _, _, _ = _priority_arrays(rr, w.travel, w.processing, w.points, w.parameters, dd)
            a['crf_native_s'] += time.perf_counter()-t
            a['crf_native_calls'] += 1
            a['native_calls'] += 1
            a['transferred_children_evaluated'] += moved
            if not math.isfinite(value):
                a['native_nonfinite'] += 1
                log.update(result='NATIVE_NONFINITE', native_cmax_s=None)
                continue
            a['native_finite'] += 1
            log.update(result='NATIVE_FINITE', native_cmax_s=float(value))
            if value < round_best.native_cmax-1.e-8:
                round_best = Individual(rr, dd, selected, float(value), cc, pp, key, np.empty(0, np.int64))
                round_trace = trace
        if round_best is not best:
            a['accepted_improvements'] += 1
            a['transferred_children_accepted'] += len(round_trace['cross_robot_transfers'])
            accepted.append(dict(round=iteration, native_before_s=best.native_cmax
                if math.isfinite(best.native_cmax) else None, native_after_s=round_best.native_cmax,
                **round_trace))
            best = round_best
        row = dict(phase='CRF_DP', round=iteration, critical=critical,
            evaluations=a['crf_native_calls'], best_native_cmax_s=best.native_cmax
                if math.isfinite(best.native_cmax) else None,
            route_periods_s=best.costs.tolist(), windows=window_log,
            accepted=round_trace is not None, elapsed_s=time.perf_counter()-started)
        progress.append(row)
        if progress_callback is not None:
            progress_callback(row)
        if a['crf_native_calls'] >= settings.max_native_evaluations:
            termination = 'NATIVE_LIMIT'
            break
        if round_trace is None:
            termination = 'NO_IMPROVEMENT' if windows else 'NO_WINDOWS'
            break
    a.update(dp_search_s=time.perf_counter()-search_started, search_complete=True,
        search_best_native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
        search_best_route_periods_s=best.costs.tolist())
    solution, vectors, schedule, native = initial.solution, initial.directions, initial.schedule, initial.native_cmax
    final = None
    if best.identity != identity and best.native_cmax < initial_native-1.e-8:
        improved = w.solution(best.routes, selected)
        expected = tuple(tuple(w.ids[int(n)] for n in row) for row in best.routes)
        if (tuple(r.block_ids for r in improved.routes) != expected or
                improved.patterns != tuple(w.patterns[p] for p in selected)):
            raise RuntimeError('canonicalization changed the fixed-mode CRF-DP candidate')
        explicit = tuple(tuple(int(d) for d in row) for row in best.directions)
        t = time.perf_counter()
        attempt = native_schedule(improved, config, scope=scope, orientations=dict(enumerate(explicit)))
        elapsed = time.perf_counter()-t
        a.update(final_coordination_calls=1, final_coordination_s=elapsed, coordination_calls=a['coordination_calls']+1,
            coordination_s=a['coordination_s']+elapsed)
        final = dict(status=attempt.status.value, cmax_s=attempt.cmax if attempt.feasible else None,
            native_cmax_s=best.native_cmax, route_periods_s=best.costs.tolist(),
            solution=solution_record(improved, explicit), diagnostics=list(attempt.diagnostics))
        if attempt.feasible and attempt.cmax < initial.cmax-1.e-8:
            if attempt.directions != explicit:
                raise RuntimeError('NATIVE changed explicit CRF-DP directions')
            solution, vectors, schedule, native = improved, explicit, attempt, best.native_cmax
            a['return_reason'] = 'IMPROVED_NATIVE'
        else:
            a.update(final_fallback=True, return_reason='FINAL_NATIVE_FAILED' if not attempt.feasible
                else 'INITIAL_NATIVE_BETTER_OR_EQUAL')
    else:
        a['return_reason'] = 'NO_NATIVE_IMPROVEMENT'
    returned_routes = w.encode(solution)
    returned_bits = [np.array(row, np.int64) for row in vectors]
    returned_costs, returned_loads = _route_statistics(returned_routes, returned_bits, w.travel, w.processing)
    original_owners = {int(n): r for r, row in enumerate(routes) for n in row}
    a.update(processing_s=returned_loads.tolist(), route_periods_s=returned_costs.tolist(),
        move_s=float(sum(returned_costs-returned_loads)),
        wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT),
        returned_cross_robot_transfers=[dict(child=w.ids[int(n)], source=original_owners[int(n)], target=r)
            for r, row in enumerate(returned_routes) for n in row if original_owners[int(n)] != r])
    return CRFDPResult('FEASIBLE', schedule.cmax, solution, vectors, schedule, native,
        time.perf_counter()-started, a['rounds'], termination, a, initial, progress, accepted, final)
