"""Parent-level rail-conflict-aware iterated greedy, seeded by an in-run SAEAX.

P+directed MOVE ranks repair choices only. Acceptance uses the unchanged native
coordinator; only the best native improvement receives one final B32 attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

import numpy as np
from numba import njit

from mrta_baselines.hga import _HGAWorkspace, _put_oriented
from mrta_baselines.wag_vns import _priority_arrays
from mrta_reference.model import OperationKind, ScientificConfig
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_search.lns import schedule_feedback_summary
from mrta_search.saeax import (Individual, _heads_legal, _route_statistics,
    complete_identity, run_saeax, validate_individual)


@dataclass(frozen=True)
class RCIGConfig:
    max_iterations: int = 500
    initial_temperature_fraction: float = .01
    cooling: float = .99

    def __post_init__(self):
        if (not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool)
                or not 1 <= self.max_iterations <= 500):
            raise ValueError('max_iterations must be an integer in [1,500]')
        if not math.isfinite(self.initial_temperature_fraction) or self.initial_temperature_fraction <= 0:
            raise ValueError('initial_temperature_fraction must be positive and finite')
        if not 0 < self.cooling < 1:
            raise ValueError('cooling must be in (0,1)')


@dataclass
class RCIGResult:
    status: str
    cmax: float | None
    solution: object | None
    directions: tuple | None
    schedule: object | None
    native_cmax: float | None
    runtime: float
    iterations: int
    termination_reason: str
    accounting: dict
    initial_result: object
    progress: list = field(default_factory=list)
    accepted_actions: list = field(default_factory=list)
    final_candidate: dict | None = None


@njit
def _insertion_choices(node, routes, bits, costs, travel, processing, eligible,
                       parent_index, points, parameters):
    """Best two distinct (robot,position) choices, optimizing local directions.

    This is the numeric counterpart of joint_block_candidates: every position,
    both inserted directions and both immediate neighbours' directions. Only
    incident arcs change. No partial MOVE value is used for pruning.
    """
    actions = np.full((2, 5), -1, np.int64)
    scores = np.full((2, 2), np.inf)
    examined = 0
    total = sum(costs)
    for r in range(4):
        if not eligible[node, r]:
            continue
        row, vector = routes[r], bits[r]
        other = 0.
        for s in range(4):
            if s != r:
                other = max(other, costs[s])
        for k in range(len(row) + 1):
            if k and parent_index[row[k-1]] == parent_index[node]:
                continue
            if k < len(row) and parent_index[row[k]] == parent_index[node]:
                continue
            best_peak, best_total = np.inf, np.inf
            best_action = np.full(5, -1, np.int64)
            for d in range(2):
                for before in range(2 if k else 1):
                    for after in range(2 if k < len(row) else 1):
                        examined += 1
                        # Match the existing four-robot free-start constraints.
                        legal = True
                        for a in range(4):
                            if a != r and not len(routes[a]):
                                continue
                            an = node if a == r and k == 0 else routes[a][0]
                            ad = (d if k == 0 else before if k == 1 else vector[0]) if a == r else bits[a][0]
                            pa = points[an, ad]
                            for b in range(a+1, 4):
                                if b != r and not len(routes[b]):
                                    continue
                                bn = node if b == r and k == 0 else routes[b][0]
                                bd = (d if k == 0 else before if k == 1 else vector[0]) if b == r else bits[b][0]
                                pb = points[bn, bd]
                                if (a//2 == b//2 and pa[0]+parameters[3] > pb[0]+parameters[5]
                                        or abs(pa[0]-pb[0]) <= parameters[3]+parameters[5]
                                        and abs(pa[1]-pb[1]) <= parameters[4]+parameters[5]):
                                    legal = False
                        if not legal:
                            continue
                        delta = processing[node]
                        if k:
                            delta += travel[row[k-1], node, before, d]
                        if k < len(row):
                            delta += travel[node, row[k], d, after]
                        if k and k < len(row):
                            delta -= travel[row[k-1], row[k], vector[k-1], vector[k]]
                        if k > 1:
                            delta += (travel[row[k-2], row[k-1], vector[k-2], before]
                                      - travel[row[k-2], row[k-1], vector[k-2], vector[k-1]])
                        if k+1 < len(row):
                            delta += (travel[row[k], row[k+1], after, vector[k+1]]
                                      - travel[row[k], row[k+1], vector[k], vector[k+1]])
                        peak, summed = max(other, costs[r]+delta), total+delta
                        if peak < best_peak or peak == best_peak and summed < best_total:
                            best_peak, best_total = peak, summed
                            best_action[:] = (r, k, d, before, after)
            if best_action[0] < 0:
                continue
            slot = -1
            for i in range(2):
                if best_peak < scores[i, 0] or best_peak == scores[i, 0] and best_total < scores[i, 1]:
                    slot = i
                    break
            if slot >= 0:
                if slot == 0:
                    actions[1], scores[1] = actions[0].copy(), scores[0].copy()
                actions[slot] = best_action
                scores[slot, 0], scores[slot, 1] = best_peak, best_total
    return actions, scores, examined


def insertion_options(node, routes, bits, workspace, accounting):
    costs, _ = _route_statistics(routes, bits, workspace.travel, workspace.processing)
    actions, scores, examined = _insertion_choices(node, routes, bits, costs,
        workspace.travel, workspace.processing, workspace.eligible, workspace.parent_index,
        workspace.points, workspace.parameters)
    accounting['insertion_states'] += examined
    rows = []
    for action, score in zip(actions, scores):
        r, k, d, before, after = map(int, action)
        if r < 0:
            continue
        rr, dd = list(routes), list(bits)
        dd[r] = dd[r].copy()
        changes = []
        for index, value in ((k-1, before), (k, after)):
            if 0 <= index < len(rr[r]):
                old = int(dd[r][index])
                if old != value:
                    changes.append((workspace.ids[int(rr[r][index])], old, value))
                dd[r][index] = value
        _put_oriented(rr, dd, int(node), d, r, k)
        detail = dict(block_id=workspace.ids[int(node)], robot=r, position=k,
            direction=d, neighbour_changes=changes)
        rows.append((rr, dd, (float(score[0]), float(score[1])), (detail,)))
    return rows


def parent_options(parent, routes, bits, workspace, accounting):
    """All catalog modes; width-two insertion beam in both child orders.

    Regret alternatives have distinct mode/owner/order structures. Direction-only
    variants of the same structure never manufacture an unconstrained choice.
    """
    options = {}
    for pattern in workspace.options[parent]:
        accounting['mode_trials'] += 1
        nodes = tuple(int(n) for n in workspace.pattern_nodes[pattern] if n >= 0)
        for order in ((nodes,) if len(nodes) == 1 else (nodes, nodes[::-1])):
            partial = [(routes, bits, (0., 0.), ())]
            for node in order:
                extended = []
                for rr, dd, _, trace in partial:
                    for nr, nd, score, action in insertion_options(node, rr, dd, workspace, accounting):
                        extended.append((nr, nd, score, trace+action))
                # Keep both first-child choices until the sibling is inserted.
                partial = extended
            for rr, dd, score, trace in partial:
                structure = (int(pattern), tuple(tuple(int(n) for n in row) for row in rr))
                rank = (score, structure, tuple(tuple(int(d) for d in row) for row in dd))
                if structure not in options or rank < options[structure][0]:
                    options[structure] = (rank, int(pattern), rr, dd, trace)
    return sorted(options.values(), key=lambda row: row[0])[:2]


def destroy_parents(item, workspace, parent_wait, robot_wait, rng):
    count = min(len(workspace.parents), max(3, min(8, round(.06*len(workspace.parents)))))
    contribution = np.zeros((len(workspace.parents), 4))
    for r, row in enumerate(item.routes):
        for node in row:
            contribution[workspace.parent_index[node], r] += workspace.processing[node]
    critical = tuple(dict.fromkeys((int(np.argmax(item.processing)), int(np.argmax(item.costs)),
                                   int(np.argmax(item.costs+robot_wait)))))
    chosen = []
    def pick(weights):
        pool = [p for p in range(len(workspace.parents)) if p not in chosen and weights[p] > 0]
        if pool:
            chosen.append(rng.choices(pool, weights=[float(weights[p]) for p in pool], k=1)[0])
    pick(contribution[:, critical[0]])
    if len(chosen) < count:
        pick(parent_wait)
    weight = (np.sum(contribution[:, critical], axis=1)/max(float(max(item.processing)), 1.)
              + np.sum(contribution, axis=1)/max(float(sum(item.processing)), 1.)
              + parent_wait/max(float(sum(parent_wait)), 1.) + 1./max(len(workspace.parents), 1))
    while len(chosen) < count:
        pick(weight)
    return tuple(chosen)


def repair_parents(item, removed, workspace, accounting):
    """Remove every child of each selected parent, then parent-level regret-2."""
    removed_set = set(removed)
    routes = [np.array([n for n in row if int(workspace.parent_index[n]) not in removed_set], np.int64)
              for row in item.routes]
    bits = [np.array([d for n, d in zip(row, dd) if int(workspace.parent_index[n]) not in removed_set], np.int64)
            for row, dd in zip(item.routes, item.directions)]
    selected, pending, trace = item.selected.copy(), list(removed), []
    while pending:
        candidates = []
        for parent in pending:
            options = parent_options(parent, routes, bits, workspace, accounting)
            if not options:
                continue
            first = options[0]
            regret = ((math.inf, math.inf) if len(options) == 1 else
                tuple(b-a for a, b in zip(first[0][0], options[1][0][0])))
            candidates.append((regret, -first[0][0][0], -parent, first))
        if not candidates:
            return None
        _, _, negative_parent, option = max(candidates, key=lambda row: row[:3])
        parent = -negative_parent
        _, pattern, routes, bits, actions = option
        trace.append(dict(parent_id=workspace.parents[parent].parent_id,
            old_pattern=workspace.patterns[item.selected[parent]].pattern_id,
            pattern=workspace.patterns[pattern].pattern_id, insertions=actions))
        selected[parent] = pattern
        pending.remove(parent)
    try:
        validate_individual(workspace, routes, bits, selected)
    except ValueError:
        return None
    if not _heads_legal(routes, bits, workspace.points, workspace.parameters):
        return None
    return routes, bits, selected, trace


def native_waits(operations, workspace):
    parents, robots = np.zeros(len(workspace.parents)), np.zeros(4)
    for r, row in enumerate(operations):
        for op in row:
            if int(op[8]) == 4:
                robots[r] += float(op[0])
                if 0 <= int(op[7]) < len(workspace.ids):
                    parents[workspace.parent_index[int(op[7])]] += float(op[0])
    return parents, robots


def solution_record(solution, directions):
    if solution is None:
        return None
    return dict(patterns=[dict(parent_id=p.parent_id, kind=p.kind.value, t=p.t,
        point_id=p.point_id, rail=p.rail.value if p.rail else None) for p in solution.patterns],
        routes=[list(r.block_ids) for r in solution.routes], directions=directions)


def run_rc_ig(parents, config=None, settings=None, *, seed=20261110,
              scope=FORMAL_SCOPE_V2, progress_callback=None):
    config, settings = config or ScientificConfig(), settings or RCIGConfig()
    if scope != FORMAL_SCOPE_V2:
        raise ValueError('RC-IG uses FORMAL_SCOPE_V2')
    started = time.perf_counter()
    def initial_progress(row):
        if progress_callback is not None:
            progress_callback(dict(phase='SAEAX_INITIALIZATION', **row))
    initial = run_saeax(parents, config, seed=seed, scope=scope, progress_callback=initial_progress)
    a = dict(initialization_s=time.perf_counter()-started, initial_saeax_runtime_s=initial.runtime,
        initial_native_calls=initial.accounting['native_calls'], initial_b32_calls=initial.accounting['b32_calls'],
        native_calls=initial.accounting['native_calls'], b32_calls=initial.accounting['b32_calls'],
        b32_s=initial.accounting['b32_s'], ig_native_calls=0, ig_native_s=0.,
        final_b32_calls=0, final_b32_s=0., ig_search_s=0., iterations=0,
        construction_failures=0, duplicate_candidates=0, mode_trials=0, insertion_states=0,
        native_feasible_candidates=0, native_nonfinite_candidates=0, numeric_failures=0,
        accepted_improvements=0, accepted_equal=0, accepted_worse=0, best_improvements=0,
        mode_changed_candidates=0, mode_changed_acceptances=0, changed_parent_modes=0,
        initial_cmax_s=initial.cmax, final_fallback=False, search_complete=False)
    if initial.status != 'FEASIBLE' or initial.schedule is None:
        a['return_reason'] = 'INITIAL_SAEAX_NOT_B32_FEASIBLE'
        return RCIGResult(initial.status, initial.cmax, initial.solution, initial.directions,
            initial.schedule, initial.native_cmax, time.perf_counter()-started, 0,
            'INITIALIZATION_FAILED', a, initial)
    search_started = time.perf_counter()
    workspace = _HGAWorkspace(parents, config, scope)
    selected = np.array([workspace.patterns.index(p) for p in initial.solution.patterns], np.int64)
    routes = workspace.encode(initial.solution)
    bits = [np.array(row, np.int64) for row in initial.directions]
    validate_individual(workspace, routes, bits, selected)
    costs, loads = _route_statistics(routes, bits, workspace.travel, workspace.processing)
    identity = complete_identity(routes, bits, selected)
    initial_native = initial.native_cmax if initial.native_cmax is not None else math.inf
    current = best = Individual(routes, bits, selected, initial_native, costs, loads, identity, np.empty(0, np.int64))
    feedback = schedule_feedback_summary(initial.solution, initial.schedule, (), config)
    waits = np.array([feedback['parent_wait'].get(p.parent_id, 0.) for p in workspace.parents])
    robot_wait = np.array(feedback['robot_wait'])
    seen, rng, progress, accepted = {identity}, random.Random(seed), [], []
    temperature = settings.initial_temperature_fraction*initial.cmax
    initial_temperature = temperature
    for iteration in range(1, settings.max_iterations+1):
        a['iterations'] = iteration
        removed = destroy_parents(current, workspace, waits, robot_wait, rng)
        repaired = repair_parents(current, removed, workspace, a)
        if repaired is None:
            a['construction_failures'] += 1
        else:
            rr, dd, ss, trace = repaired
            key = complete_identity(rr, dd, ss)
            if key in seen:
                a['duplicate_candidates'] += 1
            else:
                seen.add(key)
                t = time.perf_counter()
                value, _, operations, _ = _priority_arrays(rr, workspace.travel, workspace.processing,
                    workspace.points, workspace.parameters, dd)
                a['ig_native_s'] += time.perf_counter()-t
                a['ig_native_calls'] += 1
                a['native_calls'] += 1
                changed = int(np.count_nonzero(ss != current.selected))
                a['mode_changed_candidates'] += int(changed > 0)
                if not math.isfinite(value):
                    a['native_nonfinite_candidates'] += 1
                    a['numeric_failures'] += int(math.isnan(value))
                else:
                    a['native_feasible_candidates'] += 1
                    delta = float(value)-current.native_cmax
                    accept = delta <= 0 or rng.random() < math.exp(-delta/temperature)
                    if accept:
                        cc, pp = _route_statistics(rr, dd, workspace.travel, workspace.processing)
                        previous = current.native_cmax
                        current = Individual(rr, dd, ss, float(value), cc, pp, key, np.empty(0, np.int64))
                        waits, robot_wait = native_waits(operations, workspace)
                        a['accepted_improvements' if delta < -1.e-8 else
                          'accepted_worse' if delta > 1.e-8 else 'accepted_equal'] += 1
                        a['mode_changed_acceptances'] += int(changed > 0)
                        a['changed_parent_modes'] += changed
                        if current.native_cmax < best.native_cmax-1.e-8:
                            best = current
                            a['best_improvements'] += 1
                        accepted.append(dict(iteration=iteration, evaluation=a['ig_native_calls'],
                            native_before_s=previous if math.isfinite(previous) else None,
                            native_after_s=float(value), best_native_s=best.native_cmax,
                            temperature_s=temperature, repairs=trace, processing_s=pp.tolist(),
                            move_s=float(sum(cc-pp)), observed_native_wait_s=float(sum(waits))))
        if iteration == 1 or iteration % 50 == 0 or iteration == settings.max_iterations:
            row = dict(phase='RC_IG', iteration=iteration, evaluations=a['ig_native_calls'],
                best_native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
                current_native_cmax_s=current.native_cmax if math.isfinite(current.native_cmax) else None,
                max_processing_s=float(max(best.processing)), mode_changed_candidates=a['mode_changed_candidates'],
                construction_failures=a['construction_failures'], duplicates=a['duplicate_candidates'],
                elapsed_s=time.perf_counter()-started, temperature_s=temperature)
            progress.append(row)
            if progress_callback is not None:
                progress_callback(row)
        temperature *= settings.cooling
    a.update(ig_search_s=time.perf_counter()-search_started,
        search_s=initial.accounting['search_s']+time.perf_counter()-search_started,
        initial_temperature_s=initial_temperature, final_temperature_s=temperature,
        search_best_native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
        search_best_processing_s=best.processing.tolist(), search_best_move_s=float(sum(best.costs-best.processing)),
        search_complete=a['numeric_failures'] == 0)
    solution, vectors, schedule, native = initial.solution, initial.directions, initial.schedule, initial.native_cmax
    final = None
    if best.identity != identity and best.native_cmax < initial_native-1.e-8:
        improved_solution = workspace.solution(best.routes, best.selected)
        expected = tuple(tuple(workspace.ids[int(n)] for n in row) for row in best.routes)
        if (tuple(r.block_ids for r in improved_solution.routes) != expected or
                improved_solution.patterns != tuple(workspace.patterns[p] for p in best.selected)):
            raise RuntimeError('canonicalization changed the directed RC-IG candidate')
        improved_bits = tuple(tuple(int(d) for d in row) for row in best.directions)
        t = time.perf_counter()
        attempt = reference_schedule_formal(improved_solution, config, scope=scope,
            orientations=dict(enumerate(improved_bits)))
        elapsed = time.perf_counter()-t
        a.update(final_b32_calls=1, final_b32_s=elapsed, b32_calls=a['b32_calls']+1, b32_s=a['b32_s']+elapsed)
        final = dict(status=attempt.status.value, cmax_s=attempt.cmax if attempt.feasible else None,
            native_cmax_s=best.native_cmax, processing_s=best.processing.tolist(),
            move_s=float(sum(best.costs-best.processing)),
            wait_s=sum(op.duration for op in attempt.operations if op.kind is OperationKind.WAIT),
            wait_is_partial=not attempt.feasible, b32_s=elapsed, diagnostic=list(attempt.diagnostics),
            solution=solution_record(improved_solution, improved_bits))
        if attempt.feasible and attempt.cmax < initial.cmax-1.e-8:
            if attempt.directions != improved_bits:
                raise RuntimeError('B32 changed explicit RC-IG directions')
            solution, vectors, schedule, native = improved_solution, improved_bits, attempt, best.native_cmax
            a.update(return_reason='IMPROVED_B32', processing_s=best.processing.tolist(),
                move_s=final['move_s'], wait_s=final['wait_s'])
        else:
            a.update(final_fallback=True, return_reason='FINAL_B32_FAILED' if not attempt.feasible
                     else 'INITIAL_B32_BETTER_OR_EQUAL')
    else:
        a['return_reason'] = 'NO_NATIVE_IMPROVEMENT'
    if schedule is initial.schedule:
        a.update(processing_s=list(initial.accounting['processing_s']),
            move_s=initial.accounting['move_s'], wait_s=initial.accounting['wait_s'])
    return RCIGResult('FEASIBLE', schedule.cmax, solution, vectors, schedule, native,
        time.perf_counter()-started, a['iterations'], 'ITERATION_LIMIT', a, initial, progress, accepted, final)
