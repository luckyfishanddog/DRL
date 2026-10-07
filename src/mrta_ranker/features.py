"""Stage-available MLP features and separate supervised targets.

Extraction is pure: no scheduler, direction optimization, random draws, fitted
normalization, or state mutation. Geometry is measured in metres and travel/load
in seconds. Ordered dict keys form the numeric column order. Direction vectors
are summarized per robot (counts, alternations, endpoint orientations/geometry);
variable-length bit vectors and workbook/parent identities are not model inputs.
"""
from __future__ import annotations

import math
import statistics

from mrta_reference.geometry import oriented_endpoints
from mrta_reference.model import MoveType, SplitKind
from mrta_reference.solution import block_map
from mrta_search.lns import CandidateSourceKind, DestroyOperator, RepairOperator
from mrta_search.pipeline import decision_family


def _positive_cmax(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Current Cmax must be positive and finite")


def _summary(out, prefix, values):
    values = tuple(values)
    out[prefix + "_min"] = min(values, default=0.0)
    out[prefix + "_max"] = max(values, default=0.0)
    out[prefix + "_mean"] = statistics.fmean(values) if values else 0.0
    out[prefix + "_std"] = statistics.pstdev(values) if values else 0.0


def _solution_values(solution, config):
    blocks = block_map(solution, config)
    routes = {route.robot_id: route.block_ids for route in solution.routes}
    loads = [sum(config.process_time(blocks[b].length) for b in routes[r]) for r in range(4)]
    return blocks, routes, loads


def _travel(routes, blocks, directions, config):
    if len(directions) != 4 or any(len(directions[r]) != len(routes[r]) for r in range(4)):
        raise ValueError("Current route/direction lengths differ")
    total = 0.0
    for r in range(4):
        points = [oriented_endpoints(blocks[b], d) for b, d in zip(routes[r], directions[r])]
        total += sum(math.dist(left[1], right[0]) / config.empty_speed for left, right in zip(points, points[1:]))
    return total


def extract_c2_features(current, current_directions, current_cmax, candidate, config):
    """Use only information available before candidate direction DP.

    current_directions belong to the existing incumbent, never the candidate.
    candidate is the pre-DP CompleteSearchCandidate, which has no oracle fields.
    """
    _positive_cmax(current_cmax)
    before_blocks, before_routes, before_loads = _solution_values(current, config)
    _, after_routes, after_loads = _solution_values(candidate.solution, config)
    travel = _travel(before_routes, before_blocks, current_directions, config)
    move = candidate.atomic_move
    metadata = dict(candidate.provenance)
    out = {}
    for source in CandidateSourceKind:
        out["source_" + source.value.lower()] = float(candidate.source_kind is source)
    for kind in MoveType:
        if kind is MoveType.TWO_OPT_STAR:  # Disabled; no constant-only input column.
            continue
        out["move_" + kind.value.lower()] = float(move is not None and move.key.move_type is kind)
    family = decision_family(candidate)
    for name in ("STRUCTURAL", "TARGET_WHOLE", "TARGET_Y", "TARGET_X"):
        out["family_" + name.lower()] = float(family == name)
    for kind in DestroyOperator:
        out["destroy_" + kind.value.lower()] = float(metadata.get("destroy_operator") == kind.value)
    for kind in RepairOperator:
        out["repair_" + kind.value.lower()] = float(metadata.get("repair_operator") == kind.value)
    out.update(
        current_cmax=float(current_cmax),
        projected_process_before=max(before_loads),
        projected_process_after=float(candidate.projected_process_makespan),
        projected_process_delta=float(candidate.projected_process_makespan - max(before_loads)),
        travel_proxy_before=travel,
        travel_proxy_after=travel + candidate.local_directed_travel_delta,
        travel_delta=float(candidate.local_directed_travel_delta),
        split_count_delta=float(candidate.split_count_delta),
        load_spread_delta=float(candidate.load_spread_delta),
    )
    for label, solution, routes, loads in (
        ("before", current, before_routes, before_loads),
        ("after", candidate.solution, after_routes, after_loads),
    ):
        for r in range(4):
            out[f"{label}_r{r}_process_load"] = loads[r]
            out[f"{label}_r{r}_block_count"] = float(len(routes[r]))
        out[label + "_max_robot_load"] = max(loads)
        out[label + "_load_range"] = max(loads) - min(loads)
        out[label + "_load_std"] = statistics.pstdev(loads)
        for kind in SplitKind:
            out[label + "_" + kind.value.lower() + "_count"] = float(sum(p.kind is kind for p in solution.patterns))
    affected = set(move.key.affected_parent_ids if move is not None else metadata.get("removed_parent_ids", ()))
    robots = {r for r in range(4) if before_routes[r] != after_routes[r]}
    if move is not None:
        robots.update(r for r in (move.key.source_robot_id, move.key.destination_robot_id) if r is not None)
    for r in range(4):
        out[f"affected_r{r}"] = float(r in robots)
    out["affected_robot_count"] = float(len(robots))
    out["affected_upper_rail"] = float(bool(robots & {0, 1}))
    out["affected_lower_rail"] = float(bool(robots & {2, 3}))
    out["affected_same_rail"] = float(bool(robots) and (robots <= {0, 1} or robots <= {2, 3}))
    out["affected_parent_count"] = float(len(affected))
    if move is not None:
        positions_before = move.key.source_positions
        positions_after = move.key.destination_positions
    else:
        positions_before = tuple(i for r in range(4) for i, b in enumerate(before_routes[r]) if before_blocks[b].parent_id in affected)
        positions_after = tuple(entry[2] for _, trace in metadata.get("repair_trace", ()) for entry in trace)
    for name, positions in (("remove_position", positions_before), ("insert_position", positions_after)):
        out[name + "_count"] = float(len(positions))
        _summary(out, name, positions)
    _summary(out, "parent_length", (p.length for p in current.parents))
    parents = tuple(p for p in current.parents if p.parent_id in affected)
    _summary(out, "affected_length", (p.length for p in parents))
    points = tuple(point for p in parents for point in (p.start, p.end))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    out["affected_x_span"] = max(xs, default=0.0) - min(xs, default=0.0)
    out["affected_y_span"] = max(ys, default=0.0) - min(ys, default=0.0)
    out["affected_mid_x"] = statistics.fmean(xs) if xs else 0.0
    out["affected_mid_y"] = statistics.fmean(ys) if ys else 0.0
    mid_x = sum(config.workspace_x) / 2
    mid_y = sum(config.workspace_y) / 2
    out["affected_upper_fraction"] = sum(y >= mid_y for y in ys) / len(ys) if ys else 0.0
    out["affected_left_fraction"] = sum(x <= mid_x for x in xs) / len(xs) if xs else 0.0
    return {k: float(v) for k, v in out.items()}


def extract_c4_features(current, current_directions, current_cmax, candidate, config, *, direction, cheap_rank):
    """Add only candidate direction-DP information available before reference.

    Missing/infeasible directions use explicit availability masks and zero
    placeholders, not label penalties. Production C4 still requires feasible DP.
    """
    out = extract_c2_features(current, current_directions, current_cmax, candidate, config)
    blocks, routes, _ = _solution_values(candidate.solution, config)
    feasible = getattr(direction.status, "value", direction.status) == "FEASIBLE"
    available = feasible and direction.total_empty_travel is not None
    if available and not math.isfinite(direction.total_empty_travel):
        raise ValueError("Nonfinite direction cost")
    out["direction_feasible"] = float(feasible)
    out["direction_cost_available"] = float(available)
    out["direction_dp_travel"] = float(direction.total_empty_travel) if available else 0.0
    out["direction_dp_travel_delta"] = out["direction_dp_travel"] - out["travel_proxy_before"] if available else 0.0
    old = {b: d for route in current.routes for b, d in zip(route.block_ids, current_directions[route.robot_id])}
    flips = shared = 0
    for r in range(4):
        vector = direction.directions[r]
        valid = available and len(vector) == len(routes[r])
        if feasible and not valid:
            raise ValueError("Feasible candidate route/direction lengths differ")
        present = valid and bool(routes[r])
        prefix = f"direction_r{r}_"
        out[prefix + "route_available"] = float(present)
        out[prefix + "reverse_count"] = float(sum(vector)) if valid else 0.0
        out[prefix + "alternations"] = float(sum(a != b for a, b in zip(vector, vector[1:]))) if valid else 0.0
        out[prefix + "first_orientation"] = float(vector[0]) if present else 0.0
        out[prefix + "last_orientation"] = float(vector[-1]) if present else 0.0
        start = oriented_endpoints(blocks[routes[r][0]], vector[0])[0] if present else (0.0, 0.0)
        end = oriented_endpoints(blocks[routes[r][-1]], vector[-1])[1] if present else (0.0, 0.0)
        out[prefix + "first_x"], out[prefix + "first_y"] = start
        out[prefix + "last_x"], out[prefix + "last_y"] = end
        if valid:
            for b, d in zip(routes[r], vector):
                if b in old:
                    shared += 1
                    flips += d != old[b]
    out["direction_shared_blocks"] = float(shared)
    out["direction_flips"] = float(flips)
    # Other numeric C3 rerank components already appear in C2:
    # projected_process_after, split_count_delta. Identity is a tie-break only.
    if cheap_rank < 0 or int(cheap_rank) != cheap_rank:
        raise ValueError("cheap_rank must be a nonnegative integer")
    out["c3_cheap_rank"] = float(cheap_rank)
    return {k: float(v) for k, v in out.items()}


def make_training_targets(status, current_cmax, candidate_cmax):
    """Separate oracle labels; never called by either feature extractor."""
    status = getattr(status, "value", status)
    if status == "NUMERIC_FAILURE":
        return None
    _positive_cmax(current_cmax)
    if status == "FEASIBLE_CERTIFIED":
        if candidate_cmax is None or not math.isfinite(candidate_cmax) or candidate_cmax <= 0:
            raise ValueError("Certified candidate needs positive finite Cmax")
        return {"y_feasible": 1, "y_improvement": (current_cmax - candidate_cmax) / current_cmax}
    if status not in {"DIRECTION_INFEASIBLE", "DEADLOCK", "INFEASIBLE", "FEASIBLE"}:
        raise ValueError("Unknown evaluated-candidate status: " + str(status))
    return {"y_feasible": 0, "y_improvement": None}
