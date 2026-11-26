"""Deterministic atomic tasks and mandatory pre-ordering continuous weld jobs.

Only generate_taskset decides cuts. Search reads its saved tasks/patterns and
may change owners, job order and job directions, never the task boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections import defaultdict

from .geometry import blocks_for_pattern, whole_eligible_rails, robot_is_eligible
from .model import ParentWeld, Rail, Route, ScientificConfig, SplitKind, FixedSplitPattern, FixedWeldingBlock, canonical_routes


@dataclass(frozen=True)
class AtomicTask:
    task_id: str
    source_parent_id: str
    segment_index: int
    u_start: float
    u_end: float
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    eligible_robots: tuple[int, ...]


@dataclass(frozen=True)
class FixedTaskset:
    parents: tuple[ParentWeld, ...]
    tasks: tuple[AtomicTask, ...]
    statistics: dict

    @property
    def patterns(self):
        grouped = defaultdict(list)
        for task in self.tasks:
            grouped[task.source_parent_id].append(task)
        return tuple(FixedSplitPattern(p.parent_id, SplitKind.FIXED,
                     cuts=(0., *(t.u_end for t in grouped[p.parent_id])),
                     fixed_eligible=tuple(t.eligible_robots for t in grouped[p.parent_id])) for p in self.parents)

    def record(self):
        return dict(parents=[asdict(p) for p in self.parents], tasks=[asdict(t) for t in self.tasks],
                    statistics=self.statistics)

    @classmethod
    def from_record(cls, row):
        parents = tuple(ParentWeld(p['parent_id'], tuple(p['start']), tuple(p['end'])) for p in row['parents'])
        tasks = tuple(AtomicTask(**{**t, 'start': tuple(t['start']), 'end': tuple(t['end']),
                                  'eligible_robots': tuple(t['eligible_robots'])}) for t in row['tasks'])
        return cls(parents, tasks, row['statistics'])


def long_segment_count(length):
    """Largest integer n with L/n > 1; no tolerance rounds an integer up."""
    if length <= 5.:
        return 1
    n = math.ceil(length) - 1
    while length / n <= 1.:
        n -= 1
    return n


def generate_taskset(parents, config=ScientificConfig()):
    parents = tuple(sorted(parents, key=lambda p: p.parent_id))
    if len({p.parent_id for p in parents}) != len(parents):
        raise ValueError('duplicate original parent_id')
    stats = dict(parent_count=len(parents), y_split_count=0, y_strict_1m_count=0,
                 y_relaxed_0_5m_count=0, long_region_count=0, long_split_extra_tasks=0)
    tasks = []
    for parent in parents:
        for point in (parent.start, parent.end):
            if not (config.workspace_x[0]-config.numeric_epsilon <= point[0] <= config.workspace_x[1]+config.numeric_epsilon
                    and config.workspace_y[0]-config.numeric_epsilon <= point[1] <= config.workspace_y[1]+config.numeric_epsilon):
                raise ValueError(f'{parent.parent_id}: outside workspace: {parent.start}, {parent.end}')
        regions = (0., 1.)
        if not whole_eligible_rails(parent.start, parent.end, config):
            dy = parent.end[1]-parent.start[1]
            lo, hi = sorted((y-parent.start[1])/dy for y in config.by)
            t = max(0., lo, min(.5, hi, 1.))
            # Keep a boundary intersection inside the actual floating-point
            # shared band too, rather than accepting an epsilon-outside cut.
            while not config.by[0] <= parent.point(t)[1] <= config.by[1]:
                adjusted = math.nextafter(t,(lo+hi)/2.)
                if adjusted == t:
                    raise ValueError(f'{parent.parent_id}: Y intersection is not representable inside shared band')
                t = adjusted
            short = min(math.dist(parent.start,parent.point(t)),math.dist(parent.point(t),parent.end))
            if short <= .5 or not (0. < t < 1.):
                raise ValueError(f'{parent.parent_id}: no legal fixed Y cut with both lengths >0.5m; '
                                 f'start={parent.start}, end={parent.end}, length={parent.length}, t={t}')
            stats['y_split_count'] += 1
            stats['y_strict_1m_count' if short > 1. else 'y_relaxed_0_5m_count'] += 1
            regions = (0., t, 1.)
        index = 0
        for a, b in zip(regions, regions[1:]):
            length = math.dist(parent.point(a), parent.point(b))
            # Y responsibility is established BEFORE long splitting. A small
            # atom inside the overlap must not reopen a different rail for an
            # originally single-rail regional weld.
            regional_rails = whole_eligible_rails(parent.point(a), parent.point(b), config)
            n = long_segment_count(length)
            if length > 5.:
                # At nextafter(integer, +inf), interpolation can round an
                # actual new segment to exactly 1m. Such an unrepresentable
                # input must be reported rather than silently violating >1m.
                stats['long_region_count'] += 1
                stats['long_split_extra_tasks'] += n-1
            for j in range(n):
                u = a+(b-a)*j/n
                v = b if j == n-1 else a+(b-a)*(j+1)/n
                start, end = parent.point(u), parent.point(v)
                actual = math.dist(start, end)
                if n > 1 and actual <= 1.:
                    raise ValueError(f'{parent.parent_id}: cannot represent strict >1m equal split: L={length}, n={n}')
                rails = regional_rails
                eligible = tuple(r for r in range(4) if (Rail.UPPER if r < 2 else Rail.LOWER) in rails)
                if not eligible:
                    raise ValueError(f'{parent.parent_id}: fixed atom has no eligible robot')
                tasks.append(AtomicTask(f'{parent.parent_id}::seg_{index:03d}', parent.parent_id, index,
                                        u, v, start, end, actual, eligible))
                index += 1
        own = tasks[-index:]
        if not math.isclose(sum(t.length for t in own), parent.length, rel_tol=1.e-12, abs_tol=1.e-12):
            raise ValueError(f'{parent.parent_id}: fixed task length conservation failure')
    stats.update(atomic_task_count=len(tasks), shortest_task_m=min((t.length for t in tasks), default=0.))
    return FixedTaskset(parents, tuple(tasks), stats)


def fixed_patterns(patterns):
    return bool(patterns) and all(p.kind is SplitKind.FIXED for p in patterns)


def job_id(parent, lo, hi):
    return f'{parent}::seg_{lo:03d}' if hi == lo+1 else f'{parent}::job_{lo:03d}_{hi:03d}'


def job_span(identifier):
    parent, suffix = identifier.rsplit('::', 1)
    if suffix.startswith('seg_'):
        lo = int(suffix[4:]); hi = lo+1
    elif suffix.startswith('job_'):
        lo, hi = map(int, suffix[4:].split('_'))
    else:
        raise ValueError(f'not a fixed atom/job id: {identifier}')
    if not 0 <= lo < hi:
        raise ValueError(f'invalid fixed atom interval: {identifier}')
    return parent,lo,hi


def expand_routes(routes):
    output = [[] for _ in range(4)]
    for route in canonical_routes(routes):
        for identifier in route.block_ids:
            parent, lo, hi = job_span(identifier)
            output[route.robot_id].extend(job_id(parent, j, j+1) for j in range(lo, hi))
    return output


def canonical_fixed_routes(parents, patterns, routes, config):
    """Determine owned geometric runs FIRST, then map a proposed job order.

    If a caller supplies atoms separated in its provisional list, all adjacent
    owned atoms still form one job. Its first occurrence fixes the proposed
    job's position. Route adjacency never determines whether merging occurs.
    """
    if not fixed_patterns(patterns):
        raise ValueError('a fixed taskset cannot include dynamic patterns')
    parent_map = {p.parent_id: p for p in parents}
    pattern_map = {p.parent_id: p for p in patterns}
    if len(pattern_map) != len(patterns) or set(pattern_map) != set(parent_map):
        raise ValueError('fixed parent coverage failure')
    expanded = expand_routes(routes)
    owner = {}
    for r, identifiers in enumerate(expanded):
        for identifier in identifiers:
            parent, lo, hi = job_span(identifier)
            if parent not in pattern_map or lo < 0 or hi >= len(pattern_map[parent].cuts):
                raise ValueError(f'unknown fixed task: {identifier}')
            if identifier in owner:
                raise ValueError(f'fixed atom assigned more than once: {identifier}')
            p, cuts = parent_map[parent], pattern_map[parent].cuts
            block = FixedWeldingBlock(parent, identifier, cuts[lo], cuts[hi], p.point(cuts[lo]), p.point(cuts[hi]),
                                 pattern_map[parent].fixed_eligible[lo])
            if not robot_is_eligible(block, r, config):
                raise ValueError(f'R{r} is ineligible for {identifier}')
            owner[identifier] = r
    mapping, blocks = {}, {}
    for p in parents:
        cuts = pattern_map[p.parent_id].cuts
        lo = 0
        while lo < len(cuts)-1:
            identifier = job_id(p.parent_id, lo, lo+1)
            if identifier not in owner:
                raise ValueError(f'missing fixed atom: {identifier}')
            r = owner[identifier]; hi = lo+1
            while hi < len(cuts)-1 and owner.get(job_id(p.parent_id, hi, hi+1)) == r:
                hi += 1
            merged = job_id(p.parent_id, lo, hi)
            eligible = tuple(r for r in range(4) if all(r in pattern_map[p.parent_id].fixed_eligible[j] for j in range(lo,hi)))
            blocks[merged] = FixedWeldingBlock(p.parent_id, merged, cuts[lo], cuts[hi], p.point(cuts[lo]), p.point(cuts[hi]),eligible)
            for j in range(lo, hi):
                mapping[job_id(p.parent_id, j, j+1)] = merged
            lo = hi
    output = []
    for r, identifiers in enumerate(expanded):
        seen, ordered = set(), []
        for atom in identifiers:
            identifier = mapping[atom]
            if identifier not in seen:
                seen.add(identifier); ordered.append(identifier)
        output.append(Route(r, tuple(ordered)))
    return tuple(output), blocks
