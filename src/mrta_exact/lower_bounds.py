from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from mrta_reference.model import ParentWeld, ScientificConfig


@dataclass(frozen=True)
class AnalyticalLowerBound:
    """The scope-independent analytical LB0 defined by the experiment plan."""

    total_base_processing_work: float
    lb_work: float
    max_parent_parallel_lb: float
    lb0: float


def analytical_lower_bound(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> AnalyticalLowerBound:
    """Compute LB0 without travel, interference, eligibility, or split restrictions."""
    total_work = sum(config.process_time(parent.length) for parent in parents)
    lb_work = total_work / 4.0
    max_parent = max(
        (
            config.t_pre
            + config.t_post
            + parent.length / (2.0 * config.weld_speed)
            for parent in parents
        ),
        default=0.0,
    )
    return AnalyticalLowerBound(
        total_base_processing_work=total_work,
        lb_work=lb_work,
        max_parent_parallel_lb=max_parent,
        lb0=max(lb_work, max_parent),
    )


def eligibility_load_lower_bound(parents, config):
    """Finite-domain fractional parent-mode/assignment relaxation (V2).

    Every parent selects a convex combination of legal pattern/robot options;
    travel, order and interference are omitted. A nonnegative robot-price
    vector summing to one gives the bound sum_p min_option price.dot(load).
    We derive that vector from LP duals and recompute the bound directly,
    rather than treating a feasible primal objective as a certified lower bound.
    This is a diagnostic only, with ordinary floating-point arithmetic.
    """
    import itertools
    import numpy as np
    from scipy.optimize import linprog
    from mrta_reference.geometry import build_legal_pattern_catalog, blocks_for_pattern, robot_is_eligible
    from mrta_reference.model import SplitKind
    from mrta_reference.scope import FORMAL_SCOPE_V2
    lb0 = analytical_lower_bound(parents, config).lb0
    if not parents:
        return dict(lb_process=0.0, lp_objective=0.0, robot_prices=(0.25,)*4,
                    options_count=0, analytical_lb0=0.0)
    catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    by_parent = []
    for parent in parents:
        options = set()
        for pattern in catalog[parent.parent_id]:
            children = blocks_for_pattern(parent, pattern, config)
            legal = [((0 if pattern.rail.value == 'UPPER' else 2)+index,)
                     if pattern.kind is SplitKind.X_SPLIT else
                     tuple(r for r in range(4) if robot_is_eligible(child, r, config))
                     for index, child in enumerate(children)]
            for assignment in itertools.product(*legal):
                load = [0.0]*4
                for child, robot in zip(children, assignment):
                    load[robot] += config.process_time(child.length)
                options.add(tuple(load))
        if not options:
            raise ValueError(f'no legal load option for {parent.parent_id}')
        by_parent.append(sorted(options))
    count = sum(map(len, by_parent))
    loads = np.asarray([row for group in by_parent for row in group])
    equality = np.zeros((len(parents), count+1))
    cursor = 0
    for i, group in enumerate(by_parent):
        equality[i, cursor:cursor+len(group)] = 1
        cursor += len(group)
    objective = np.zeros(count+1); objective[-1] = 1
    inequalities = np.column_stack((loads.T, -np.ones(4)))
    result = linprog(objective, A_ub=inequalities, b_ub=np.zeros(4),
                     A_eq=equality, b_eq=np.ones(len(parents)), bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(f'load relaxation failed: {result.message}')
    prices = np.maximum(-result.ineqlin.marginals, 0.0)
    prices = prices/prices.sum() if prices.sum() > 0 else np.full(4, 0.25)
    dual_bound = sum(min(sum(p*v for p,v in zip(prices, option)) for option in group)
                     for group in by_parent)
    return dict(lb_process=max(lb0, float(dual_bound)), lp_objective=float(result.fun),
                robot_prices=tuple(map(float, prices)), options_count=count, analytical_lb0=lb0)
