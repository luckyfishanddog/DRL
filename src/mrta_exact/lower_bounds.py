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
