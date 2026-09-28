"""Phase 2B-1 bounded SA-OI neighborhood-search backbone.

This package is intentionally not a complete ALNS layer: it has no destroy/repair
operators and no adaptive operator selection.
"""

from .direction import (
    ConstrainedDirectionResult,
    DirectionStatus,
    DirectionVectors,
    optimize_directions_with_initial_feasibility,
)
from .stats import ACTIVE_MOVE_TYPES, SearchStats
from .initialization import (
    InitializationResult,
    InitializationStatus,
    bounded_insertion_positions,
    build_initial_solution,
)
from .neighborhood import (
    RawAttempt,
    ScreenedCandidate,
    balanced_move_attempt_order,
    generate_raw_attempts,
    screen_raw_attempts,
)
from .pipeline import (
    MicroGapResult,
    SearchConfig,
    SearchResult,
    SearchStatus,
    evaluate_iteration,
    micro_gap_decomposition,
    rerank_c3,
    run_bounded_sa_oi,
    sa_accept,
)

__all__ = [
    "ACTIVE_MOVE_TYPES",
    "ConstrainedDirectionResult",
    "DirectionStatus",
    "DirectionVectors",
    "InitializationResult",
    "InitializationStatus",
    "MicroGapResult",
    "RawAttempt",
    "ScreenedCandidate",
    "SearchConfig",
    "SearchResult",
    "SearchStatus",
    "SearchStats",
    "balanced_move_attempt_order",
    "bounded_insertion_positions",
    "build_initial_solution",
    "evaluate_iteration",
    "generate_raw_attempts",
    "optimize_directions_with_initial_feasibility",
    "micro_gap_decomposition",
    "rerank_c3",
    "run_bounded_sa_oi",
    "sa_accept",
    "screen_raw_attempts",
]
