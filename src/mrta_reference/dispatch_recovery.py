"""Deterministic dispatch traces and complete-rollout discrepancy recovery.

This module owns search bookkeeping only.  Scientific timing, ESS, WAIT, and
conflict semantics stay in :mod:`mrta_reference.scheduler` and are supplied by
the scheduler kernel used by ``run_dispatch_rollout``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import time
from collections.abc import Callable
from typing import Protocol

from .model import Operation, ScheduleResult, ScheduleStatus


def _operation_payload(operation: Operation) -> dict[str, object]:
    return {
        "id": operation.operation_id,
        "robot": operation.robot_id,
        "kind": operation.kind.value,
        "start_time": operation.start_time,
        "end_time": operation.end_time,
        "start": operation.start,
        "end": operation.end,
        "sequence": operation.sequence_index,
        "block": operation.block_id,
    }


@dataclass(frozen=True)
class DispatchState:
    next_index: tuple[int, int, int, int] = (0, 0, 0, 0)
    completion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    current_point: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None, None, None)
    fixed_operations: tuple[Operation, ...] = ()
    wait_count: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def depth(self) -> int:
        return sum(self.next_index)

    @property
    def canonical_json(self) -> str:
        payload = {
            "next_index": self.next_index,
            "completion": self.completion,
            "current_point": self.current_point,
            "wait_count": self.wait_count,
            "fixed_operations": [
                _operation_payload(operation) for operation in self.fixed_operations
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DispatchChoice:
    priority: tuple[float, float, float, int]
    robot_id: int
    template_index: int
    operation_id: str
    start_time: float

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "priority": self.priority,
                "robot_id": self.robot_id,
                "template_index": self.template_index,
                "operation_id": self.operation_id,
                "start_time": self.start_time,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ForcedDispatchDecision:
    state_identity: str
    choice_identity: str

    @property
    def key(self) -> tuple[str, str]:
        return self.state_identity, self.choice_identity


@dataclass(frozen=True)
class DispatchChoiceSet:
    ordered_choices: tuple[DispatchChoice, ...]
    blockers_by_robot: tuple[tuple[int, tuple[int, ...]], ...] = ()
    blocker_ids_by_robot: tuple[tuple[int, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class BranchPoint:
    depth: int
    ordinal: int
    state: DispatchState
    ordered_choices: tuple[DispatchChoice, ...]
    chosen_choice_rank: int

    @property
    def identity(self) -> str:
        payload = (
            self.depth,
            self.ordinal,
            self.state.identity,
            tuple(choice.identity for choice in self.ordered_choices),
        )
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RolloutTrace:
    branch_points: tuple[BranchPoint, ...]
    initial_depth: int
    terminal_depth: int
    forced_decisions: tuple[ForcedDispatchDecision, ...] = ()
    terminal_blocker_operation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RolloutOutcome:
    result: ScheduleResult
    trace: RolloutTrace
    terminal_state: DispatchState


class DispatchKernel(Protocol):
    def start_runtime(self, state: DispatchState): ...

    def is_complete(self, state: DispatchState) -> bool: ...

    def choices(self, state: DispatchState, runtime) -> DispatchChoiceSet: ...

    def commit(self, state: DispatchState, choice: DispatchChoice, runtime) -> DispatchState: ...

    def feasible_result(self, state: DispatchState) -> ScheduleResult: ...

    def deadlock_result(
        self, state: DispatchState, choices: DispatchChoiceSet
    ) -> ScheduleResult: ...


def run_dispatch_rollout(
    kernel: DispatchKernel,
    *,
    initial_state: DispatchState | None = None,
    forced_decisions: tuple[ForcedDispatchDecision, ...] = (),
    collect_trace: bool = True,
    choice_selector: Callable[
        [DispatchState, tuple[DispatchChoice, ...]], int
    ] | None = None,
    profile=None,
) -> RolloutOutcome:
    """Run a complete deterministic continuation to FEASIBLE or DEADLOCK."""
    state = DispatchState() if initial_state is None else initial_state
    initial_depth = state.depth
    runtime = kernel.start_runtime(state)
    trace: list[BranchPoint] = []
    forced_index = 0

    while True:
        if kernel.is_complete(state):
            if forced_index != len(forced_decisions):
                raise ValueError("forced dispatch plan was not encountered")
            return RolloutOutcome(
                kernel.feasible_result(state),
                RolloutTrace(
                    tuple(trace), initial_depth, state.depth, forced_decisions, ()
                ),
                state,
            )

        choice_set = kernel.choices(state, runtime)
        choices = choice_set.ordered_choices
        if not choices:
            if forced_index != len(forced_decisions):
                raise ValueError("forced dispatch plan was not encountered")
            return RolloutOutcome(
                kernel.deadlock_result(state, choice_set),
                RolloutTrace(
                    tuple(trace),
                    initial_depth,
                    state.depth,
                    forced_decisions,
                    tuple(
                        sorted(
                            {
                                operation_id
                                for _, operation_ids in choice_set.blocker_ids_by_robot
                                for operation_id in operation_ids
                            }
                        )
                    ),
                ),
                state,
            )

        chosen_rank = 0
        if forced_index < len(forced_decisions):
            forced = forced_decisions[forced_index]
            identity_started = time.perf_counter()
            state_identity = state.identity
            if profile is not None:
                profile.state_identity_time += time.perf_counter() - identity_started
            if forced.state_identity == state_identity:
                identity_started = time.perf_counter()
                matches = [
                    index
                    for index, choice in enumerate(choices)
                    if choice.identity == forced.choice_identity
                ]
                if profile is not None:
                    profile.state_identity_time += time.perf_counter() - identity_started
                if len(matches) != 1:
                    raise ValueError("forced dispatch choice is unavailable or ambiguous")
                chosen_rank = matches[0]
                forced_index += 1
        elif choice_selector is not None:
            chosen_rank = choice_selector(state, choices)
            if not 0 <= chosen_rank < len(choices):
                raise ValueError("dispatch choice selector returned an invalid rank")

        if collect_trace and len(choices) >= 2:
            trace_started = time.perf_counter()
            trace.append(
                BranchPoint(
                    state.depth,
                    len(trace),
                    state,
                    choices,
                    chosen_rank,
                )
            )
            if profile is not None:
                elapsed = time.perf_counter() - trace_started
                profile.branch_trace_time += elapsed
                profile.branch_snapshot_preparation_time += elapsed
        state = kernel.commit(state, choices[chosen_rank], runtime)


@dataclass(order=True)
class _FrontierItem:
    priority: tuple[object, ...]
    branch_point: BranchPoint = field(compare=False)
    alternative_rank: int = field(compare=False)
    discrepancy_count: int = field(compare=False)
    plan: tuple[ForcedDispatchDecision, ...] = field(compare=False)
    causal_blocker_operation_ids: tuple[str, ...] = field(
        compare=False, default=()
    )


@dataclass(frozen=True)
class RecoverySummary:
    result: ScheduleResult
    recovery_rollouts: int
    rollout_budget: int
    max_discrepancies_used: int
    branch_points_considered: int
    frontier_exhausted: bool
    best_plan: tuple[ForcedDispatchDecision, ...] = ()


def limited_discrepancy_recovery(
    baseline: RolloutOutcome,
    *,
    rollout_budget: int,
    rollout_from_snapshot,
    profile=None,
) -> RecoverySummary:
    """Search deterministic complete continuations with a rollout-count budget."""
    if rollout_budget < 1:
        raise ValueError("deadlock rollout budget must be positive")
    if baseline.result.status is not ScheduleStatus.DEADLOCK:
        return RecoverySummary(baseline.result, 0, rollout_budget, 0, 0, True)

    frontier: list[_FrontierItem] = []
    enqueued: set[tuple[str, str, tuple[str, ...]]] = set()
    considered: set[str] = set()

    def enqueue_trace(
        trace: RolloutTrace,
        discrepancy_count: int,
        prefix_plan: tuple[ForcedDispatchDecision, ...],
        *,
        after_depth: int,
    ) -> None:
        enqueue_started = time.perf_counter()
        eligible = tuple(
            point for point in trace.branch_points if point.depth > after_depth
        )
        terminal_blockers = set(trace.terminal_blocker_operation_ids)
        causal = tuple(
            point
            for point in eligible
            if point.ordered_choices[point.chosen_choice_rank].operation_id
            in terminal_blockers
        )
        branch_points = sorted(
            causal if causal else eligible,
            key=lambda point: (-point.depth, point.ordinal, point.identity),
        )
        for point in branch_points:
            considered.add(point.identity)
            for rank in range(1, len(point.ordered_choices)):
                choice = point.ordered_choices[rank]
                decision = ForcedDispatchDecision(point.state.identity, choice.identity)
                dedup_key = (*decision.key, ())
                if dedup_key in enqueued:
                    continue
                enqueued.add(dedup_key)
                plan = prefix_plan + (decision,)
                plan_key = tuple(item.key for item in plan)
                heapq.heappush(
                    frontier,
                    _FrontierItem(
                        (
                            discrepancy_count,
                            -point.depth,
                            rank,
                            plan_key,
                        ),
                        point,
                        rank,
                        discrepancy_count,
                        plan,
                    ),
                )
        if profile is not None:
            elapsed = time.perf_counter() - enqueue_started
            profile.frontier_time += elapsed
            profile.dedup_time += elapsed
            profile.peak_branch_states = max(
                profile.peak_branch_states, len(frontier)
            )
            profile.peak_dedup_entries = max(
                profile.peak_dedup_entries, len(enqueued)
            )

    # The baseline root has depth -1 for filtering so every branch is eligible.
    enqueue_trace(baseline.trace, 1, (), after_depth=-1)
    rollouts = 0
    max_discrepancies = 0
    best: ScheduleResult | None = None
    best_plan: tuple[ForcedDispatchDecision, ...] = ()

    while frontier and rollouts < rollout_budget:
        frontier_started = time.perf_counter()
        item = heapq.heappop(frontier)
        if profile is not None:
            profile.frontier_time += time.perf_counter() - frontier_started
        point = item.branch_point
        choice = point.ordered_choices[item.alternative_rank]
        decision = ForcedDispatchDecision(point.state.identity, choice.identity)
        outcome = rollout_from_snapshot(
            point.state, decision, item.causal_blocker_operation_ids
        )
        rollouts += 1
        actual_discrepancies = sum(
            branch.chosen_choice_rank != 0 for branch in outcome.trace.branch_points
        )
        max_discrepancies = max(
            max_discrepancies, item.discrepancy_count, actual_discrepancies
        )

        if outcome.result.status is ScheduleStatus.NUMERIC_FAILURE:
            return RecoverySummary(
                outcome.result,
                rollouts,
                rollout_budget,
                max_discrepancies,
                len(considered),
                not frontier,
                (),
            )
        if outcome.result.status is ScheduleStatus.FEASIBLE:
            candidate_key = (outcome.result.cmax, outcome.result.canonical_json())
            if best is None or candidate_key < (best.cmax, best.canonical_json()):
                best = outcome.result
                best_plan = item.plan

        if (
            outcome.result.status is ScheduleStatus.DEADLOCK
            and not item.causal_blocker_operation_ids
            and outcome.trace.terminal_blocker_operation_ids
        ):
            # After the complete single-discrepancy layer, one causal chain
            # suppresses the operations that directly caused that rollout's
            # terminal DEADLOCK. The selector is derived only from the
            # rollout's wait-for evidence; it has no instance-specific cases.
            blockers = outcome.trace.terminal_blocker_operation_ids
            chain_key = (*decision.key, blockers)
            if chain_key not in enqueued:
                enqueued.add(chain_key)
                plan_key = tuple(entry.key for entry in item.plan)
                heapq.heappush(
                    frontier,
                    _FrontierItem(
                        (
                            item.discrepancy_count + 1,
                            -point.depth,
                            item.alternative_rank,
                            plan_key,
                        ),
                        point,
                        item.alternative_rank,
                        item.discrepancy_count + 1,
                        item.plan,
                        blockers,
                    ),
                )

    return RecoverySummary(
        baseline.result if best is None else best,
        rollouts,
        rollout_budget,
        max_discrepancies,
        len(considered),
        not frontier,
        best_plan,
    )
