"""Generic deterministic dispatch traces for the independent Python tools.

This module owns search bookkeeping only.  Scientific timing, ESS, WAIT, and
conflict semantics stay in :mod:`mrta_reference.scheduler` and are supplied by
the scheduler kernel used by ``run_dispatch_rollout``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
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






