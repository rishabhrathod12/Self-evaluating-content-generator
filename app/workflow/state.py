"""LangGraph workflow state schema.

State holds only plain, JSON-serializable values (lessons and evaluations are
stored as dicts produced by ``model_dump()``).
"""

from typing import Any, Literal, TypedDict

# Hard upper bound on improvement attempts, whatever the configuration says.
MAX_IMPROVEMENT_ATTEMPTS = 2

WorkflowStatus = Literal["in_progress", "passed", "rejected", "error"]


class RejectionRecord(TypedDict):
    """One failed evaluation."""

    attempt_number: int  # 1 = initial lesson, 2 = after the first improvement, ...
    failed_criteria: list[str]
    failure_reasons: list[str]
    improvement_feedback: str
    another_attempt_made: bool
    changes: list[str]  # what the improver changed in response (filled in later)
    missing_concepts: list[str]  # ids of concepts not taught at all
    weak_concepts: list[str]  # ids of concepts only partly taught
    problem_sentences: list[str]  # exact lesson text the evaluator flagged
    # What happened after this failure. Empty ``changes`` is NOT a failed
    # improvement: the final entry never has one, because no improvement
    # follows it. ``change_summary`` always says which case applies.
    change_summary: str
    improvement_unaddressed: list[str]  # points the revision did not state
    concepts_resolved_next: list[str]  # fixed per the next evaluation
    concepts_open_next: list[str]  # still open per the next evaluation


class ErrorInfo(TypedDict):
    """A failure raised by an agent."""

    node: str
    error_type: str
    message: str


class WorkflowState(TypedDict):
    """Shared state passed between workflow nodes."""

    topic: str
    lesson: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    evaluation_count: int  # evaluations run so far
    improvement_attempts: int  # improvements run so far (separate from the above)
    maximum_attempts: int  # improvement attempts allowed
    status: WorkflowStatus
    rejection_log: list[RejectionRecord]
    error: ErrorInfo | None
    best_lesson: dict[str, Any] | None  # highest-scoring evaluated lesson
    best_evaluation: dict[str, Any] | None
    best_evaluation_number: int  # which evaluation produced it (0 = none yet)
    latest_lesson: dict[str, Any] | None  # set when a rejected run returns
    latest_evaluation: dict[str, Any] | None  # an earlier, better version
    rejection_reason: str  # why the run was rejected (empty otherwise)


def initial_state(topic: str, maximum_attempts: int) -> WorkflowState:
    """Create the starting state, clamping the retry limit to 0..2."""
    return WorkflowState(
        topic=topic,
        lesson=None,
        evaluation=None,
        evaluation_count=0,
        improvement_attempts=0,
        maximum_attempts=max(0, min(maximum_attempts, MAX_IMPROVEMENT_ATTEMPTS)),
        status="in_progress",
        rejection_log=[],
        error=None,
        best_lesson=None,
        best_evaluation=None,
        best_evaluation_number=0,
        latest_lesson=None,
        latest_evaluation=None,
        rejection_reason="",
    )
