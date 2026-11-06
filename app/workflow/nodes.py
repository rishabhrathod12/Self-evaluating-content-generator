"""LangGraph workflow nodes. Each node delegates to an existing agent."""

from typing import Any

from app.agents.evaluator import EvaluationError, evaluate_lesson
from app.agents.generator import LessonGenerationError, generate_lesson
from app.agents.improver import ImprovementError, improve_lesson
from app.config import ConfigurationError
from app.evaluation.schemas import EvaluationResult, LessonContent
from app.evaluation.topics import get_requirements
from app.llm import LLMError
from app.workflow.state import RejectionRecord, WorkflowState

# Agent failures we turn into an "error" status. Anything else is a bug and is
# allowed to propagate.
AGENT_ERRORS = (
    LessonGenerationError,
    EvaluationError,
    ImprovementError,
    LLMError,
    ConfigurationError,
)


def _error_update(
    node: str,
    exc: Exception,
    state: WorkflowState | None = None,
    current_evaluated: bool = False,
) -> dict[str, Any]:
    """The state update for an agent failure.

    If a lesson has already been evaluated, the best lesson is returned together
    with its own evaluation. The current lesson, when it differs, is kept in
    ``latest_lesson`` with its evaluation (``None`` if it was never evaluated).
    Before any evaluation exists there is nothing to pair, so only the error is
    recorded.
    """
    update: dict[str, Any] = {
        "status": "error",
        "error": {
            "node": node,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    }
    if state is None or state["best_evaluation"] is None:
        return update
    best_lesson = state["best_lesson"]
    update.update(lesson=best_lesson, evaluation=state["best_evaluation"])
    if state["lesson"] != best_lesson:
        update.update(
            latest_lesson=state["lesson"],
            latest_evaluation=state["evaluation"] if current_evaluated else None,
        )
    return update


def evaluation_score(evaluation: dict[str, Any]) -> tuple[int, int]:
    """Rank evaluations: more passed checks first, then more covered concepts."""
    passed = sum(1 for check in evaluation["checks"] if check["passed"])
    covered = sum(1 for c in evaluation.get("concept_checks", []) if c["covered"])
    return passed, covered


def _problem_sentences(evaluation: EvaluationResult) -> list[str]:
    """Exact lesson text the evaluator flagged, without duplicates."""
    sentences: list[str] = []
    for check in evaluation.checks:
        if check.passed:
            continue
        for issue in check.issues:
            quote = issue.quote.strip()[:300]
            if quote and quote not in sentences:
                sentences.append(quote)
    return sentences


def summarize_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Describe, field by field, how an improved lesson differs from the old one."""
    changes = []
    for field in ("title", "learning_objectives", "key_takeaways", "summary"):
        if old[field] != new[field]:
            changes.append(f"{field} changed")
    old_sections, new_sections = old["sections"], new["sections"]
    differing = sum(a != b for a, b in zip(old_sections, new_sections, strict=False))
    differing += abs(len(old_sections) - len(new_sections))
    if differing:
        changes.append(
            f"sections: {len(old_sections)} -> {len(new_sections)}, {differing} changed"
        )
    return changes or ["no differences detected"]


async def generate_node(state: WorkflowState) -> dict[str, Any]:
    """Generate the first lesson for the topic."""
    try:
        lesson = await generate_lesson(state["topic"])
    except AGENT_ERRORS as exc:
        return _error_update("generate", exc)
    return {"lesson": lesson.model_dump()}


async def evaluate_node(state: WorkflowState) -> dict[str, Any]:
    """Evaluate the current lesson, track the best one and set the status.

    PASS -> "passed". FAIL with attempts left -> "in_progress" (route to improve).
    FAIL with no attempts left -> "rejected", returning the best-scoring lesson
    evaluated so far (not necessarily the latest). Every evaluation covers all
    seven criteria.
    """
    try:
        evaluation = await evaluate_lesson(
            LessonContent.model_validate(state["lesson"]),
            requirements=get_requirements(state["topic"]),
        )
    except AGENT_ERRORS as exc:
        return _error_update("evaluate", exc, state)

    count = state["evaluation_count"] + 1
    evaluation_dict = evaluation.model_dump()
    update: dict[str, Any] = {
        "evaluation": evaluation_dict,
        "evaluation_count": count,
    }
    best_lesson = state["best_lesson"]
    best_evaluation = state["best_evaluation"]
    best_number = state["best_evaluation_number"]
    if best_evaluation is None or (
        evaluation_score(evaluation_dict) > evaluation_score(best_evaluation)
    ):
        best_lesson, best_evaluation = state["lesson"], evaluation_dict
        best_number = count
        update.update(
            best_lesson=best_lesson,
            best_evaluation=best_evaluation,
            best_evaluation_number=best_number,
        )
    if evaluation.overall_pass:
        update["status"] = "passed"
        return update

    attempts_remain = state["improvement_attempts"] < state["maximum_attempts"]
    record = RejectionRecord(
        attempt_number=count,
        failed_criteria=[c.criterion for c in evaluation.checks if not c.passed],
        failure_reasons=list(evaluation.failure_reasons),
        improvement_feedback=evaluation.improvement_feedback,
        another_attempt_made=attempts_remain,
        changes=[],
        missing_concepts=[
            c.concept_id for c in evaluation.concept_checks if not c.covered
        ],
        problem_sentences=_problem_sentences(evaluation),
    )
    update["rejection_log"] = [*state["rejection_log"], record]
    update["status"] = "in_progress" if attempts_remain else "rejected"
    if not attempts_remain and best_number != count:
        # The latest version is not the best one: return the best, keep the latest.
        update.update(
            lesson=best_lesson,
            evaluation=best_evaluation,
            latest_lesson=state["lesson"],
            latest_evaluation=evaluation_dict,
        )
    return update


async def improve_node(state: WorkflowState) -> dict[str, Any]:
    """Improve the lesson using the latest evaluation and log what changed."""
    lesson = LessonContent.model_validate(state["lesson"])
    evaluation = EvaluationResult.model_validate(state["evaluation"])
    try:
        # The last log entry is the current evaluation; the rest are earlier rounds.
        improved = await improve_lesson(
            lesson,
            evaluation,
            history=state["rejection_log"][:-1],
            requirements=get_requirements(state["topic"]),
        )
    except AGENT_ERRORS as exc:
        return _error_update("improve", exc, state, current_evaluated=True)

    new_lesson = improved.model_dump()
    log = [dict(record) for record in state["rejection_log"]]
    log[-1]["changes"] = summarize_changes(state["lesson"], new_lesson)
    return {
        "lesson": new_lesson,
        "improvement_attempts": state["improvement_attempts"] + 1,
        "rejection_log": log,
    }
