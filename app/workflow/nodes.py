"""LangGraph workflow nodes. Each node delegates to an existing agent."""

from typing import Any

from app.agents.evaluator import EvaluationError, evaluate_lesson
from app.agents.generator import LessonGenerationError, generate_lesson
from app.agents.improver import ImprovementError, improve_lesson
from app.config import ConfigurationError
from app.evaluation.schemas import EvaluationResult, LessonContent
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


def _error_update(node: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "error": {
            "node": node,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    }


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
    """Evaluate the current lesson and decide the workflow status.

    PASS -> "passed". FAIL with attempts left -> "in_progress" (route to improve).
    FAIL with no attempts left -> "rejected".
    """
    try:
        evaluation = await evaluate_lesson(
            LessonContent.model_validate(state["lesson"])
        )
    except AGENT_ERRORS as exc:
        return _error_update("evaluate", exc)

    count = state["evaluation_count"] + 1
    update: dict[str, Any] = {
        "evaluation": evaluation.model_dump(),
        "evaluation_count": count,
    }
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
    )
    update["rejection_log"] = [*state["rejection_log"], record]
    update["status"] = "in_progress" if attempts_remain else "rejected"
    return update


async def improve_node(state: WorkflowState) -> dict[str, Any]:
    """Improve the lesson using the latest evaluation and log what changed."""
    lesson = LessonContent.model_validate(state["lesson"])
    evaluation = EvaluationResult.model_validate(state["evaluation"])
    try:
        improved = await improve_lesson(lesson, evaluation)
    except AGENT_ERRORS as exc:
        return _error_update("improve", exc)

    new_lesson = improved.model_dump()
    log = [dict(record) for record in state["rejection_log"]]
    log[-1]["changes"] = summarize_changes(state["lesson"], new_lesson)
    return {
        "lesson": new_lesson,
        "improvement_attempts": state["improvement_attempts"] + 1,
        "rejection_log": log,
    }
