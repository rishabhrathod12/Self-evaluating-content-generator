"""Turn stored history into concise feedback for the generator and improver.

The context is a small JSON-serializable dict kept in the workflow state:

    {"feedback": ["...", ...], "reference_lesson": {...} | None,
     "source_run_ids": [3, 2]}

It is advice, never a template: the prompts tell the agents to use it only where it
is relevant, and the curriculum, rubric and the current evaluation always take
priority. An empty dict means "no history", and the agents then behave as before.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.memory.repository import MemoryRepository

MAX_RUNS = 5  # newest stored runs of the topic that are looked at
MAX_FEEDBACK_ITEMS = 6
MAX_ITEM_CHARS = 350
MAX_REFERENCE_SECTIONS = 12


def _clip(text: str, limit: int = MAX_ITEM_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _run_feedback(run: Any) -> list[str]:
    """Concise feedback lines from one stored run (newest information first)."""
    lines: list[str] = []
    if run.status == "rejected" and run.rejection_reason:
        lines.append(f"A run was rejected: {_clip(run.rejection_reason)}")
    for record in reversed(run.rejection_log):
        criteria = ", ".join(record.get("failed_criteria", []))
        advice = _clip(record.get("improvement_feedback", ""))
        if criteria and advice:
            lines.append(f"A draft failed {criteria}. Evaluator feedback: {advice}")
    return lines


def build_memory_context(repository: "MemoryRepository", topic: str) -> dict[str, Any]:
    """Collect feedback and a reference outline for ``topic`` from stored runs.

    Returns ``{}`` when nothing relevant is stored.
    """
    runs = repository.list_runs(topic=topic, limit=MAX_RUNS)
    feedback: list[str] = []
    used: list[int] = []
    for run in runs:
        for line in _run_feedback(run):
            if line not in feedback and len(feedback) < MAX_FEEDBACK_ITEMS:
                feedback.append(line)
                if run.id not in used:
                    used.append(run.id)
    lesson = repository.latest_passed_lesson(topic)
    reference = None
    if lesson is not None:
        reference = {
            "title": lesson.title,
            "section_headings": [s.heading for s in lesson.sections][
                :MAX_REFERENCE_SECTIONS
            ],
            "key_takeaways": [_clip(t, 200) for t in lesson.key_takeaways[:3]],
        }
    if not feedback and reference is None:
        return {}
    return {
        "feedback": feedback,
        "reference_lesson": reference,
        "source_run_ids": used,
    }


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def generator_memory_block(memory: Mapping[str, Any] | None) -> str:
    """Prompt text for the generator; empty when there is no history."""
    if not memory:
        return ""
    parts = [
        "Notes from earlier runs on this topic. Use them only where they are "
        "relevant; the requirements above always come first, and do not copy any "
        "earlier lesson, write fresh explanations and examples:\n"
    ]
    if memory.get("feedback"):
        parts.append(
            "Problems earlier drafts were criticised for (avoid repeating them):\n"
            f"{_bullets(memory['feedback'])}\n"
        )
    reference = memory.get("reference_lesson")
    if reference:
        parts.append(
            "An earlier lesson that passed evaluation was titled "
            f'"{reference["title"]}" with sections: '
            f"{'; '.join(reference['section_headings'])}. "
            "Treat it only as a loose reference for coverage and ordering.\n"
        )
    return "".join(parts) + "\n"


def improver_memory_block(memory: Mapping[str, Any] | None) -> str:
    """Prompt text for the improver; empty when there is no feedback."""
    if not memory or not memory.get("feedback"):
        return ""
    return (
        "Feedback from earlier runs on this topic. Consider it only where it is "
        "relevant to the failed checkpoints above; the current evaluation takes "
        "priority:\n"
        f"{_bullets(memory['feedback'])}\n\n"
    )
