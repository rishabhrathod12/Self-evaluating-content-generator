"""Application entry point: run the workflow with persistent memory.

    python -m app.pipeline "Introduction to RAG"

One call does the whole memory-enabled flow:

1. open the SQLite memory and read concise feedback from earlier runs of the topic;
2. run the existing LangGraph workflow with that feedback in its state;
3. save the finished run (passed, rejected or error) to the memory.

Database problems never cost you a finished run: opening or reading the memory fails
*before* any model call (so nothing is spent), while a failure to save is reported in
``MemoryRunResult.save_error`` and the run's state is still returned.
"""

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from app.llm import close_client
from app.memory.context import build_memory_context
from app.memory.database import MemoryDatabaseError
from app.memory.evolution import propose_refinements
from app.memory.repository import MemoryRepository
from app.workflow.graph import run_workflow
from app.workflow.state import WorkflowState


@dataclass(frozen=True)
class MemoryRunResult:
    """The outcome of ``run_with_memory``."""

    state: WorkflowState
    run_id: int | None  # None if the run could not be saved
    memory_context: dict[str, Any]  # what was retrieved ({} = no history)
    save_error: str | None = None
    new_proposals: tuple[int, ...] = ()  # ids of new, unapproved prompt versions


async def run_with_memory(
    topic: str,
    maximum_attempts: int | None = None,
    repository: MemoryRepository | None = None,
) -> MemoryRunResult:
    """Retrieve history, run the workflow, and save the finished run.

    Args:
        topic: The lesson topic.
        maximum_attempts: Improvement attempts (clamped to 0..2, default MAX_RETRIES).
        repository: The memory to use; defaults to the ``DATABASE_URL`` setting.

    Raises:
        MemoryDatabaseError: The memory could not be opened or read (nothing ran).
    """
    memory = repository or MemoryRepository.from_settings()
    context = build_memory_context(memory, topic)
    state = await run_workflow(topic, maximum_attempts, memory_context=context)
    try:
        run_id = memory.save_run(state)
    except MemoryDatabaseError as exc:
        return MemoryRunResult(state, None, context, save_error=str(exc))
    # Proposals only: they have no effect until a person approves them. A failure
    # here must not hide that the run itself was saved.
    try:
        proposals = tuple(v.id for v in propose_refinements(memory, topic))
    except MemoryDatabaseError:
        proposals = ()
    return MemoryRunResult(state, run_id, context, new_proposals=proposals)


async def _main(topic: str) -> int:
    try:
        result = await run_with_memory(topic)
    except MemoryDatabaseError as exc:
        print(f"Memory error (no model call was made): {exc}", file=sys.stderr)
        return 2
    finally:
        await close_client()
    state = result.state
    used = result.memory_context.get("source_run_ids", [])
    print(
        f"status: {state['status']}  evaluations: {state['evaluation_count']}  "
        f"improvements: {state['improvement_attempts']}"
    )
    print(f"history used from runs: {used or 'none'}")
    if result.save_error:
        print(f"NOT SAVED: {result.save_error}", file=sys.stderr)
    else:
        print(f"saved as run #{result.run_id}")
    if result.new_proposals:
        print(
            f"new prompt proposals (unapproved, not used): {list(result.new_proposals)}"
            " - review with: python -m app.evolve list"
        )
    if state["rejection_reason"]:
        print("rejected because:", state["rejection_reason"])
    if state["error"]:
        print("error:", state["error"])
    lesson = state["lesson"]
    if lesson:
        print(f"\n# {lesson['title']}\n")
        for number, section in enumerate(lesson["sections"], 1):
            print(f"## {number}. {section['heading']}\n{section['explanation']}\n")
    return 0 if state["status"] == "passed" and not result.save_error else 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # lessons may contain non-ASCII text
    topic = " ".join(sys.argv[1:]).strip() or "Introduction to RAG"
    raise SystemExit(asyncio.run(_main(topic)))


if __name__ == "__main__":
    main()
