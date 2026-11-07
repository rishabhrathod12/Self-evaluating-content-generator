"""End-to-end memory flow: retrieve history, run the workflow, save the run."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from app import pipeline
from app.agents import generator, improver
from app.agents.evaluator import _finish
from app.agents.generator import LessonGenerationError
from app.evaluation.schemas import EvaluationResult, LessonContent
from app.evaluation.topics import RAG_REQUIREMENTS
from app.memory import MemoryDatabaseError, MemoryRepository
from app.memory.context import (
    MAX_FEEDBACK_ITEMS,
    MAX_ITEM_CHARS,
    build_memory_context,
    generator_memory_block,
    improver_memory_block,
)
from app.pipeline import run_with_memory
from app.workflow import nodes
from app.workflow.state import MAX_IMPROVEMENT_ATTEMPTS, initial_state

from tests.rag_fixtures import as_result, concept_points, eval_data, rag_lesson

TOPIC = "Introduction to RAG"
GEN = concept_points("generation")


def failed():
    return _finish(
        as_result(eval_data(uncovered=GEN)),
        rag_lesson(omit=("generation",)),
        RAG_REQUIREMENTS,
    )


def passed():
    return _finish(as_result(eval_data()), rag_lesson(), RAG_REQUIREMENTS)


@pytest.fixture
def repo(tmp_path):
    return MemoryRepository(tmp_path / "memory.db")


@pytest.fixture
def agents(monkeypatch):
    gen, ev, imp = AsyncMock(), AsyncMock(), AsyncMock()
    gen.return_value = rag_lesson(omit=("generation",))
    imp.return_value = rag_lesson(omit=("generation",))
    monkeypatch.setattr(nodes, "generate_lesson", gen)
    monkeypatch.setattr(nodes, "evaluate_lesson", ev)
    monkeypatch.setattr(nodes, "improve_lesson", imp)
    return gen, ev, imp


def go(repo, max_attempts=2, topic=TOPIC):
    return asyncio.run(run_with_memory(topic, max_attempts, repository=repo))


# --- the orchestration entry point -------------------------------------------------


def test_the_first_run_is_saved_automatically(repo, agents):
    gen, ev, imp = agents
    ev.side_effect = [failed(), failed(), failed()]

    result = go(repo)

    assert result.run_id == 1 and result.save_error is None
    assert result.memory_context == {}
    stored = repo.get_run(1)
    assert stored.status == "rejected" == result.state["status"]
    assert stored.lesson == result.state["lesson"]
    # With no history nothing extra is passed to the agents.
    gen.assert_awaited_once_with(TOPIC)
    for call in imp.await_args_list:
        assert "memory" not in call.kwargs


def test_the_second_run_retrieves_the_first_runs_feedback(repo, agents):
    gen, ev, imp = agents
    ev.side_effect = [failed(), failed(), failed()]
    first = go(repo)
    first_feedback = first.state["rejection_log"][0]["improvement_feedback"]

    gen.reset_mock()
    imp.reset_mock()
    ev.side_effect = [failed(), passed()]
    second = go(repo)

    context = second.memory_context
    assert context["source_run_ids"] == [1]
    assert any(first.state["rejection_reason"][:40] in f for f in context["feedback"])
    assert any(first_feedback[:40] in f for f in context["feedback"])
    assert second.state["memory_context"] == context
    assert second.run_id == 2 and repo.count_runs() == 2
    assert second.state["status"] == "passed"


def test_retrieved_feedback_reaches_the_generator_and_improver(repo, agents):
    gen, ev, imp = agents
    ev.side_effect = [failed(), failed(), failed()]
    go(repo)

    gen.reset_mock()
    imp.reset_mock()
    ev.side_effect = [failed(), passed()]
    second = go(repo)

    assert gen.await_args.kwargs["memory"] == second.memory_context
    assert imp.await_args.kwargs["memory"] == second.memory_context
    assert second.memory_context["feedback"]


def test_history_survives_reopening_the_database(tmp_path, agents):
    _, ev, _ = agents
    path = tmp_path / "reopen.db"
    ev.side_effect = [failed(), failed(), failed()]
    go(MemoryRepository(path))

    reopened = MemoryRepository(path)  # a brand-new repository and connections
    context = build_memory_context(reopened, TOPIC)
    assert context["source_run_ids"] == [1]
    assert reopened.get_run(1).status == "rejected"

    ev.side_effect = [passed()]
    result = go(reopened)
    assert result.memory_context["source_run_ids"] == [1]
    assert MemoryRepository(path).count_runs() == 2


def test_a_passed_lesson_becomes_a_reference_outline(repo, agents):
    gen, ev, _ = agents
    ev.side_effect = [passed()]
    gen.return_value = rag_lesson()
    go(repo)

    context = build_memory_context(repo, TOPIC)
    reference = context["reference_lesson"]
    assert reference["title"] == rag_lesson().title
    assert reference["section_headings"] == [s.heading for s in rag_lesson().sections]
    # Only an outline is passed on, never the lesson text.
    assert "explanation" not in str(reference)


def test_error_runs_are_saved_too(repo, agents):
    gen, _, _ = agents
    gen.side_effect = LessonGenerationError("bad json")

    result = go(repo)

    assert result.state["status"] == "error"
    assert repo.get_run(result.run_id).error["message"] == "bad json"


def test_other_topics_do_not_leak_history(repo, agents):
    _, ev, _ = agents
    ev.side_effect = [failed(), failed(), failed()]
    go(repo)
    ev.side_effect = [passed()]
    assert go(repo, topic="Photosynthesis").memory_context == {}


def test_the_retry_limit_is_unchanged_with_history(repo, agents):
    _, ev, imp = agents
    ev.side_effect = [failed()] * 3
    go(repo)
    imp.reset_mock()
    ev.side_effect = [failed()] * 3
    result = go(repo, max_attempts=99)
    assert result.state["maximum_attempts"] == MAX_IMPROVEMENT_ATTEMPTS == 2
    assert result.state["improvement_attempts"] == 2
    assert imp.await_count == 2


# --- database errors ---------------------------------------------------------------


def test_a_save_failure_returns_the_run_with_a_clear_message(repo, agents, monkeypatch):
    _, ev, _ = agents
    ev.side_effect = [passed()]

    def boom(state):
        raise MemoryDatabaseError("disk full")

    monkeypatch.setattr(repo, "save_run", boom)
    result = go(repo)

    assert result.run_id is None and result.save_error == "disk full"
    assert result.state["status"] == "passed"  # the finished run is not lost


def test_an_unusable_database_fails_before_any_model_call(monkeypatch):
    workflow = AsyncMock()
    monkeypatch.setattr(pipeline, "run_workflow", workflow)

    def broken():
        raise MemoryDatabaseError("unable to open database file")

    monkeypatch.setattr(pipeline.MemoryRepository, "from_settings", broken)
    with pytest.raises(MemoryDatabaseError, match="unable to open"):
        asyncio.run(run_with_memory(TOPIC))
    workflow.assert_not_awaited()


def test_sqlite_errors_become_memory_database_errors(tmp_path):
    with pytest.raises(MemoryDatabaseError, match="failed"):
        MemoryRepository(tmp_path)  # a directory is not a database file


def test_a_failure_while_reading_history_is_reported(repo, monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryDatabaseError("locked")

    monkeypatch.setattr(repo, "list_runs", boom)
    with pytest.raises(MemoryDatabaseError):
        asyncio.run(run_with_memory(TOPIC, repository=repo))


# --- secrets -----------------------------------------------------------------------


def test_credentials_in_error_messages_are_not_stored(repo, agents):
    gen, _, _ = agents
    gen.side_effect = LessonGenerationError(
        "Incorrect API key provided: sk-abcdefgh1234567890XYZ"
    )
    result = go(repo)

    stored = repo.get_run(result.run_id)
    assert "sk-abcdefgh1234567890XYZ" not in stored.error["message"]
    with open(repo.path, "rb") as handle:
        assert b"sk-abcdefgh1234567890XYZ" not in handle.read()


# --- prompts and context -----------------------------------------------------------

MEMORY = {
    "feedback": ["A draft failed Clear explanations. Evaluator feedback: Define RAG."],
    "reference_lesson": {
        "title": "RAG for Beginners",
        "section_headings": ["What is RAG", "Why it matters"],
        "key_takeaways": ["RAG adds documents."],
    },
    "source_run_ids": [4],
}


def test_without_history_prompts_are_unchanged():
    plain = generator.build_prompt(TOPIC, RAG_REQUIREMENTS)
    assert generator.build_prompt(TOPIC, RAG_REQUIREMENTS, None) == plain
    assert generator.build_prompt(TOPIC, RAG_REQUIREMENTS, {}) == plain
    assert generator_memory_block({}) == "" == improver_memory_block({})
    assert initial_state(TOPIC, 2)["memory_context"] == {}


def test_generator_prompt_uses_history_as_advice_not_a_template():
    prompt = generator.build_prompt(TOPIC, RAG_REQUIREMENTS, MEMORY)
    assert "Define RAG." in prompt
    assert "RAG for Beginners" in prompt and "What is RAG" in prompt
    assert "do not copy" in prompt and "only where they are relevant" in prompt
    assert "RAG adds documents." not in prompt  # takeaways are not passed on


def test_improver_prompt_includes_feedback_but_current_evaluation_wins():
    lesson = LessonContent.model_validate(rag_lesson(omit=("generation",)))
    evaluation = EvaluationResult.model_validate(failed().model_dump())
    plain = improver.build_prompt(lesson, evaluation, None, RAG_REQUIREMENTS)
    with_memory = improver.build_prompt(
        lesson, evaluation, None, RAG_REQUIREMENTS, MEMORY
    )
    assert with_memory != plain
    assert "Define RAG." in with_memory
    assert "current evaluation takes priority" in with_memory
    assert "RAG for Beginners" not in with_memory  # the improver gets feedback only


def test_context_is_concise(repo, agents):
    _, ev, _ = agents
    for _ in range(4):
        ev.side_effect = [failed(), failed(), failed()]
        go(repo)
    context = build_memory_context(repo, TOPIC)
    assert 0 < len(context["feedback"]) <= MAX_FEEDBACK_ITEMS
    assert all(len(line) < MAX_ITEM_CHARS + 120 for line in context["feedback"])
    assert len(set(context["feedback"])) == len(context["feedback"])
