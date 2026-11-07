"""SQLite memory: schema, saving finished runs, and retrieving them."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from app.agents.evaluator import _finish
from app.agents.generator import LessonGenerationError
from app.evaluation.schemas import LessonContent
from app.evaluation.topics import RAG_REQUIREMENTS
from app.memory import MemoryRepository
from app.memory.database import SCHEMA_VERSION, connect, init_schema, sqlite_path
from app.workflow import nodes
from app.workflow.graph import run_workflow

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
    monkeypatch.setattr(nodes, "generate_lesson", gen)
    monkeypatch.setattr(nodes, "evaluate_lesson", ev)
    monkeypatch.setattr(nodes, "improve_lesson", imp)
    return gen, ev, imp


def run(topic=TOPIC, max_attempts=2):
    return asyncio.run(run_workflow(topic, maximum_attempts=max_attempts))


def passing_state(agents):
    gen, ev, imp = agents
    gen.return_value = rag_lesson(omit=("generation",))
    imp.return_value = rag_lesson()
    ev.side_effect = [failed(), passed()]
    return run()


def rejected_state(agents):
    gen, ev, imp = agents
    gen.return_value = rag_lesson(omit=("generation",))
    imp.return_value = rag_lesson(omit=("generation",))
    ev.side_effect = [failed(), failed(), failed()]
    return run()


# --- database setup ---------------------------------------------------------------


def test_sqlite_path_parsing():
    assert sqlite_path("sqlite:///./lesson_memory.db") == "./lesson_memory.db"
    assert sqlite_path("sqlite:///:memory:") == ":memory:"
    assert sqlite_path("sqlite:////tmp/x.db") == "/tmp/x.db"
    for bad in ("postgresql://x/y", "sqlite://", "sqlite:///", "x.db"):
        with pytest.raises(ValueError, match="sqlite:///"):
            sqlite_path(bad)


def test_schema_is_created_and_idempotent(tmp_path):
    path = tmp_path / "nested" / "m.db"
    MemoryRepository(path)
    MemoryRepository(path)  # opening again must not fail or wipe anything
    with connect(path) as connection:
        tables = {
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"runs", "lessons", "evaluations", "rejection_log"} <= tables
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    connection.close()


def test_a_newer_schema_version_is_refused(tmp_path):
    path = tmp_path / "m.db"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(RuntimeError, match="newer"):
        MemoryRepository(path)


def test_from_settings_uses_database_url(tmp_path, monkeypatch):
    from app import config

    url = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    monkeypatch.setattr(
        config, "get_settings", lambda: config.Settings(database_url=url)
    )
    from app.memory import repository

    monkeypatch.setattr(repository, "get_settings", config.get_settings)
    repo = repository.MemoryRepository.from_settings()
    assert repo.count_runs() == 0
    assert (tmp_path / "s.db").exists()


def test_foreign_keys_are_enforced(tmp_path):
    path = tmp_path / "m.db"
    MemoryRepository(path)
    connection = connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO lessons (run_id, role, title, topic, content_json) "
            "VALUES (999, 'returned', 't', 't', '{}')"
        )
    connection.close()


# --- saving and reading back ------------------------------------------------------


def test_a_passed_run_round_trips(repo, agents):
    state = passing_state(agents)
    run_id = repo.save_run(state)
    stored = repo.get_run(run_id)

    assert stored is not None
    assert stored.status == "passed"
    assert stored.topic == TOPIC
    assert stored.evaluation_count == state["evaluation_count"] == 2
    assert stored.improvement_attempts == 1
    assert stored.lesson == state["lesson"]
    assert stored.evaluation == state["evaluation"]
    assert stored.evaluation["overall_pass"] is True
    assert stored.rejection_log == state["rejection_log"]
    assert len(stored.rejection_log) == 1


def test_a_rejected_run_keeps_best_and_latest_separately(repo, agents):
    state = rejected_state(agents)
    stored = repo.get_run(repo.save_run(state))

    assert stored.status == "rejected"
    assert stored.rejection_reason == state["rejection_reason"] != ""
    assert stored.lesson == state["lesson"]
    assert stored.evaluation["overall_pass"] is False
    assert stored.latest_lesson == state["latest_lesson"]
    assert stored.latest_evaluation == state["latest_evaluation"]
    assert stored.best_evaluation_number == state["best_evaluation_number"]


def test_an_error_run_is_stored_with_its_error(repo, agents):
    gen, _, _ = agents
    gen.side_effect = LessonGenerationError("boom")
    state = run()
    assert state["status"] == "error"
    stored = repo.get_run(repo.save_run(state))

    assert stored.status == "error"
    assert stored.error == state["error"]
    assert stored.lesson is None and stored.evaluation is None


def test_unfinished_runs_are_refused(repo, agents):
    state = dict(passing_state(agents), status="in_progress")
    with pytest.raises(ValueError, match="finished"):
        repo.save_run(state)
    assert repo.count_runs() == 0


def test_a_failed_save_leaves_nothing_behind(repo, agents):
    state = dict(passing_state(agents))
    state["rejection_log"] = [{"attempt_number": 1}]  # missing failed_criteria
    with pytest.raises(KeyError):
        repo.save_run(state)
    assert repo.count_runs() == 0
    with connect(repo.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 0
    connection.close()


def test_get_run_of_unknown_id_is_none(repo):
    assert repo.get_run(123) is None


def test_delete_run_cascades(repo, agents):
    run_id = repo.save_run(passing_state(agents))
    assert repo.delete_run(run_id) is True
    assert repo.delete_run(run_id) is False
    with connect(repo.path) as connection:
        for table in ("lessons", "evaluations", "rejection_log"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0
    connection.close()


def test_data_survives_a_new_repository_instance(tmp_path, agents):
    path = tmp_path / "m.db"
    run_id = MemoryRepository(path).save_run(passing_state(agents))
    assert MemoryRepository(path).get_run(run_id) is not None


# --- retrieval --------------------------------------------------------------------


def test_list_runs_filters_and_orders_newest_first(repo, agents):
    first = repo.save_run(passing_state(agents))
    second = repo.save_run(rejected_state(agents))

    assert [r.id for r in repo.list_runs()] == [second, first]
    assert [r.id for r in repo.list_runs(status="passed")] == [first]
    assert [r.id for r in repo.list_runs(topic="  introduction to rag ")] == [
        second,
        first,
    ]
    assert repo.list_runs(topic="Something else") == []
    assert len(repo.list_runs(limit=1)) == 1


def test_latest_passed_lesson_returns_newest_passing_lesson(repo, agents):
    assert repo.latest_passed_lesson(TOPIC) is None
    repo.save_run(rejected_state(agents))
    assert repo.latest_passed_lesson(TOPIC) is None  # rejected lessons never count

    passing_state_run = passing_state(agents)
    repo.save_run(passing_state_run)
    lesson = repo.latest_passed_lesson("introduction to rag")
    assert isinstance(lesson, LessonContent)
    assert lesson.model_dump() == passing_state_run["lesson"]
    assert repo.latest_passed_lesson("another topic") is None


def test_evaluations_for_topic_summaries(repo, agents):
    repo.save_run(rejected_state(agents))
    repo.save_run(passing_state(agents))

    everything = repo.evaluations_for_topic(TOPIC)
    only_passed = repo.evaluations_for_topic(TOPIC, only_passed=True)

    assert [e["overall_pass"] for e in everything].count(True) == 1
    assert [e["overall_pass"] for e in only_passed] == [True]
    summary = only_passed[0]
    assert summary["passed_checks"] == 7
    assert summary["covered_concepts"] == len(RAG_REQUIREMENTS.concepts)
    assert isinstance(summary["notes"], list)
    assert repo.evaluations_for_topic("other") == []


def test_in_memory_path_is_rejected_for_the_repository():
    with pytest.raises(ValueError, match="file"):
        MemoryRepository(":memory:")


def test_init_schema_can_run_on_an_open_connection():
    connection = connect(":memory:")
    init_schema(connection)
    init_schema(connection)
    connection.close()
