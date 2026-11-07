"""Replays of REAL model output (gpt-4o-mini, live runs on 2026-11-07).

Only the LLM call is mocked; everything else is the real evaluation code.

* ``real_run_rag.json``: the lesson genuinely teaches all eight concepts. The
  evaluator judged every point covered but quoted partial passages, used concept ids
  on its first response and made vague, quote-less absence claims. An earlier
  version rejected the lesson because of how the evaluator quoted, not because of
  the lesson.
* ``real_run_rag_run3.json``: the confirmation run, which ended in an
  ``EvidenceError`` because the evaluator copied requirement text as its quotes and
  misread "does not guarantee". Lessons 2 and 3 really do say "real-time
  information", which is a genuine accuracy problem.
"""

import asyncio
import json
import pathlib
from unittest.mock import AsyncMock

import pytest
from app.agents import evaluator
from app.agents.evaluator import evaluate_lesson
from app.evaluation.lesson_text import body_text
from app.evaluation.schemas import LessonContent
from app.evaluation.topics import RAG_REQUIREMENTS

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
RUN_A = json.loads((FIXTURES / "real_run_rag.json").read_text(encoding="utf-8"))
RUN_C = json.loads((FIXTURES / "real_run_rag_run3.json").read_text(encoding="utf-8"))
LESSON_A = LessonContent.model_validate(RUN_A["lesson"])


def replay(monkeypatch, lesson, *raw_responses):
    mock = AsyncMock(side_effect=list(raw_responses))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))
    return result, mock


# --- run A ----------------------------------------------------------------------------


def test_the_recorded_lesson_really_states_every_teaching_point():
    """Guard the premise: the lesson is genuinely complete."""
    body = body_text(LESSON_A)
    for phrase in (
        "RAG stands for Retrieval-Augmented Generation",
        "language model reads the question and the context",
        "It then writes an answer using both pieces of information",
        "retrieved information is added to the user's question as context",
        "RAG does not guarantee",
        "First, the user asks a question",
        "Finally, the model writes the answer",
    ):
        assert phrase in body, phrase


@pytest.mark.parametrize("which", ["evaluator_attempt_1", "evaluator_attempt_2"])
def test_either_real_evaluator_response_gives_a_passing_evaluation(monkeypatch, which):
    """Even the first response (concept ids, requirement text as quotes) works."""
    result, mock = replay(monkeypatch, LESSON_A, RUN_A[which])
    assert mock.await_count == 1  # no re-ask: code decides the points
    assert result.overall_pass is True
    assert [c.passed for c in result.checks] == [True] * 7
    assert [c.status for c in result.concept_checks] == ["covered"] * 8


def test_point_evidence_is_real_lesson_text_found_by_code(monkeypatch):
    result, _ = replay(monkeypatch, LESSON_A, RUN_A["evaluator_attempt_1"])
    body = body_text(LESSON_A)
    evidence = {p.point_id: p.evidence_quote for p in result.point_checks}
    assert (
        "RAG stands for Retrieval-Augmented Generation" in evidence["definition.name"]
    )
    assert "It then writes an answer" in evidence["generation.writes_from_both"]
    assert "First, the user asks a question" in evidence["flow.sequence"]
    for point_id, passage in evidence.items():
        assert passage in body, point_id  # never text the model supplied
    # the requirement text the model copied is not what was recorded
    assert not any(
        "combines looking up information with a language model" in e
        for e in evidence.values()
    )


def test_every_claim_the_code_overrode_is_recorded_in_the_notes(monkeypatch):
    result, _ = replay(monkeypatch, LESSON_A, RUN_A["evaluator_attempt_2"])
    notes = " ".join(result.notes)
    assert "Resolved unexplained-term issue 'LLM'" in notes  # "also known as"
    assert "Resolved accuracy issue" in notes  # "does not guarantee" is correct
    assert "'Covers key points' was marked passed" in notes
    assert "'Overall lesson completeness' was marked passed" in notes
    assert "recomputed" in notes


def test_the_overrides_do_not_apply_when_the_lesson_really_lacks_the_statements(
    monkeypatch,
):
    data = json.loads(json.dumps(RUN_A["lesson"]))
    for section in data["sections"]:
        if section["heading"] in ("Generation", "End-to-End Workflow"):
            section["explanation"] = "This step matters."
            section["example"] = "An illustration."
    data["key_takeaways"] = ["RAG is useful.", "It has parts.", "It has limits."]
    data["summary"] = "RAG is a topic with several parts."
    lesson = LessonContent.model_validate(data)

    result, _ = replay(monkeypatch, lesson, RUN_A["evaluator_attempt_2"])

    statuses = {c.concept_id: c.status for c in result.concept_checks}
    # (Generation can still be matched loosely by the language-model section; the
    # workflow and worked example, which are gone, are what code reliably catches.)
    assert statuses["end_to_end"] == "missing"
    assert result.overall_pass is False
    assert result.checks[6].passed is False  # completeness stays failed
    assert result.checks[4].passed is False  # so do the key points


# --- run C: the confirmation run that ended in an EvidenceError ---

LESSONS_C = [LessonContent.model_validate(raw) for raw in RUN_C["lessons"]]
RESPONSES_C = RUN_C["evaluator_responses"]


@pytest.mark.parametrize("index", range(6))
def test_no_recorded_response_can_end_the_run_in_an_error_any_more(monkeypatch, index):
    result, mock = replay(monkeypatch, LESSONS_C[index // 2], RESPONSES_C[index])
    assert mock.await_count == 1
    assert len(result.checks) == 7
    assert [c.status for c in result.concept_checks] == ["covered"] * 8


def test_the_first_lesson_now_passes(monkeypatch):
    for response in RESPONSES_C[:2]:
        result, _ = replay(monkeypatch, LESSONS_C[0], response)
        assert result.overall_pass is True
        assert any("already negates the claim" in n for n in result.notes)


@pytest.mark.parametrize("index", [2, 3, 4, 5])
def test_later_lessons_fail_only_on_the_genuine_real_time_overclaim(monkeypatch, index):
    result, _ = replay(monkeypatch, LESSONS_C[index // 2], RESPONSES_C[index])
    failed = [c.criterion for c in result.checks if not c.passed]
    assert failed == ["Accuracy and groundedness"]
    quotes = " ".join(i.quote for i in result.checks[0].issues)
    assert "real-time" in quotes
    assert "real-time" in body_text(LESSONS_C[index // 2])  # it is really there


def test_the_recorded_fixtures_are_well_formed():
    assert len(RUN_C["lessons"]) == 3
    assert len(RESPONSES_C) == 6
    for raw in RESPONSES_C:
        assert len(json.loads(raw)["checks"]) == 7
