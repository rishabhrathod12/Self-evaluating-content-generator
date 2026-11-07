"""The rejection log and rejection reason: exactly what happened and why."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents.evaluator import _finish
from app.evaluation.topics import RAG_REQUIREMENTS
from app.workflow import nodes
from app.workflow.graph import run_workflow

from tests.rag_fixtures import as_result, concept_points, eval_data, rag_lesson

GEN = concept_points("generation")
VAGUE = ("Overview", "The parts of RAG work together.", "X")


def failed(uncovered=()):
    """A realistic failed evaluation with derived concept statuses."""
    return _finish(
        as_result(eval_data(uncovered=uncovered)), rag_lesson(), RAG_REQUIREMENTS
    )


@pytest.fixture
def agents(monkeypatch):
    gen = AsyncMock(return_value=rag_lesson(omit=("generation",)))
    ev = AsyncMock()
    imp = AsyncMock()
    monkeypatch.setattr(nodes, "generate_lesson", gen)
    monkeypatch.setattr(nodes, "evaluate_lesson", ev)
    monkeypatch.setattr(nodes, "improve_lesson", imp)
    return gen, ev, imp


def run(max_attempts: int = 2) -> dict:
    return asyncio.run(
        run_workflow("Introduction to RAG", maximum_attempts=max_attempts)
    )


def improved_lesson():
    return rag_lesson(
        omit=("generation",), replace={"limitations": ("Limits", "New text here.", "X")}
    )


# --- the empty "changes" on the final entry is by design, and now says so ---


def test_the_final_entry_explains_why_it_has_no_changes(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), failed(GEN)]
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)

    first, last = state["rejection_log"]
    assert first["another_attempt_made"] is True
    assert first["changes"] != []
    assert first["change_summary"].startswith("Improvement attempt 1 changed:")
    assert last["another_attempt_made"] is False
    assert last["changes"] == []
    assert "No improvement followed this evaluation" in last["change_summary"]
    assert "limit was reached" in last["change_summary"]


def test_an_entry_waiting_for_its_improvement_says_so(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN)]
    imp.side_effect = RuntimeError("never reached")  # not used: attempts == 0
    state = run(max_attempts=0)
    (entry,) = state["rejection_log"]
    assert entry["another_attempt_made"] is False
    assert "No improvement followed" in entry["change_summary"]


def test_unaddressed_points_after_a_revision_are_recorded_explicitly(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), failed(GEN)]
    # The revision changes the lesson but never states the generation point.
    imp.return_value = rag_lesson(omit=("generation",), replace={"end_to_end": VAGUE})
    state = run(max_attempts=1)

    first = state["rejection_log"][0]
    assert first["improvement_unaddressed"] == ["generation.writes_from_both"]
    assert "did NOT state these requested points" in first["change_summary"]
    assert "generation.writes_from_both" in first["change_summary"]


def test_a_revision_that_states_the_point_records_nothing_unaddressed(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), eval_passing()]
    imp.return_value = rag_lesson()
    state = run()
    assert state["rejection_log"][0]["improvement_unaddressed"] == []
    assert state["status"] == "passed"


def eval_passing():
    return _finish(as_result(eval_data()), rag_lesson(), RAG_REQUIREMENTS)


# --- what the next evaluation says about the concepts that were open ---


def test_the_next_evaluation_marks_concepts_as_resolved_or_still_open(agents):
    _, ev, imp = agents
    ev.side_effect = [
        failed(("why.hallucination", *GEN)),  # why_needed weak, generation missing
        failed(GEN),  # why_needed fixed, generation still missing
    ]
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)

    first = state["rejection_log"][0]
    assert first["missing_concepts"] == ["generation"]
    assert first["weak_concepts"] == ["why_needed"]
    assert first["concepts_resolved_next"] == ["why_needed"]
    assert first["concepts_open_next"] == ["generation"]


def test_a_passing_evaluation_also_closes_the_previous_entry(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), eval_passing()]
    imp.return_value = rag_lesson()
    state = run()
    entry = state["rejection_log"][0]
    assert entry["concepts_resolved_next"] == ["generation"]
    assert entry["concepts_open_next"] == []
    assert state["rejection_reason"] == ""


# --- exactly why a run was rejected ---


def test_rejection_reason_names_the_failed_criterion_and_the_open_concept(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), failed(GEN)]
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)

    reason = state["rejection_reason"]
    assert state["status"] == "rejected"
    assert (
        "Rejected after 1 improvement attempt(s) and 2 evaluation(s) (limit 1)"
        in reason
    )
    assert "Failed criteria: Overall lesson completeness." in reason
    assert "Concept 'Generation' is missing" in reason
    assert "The lesson never states" in reason  # the evaluator's own words


def test_rejection_reason_says_when_an_earlier_lesson_is_returned(agents):
    _, ev, imp = agents
    ev.side_effect = [
        failed(("why.hallucination",)),  # weak concept: the better evaluation
        failed(("why.hallucination", *GEN)),  # worse
    ]
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)

    reason = state["rejection_reason"]
    assert "evaluation #1 scored best and is returned" in reason
    assert "latest_lesson" in reason
    assert "Why RAG is needed" in reason or "why_needed" in reason
    assert state["best_evaluation_number"] == 1
    assert state["lesson"] == state["best_lesson"]
    assert state["latest_lesson"] is not None


def test_a_weak_concept_is_reported_as_weak_not_missing(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(("why.hallucination",)), failed(("why.hallucination",))]
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)
    assert "is weak" in state["rejection_reason"]
    assert "is missing" not in state["rejection_reason"]


# --- convergence and safety ---


def test_every_improvement_is_followed_by_a_full_re_evaluation_of_the_revised_lesson(
    agents,
):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), failed(GEN), eval_passing()]
    versions = [
        rag_lesson(
            omit=("generation",), replace={"limitations": ("L", "Text one.", "X")}
        ),
        rag_lesson(),
    ]
    imp.side_effect = versions
    state = run()

    evaluated = [call.args[0] for call in ev.await_args_list]
    assert evaluated[1] == versions[0] and evaluated[2] == versions[1]
    assert all(
        call.kwargs["requirements"] is RAG_REQUIREMENTS for call in ev.await_args_list
    )
    assert len(state["evaluation"]["checks"]) == 7
    assert state["status"] == "passed"


def test_the_improvement_cap_is_two_even_if_configured_higher(agents):
    _, ev, imp = agents
    ev.return_value = failed(GEN)
    imp.side_effect = [
        rag_lesson(
            omit=("generation",), replace={"limitations": ("L", "Text one.", "X")}
        ),
        rag_lesson(
            omit=("generation",), replace={"limitations": ("L", "Text two.", "X")}
        ),
    ]
    state = run(max_attempts=99)
    assert state["improvement_attempts"] == 2
    assert state["evaluation_count"] == 3
    assert state["status"] == "rejected"
    assert imp.await_count == 2


def test_an_incomplete_lesson_is_never_reported_as_passed(agents):
    _, ev, imp = agents
    ev.return_value = failed(GEN)
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)
    assert state["status"] == "rejected"
    assert state["evaluation"]["overall_pass"] is False


def test_the_state_with_the_new_log_fields_is_serializable(agents):
    _, ev, imp = agents
    ev.side_effect = [failed(GEN), failed(GEN)]
    imp.return_value = improved_lesson()
    state = run(max_attempts=1)
    json.dumps(state)
    for key in ("rejection_reason", "best_lesson", "latest_lesson"):
        assert key in state
