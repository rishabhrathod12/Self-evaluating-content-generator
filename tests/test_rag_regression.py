"""Offline end-to-end scenarios: the REAL generator, evaluator, improver and workflow.

Only the LLM text is scripted (one mock per agent). These reproduce the failure
modes found in real runs and prove the pieces work together.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import evaluator, generator, improver
from app.workflow.graph import run_workflow

from tests.rag_fixtures import (
    NEUTRAL_SUMMARY,
    NEUTRAL_TAKEAWAYS,
    concept_points,
    eval_json,
    lacking_variant,
    lesson_lacking,
    rag_lesson,
)

GEN = concept_points("generation")
VAGUE = ("Overview", "The parts of RAG work together.", "X")


def lesson_json(lesson) -> str:
    return json.dumps(lesson.model_dump())


@pytest.fixture
def llms(monkeypatch):
    """One scripted LLM per agent."""
    gen, ev, imp = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(generator, "generate_text", gen)
    monkeypatch.setattr(evaluator, "generate_text", ev)
    monkeypatch.setattr(improver, "generate_text", imp)
    return gen, ev, imp


def run(max_attempts=2):
    return asyncio.run(
        run_workflow("Introduction to RAG", maximum_attempts=max_attempts)
    )


def test_a_complete_lesson_in_different_wording_passes_in_one_evaluation(llms):
    """The failure of the last real run: an adequate lesson must simply pass."""
    gen, ev, imp = llms
    gen.return_value = lesson_json(rag_lesson())
    ev.return_value = eval_json()

    state = run()

    assert state["status"] == "passed"
    assert state["evaluation_count"] == 1
    assert state["improvement_attempts"] == 0
    assert ev.await_count == 1  # no confirmation re-ask that could flip verdicts
    assert [c["status"] for c in state["evaluation"]["concept_checks"]] == [
        "covered"
    ] * 8
    imp.assert_not_awaited()


def test_the_generator_receives_the_curriculum_and_the_evaluator_the_points(llms):
    gen, ev, _ = llms
    gen.return_value = lesson_json(rag_lesson())
    ev.return_value = eval_json()
    run()
    generator_prompt = gen.await_args.args[0]
    assert "reads the question together with the retrieved context" in generator_prompt
    assert "1. The user asks a question." in generator_prompt
    assert "single user question" in generator_prompt
    evaluator_prompt = ev.await_args.args[0]
    assert "generation.writes_from_both" in evaluator_prompt
    assert ev.await_args.kwargs["temperature"] == 0


def test_missing_generation_is_found_fixed_and_the_fix_is_confirmed(llms):
    gen, ev, imp = llms
    gen.return_value = lesson_json(lesson_lacking("generation"))
    ev.side_effect = [eval_json(), eval_json()]  # the model's verdicts do not decide
    imp.return_value = lesson_json(rag_lesson())

    state = run()

    assert state["status"] == "passed"
    assert state["improvement_attempts"] == 1
    assert state["evaluation_count"] == 2
    first = state["rejection_log"][0]
    assert "generation" in first["missing_concepts"]
    assert first["improvement_unaddressed"] == []
    assert "generation" in first["concepts_resolved_next"]
    assert first["concepts_open_next"] == []
    assert "sections" in first["change_summary"]
    # the improver was told exactly what to teach
    improver_prompt = imp.await_args.args[0]
    assert "[generation] Generation: missing." in improver_prompt
    assert "reads the question together with the retrieved context" in improver_prompt
    # the revised lesson, in full, is what was evaluated next
    assert len(state["lesson"]["sections"]) == 8


def test_missing_augmentation_is_found_and_reported_by_name(llms):
    gen, ev, imp = llms
    gen.return_value = lesson_json(lesson_lacking("augmentation"))
    ev.return_value = eval_json()
    imp.return_value = lesson_json(lacking_variant(1, "augmentation"))
    state = run(max_attempts=1)
    assert state["status"] == "rejected"
    assert "Concept 'Augmentation' is missing" in state["rejection_reason"]
    assert "augmentation" in state["rejection_log"][0]["missing_concepts"]


def test_the_improver_returning_the_lesson_unchanged_ends_in_a_clear_error(llms):
    gen, ev, imp = llms
    original = lesson_lacking("generation")
    gen.return_value = lesson_json(original)
    ev.return_value = eval_json()
    imp.return_value = lesson_json(original)  # "improved" to the same text

    state = run()

    assert imp.await_count == 2  # one re-ask, then the error
    assert state["status"] == "error"
    assert state["error"]["node"] == "improve"
    assert state["error"]["error_type"] == "ImprovementValidationError"
    assert "unchanged" in state["error"]["message"]
    # the best evaluated pair is still returned together
    assert state["lesson"] == state["best_lesson"]
    assert state["evaluation"] == state["best_evaluation"]
    assert state["evaluation"]["overall_pass"] is False


def test_a_keyword_only_revision_is_recorded_as_not_stating_the_point(llms):
    gen, ev, imp = llms
    original = lesson_lacking("generation")
    gen.return_value = lesson_json(original)
    ev.return_value = eval_json()
    keyword_only = rag_lesson(
        replace={
            "generation": ("Generation", "Generation is an important step.", "X"),
            "end_to_end": ("Overview", "The parts of RAG work together.", "X"),
        },
        takeaways=NEUTRAL_TAKEAWAYS,
        summary=NEUTRAL_SUMMARY,
    )
    imp.return_value = lesson_json(keyword_only)

    state = run(max_attempts=1)

    entry = state["rejection_log"][0]
    assert imp.await_count == 2  # the improver was asked to try again once
    assert "generation.writes_from_both" in entry["improvement_unaddressed"]
    assert "did NOT state" in entry["change_summary"]
    assert "generation" in entry["concepts_open_next"]
    assert state["status"] == "rejected"
    assert "Generation" in state["rejection_reason"]


def test_an_evaluator_claiming_coverage_for_a_keyword_only_lesson_cannot_pass(llms):
    gen, ev, imp = llms
    lesson = rag_lesson(
        replace={
            "generation": ("Generation", "Generation is an important step.", "X"),
            "end_to_end": ("Overview", "The parts of RAG work together.", "X"),
        },
        takeaways=NEUTRAL_TAKEAWAYS,
        summary=NEUTRAL_SUMMARY,
    )
    gen.return_value = lesson_json(lesson)
    ev.return_value = eval_json()  # the model claims every point is covered
    imp.return_value = lesson_json(
        rag_lesson(
            replace={
                "generation": ("Generation", "Generation is an important step.", "X"),
                "end_to_end": ("Overview", "The parts of RAG work together.", "X"),
                "limitations": ("L", "Changed text.", "X"),
            },
            takeaways=NEUTRAL_TAKEAWAYS,
            summary=NEUTRAL_SUMMARY,
        )
    )

    state = run(max_attempts=1)

    assert ev.await_count == 2  # one per lesson: no re-asks about claims
    assert "generation" in state["rejection_log"][0]["missing_concepts"]
    assert any(
        "evaluator said covered, but no passage" in note
        for note in state["evaluation"]["notes"]
    )
    assert state["evaluation"]["overall_pass"] is False


def test_an_overclaim_is_a_genuine_failure_and_a_negated_sentence_is_not(llms):
    gen, ev, imp = llms
    good = "RAG does not guarantee the accuracy of the answers provided."
    bad = "RAG always learns from every conversation and improves itself."
    lesson = rag_lesson(
        replace={
            "limitations": (
                "Limits of RAG",
                "The retrieved information can be incomplete, outdated or "
                "incorrect. RAG does not guarantee that answers are accurate. "
                f"{good} {bad}",
                "If the policy page is old, the answer will be old too.",
            )
        }
    )
    gen.return_value = lesson_json(lesson)
    data = json.loads(eval_json(failing=(0,)))
    data["checks"][0]["issues"] = [
        {
            "problem": "Claims that RAG guarantees accuracy.",
            "quote": good,
            "term": "",
            "correction": "RAG does not ensure the accuracy of the answers provided.",
        },
        {
            "problem": "Says RAG learns.",
            "quote": bad,
            "term": "",
            "correction": "RAG does not learn; it only supplies information.",
        },
    ]
    ev.return_value = json.dumps(data)
    imp.return_value = lesson_json(
        rag_lesson(
            replace={
                "limitations": (
                    "Limits of RAG",
                    "The retrieved information can be incomplete, outdated or "
                    "incorrect. RAG does not guarantee that answers are accurate. "
                    f"{good}",
                    "If the policy page is old, the answer will be old too.",
                )
            }
        )
    )

    state = run(max_attempts=1)

    first_issues = state["rejection_log"][0]["problem_sentences"]
    assert first_issues == [bad]  # the false positive was resolved by code
    assert (
        "Correction: RAG does not learn"
        in state["rejection_log"][0]["improvement_feedback"]
    )


def test_best_lesson_and_evaluation_stay_paired_through_a_rejected_run(llms):
    gen, ev, imp = llms
    gen.return_value = lesson_json(lesson_lacking("generation"))
    ev.return_value = eval_json()
    # Each revision loses more concepts, so evaluation #1 stays the best.
    imp.side_effect = [
        lesson_json(lacking_variant(1, "generation", "why_needed")),
        lesson_json(lacking_variant(2, "generation", "why_needed")),
        lesson_json(lacking_variant(3, "generation", "why_needed", "limitations")),
        lesson_json(lacking_variant(4, "generation", "why_needed", "limitations")),
    ]

    state = run()

    assert state["status"] == "rejected"
    assert state["evaluation_count"] == 3
    assert state["best_evaluation_number"] == 1
    assert state["lesson"] == state["best_lesson"]
    assert state["evaluation"] == state["best_evaluation"]
    assert state["latest_lesson"] != state["lesson"]
    assert state["latest_evaluation"] != state["evaluation"]
    assert "evaluation #1 scored best" in state["rejection_reason"]


def test_an_evaluator_failure_returns_the_best_pair_not_a_half_evaluated_lesson(llms):
    gen, ev, imp = llms
    gen.return_value = lesson_json(lesson_lacking("generation"))
    ev.side_effect = [eval_json(), "not json", "still not json"]
    imp.return_value = lesson_json(rag_lesson())

    state = run()

    assert state["status"] == "error"
    assert state["error"]["node"] == "evaluate"
    assert state["lesson"] == state["best_lesson"]
    assert state["evaluation"]["overall_pass"] is False
    assert state["latest_lesson"] is not None  # the improved, unevaluated lesson
    assert state["latest_evaluation"] is None
    assert state["status"] != "passed"


def test_the_whole_run_never_exceeds_two_improvements_or_loops(llms):
    gen, ev, imp = llms
    gen.return_value = lesson_json(lesson_lacking("generation"))
    ev.return_value = eval_json()
    imp.side_effect = [
        lesson_json(lacking_variant(i, "generation")) for i in range(1, 20)
    ]
    state = run(max_attempts=2)
    assert state["improvement_attempts"] == 2
    assert state["evaluation_count"] == 3
    assert imp.await_count <= 4  # two improvements, at most one re-ask each
    assert ev.await_count == 3
    assert state["status"] == "rejected"
