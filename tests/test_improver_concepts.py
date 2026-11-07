"""The improver and missing teaching points: what it receives, what it must do."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import improver
from app.agents.improver import (
    ImprovementValidationError,
    improve_lesson,
    no_change_problems,
    unaddressed_messages,
)
from app.evaluation.concepts import requested_point_ids
from app.evaluation.schemas import LessonContent
from app.evaluation.topics import RAG_REQUIREMENTS

from tests.rag_fixtures import as_result, concept_points, eval_data, rag_lesson


def failed_evaluation(*, uncovered=(), lesson=None):
    """A realistic failed evaluation, built through the evaluator's own pipeline."""
    from app.agents.evaluator import _finish

    data = eval_data(uncovered=uncovered)
    return _finish(as_result(data), rag_lesson(), RAG_REQUIREMENTS)


def lesson_json(lesson: LessonContent) -> str:
    return json.dumps(lesson.model_dump())


@pytest.fixture
def llm(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(improver, "generate_text", mock)
    return mock


def improve(original, evaluation, requirements=RAG_REQUIREMENTS):
    return asyncio.run(improve_lesson(original, evaluation, requirements=requirements))


GEN = concept_points("generation")
VAGUE = ("Overview", "The parts of RAG work together.", "X")


# --- what the improver receives ---


def test_the_prompt_names_the_exact_missing_concept_and_point(llm):
    original = rag_lesson(omit=("generation",))
    evaluation = failed_evaluation(uncovered=GEN)
    llm.return_value = lesson_json(rag_lesson())
    improve(original, evaluation)

    prompt = llm.await_args.args[0]
    assert "Concepts that need work" in prompt
    assert "[generation] Generation: missing." in prompt
    assert "(generation.writes_from_both)" in prompt
    assert "reads the question together with the retrieved context" in prompt
    assert "The lesson never states" in prompt  # the evaluator's own note


def test_the_prompt_carries_failed_criteria_the_curriculum_and_the_example(llm):
    original = rag_lesson(omit=("generation",))
    llm.return_value = lesson_json(rag_lesson())
    improve(original, failed_evaluation(uncovered=GEN))
    prompt = llm.await_args.args[0]
    assert "Overall lesson completeness" in prompt
    assert "Failed checkpoints" in prompt
    assert "1. The user asks a question." in prompt
    assert "single user question" in prompt
    assert "public internet" in prompt


def test_weak_concepts_list_only_their_missing_points(llm):
    original = rag_lesson()
    evaluation = failed_evaluation(uncovered=("why.hallucination",))
    llm.return_value = lesson_json(rag_lesson())
    with pytest.raises(ImprovementValidationError):  # unchanged: not the point here
        improve(original, evaluation)
    prompt = llm.await_args_list[0].args[0]
    block = prompt.split("Concepts that need work")[1].split("Earlier rounds")[0]
    block = block.split("The finished lesson")[0]
    assert "[why_needed] Why RAG is needed: weak." in block
    assert "(why.hallucination)" in block
    assert "(why.knowledge_gap)" not in block


def test_instructions_require_stating_the_missing_points_explicitly():
    text = improver.SYSTEM_INSTRUCTIONS
    assert 'under "Concepts that need work"' in text
    assert "STATED explicitly" in text
    assert "renaming a heading" in text
    assert "Never add filler" in text


# --- fixing a missing concept ---


def test_the_improver_fixes_a_missing_generation_concept(llm):
    original = rag_lesson(omit=("generation",))
    evaluation = failed_evaluation(uncovered=GEN)
    fixed = rag_lesson()
    llm.return_value = lesson_json(fixed)

    result = improve(original, evaluation)

    assert len(result.sections) == len(original.sections) + 1
    assert any(
        "reads the question together with the retrieved context" in s.explanation
        for s in result.sections
    )
    assert llm.await_count == 1
    assert (
        unaddressed_messages(RAG_REQUIREMENTS, result, requested_point_ids(evaluation))
        == []
    )


def test_preservation_rules_do_not_block_a_required_correction(llm):
    """A missing concept may need any section changed, plus new objectives."""
    original = rag_lesson(omit=("generation",))
    evaluation = failed_evaluation(uncovered=GEN)
    fixed = rag_lesson(
        replace={
            "end_to_end": (
                "The whole workflow",
                SECTIONS_END_TO_END,
                "Suppose a user asks, 'What is our refund policy?' The system "
                "retrieves the policy page. It is added to the question as context. "
                "The language model writes the answer.",
            )
        },
        objectives=(
            "Understand what RAG is and why it is needed.",
            "Follow how a question becomes an answer in RAG.",
            "Explain how the model uses the context to write an answer.",
        ),
    )
    llm.return_value = lesson_json(fixed)
    result = improve(original, evaluation)
    assert llm.await_count == 1  # no preservation re-ask
    assert len(result.learning_objectives) == 3


SECTIONS_END_TO_END = (
    "Step by step: first the user asks a question, then relevant information is "
    "retrieved, then it is added as context, and finally the language model writes "
    "the answer. Each step feeds the next."
)


def test_a_flagged_objective_can_be_corrected_while_a_concept_is_missing(llm):
    wrong = "Understand how RAG trains the model on new data."
    original = rag_lesson(omit=("generation",), objectives=(wrong, "Know what RAG is."))
    data = eval_data(uncovered=GEN, failing=(0,))
    data["checks"][0]["issues"][0].update(
        quote=wrong, correction="RAG does not train the model."
    )
    from app.agents.evaluator import _finish

    evaluation = _finish(as_result(data), rag_lesson(), RAG_REQUIREMENTS)
    fixed = rag_lesson(
        objectives=("Understand how RAG supplies information.", "Know what RAG is.")
    )
    llm.return_value = lesson_json(fixed)
    result = improve(original, evaluation)
    assert result.learning_objectives[0] == "Understand how RAG supplies information."
    assert llm.await_count == 1


# --- unchanged content is never accepted ---


def test_unchanged_content_despite_a_required_correction_is_rejected(llm):
    original = rag_lesson(omit=("generation",))
    evaluation = failed_evaluation(uncovered=GEN)
    llm.return_value = lesson_json(original)  # the same lesson, again

    with pytest.raises(ImprovementValidationError, match="unchanged"):
        improve(original, evaluation)

    assert llm.await_count == 2  # one re-ask, then the error
    assert "returned the lesson unchanged" in llm.await_args_list[1].args[0]


def test_changing_only_title_and_summary_is_not_a_fix_for_a_missing_concept(llm):
    original = rag_lesson(omit=("generation",))
    evaluation = failed_evaluation(uncovered=GEN)
    cosmetic = original.model_copy(update={"title": "New title", "summary": "New."})
    llm.side_effect = [lesson_json(cosmetic), lesson_json(rag_lesson())]

    result = improve(original, evaluation)

    assert llm.await_count == 2
    retry = llm.await_args_list[1].args[0]
    assert "no section changed and no section was added" in retry
    assert len(result.sections) == len(original.sections) + 1


def test_no_change_problems_unit():
    lesson = rag_lesson()
    assert no_change_problems(lesson, lesson, ["x"])
    assert no_change_problems(lesson, lesson.model_copy(update={"title": "T"}), ["x"])
    assert (
        no_change_problems(lesson, lesson.model_copy(update={"title": "T"}), []) == []
    )
    other = rag_lesson(replace={"generation": ("G", "Different text here.", "X")})
    assert no_change_problems(lesson, other, ["x"]) == []


# --- the revision must really state the missing point ---


def test_a_revision_that_only_names_the_concept_is_questioned_then_returned_flagged(
    llm,
):
    original = rag_lesson(omit=("generation",), replace={"end_to_end": VAGUE})
    evaluation = failed_evaluation(uncovered=GEN)
    keyword_only = rag_lesson(
        replace={
            "generation": ("Generation", "Generation is an important step.", "X"),
            "end_to_end": VAGUE,
        }
    )
    llm.return_value = lesson_json(keyword_only)

    result = improve(original, evaluation)

    assert llm.await_count == 2  # one targeted re-ask
    retry = llm.await_args_list[1].args[0]
    assert "still has no passage that states" in retry
    assert "reads the question together with the retrieved context" in retry
    # Returned anyway, but the caller can see exactly what remains open.
    still_open = unaddressed_messages(
        RAG_REQUIREMENTS, result, requested_point_ids(evaluation)
    )
    assert still_open
    assert "Your revision still has no passage that states" in still_open[0]


def test_a_revision_that_states_the_point_in_different_words_is_accepted(llm):
    original = rag_lesson(omit=("generation",), replace={"end_to_end": VAGUE})
    evaluation = failed_evaluation(uncovered=GEN)
    different = rag_lesson(
        replace={
            "generation": (
                "How the answer is produced",
                "The model is shown the user's question and the retrieved passages "
                "side by side, and it composes a reply based on them.",
                "X",
            ),
            "end_to_end": VAGUE,
        }
    )
    llm.return_value = lesson_json(different)
    improve(original, evaluation)
    assert llm.await_count == 1


def test_no_point_check_without_requirements(llm):
    original = rag_lesson()
    evaluation = failed_evaluation(uncovered=("why.hallucination",))
    changed = rag_lesson(
        replace={"generation": ("G", "Another wording entirely.", "X")}
    )
    llm.return_value = lesson_json(changed)
    result = asyncio.run(improve_lesson(original, evaluation, requirements=None))
    assert result.sections != original.sections
    assert llm.await_count == 1


def test_blocking_and_unaddressed_problems_are_logged(llm, monkeypatch):
    captured = []
    monkeypatch.setattr(
        improver, "log_event", lambda name, **fields: captured.append((name, fields))
    )
    original = rag_lesson(omit=("generation",), replace={"end_to_end": VAGUE})
    evaluation = failed_evaluation(uncovered=GEN)
    llm.return_value = lesson_json(
        rag_lesson(
            replace={
                "generation": ("Generation", "Generation matters.", "X"),
                "end_to_end": VAGUE,
            }
        )
    )
    improve(original, evaluation)
    problems = [f for n, f in captured if n == "improver_problems"]
    assert problems and problems[0]["unaddressed"]
    assert problems[0]["blocking"] == []
