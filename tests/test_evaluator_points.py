"""Point-level concept evaluation: evidence, relevance, re-ask and accuracy."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import evaluator
from app.agents.evaluator import (
    evaluate_lesson,
    quote_states_correction,
)
from app.evaluation.schemas import CRITERIA, EvaluationIssue
from app.evaluation.topics import RAG_REQUIREMENTS

from tests.rag_fixtures import (
    EVIDENCE,
    NEUTRAL_SUMMARY,
    NEUTRAL_TAKEAWAYS,
    VAGUE,
    WEAK_WHY,
    concept_points,
    eval_data,
    lesson_lacking,
    rag_lesson,
)

DEFINITION_QUOTE = "RAG stands for Retrieval-Augmented Generation"


def run_eval(monkeypatch, lesson, *responses, requirements=RAG_REQUIREMENTS):
    mock = AsyncMock(side_effect=[json.dumps(r) for r in responses])
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(lesson, requirements=requirements))
    return result, mock


def concept(result, concept_id):
    return next(c for c in result.concept_checks if c.concept_id == concept_id)


# --- a complete lesson, in different wording, is accepted --------------------------


def test_complete_lesson_with_different_wording_passes_without_a_reask(monkeypatch):
    result, mock = run_eval(monkeypatch, rag_lesson(), eval_data())
    assert result.overall_pass is True
    assert mock.await_count == 1
    assert [c.status for c in result.concept_checks] == ["covered"] * 8
    assert result.notes == []
    assert result.improvement_feedback == ""


def test_the_evaluator_runs_at_temperature_zero_with_the_point_schema(monkeypatch):
    _, mock = run_eval(monkeypatch, rag_lesson(), eval_data())
    kwargs = mock.await_args.kwargs
    assert kwargs["temperature"] == 0
    schema = kwargs["json_schema"]
    assert "point_checks" in schema["properties"]
    assert "concept_checks" not in schema["properties"]  # derived by code
    assert "notes" not in schema["properties"]
    assert schema["required"] == list(schema["properties"])


def test_a_language_model_using_retrieved_context_to_answer_is_recognised(monkeypatch):
    data = eval_data()
    result, mock = run_eval(monkeypatch, rag_lesson(), data)
    generation = concept(result, "generation")
    assert generation.status == "covered"
    assert (
        "reads the question together with the retrieved context"
        in (generation.evidence_quotes[0])
    )
    assert mock.await_count == 1


def test_a_correct_statement_in_unlisted_wording_is_recognised(monkeypatch):
    statement = (
        "Once the model has the question and the passages it fetched, it uses both "
        "to compose a reply."
    )
    lesson = rag_lesson(  # the only statement of generation in the lesson
        replace={
            "generation": ("Writing the answer", statement, "X"),
            "end_to_end": VAGUE,
        },
        takeaways=NEUTRAL_TAKEAWAYS,
        summary=NEUTRAL_SUMMARY,
    )
    result, mock = run_eval(monkeypatch, lesson, eval_data())
    assert concept(result, "generation").status == "covered"
    assert statement in concept(result, "generation").evidence_quotes[0]
    assert mock.await_count == 1


# --- missing, weak and keyword-only coverage ----------------------------------------


def test_missing_augmentation_fails_completeness_and_says_exactly_what_to_teach(
    monkeypatch,
):
    result, _ = run_eval(monkeypatch, lesson_lacking("augmentation"), eval_data())

    assert concept(result, "augmentation").status == "missing"
    completeness = result.checks[6]
    assert completeness.passed is False
    assert "Augmentation (missing)" in completeness.reason
    assert "added to the user's question as context" in (
        completeness.improvement_suggestion
    )
    assert any(
        "Mandatory concept missing: Augmentation" in i.problem
        for i in completeness.issues
    )
    assert result.overall_pass is False
    assert "Augmentation" in result.improvement_feedback
    assert any("marked failed by code" in n for n in result.notes)


def test_missing_generation_fails_completeness_and_names_the_point(monkeypatch):
    result, _ = run_eval(monkeypatch, lesson_lacking("generation"), eval_data())

    assert concept(result, "generation").status == "missing"
    assert concept(result, "generation").missing_points == [
        "generation.writes_from_both"
    ]
    feedback = result.improvement_feedback
    assert "Generation" in feedback
    assert "reads the question together with the retrieved context" in feedback
    assert result.checks[6].passed is False


def test_a_partly_taught_concept_is_weak_not_missing(monkeypatch):
    lesson = rag_lesson(replace={"why_needed": WEAK_WHY})
    result, _ = run_eval(monkeypatch, lesson, eval_data())
    why = concept(result, "why_needed")
    assert why.status == "weak"
    assert why.missing_points == ["why.hallucination"]
    assert "Why RAG is needed (weak)" in result.checks[6].reason
    # only the unexplained point is asked for
    assert "unsupported or made-up answers" in result.checks[6].improvement_suggestion
    assert "recent or private information" not in (
        result.checks[6].improvement_suggestion
    )


def test_weak_and_missing_concepts_are_reported_separately(monkeypatch):
    lesson = rag_lesson(
        omit=("generation",),
        replace={"why_needed": WEAK_WHY, "end_to_end": VAGUE},
        takeaways=NEUTRAL_TAKEAWAYS,
        summary=NEUTRAL_SUMMARY,
    )
    result, _ = run_eval(monkeypatch, lesson, eval_data())
    statuses = {c.concept_id: c.status for c in result.concept_checks}
    assert statuses["why_needed"] == "weak"
    assert statuses["generation"] == "missing"
    reason = result.checks[6].reason
    assert "Why RAG is needed (weak)" in reason
    assert "Generation (missing)" in reason


def test_a_keyword_only_lesson_cannot_pass_even_if_the_model_claims_coverage(
    monkeypatch,
):
    """The model says covered, but no passage of the lesson states the point."""
    lesson = rag_lesson(
        replace={
            "generation": ("Generation", "Generation is an important step.", "X"),
            "end_to_end": VAGUE,
        },
        takeaways=NEUTRAL_TAKEAWAYS,
        summary=NEUTRAL_SUMMARY,
    )
    result, mock = run_eval(monkeypatch, lesson, eval_data())

    assert mock.await_count == 1  # nothing to ask again: code decides
    assert concept(result, "generation").status == "missing"
    assert result.checks[6].passed is False
    assert result.overall_pass is False
    assert any(
        "evaluator said covered, but no passage" in n and "generation.writes" in n
        for n in result.notes
    )


# --- evidence must be real ----------------------------------------------------------


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "unknown"])
def test_wrong_point_ids_from_the_model_cannot_break_the_evaluation(
    monkeypatch, mutation
):
    """Coverage is decided by code, so the model's ids only matter for wording."""
    data = eval_data()
    if mutation == "drop":
        data["point_checks"].pop()
    elif mutation == "duplicate":
        data["point_checks"][1] = dict(data["point_checks"][0])
    else:
        data["point_checks"][0]["point_id"] = "made.up"
    result, mock = run_eval(monkeypatch, rag_lesson(), data)
    assert mock.await_count == 1
    assert [c.status for c in result.concept_checks] == ["covered"] * 8
    assert result.overall_pass is True


def test_points_sent_without_requirements_are_discarded(monkeypatch):
    result, mock = run_eval(monkeypatch, rag_lesson(), eval_data(), requirements=None)
    assert mock.await_count == 1
    assert result.point_checks == []
    assert result.concept_checks == []


def test_no_requirements_means_empty_points_and_no_concept_checks(monkeypatch):
    data = eval_data()
    data["point_checks"] = []
    result, _ = run_eval(monkeypatch, rag_lesson(), data, requirements=None)
    assert result.concept_checks == []
    assert result.overall_pass is True


def test_model_supplied_concept_checks_and_notes_are_ignored(monkeypatch):
    data = eval_data()
    data["concept_checks"] = [{"concept_id": "generation", "status": "covered"}]
    data["notes"] = ["everything is fine"]
    result, _ = run_eval(monkeypatch, lesson_lacking("generation"), data)
    assert concept(result, "generation").status == "missing"  # derived by code
    assert "everything is fine" not in result.notes


def test_prompt_lists_points_workflow_and_rules(monkeypatch):
    prompt = evaluator.build_prompt(rag_lesson(), RAG_REQUIREMENTS)
    assert "generation.writes_from_both" in prompt
    assert "1. The user asks a question." in prompt
    assert "public internet" in prompt
    assert "No teaching points apply" in evaluator.build_prompt(rag_lesson())


def test_instructions_demand_explicit_statements_not_mentions():
    text = evaluator.SYSTEM_INSTRUCTIONS
    assert "STATES the point explicitly" in text
    assert "Merely mentioning" in text
    assert "Do \nNOT quote the lesson for points" in text.replace("a \\\n", "a ") or (
        "NOT quote the lesson for points" in text.replace("\n", " ")
    )
    assert "code finds the evidence" in text.replace("\n", " ")


# --- accuracy: the actual claim and its correction ----------------------------------


def accuracy_data(quote: str, correction: str, problem: str = "Inaccurate claim."):
    data = eval_data(failing=(0,))
    data["checks"][0]["issues"] = [
        {"problem": problem, "quote": quote, "term": "", "correction": correction}
    ]
    return data


def lesson_with_claim(sentence: str):
    return rag_lesson(
        replace={
            "limitations": (
                "Limits of RAG",
                "The retrieved information can be incomplete, outdated or "
                "incorrect. RAG does not guarantee that answers are accurate. "
                + sentence,
                "If the policy page is old, the answer will be old too.",
            )
        }
    )


def test_a_genuine_overclaim_is_kept_as_an_accuracy_failure(monkeypatch):
    claim = "RAG guarantees that every answer is correct."
    data = accuracy_data(claim, "RAG does not guarantee that answers are correct.")
    result, _ = run_eval(monkeypatch, lesson_with_claim(claim), data)
    assert result.checks[0].passed is False
    assert result.checks[0].issues[0].quote == claim
    assert result.overall_pass is False
    assert "Correction: RAG does not guarantee" in result.improvement_feedback


def test_an_unsupported_product_claim_is_a_genuine_accuracy_failure(monkeypatch):
    claim = "A well-known voice assistant uses RAG to find the weather."
    data = accuracy_data(
        claim,
        "Do not claim that a named product uses RAG; no source supports it.",
        problem="Unsupported claim about a real product.",
    )
    result, _ = run_eval(monkeypatch, lesson_with_claim(claim), data)
    assert result.checks[0].passed is False
    assert "Unsupported claim about a real product" in result.improvement_feedback


def test_a_sentence_that_already_states_the_correction_is_not_an_inaccuracy(
    monkeypatch,
):
    claim = "RAG does not guarantee the accuracy of the answers provided."
    data = accuracy_data(
        claim,
        "RAG does not ensure the accuracy of the answers, as the retrieved "
        "information may be outdated or incorrect.",
        problem="Claims that RAG guarantees accuracy.",
    )
    result, mock = run_eval(monkeypatch, lesson_with_claim(claim), data)
    assert mock.await_count == 1  # resolved by code, not sent back
    assert result.checks[0].passed is True
    assert result.overall_pass is True
    assert any("already states the correction" in n for n in result.notes)


def test_overclaim_and_false_positive_together_keep_only_the_real_problem(
    monkeypatch,
):
    good = "RAG does not guarantee the accuracy of the answers provided."
    bad = "RAG always learns from every conversation and improves itself."
    data = eval_data(failing=(0,))
    data["checks"][0]["issues"] = [
        {
            "problem": "Claims RAG guarantees accuracy.",
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
    lesson = lesson_with_claim(f"{good} {bad}")
    result, _ = run_eval(monkeypatch, lesson, data)
    assert result.checks[0].passed is False
    assert [i.quote for i in result.checks[0].issues] == [bad]


def test_accuracy_claims_are_still_verified_against_the_lesson(monkeypatch):
    data = accuracy_data("RAG trains the model.", "RAG does not train the model.")
    result, mock = run_eval(monkeypatch, rag_lesson(), data)
    assert mock.await_count == 1
    assert result.checks[0].passed is True  # nothing in the lesson says that
    assert any(
        "dropped an issue whose quote is not in the lesson" in n for n in result.notes
    )


@pytest.mark.parametrize(
    ("quote", "correction", "expected"),
    [
        (
            "RAG does not guarantee the accuracy of the answers provided.",
            "RAG does not ensure the accuracy of the answers, as the "
            "information may be wrong.",
            True,
        ),
        (  # opposite polarity: a real error
            "RAG guarantees accurate answers every time.",
            "RAG does not guarantee accurate answers.",
            False,
        ),
        (  # same polarity but unrelated words
            "RAG never forgets anything it reads.",
            "RAG cannot update the language model itself.",
            False,
        ),
        (  # too short to compare
            "It is not so.",
            "It is not that.",
            False,
        ),
        ("", "RAG does not guarantee answers.", False),
        ("RAG does not guarantee answers.", "", False),
    ],
)
def test_quote_states_correction_heuristic(quote, correction, expected):
    issue = EvaluationIssue(problem="p", quote=quote, term="", correction=correction)
    assert quote_states_correction(issue) is expected


# --- the final feedback --------------------------------------------------------------


def test_feedback_for_missing_concepts_names_the_concept_and_the_point_to_state(
    monkeypatch,
):
    result, _ = run_eval(
        monkeypatch,
        lesson_lacking("generation"),
        eval_data(uncovered=concept_points("generation")),
    )
    feedback = result.improvement_feedback
    assert CRITERIA[6] in feedback
    assert "Mandatory concept missing: Generation" in feedback
    assert "The lesson never states" in feedback  # the evaluator's own note
    assert "Mentioning a term is not enough" in feedback


def test_every_evidence_quote_in_the_fixture_is_in_the_lesson():
    from app.agents.evaluator import _normalize, _quote_in_lesson
    from app.evaluation.lesson_text import render_lesson

    normalized = _normalize(render_lesson(rag_lesson()))
    for point_id, quote in EVIDENCE.items():
        assert _quote_in_lesson(quote, normalized), point_id


# --- schema enum, wording, reconciling unverifiable absence claims ---

from app.agents.evaluator import align_with_evidence  # noqa: E402
from app.evaluation.topics import all_points  # noqa: E402

from tests.rag_fixtures import as_result  # noqa: E402


def test_point_ids_are_an_enum_in_the_schema_sent_to_the_model():
    schema = evaluator.api_schema(RAG_REQUIREMENTS)
    enum = schema["$defs"]["PointCheck"]["properties"]["point_id"]["enum"]
    assert enum == [p.id for _, p in all_points(RAG_REQUIREMENTS)]
    assert "retrieval" not in enum  # a concept id can no longer be answered
    assert (
        "enum"
        not in evaluator.api_schema()["$defs"]["PointCheck"]["properties"]["point_id"]
    )


def test_the_request_carries_the_enum_and_the_prompt_embeds_it(monkeypatch):
    _, mock = run_eval(monkeypatch, rag_lesson(), eval_data())
    schema = mock.await_args.kwargs["json_schema"]
    assert (
        "generation.writes_from_both"
        in (schema["$defs"]["PointCheck"]["properties"]["point_id"]["enum"])
    )
    assert '"enum"' in mock.await_args.args[0]


def test_the_prompt_says_requirements_are_not_lesson_text_and_ids_are_exact():
    prompt = evaluator.build_prompt(rag_lesson(), RAG_REQUIREMENTS)
    assert "REQUIREMENTS, not lesson text" in prompt
    assert "never the concept id" in prompt
    assert "do not quote the lesson for points" in prompt
    text = " ".join(evaluator.SYSTEM_INSTRUCTIONS.split())
    assert "never a concept id" in text
    assert "NOT quote the lesson for points" in text
    assert "code finds the evidence" in text


def test_point_evidence_is_always_a_real_lesson_passage(monkeypatch):
    from app.agents.evaluator import _normalize
    from app.evaluation.lesson_text import render_lesson

    data = eval_data()
    for point in data["point_checks"]:
        point["evidence_quote"] = "A sentence the model made up."  # ignored
    result, _ = run_eval(monkeypatch, rag_lesson(), data)
    text = _normalize(render_lesson(rag_lesson()))
    assert len(result.point_checks) == 12
    for point in result.point_checks:
        assert point.covered
        assert point.evidence_quote != "A sentence the model made up."
        assert _normalize(point.evidence_quote) in text


def test_a_lesson_that_never_states_a_point_is_missing_it_whatever_the_model_says(
    monkeypatch,
):
    result, _ = run_eval(monkeypatch, lesson_lacking("augmentation"), eval_data())
    assert concept(result, "augmentation").status == "missing"
    assert result.checks[6].passed is False


def test_a_passage_the_code_finds_beats_a_model_not_covered(monkeypatch):
    data = eval_data(uncovered=("limits.no_guarantee",))  # the lesson does state it
    result, _ = run_eval(monkeypatch, rag_lesson(), data)
    assert concept(result, "limitations").status == "covered"
    assert result.overall_pass is True
    assert any(
        "evaluator said not covered, but the lesson states it" in n
        for n in result.notes
    )


def key_point_result(absence_issue_quote="", uncovered=(), failing_key=True):
    data = eval_data(uncovered=uncovered, failing=(4,) if failing_key else ())
    if failing_key:
        data["checks"][4]["issues"] = [
            {
                "problem": "Missing explanation of importance.",
                "quote": absence_issue_quote,
                "term": "",
                "correction": "",
            }
        ]
    return as_result(data)


def derive(result):
    from app.evaluation.concepts import derive_concept_checks

    return result.model_copy(
        update={
            "concept_checks": derive_concept_checks(
                RAG_REQUIREMENTS, result.point_checks
            )
        }
    )


def test_an_unquoted_key_points_claim_cannot_outvote_verified_evidence():
    aligned = align_with_evidence(
        derive(key_point_result()), rag_lesson(), RAG_REQUIREMENTS
    )
    assert aligned.checks[4].passed is True
    assert any("'Covers key points' was marked passed" in n for n in aligned.notes)
    assert aligned.overall_pass is True


def test_a_quoted_key_points_issue_is_never_dropped():
    result = key_point_result(absence_issue_quote="Introduction to RAG")
    aligned = align_with_evidence(derive(result), rag_lesson(), RAG_REQUIREMENTS)
    assert aligned.checks[4].passed is False


def test_key_points_fail_by_code_when_a_key_point_is_not_covered():
    result = key_point_result(uncovered=("flow.example",), failing_key=False)
    assert result.checks[4].passed is True  # the evaluator passed it
    aligned = align_with_evidence(derive(result), rag_lesson(), RAG_REQUIREMENTS)
    assert aligned.checks[4].passed is False
    assert "flow.example" in aligned.checks[4].reason
    assert any("marked failed by code" in n for n in aligned.notes)


def test_key_points_claim_is_kept_when_a_key_point_is_missing():
    result = key_point_result(uncovered=("why.knowledge_gap",))
    aligned = align_with_evidence(derive(result), rag_lesson(), RAG_REQUIREMENTS)
    assert aligned.checks[4].passed is False


def completeness_result(issue_quote="", uncovered=()):
    data = eval_data(uncovered=uncovered, failing=(6,))
    data["checks"][6]["issues"] = [
        {
            "problem": "Learning objective not fully addressed.",
            "quote": issue_quote,
            "term": "",
            "correction": "",
        }
    ]
    return as_result(data)


def test_a_vague_completeness_claim_is_dropped_when_everything_is_evidenced():
    aligned = align_with_evidence(
        derive(completeness_result()), rag_lesson(), RAG_REQUIREMENTS
    )
    assert aligned.checks[6].passed is True
    assert any(
        "'Overall lesson completeness' was marked passed" in n for n in aligned.notes
    )


def test_a_quoted_completeness_issue_is_never_dropped():
    result = completeness_result(issue_quote="Introduction to RAG")
    aligned = align_with_evidence(derive(result), rag_lesson(), RAG_REQUIREMENTS)
    assert aligned.checks[6].passed is False


def test_the_completeness_claim_is_kept_when_an_objective_is_unsupported():
    lesson = rag_lesson(
        objectives=("Explain quantum computing hardware design.", "Know RAG.")
    )
    aligned = align_with_evidence(
        derive(completeness_result()), lesson, RAG_REQUIREMENTS
    )
    assert aligned.checks[6].passed is False


def test_the_completeness_claim_is_kept_when_a_point_is_not_covered():
    result = completeness_result(uncovered=("limits.no_guarantee",))
    aligned = align_with_evidence(derive(result), rag_lesson(), RAG_REQUIREMENTS)
    assert aligned.checks[6].passed is False


def test_alignment_does_nothing_without_requirements_or_points():
    result = as_result(eval_data(failing=(4, 6)))
    assert align_with_evidence(result, rag_lesson(), None) is result
    assert (
        align_with_evidence(
            result.model_copy(update={"point_checks": []}),
            rag_lesson(),
            RAG_REQUIREMENTS,
        )
        .checks[4]
        .passed
        is False
    )
