import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import improver
from app.agents.improver import (
    ImprovementValidationError,
    InvalidImprovementJSONError,
    MissingFeedbackError,
    improve_lesson,
)
from app.evaluation.schemas import CRITERIA, EvaluationResult, LessonContent
from app.llm import LLMError


def lesson_data(title: str = "Intro to RAG") -> dict:
    section = {"heading": "H", "explanation": "E", "example": "X"}
    return {
        "title": title,
        "topic": "RAG",
        "learner_level": "Beginner",
        "learning_objectives": ["Know what RAG is", "Know why it matters"],
        "sections": [section, section, section],
        "key_takeaways": ["a", "b", "c"],
        "summary": "s",
    }


def make_lesson() -> LessonContent:
    return LessonContent.model_validate(lesson_data())


def make_evaluation(failing: tuple[int, ...] = (2,)) -> EvaluationResult:
    checks = [
        {
            "criterion": name,
            "passed": i not in failing,
            "reason": f"Reason for {name}.",
            "improvement_suggestion": f"Fix {name}." if i in failing else "",
            "evidence_quote": "" if i in failing else "ok",
            "issues": (
                [
                    {
                        "problem": f"Problem in {name}.",
                        "quote": "Intro",
                        "term": "RAG",
                        "correction": "The correct statement.",
                    }
                ]
                if i in failing
                else []
            ),
        }
        for i, name in enumerate(CRITERIA)
    ]
    return EvaluationResult.model_validate(
        {
            "overall_pass": not failing,
            "checks": checks,
            "failure_reasons": [f"{CRITERIA[i]} failed" for i in failing],
            "improvement_feedback": "Add better examples." if failing else "",
        }
    )


@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock(return_value=json.dumps(lesson_data("Better Intro to RAG")))
    monkeypatch.setattr(improver, "generate_text", mock)
    return mock


def run(lesson=None, evaluation=None) -> LessonContent:
    return asyncio.run(
        improve_lesson(lesson or make_lesson(), evaluation or make_evaluation())
    )


def test_successful_improvement(mock_llm):
    result = run()
    assert isinstance(result, LessonContent)
    assert result.title == "Better Intro to RAG"
    mock_llm.assert_awaited_once()


def test_topic_and_level_preserved(mock_llm):
    original = make_lesson()
    result = run(original)
    assert result.topic == original.topic
    assert result.learner_level == original.learner_level


def test_changed_topic_rejected(mock_llm):
    data = lesson_data()
    data["topic"] = "Something else"
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(ImprovementValidationError, match="topic"):
        run()


def test_changed_learner_level_rejected(mock_llm):
    data = lesson_data()
    data["learner_level"] = "Expert"
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(ImprovementValidationError, match="learner level"):
        run()


def test_prompt_uses_failure_feedback_and_lesson(mock_llm):
    lesson = make_lesson()
    run(lesson, make_evaluation(failing=(2, 3)))
    args, kwargs = mock_llm.await_args
    prompt = args[0]
    assert lesson.model_dump_json(indent=2) in prompt
    assert f"Fix {CRITERIA[2]}." in prompt
    assert f"Fix {CRITERIA[3]}." in prompt
    assert "Add better examples." in prompt
    assert kwargs["instructions"] == improver.SYSTEM_INSTRUCTIONS
    # Passing checkpoints are listed as "preserve", not as problems to fix.
    assert f"Fix {CRITERIA[0]}." not in prompt
    assert f"- {CRITERIA[0]}" in prompt.split("Passing checkpoints")[1]


def test_passing_evaluation_skips_llm(mock_llm):
    lesson = make_lesson()
    result = run(lesson, make_evaluation(failing=()))
    assert result is lesson
    mock_llm.assert_not_awaited()


def test_evaluation_not_modified(mock_llm):
    evaluation = make_evaluation()
    before = evaluation.model_dump()
    run(evaluation=evaluation)
    assert evaluation.model_dump() == before


def test_failed_evaluation_without_feedback_rejected(mock_llm):
    evaluation = make_evaluation().model_copy(update={"improvement_feedback": ""})
    with pytest.raises(MissingFeedbackError):
        run(evaluation=evaluation)
    mock_llm.assert_not_awaited()


def test_invalid_json(mock_llm):
    mock_llm.return_value = "here is your lesson!"
    with pytest.raises(InvalidImprovementJSONError):
        run()


def test_pydantic_validation_failure(mock_llm):
    data = lesson_data()
    data["sections"] = data["sections"][:2]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(ImprovementValidationError):
        run()


def test_llm_error_propagates_unchanged(mock_llm):
    error = LLMError("boom")
    mock_llm.side_effect = error
    with pytest.raises(LLMError) as info:
        run()
    assert info.value is error


# --- Issues, rubric and history in the prompt ------------------------------------


def test_prompt_lists_issues_and_pass_conditions(mock_llm):
    run(evaluation=make_evaluation(failing=(3,)))
    prompt = mock_llm.await_args.args[0]
    assert f"Problem in {CRITERIA[3]}." in prompt
    assert "Pass conditions:" in prompt
    assert "EVERY technical term" in prompt


def test_history_included_when_given(mock_llm):
    history = [
        {
            "attempt_number": 1,
            "failed_criteria": [CRITERIA[3]],
            "failure_reasons": ["x"],
            "improvement_feedback": "Define 'augmented'.",
            "changes": ["sections: 3 -> 5, 5 changed"],
        }
    ]
    asyncio.run(improve_lesson(make_lesson(), make_evaluation(), history=history))
    prompt = mock_llm.await_args.args[0]
    assert "Earlier rounds that also failed" in prompt
    assert "Define 'augmented'." in prompt
    assert "sections: 3 -> 5, 5 changed" in prompt


def test_no_history_section_when_none(mock_llm):
    run()
    assert "Earlier rounds" not in mock_llm.await_args.args[0]


def test_instructions_require_sweep_and_scope_control():
    text = improver.SYSTEM_INSTRUCTIONS
    assert "sweep the WHOLE lesson" in text
    assert "do not change learning_objectives" in text


def test_prompt_includes_mandatory_concepts_when_given(mock_llm):
    from app.evaluation.topics import RAG_REQUIREMENTS

    asyncio.run(
        improve_lesson(make_lesson(), make_evaluation(), requirements=RAG_REQUIREMENTS)
    )
    prompt = mock_llm.await_args.args[0]
    assert "Mandatory concepts the finished lesson must teach" in prompt
    assert "[augmentation]" in prompt
    assert "do not rewrite sections that already teach theirs" in prompt


def test_prompt_has_no_concept_block_without_requirements(mock_llm):
    run()
    assert "Mandatory concepts" not in mock_llm.await_args.args[0]


# --- Phase 2: accuracy guardrails --------------------------------------------------


def test_improver_instructions_contain_general_guardrails():
    from app.evaluation.topics import GENERAL_GUARDRAILS

    assert "{guardrails}" not in improver.SYSTEM_INSTRUCTIONS
    for rule in GENERAL_GUARDRAILS:
        assert rule in improver.SYSTEM_INSTRUCTIONS


def test_improver_prompt_has_topic_guardrails_with_requirements(mock_llm):
    from app.evaluation.topics import RAG_REQUIREMENTS

    asyncio.run(
        improve_lesson(make_lesson(), make_evaluation(), requirements=RAG_REQUIREMENTS)
    )
    prompt = mock_llm.await_args.args[0]
    assert "trains or updates the language model" in prompt
    assert "public internet" in prompt


def test_improver_prompt_shows_exact_claim_and_correction(mock_llm):
    run(evaluation=make_evaluation(failing=(0,)))
    prompt = mock_llm.await_args.args[0]
    assert '(in the text: "Intro")' in prompt
    assert "Correction: The correct statement." in prompt
    assert "fix exactly that claim" in improver.SYSTEM_INSTRUCTIONS


# --- Phase 3: preservation, re-ask and logging -------------------------------------

from app.agents.improver import (  # noqa: E402
    ImprovementValidationError as _ImprovementValidationError,
)
from app.agents.improver import preservation_problems, target_sections  # noqa: E402


def distinct_lesson(objectives=None) -> LessonContent:
    data = lesson_data()
    data["sections"] = [
        {
            "heading": f"Heading {i}",
            "explanation": f"Explanation {i}.",
            "example": f"Ex {i}",
        }
        for i in (1, 2, 3)
    ]
    if objectives:
        data["learning_objectives"] = objectives
    return LessonContent.model_validate(data)


def localized_evaluation(quote: str, failing: int = 2) -> EvaluationResult:
    data = make_evaluation(failing=(failing,)).model_dump()
    data["checks"][failing]["issues"][0]["quote"] = quote
    return EvaluationResult.model_validate(data)


def improved_json(lesson: LessonContent, **changes) -> str:
    data = lesson.model_dump()
    data.update(changes)
    return json.dumps(data)


def test_target_sections_found_from_issue_quotes():
    lesson = distinct_lesson()
    assert target_sections(lesson, localized_evaluation("Explanation 2.")) == {1}


def test_target_sections_none_when_not_localizable():
    lesson = distinct_lesson()
    assert target_sections(lesson, localized_evaluation("not in any section")) is None
    assert target_sections(lesson, localized_evaluation("")) is None


def test_prompt_names_sections_to_edit_and_objective_rule(mock_llm):
    lesson = distinct_lesson()
    sections = [dict(s) for s in lesson.model_dump()["sections"]]
    sections[1]["explanation"] = "Better."
    mock_llm.return_value = improved_json(lesson, sections=sections)
    asyncio.run(improve_lesson(lesson, localized_evaluation("Explanation 2.")))
    prompt = mock_llm.await_args.args[0]
    assert "Sections to edit: 2." in prompt
    assert "Keep every other section word for word" in prompt
    assert "Keep the existing learning objectives word for word" in prompt


def test_prompt_has_no_section_scope_when_not_localizable(mock_llm):
    run()
    prompt = mock_llm.await_args.args[0]
    assert "Sections to edit" not in prompt
    assert "Keep the existing learning objectives word for word" in prompt


def test_changing_an_untargeted_section_is_re_asked_then_rejected(mock_llm):
    lesson = distinct_lesson()
    sections = [dict(s) for s in lesson.model_dump()["sections"]]
    sections[0]["explanation"] = "Rewritten."
    mock_llm.return_value = improved_json(lesson, sections=sections)
    with pytest.raises(_ImprovementValidationError, match="section"):
        asyncio.run(improve_lesson(lesson, localized_evaluation("Explanation 2.")))
    assert mock_llm.await_count == 2
    assert "rejected" in mock_llm.await_args_list[1].args[0]


def test_editing_only_the_targeted_section_is_accepted(mock_llm):
    lesson = distinct_lesson()
    sections = [dict(s) for s in lesson.model_dump()["sections"]]
    sections[1]["explanation"] = "Better explanation."
    mock_llm.return_value = improved_json(lesson, sections=sections)
    result = asyncio.run(improve_lesson(lesson, localized_evaluation("Explanation 2.")))
    assert result.sections[1].explanation == "Better explanation."
    assert mock_llm.await_count == 1


def test_adding_sections_is_allowed_but_removing_is_not(mock_llm):
    lesson = distinct_lesson()
    sections = [dict(s) for s in lesson.model_dump()["sections"]]
    extra = {"heading": "New", "explanation": "New one.", "example": "Ex"}
    mock_llm.return_value = improved_json(lesson, sections=[*sections, extra])
    assert (
        len(
            asyncio.run(
                improve_lesson(lesson, localized_evaluation("Explanation 2."))
            ).sections
        )
        == 4
    )

    mock_llm.return_value = improved_json(lesson, sections=[*sections[:2], extra])
    problems = preservation_problems(
        lesson,
        LessonContent.model_validate(json.loads(mock_llm.return_value)),
        localized_evaluation("Explanation 2."),
    )
    assert any("section" in p for p in problems)


def test_learning_objectives_must_be_kept_unless_flow_failed():
    lesson = distinct_lesson(["Objective A", "Objective B"])
    changed = LessonContent.model_validate(
        {**lesson.model_dump(), "learning_objectives": ["Objective A", "Other"]}
    )
    appended = LessonContent.model_validate(
        {
            **lesson.model_dump(),
            "learning_objectives": ["Objective A", "Objective B", "New"],
        }
    )
    evaluation = localized_evaluation("Explanation 2.")
    assert any(
        "Objective B" in p for p in preservation_problems(lesson, changed, evaluation)
    )
    assert preservation_problems(lesson, appended, evaluation) == []

    flow_failed = make_evaluation(failing=(5,))  # Coherent teaching flow
    assert preservation_problems(lesson, changed, flow_failed) == []


def test_changed_objectives_are_re_asked_once_then_accepted(mock_llm):
    lesson = distinct_lesson(["Objective A", "Objective B"])
    bad = improved_json(lesson, learning_objectives=["Objective A", "Rewritten"])
    good = improved_json(lesson, title="Better title")
    mock_llm.side_effect = [bad, good]
    result = asyncio.run(improve_lesson(lesson, make_evaluation()))
    assert result.title == "Better title"
    assert mock_llm.await_count == 2
    assert "Objective B" in mock_llm.await_args_list[1].args[0]


def test_invalid_json_is_re_asked_once(mock_llm):
    mock_llm.side_effect = ["not json", json.dumps(lesson_data("Fixed"))]
    assert run().title == "Fixed"
    assert mock_llm.await_count == 2


@pytest.fixture
def events(monkeypatch):
    captured = []
    monkeypatch.setattr(
        improver, "log_event", lambda name, **fields: captured.append((name, fields))
    )
    return captured


def test_improver_logs_responses_and_reasks(mock_llm, events):
    lesson = distinct_lesson(["Objective A", "Objective B"])
    mock_llm.side_effect = [
        improved_json(lesson, learning_objectives=["Objective A", "Rewritten"]),
        improved_json(lesson, title="Better"),
    ]
    asyncio.run(improve_lesson(lesson, make_evaluation()))
    names = [name for name, _ in events]
    assert names == [
        "improver_response",
        "improver_preservation_problems",
        "improver_reask",
        "improver_response",
    ]


def test_improver_logs_validation_failures(mock_llm, events):
    mock_llm.return_value = "not json"
    with pytest.raises(InvalidImprovementJSONError):
        run()
    names = [name for name, _ in events]
    assert names.count("improver_validation_failure") == 2


# --- Fix 1: objectives containing a flagged problem may be corrected ---------------

WRONG_OBJECTIVE = "Understand how RAG trains the model on new data."
JARGON_OBJECTIVE = "Explain how vectorization powers retrieval."
GOOD_OBJECTIVE = "Know what RAG is."


def objectives_evaluation(failing: int, quote: str, **issue_fields) -> EvaluationResult:
    data = make_evaluation(failing=(failing,)).model_dump()
    issue = data["checks"][failing]["issues"][0]
    issue.update(quote=quote, **issue_fields)
    return EvaluationResult.model_validate(data)


def with_objectives(*objectives: str) -> LessonContent:
    return LessonContent.model_validate(
        {**lesson_data(), "learning_objectives": list(objectives)}
    )


def test_inaccurate_objective_may_be_corrected():
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    fixed = with_objectives("Understand how RAG supplies information.", GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(
        0, WRONG_OBJECTIVE, correction="RAG does not train the model."
    )
    assert preservation_problems(original, fixed, evaluation) == []


def test_jargon_containing_objective_may_be_corrected():
    original = with_objectives(JARGON_OBJECTIVE, GOOD_OBJECTIVE)
    fixed = with_objectives("Explain how RAG finds useful passages.", GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(3, JARGON_OBJECTIVE, term="vectorization")
    assert preservation_problems(original, fixed, evaluation) == []


def test_quote_inside_an_objective_exempts_that_objective():
    original = with_objectives(
        "Understand how RAG trains the model on new data.", GOOD_OBJECTIVE
    )
    fixed = with_objectives("Understand RAG.", GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(0, "trains the model on new data")
    assert preservation_problems(original, fixed, evaluation) == []


def test_other_objectives_are_still_preserved():
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    fixed = with_objectives(
        "Understand how RAG supplies information.", "Something else."
    )
    evaluation = objectives_evaluation(0, WRONG_OBJECTIVE, correction="Fix.")
    problems = preservation_problems(original, fixed, evaluation)
    assert len(problems) == 1
    assert GOOD_OBJECTIVE in problems[0]
    assert WRONG_OBJECTIVE not in problems[0]


def test_unflagged_objective_cannot_change_when_quote_is_elsewhere():
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    fixed = with_objectives("Changed.", GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(0, "text that is in no objective")
    problems = preservation_problems(original, fixed, evaluation)
    assert problems and WRONG_OBJECTIVE in problems[0]


def test_removing_a_flagged_objective_outright_is_also_allowed():
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    fixed = with_objectives(GOOD_OBJECTIVE, "A replacement.")
    evaluation = objectives_evaluation(0, WRONG_OBJECTIVE, correction="Fix.")
    assert preservation_problems(original, fixed, evaluation) == []


def test_prompt_allows_correcting_flagged_objectives(mock_llm):
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(0, WRONG_OBJECTIVE, correction="Fix.")
    mock_llm.return_value = improved_json(
        original, learning_objectives=["Understand RAG.", GOOD_OBJECTIVE]
    )
    asyncio.run(improve_lesson(original, evaluation))
    prompt = mock_llm.await_args.args[0]
    assert "except any that contain a reported problem" in prompt


def test_fixing_a_flagged_objective_is_accepted_without_a_reask(mock_llm):
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(0, WRONG_OBJECTIVE, correction="Fix.")
    mock_llm.return_value = improved_json(
        original, learning_objectives=["Understand RAG.", GOOD_OBJECTIVE]
    )
    result = asyncio.run(improve_lesson(original, evaluation))
    assert result.learning_objectives[0] == "Understand RAG."
    assert mock_llm.await_count == 1


def test_one_reask_still_applies_when_an_unflagged_objective_changes(mock_llm):
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(0, WRONG_OBJECTIVE, correction="Fix.")
    bad = improved_json(original, learning_objectives=["Understand RAG.", "Rewritten."])
    good = improved_json(
        original, learning_objectives=["Understand RAG.", GOOD_OBJECTIVE]
    )
    mock_llm.side_effect = [bad, good]
    result = asyncio.run(improve_lesson(original, evaluation))
    assert result.learning_objectives == ["Understand RAG.", GOOD_OBJECTIVE]
    assert mock_llm.await_count == 2
    assert GOOD_OBJECTIVE in mock_llm.await_args_list[1].args[0]


def test_persistent_violation_after_the_reask_still_raises(mock_llm):
    original = with_objectives(WRONG_OBJECTIVE, GOOD_OBJECTIVE)
    evaluation = objectives_evaluation(0, WRONG_OBJECTIVE, correction="Fix.")
    mock_llm.return_value = improved_json(
        original, learning_objectives=["Understand RAG.", "Rewritten."]
    )
    with pytest.raises(_ImprovementValidationError, match="objectives"):
        asyncio.run(improve_lesson(original, evaluation))
    assert mock_llm.await_count == 2
