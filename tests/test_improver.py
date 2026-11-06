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
            "issues": [f"Problem in {name}."] if i in failing else [],
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
