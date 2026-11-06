import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import evaluator
from app.agents.evaluator import (
    EvaluationValidationError,
    InvalidEvaluationJSONError,
    evaluate_lesson,
)
from app.evaluation.schemas import CRITERIA, EvaluationResult, LessonContent
from app.llm import LLMError


def make_lesson() -> LessonContent:
    section = {"heading": "H", "explanation": "E", "example": "X"}
    return LessonContent.model_validate(
        {
            "title": "Intro to RAG",
            "topic": "RAG",
            "learner_level": "Beginner",
            "learning_objectives": ["a", "b"],
            "sections": [section, section, section],
            "key_takeaways": ["a", "b", "c"],
            "summary": "s",
        }
    )


def make_result(failing: tuple[int, ...] = ()) -> dict:
    checks = []
    for i, name in enumerate(CRITERIA):
        failed = i in failing
        checks.append(
            {
                "criterion": name,
                "passed": not failed,
                "reason": "Because.",
                "improvement_suggestion": "Fix it." if failed else "",
            }
        )
    return {
        "overall_pass": not failing,
        "checks": checks,
        "failure_reasons": [f"{CRITERIA[i]} failed" for i in failing],
        "improvement_feedback": "Fix the failures." if failing else "",
    }


@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock(return_value=json.dumps(make_result()))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    return mock


def run(raw_data) -> EvaluationResult:
    return asyncio.run(evaluate_lesson(make_lesson()))


def test_all_seven_pass(mock_llm):
    result = run(None)
    assert result.overall_pass is True
    assert len(result.checks) == 7
    assert all(c.passed for c in result.checks)


def test_one_check_fails(mock_llm):
    mock_llm.return_value = json.dumps(make_result(failing=(2,)))
    result = run(None)
    assert result.overall_pass is False
    assert [c.passed for c in result.checks].count(False) == 1
    assert len(result.failure_reasons) == 1


def test_multiple_checks_fail(mock_llm):
    mock_llm.return_value = json.dumps(make_result(failing=(0, 3, 6)))
    result = run(None)
    assert result.overall_pass is False
    assert [c.passed for c in result.checks].count(False) == 3
    assert len(result.failure_reasons) == 3
    assert result.improvement_feedback


def test_overall_pass_true_with_failed_check_rejected(mock_llm):
    data = make_result(failing=(1,))
    data["overall_pass"] = True
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError, match="inconsistent"):
        run(None)


def test_overall_pass_false_with_all_passing_rejected(mock_llm):
    data = make_result()
    data["overall_pass"] = False
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError):
        run(None)


def test_failed_check_without_suggestion_rejected(mock_llm):
    data = make_result(failing=(4,))
    data["checks"][4]["improvement_suggestion"] = ""
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError, match="improvement_suggestion"):
        run(None)


def test_missing_checkpoint_rejected(mock_llm):
    data = make_result()
    data["checks"].pop()
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError):
        run(None)


def test_extra_checkpoint_rejected(mock_llm):
    data = make_result()
    data["checks"].append(data["checks"][0])
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError):
        run(None)


def test_duplicate_criterion_rejected(mock_llm):
    data = make_result()
    data["checks"][6]["criterion"] = data["checks"][0]["criterion"]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError, match="exactly once"):
        run(None)


def test_unknown_criterion_rejected(mock_llm):
    data = make_result()
    data["checks"][0]["criterion"] = "Vibes"
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError):
        run(None)


def test_invalid_json(mock_llm):
    mock_llm.return_value = "looks great, pass!"
    with pytest.raises(InvalidEvaluationJSONError):
        run(None)


def test_llm_error_propagates_unchanged(mock_llm):
    error = LLMError("boom")
    mock_llm.side_effect = error
    with pytest.raises(LLMError) as info:
        run(None)
    assert info.value is error


def test_full_lesson_and_instructions_passed(mock_llm):
    lesson = make_lesson()
    asyncio.run(evaluate_lesson(lesson))
    args, kwargs = mock_llm.await_args
    assert lesson.model_dump_json(indent=2) in args[0]
    assert kwargs["instructions"] == evaluator.SYSTEM_INSTRUCTIONS
    for name in CRITERIA:
        assert name in evaluator.SYSTEM_INSTRUCTIONS
