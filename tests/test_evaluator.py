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
from app.evaluation.schemas import (
    CRITERIA,
    EvaluationResult,
    LessonContent,
    to_strict_json_schema,
)
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
                "issues": ["A specific problem."] if failed else [],
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


# --- Structured output / real-world JSON handling regression tests ---------------


def _walk_objects(node):
    if isinstance(node, dict):
        if "properties" in node:
            yield node
            for sub in node["properties"].values():
                yield from _walk_objects(sub)
        for key, value in node.items():
            if key != "properties":
                yield from _walk_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_objects(item)


def _all_keywords(node, found=None):
    found = set() if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties":
                for sub in value.values():
                    _all_keywords(sub, found)
            else:
                found.add(key)
                _all_keywords(value, found)
    elif isinstance(node, list):
        for item in node:
            _all_keywords(item, found)
    return found


def test_strict_schema_meets_openai_strict_rules():
    schema = to_strict_json_schema(EvaluationResult)
    objects = list(_walk_objects(schema))
    assert len(objects) >= 2  # EvaluationResult and EvaluationCheck
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])
    unsupported = {"default", "minLength", "maxLength", "minItems", "maxItems"}
    assert not unsupported & _all_keywords(schema)
    assert set(CRITERIA) == set(
        schema["$defs"]["EvaluationCheck"]["properties"]["criterion"]["enum"]
    )


def test_strict_schema_keeps_fields_named_like_keywords():
    schema = to_strict_json_schema(LessonContent)
    assert "title" in schema["properties"]
    assert "title" in schema["required"]


def test_evaluator_requests_structured_output(mock_llm):
    asyncio.run(evaluate_lesson(make_lesson()))
    kwargs = mock_llm.await_args.kwargs
    assert kwargs["schema_name"] == "evaluation_result"
    assert kwargs["json_schema"] == to_strict_json_schema(EvaluationResult)


def test_valid_structured_response_accepted(mock_llm):
    mock_llm.return_value = json.dumps(make_result(failing=(1,)))
    result = run(None)
    assert result.overall_pass is False
    assert result.checks[1].criterion == CRITERIA[1]


def test_unescaped_quote_in_string_rejected_without_leaking_text(mock_llm):
    mock_llm.return_value = (
        '{"overall_pass": true, "checks": [{"criterion": "x", '
        '"reason": "The phrase "retrieval" is vague"}]}'
    )
    with pytest.raises(InvalidEvaluationJSONError) as info:
        run(None)
    assert "retrieval" not in str(info.value)
    assert "response length" in str(info.value)


def test_truncated_json_rejected(mock_llm):
    mock_llm.return_value = json.dumps(make_result())[:200]
    with pytest.raises(InvalidEvaluationJSONError):
        run(None)


def test_empty_response_rejected(mock_llm):
    mock_llm.return_value = ""
    with pytest.raises(InvalidEvaluationJSONError):
        run(None)


@pytest.mark.parametrize("missing", ["overall_pass", "checks"])
def test_missing_top_level_field_rejected(mock_llm, missing):
    data = make_result()
    del data[missing]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError):
        run(None)


@pytest.mark.parametrize("missing", ["criterion", "passed", "reason"])
def test_missing_check_field_rejected(mock_llm, missing):
    data = make_result()
    del data["checks"][0][missing]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError):
        run(None)


# --- Rubric, exhaustive issues, derived failure reasons --------------------------


def test_failed_check_without_issues_rejected(mock_llm):
    data = make_result(failing=(3,))
    data["checks"][3]["issues"] = []
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError, match="issues"):
        run(None)


def test_missing_failure_reasons_is_derived_not_rejected(mock_llm):
    data = make_result(failing=(2,))
    del data["failure_reasons"]
    mock_llm.return_value = json.dumps(data)
    assert len(run(None).failure_reasons) == 1


def test_failure_reasons_derived_from_check_reasons(mock_llm):
    data = make_result(failing=(1, 3))
    data["failure_reasons"] = [CRITERIA[1], CRITERIA[3]]  # bare names, as seen live
    data["checks"][1]["reason"] = "Uses the word 'ubiquitous'."
    mock_llm.return_value = json.dumps(data)
    result = run(None)
    assert result.failure_reasons[0] == f"{CRITERIA[1]}: Uses the word 'ubiquitous'."
    assert len(result.failure_reasons) == 2


def test_instructions_contain_rubric_and_exhaustiveness_rule():
    from app.evaluation.rubric import RUBRIC

    text = evaluator.SYSTEM_INSTRUCTIONS
    assert "{rubric}" not in text
    for item in RUBRIC:
        assert item.name in text
        assert item.pass_when[0] in text
    assert "EVERY problem" in text
