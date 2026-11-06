import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import evaluator
from app.agents.evaluator import (
    EvaluationValidationError,
    EvidenceError,
    InvalidEvaluationJSONError,
    body_text,
    collect_evidence_problems,
    evaluate_lesson,
    find_definition,
    render_lesson,
    resolve_contradictions,
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
                "evidence_quote": "" if failed else "Intro to RAG",
                "issues": (
                    [
                        {
                            "problem": "A specific problem.",
                            "quote": "Intro to RAG",
                            "term": "RAG",
                            "correction": "The correct statement.",
                        }
                    ]
                    if failed
                    else []
                ),
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
    assert render_lesson(lesson) in args[0]
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
    assert kwargs["json_schema"] == evaluator.api_schema()


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


# --- Evidence verification, definition cross-check and re-ask ---------------------


def jargon_failure(term: str, quote: str) -> dict:
    data = make_result(failing=(3,))
    data["checks"][3]["issues"] = [
        {"problem": "Unexplained.", "quote": quote, "term": term}
    ]
    return data


def test_lesson_rendered_as_plain_text_in_prompt(mock_llm):
    lesson = make_lesson()
    asyncio.run(evaluate_lesson(lesson))
    prompt = mock_llm.await_args.args[0]
    assert "Section 1: H" in prompt
    assert "Learning objectives:" in prompt
    assert '"learning_objectives"' not in prompt.split("JSON schema")[0]


def test_passed_check_needs_evidence_quote(mock_llm):
    data = make_result()
    data["checks"][0]["evidence_quote"] = ""
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError, match="evidence_quote"):
        run(None)


def test_jargon_issue_needs_term_and_quote(mock_llm):
    data = jargon_failure("", "Intro to RAG")
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvaluationValidationError, match="exact term"):
        run(None)


def test_invented_quote_is_rejected(mock_llm):
    data = make_result()
    data["checks"][0]["evidence_quote"] = "This sentence is not in the lesson."
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvidenceError, match="does not appear"):
        run(None)
    assert mock_llm.await_count == 2  # asked again once, then gave up


def test_quote_matching_ignores_case_whitespace_curly_quotes_and_ellipsis(mock_llm):
    data = make_result()
    data["checks"][0]["evidence_quote"] = "INTRO   to RAG ... Topic: RAG"
    mock_llm.return_value = json.dumps(data)
    assert run(None).overall_pass is True


def test_invalid_evaluation_is_re_asked_once_then_accepted(mock_llm):
    bad = make_result()
    bad["checks"][0]["evidence_quote"] = "Made up text."
    mock_llm.side_effect = [json.dumps(bad), json.dumps(make_result())]
    result = run(None)
    assert result.overall_pass is True
    assert mock_llm.await_count == 2
    second_prompt = mock_llm.await_args_list[1].args[0]
    assert "Your previous evaluation was rejected" in second_prompt
    assert "Made up text." in second_prompt


def test_claim_contradicted_by_lesson_is_rejected_then_corrected(monkeypatch):
    section = {
        "heading": "H",
        "explanation": (
            "Data is often called a 'database.' A database is just a way to store "
            "information in an organized manner."
        ),
        "example": "X",
    }
    lesson = LessonContent.model_validate(
        {
            **make_lesson().model_dump(),
            "sections": [section, section, section],
        }
    )
    bad = jargon_failure("database", "often called a 'database.'")
    mock = AsyncMock(side_effect=[json.dumps(bad), json.dumps(make_result())])
    monkeypatch.setattr(evaluator, "generate_text", mock)

    result = asyncio.run(evaluate_lesson(lesson))

    assert result.overall_pass is True
    assert "defines it" in mock.await_args_list[1].args[0]


def test_persistent_contradiction_is_resolved_in_code(monkeypatch):
    section = {
        "heading": "H",
        "explanation": "Information retrieval is the term used when a system searches.",
        "example": "X",
    }
    lesson = LessonContent.model_validate(
        {**make_lesson().model_dump(), "sections": [section, section, section]}
    )
    bad = jargon_failure("information retrieval", "Information retrieval is the term")
    mock = AsyncMock(return_value=json.dumps(bad))
    monkeypatch.setattr(evaluator, "generate_text", mock)

    result = asyncio.run(evaluate_lesson(lesson))

    assert mock.await_count == 2  # one re-ask, then resolved in code
    assert result.overall_pass is True
    assert result.checks[3].passed is True
    assert result.notes


def test_genuinely_unexplained_term_is_still_reported(mock_llm):
    mock_llm.return_value = json.dumps(jargon_failure("Intro", "Intro to RAG"))
    result = run(None)
    assert result.overall_pass is False
    assert result.checks[3].issues[0].term == "Intro"


@pytest.mark.parametrize(
    ("term", "text", "expected"),
    [
        ("database", "often called a 'database.' A database is just a way", True),
        (
            "information retrieval",
            "'Information retrieval' is the term used when",
            True,
        ),
        ("RAG", "RAG stands for Retrieval-Augmented Generation.", True),
        ("RAG", "RAG is important because it helps.", False),
        (
            "augmented",
            "The word means nothing here, augmented stays unexplained.",
            False,
        ),
        ("retrieval", "Retrieval means finding information.", True),
    ],
)
def test_find_definition_heuristic(term, text, expected):
    assert (find_definition(term, text) is not None) is expected


def test_curly_quotes_normalized():
    from app.agents.evaluator import _normalize

    curly = "Ravi\N{RIGHT SINGLE QUOTATION MARK}s \N{LEFT DOUBLE QUOTATION MARK}x"
    assert _normalize(curly) == "ravi's \"x"


# --- Edge-punctuation tolerance, closest-sentence hint, deduplication -------------


def _lesson_with_sentence(sentence: str) -> LessonContent:
    section = {"heading": "H", "explanation": sentence, "example": "X"}
    return LessonContent.model_validate(
        {**make_lesson().model_dump(), "sections": [section, section, section]}
    )


SENTENCE = (
    "In simple terms, RAG helps a computer to look for information, like "
    "searching in a library, and then use that information to write something."
)


@pytest.mark.parametrize(
    "quote",
    [
        "RAG helps a computer to look for information, like searching in a library.",
        "rag helps a computer to look for information, like searching in a library",
        '"RAG helps a computer to look for information, like searching in a library,"',
        "In simple terms, RAG helps a computer",
    ],
)
def test_quote_edge_punctuation_is_tolerated(monkeypatch, quote):
    data = make_result()
    data["checks"][1]["evidence_quote"] = quote
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    result = asyncio.run(evaluate_lesson(_lesson_with_sentence(SENTENCE)))
    assert result.overall_pass is True


@pytest.mark.parametrize(
    "quote",
    [
        "RAG helps a computer to find information, like searching in a library.",
        "RAG helps a computer to look for information like searching in a library",
        "RAG helps computers to look for information, like searching in a library.",
    ],
)
def test_changed_words_or_inner_punctuation_still_rejected(monkeypatch, quote):
    data = make_result()
    data["checks"][1]["evidence_quote"] = quote
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    with pytest.raises(EvidenceError, match="does not appear"):
        asyncio.run(evaluate_lesson(_lesson_with_sentence(SENTENCE)))


def test_error_and_retry_prompt_show_closest_real_sentence(monkeypatch):
    data = make_result()
    data["checks"][1]["evidence_quote"] = (
        "RAG helps a computer find stuff in a library."
    )
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    with pytest.raises(EvidenceError) as info:
        asyncio.run(evaluate_lesson(_lesson_with_sentence(SENTENCE)))
    assert "Closest text in the lesson" in str(info.value)
    assert "like searching in a library" in str(info.value)
    assert "Closest text in the lesson" in mock.await_args_list[1].args[0]


def test_repeated_problems_are_deduplicated(mock_llm):
    data = make_result(failing=(1,))
    data["checks"][1]["issues"] = [
        {"problem": "p", "quote": "Not in lesson", "term": ""},
        {"problem": "q", "quote": "Not in lesson", "term": ""},
    ]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvidenceError) as info:
        run(None)
    assert str(info.value).count("Not in lesson") == 1


# --- Acronym definitions, deterministic resolution, notes -------------------------

RAG_BODY = (
    "RAG stands for Retrieval-Augmented Generation. It is a method used in AI. "
    "A database is a collection of organized data."
)


def lesson_with(*sentences: str, topic_line: str = "RAG") -> LessonContent:
    base = make_lesson().model_dump()
    section = {"heading": "H", "explanation": " ".join(sentences), "example": "X"}
    return LessonContent.model_validate(
        {**base, "topic": topic_line, "sections": [section, section, section]}
    )


def result_with_issues(issues: list[dict], extra_failing: tuple[int, ...] = ()):
    """An EvaluationResult whose jargon check fails with the given issues."""
    data = make_result(failing=(3, *extra_failing))
    data["checks"][3]["issues"] = issues
    return EvaluationResult.model_validate(
        {**data, "improvement_feedback": "Original feedback."}
    )


def issue(
    term: str, quote: str = "RAG stands for Retrieval-Augmented Generation."
) -> dict:
    return {"problem": f"{term} unexplained", "quote": quote, "term": term}


@pytest.mark.parametrize(
    ("term", "text", "kind_contains"),
    [
        ("RAG", "RAG stands for Retrieval-Augmented Generation.", ""),
        ("RAG", "RAG means Retrieval-Augmented Generation.", ""),
        (
            "Retrieval-Augmented Generation",
            "RAG stands for Retrieval-Augmented Generation.",
            "stands for",
        ),
        (
            "Retrieval-Augmented Generation",
            "RAG means Retrieval-Augmented Generation.",
            "stands for",
        ),
        ("RAG", "Retrieval-Augmented Generation (RAG) is a method.", "Long Form"),
        (
            "Retrieval-Augmented Generation",
            "Retrieval-Augmented Generation (RAG) is a method.",
            "Long Form",
        ),
        ("RAG", "RAG (Retrieval-Augmented Generation) is a method.", "ACRONYM"),
        (
            "Retrieval-Augmented Generation",
            "RAG (Retrieval-Augmented Generation) is a method.",
            "ACRONYM",
        ),
        ("AI", "Artificial Intelligence (AI) is everywhere.", "Long Form"),
        (
            "database",
            "A database (a collection of organized data) is used.",
            "explanation",
        ),
        (
            "retrieval augmented generation",
            "RAG stands for Retrieval-Augmented Generation.",
            "stands for",
        ),
    ],
)
def test_acronym_and_plain_definitions_recognized(term, text, kind_contains):
    found = find_definition(term, text)
    assert found is not None
    assert kind_contains in found.kind
    assert found.sentence in text


@pytest.mark.parametrize(
    ("term", "text"),
    [
        ("RAG", "RAG is important because it helps."),
        ("RAG", "RAG (see below) is covered later."),
        ("Retrieval", "Information retrieval is the term used when searching."),
        ("Augmented", "RAG stands for Retrieval-Augmented Generation."),
        ("AI", "Retrieval-Augmented Generation (RAG) is a method."),
        ("RAG", "Retrieval-Augmented Generation (XYZ) is a method."),
        ("LLM", "RAG stands for Retrieval-Augmented Generation."),
        ("", "RAG stands for Retrieval-Augmented Generation."),
    ],
)
def test_non_definitions_are_not_recognized(term, text):
    assert find_definition(term, text) is None


def test_definition_sentence_keeps_original_capitalization():
    found = find_definition(
        "rag", "Intro. RAG stands for Retrieval-Augmented Generation."
    )
    assert found.sentence == "RAG stands for Retrieval-Augmented Generation."


def test_label_lines_are_not_body_text():
    lesson = lesson_with(
        "Body sentence.", topic_line="RAG (Retrieval-Augmented Generation)"
    )
    body = body_text(lesson)
    assert "Title:" not in body
    assert "Topic:" not in body
    assert "Learner level:" not in body
    assert "Section 1:" not in body
    assert "Body sentence." in body


def test_definition_only_in_topic_label_does_not_count():
    lesson = lesson_with(
        "Machines look things up.", topic_line="RAG (Retrieval-Augmented Generation)"
    )
    result = result_with_issues([issue("RAG", "Machines look things up.")])
    _, contradicted = collect_evidence_problems(result, lesson)
    assert contradicted == []
    assert resolve_contradictions(result, lesson) == result


def test_resolve_sole_issue_passes_jargon_and_recomputes_everything():
    lesson = lesson_with(RAG_BODY)
    result = result_with_issues([issue("RAG")])
    assert result.overall_pass is False

    resolved = resolve_contradictions(result, lesson)

    check = resolved.checks[3]
    assert check.passed is True
    assert check.issues == []
    assert check.evidence_quote == "RAG stands for Retrieval-Augmented Generation."
    assert resolved.overall_pass is True
    assert resolved.failure_reasons == []
    assert resolved.improvement_feedback == ""
    assert len(resolved.notes) == 3
    assert "'RAG'" in resolved.notes[0]
    assert "RAG stands for Retrieval-Augmented Generation." in resolved.notes[0]
    assert "contradicted" in resolved.notes[0]
    assert "marked passed" in resolved.notes[1]
    assert "recomputed" in resolved.notes[2]
    # The original object is not mutated.
    assert result.overall_pass is False
    assert result.notes == []


def test_resolve_mixed_issues_keeps_jargon_failed():
    lesson = lesson_with(RAG_BODY)
    result = result_with_issues([issue("RAG"), issue("Augmented", "It is a method")])

    resolved = resolve_contradictions(result, lesson)

    check = resolved.checks[3]
    assert check.passed is False
    assert [i.term for i in check.issues] == ["Augmented"]
    assert "Augmented" in check.improvement_suggestion
    assert "RAG" not in check.improvement_suggestion
    assert resolved.overall_pass is False
    assert resolved.failure_reasons == [f"{CRITERIA[3]}: {check.reason}"]
    assert "Augmented" in resolved.improvement_feedback
    assert "Original feedback." not in resolved.improvement_feedback
    assert len(resolved.notes) == 2  # one resolution + the recompute note


def test_resolve_keeps_other_failed_checks_and_rebuilds_feedback():
    lesson = lesson_with(RAG_BODY)
    result = result_with_issues([issue("RAG")], extra_failing=(6,))

    resolved = resolve_contradictions(result, lesson)

    assert resolved.checks[3].passed is True
    assert resolved.checks[6].passed is False
    assert resolved.overall_pass is False
    assert resolved.failure_reasons == [f"{CRITERIA[6]}: Because."]
    assert resolved.improvement_feedback.startswith(f"{CRITERIA[6]}: Because.")
    assert "Fix: Fix it." in resolved.improvement_feedback
    assert CRITERIA[3] not in resolved.improvement_feedback
    assert [c.passed for c in resolved.checks].count(False) == 1


def test_resolve_leaves_unrelated_checks_untouched():
    lesson = lesson_with(RAG_BODY)
    result = result_with_issues([issue("RAG")], extra_failing=(1,))
    resolved = resolve_contradictions(result, lesson)
    assert resolved.checks[1] == result.checks[1]
    assert resolved.checks[0] == result.checks[0]


def test_resolve_does_nothing_when_no_term_is_defined():
    lesson = lesson_with("Machines look things up.")
    result = result_with_issues([issue("Augmented", "Machines look things up.")])
    assert resolve_contradictions(result, lesson) is result


def test_resolve_does_not_treat_a_bare_mention_as_a_definition():
    lesson = lesson_with("RAG is important because it helps.")
    result = result_with_issues([issue("RAG", "RAG is important because it helps.")])
    assert resolve_contradictions(result, lesson) is result


def test_retry_prompt_has_original_sentence_and_removal_instruction(monkeypatch):
    lesson = lesson_with(RAG_BODY)
    bad = result_with_issues([issue("RAG")]).model_dump()
    mock = AsyncMock(return_value=json.dumps(bad))
    monkeypatch.setattr(evaluator, "generate_text", mock)

    asyncio.run(evaluate_lesson(lesson))

    retry_prompt = mock.await_args_list[1].args[0]
    assert '"RAG stands for Retrieval-Augmented Generation."' in retry_prompt
    assert "Remove this issue" in retry_prompt
    assert "mark that check as passed" in retry_prompt


def test_second_attempt_with_invalid_quote_and_contradiction_still_raises(monkeypatch):
    lesson = lesson_with(RAG_BODY)
    data = result_with_issues([issue("RAG")]).model_dump()
    data["checks"][0]["evidence_quote"] = "Invented sentence."
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    with pytest.raises(EvidenceError, match="does not appear"):
        asyncio.run(evaluate_lesson(lesson))


def test_invented_quote_in_a_resolvable_issue_is_not_resolved(monkeypatch):
    lesson = lesson_with(RAG_BODY)
    data = result_with_issues([issue("RAG", "Made up quote")]).model_dump()
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    with pytest.raises(EvidenceError, match="does not appear"):
        asyncio.run(evaluate_lesson(lesson))


def test_resolution_through_evaluate_lesson_records_notes(monkeypatch):
    lesson = lesson_with(RAG_BODY)
    bad = result_with_issues([issue("RAG")]).model_dump()
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(bad))
    )
    result = asyncio.run(evaluate_lesson(lesson))
    assert result.overall_pass is True
    assert any("Resolved unexplained-term issue 'RAG'" in n for n in result.notes)


def test_model_supplied_notes_are_ignored(mock_llm):
    data = make_result()
    data["notes"] = ["The model says everything is fine."]
    mock_llm.return_value = json.dumps(data)
    assert run(None).notes == []


def test_api_schema_omits_code_only_notes_and_stays_strict():
    schema = evaluator.api_schema()
    assert "notes" not in schema["properties"]
    assert "notes" not in schema["required"]
    assert schema["required"] == list(schema["properties"])
    assert "notes" in to_strict_json_schema(EvaluationResult)["properties"]


def test_rubric_says_labels_are_not_first_use():
    from app.evaluation.rubric import get_item

    text = " ".join(get_item(CRITERIA[3]).pass_when)
    assert "labels" in text
    assert "lesson body" in text


# --- Phase 1: mandatory concepts ---------------------------------------------------

from app.agents.evaluator import apply_concept_gaps  # noqa: E402
from app.evaluation.schemas import ConceptCheck  # noqa: E402
from app.evaluation.topics import RAG_REQUIREMENTS  # noqa: E402

RAG_SENTENCES = (
    "Retrieval-Augmented Generation (RAG) lets a language model (LLM) use recent "
    "and private documents. A language model is an AI system that writes text, and "
    "it can make up answers. First, retrieval finds documents in a knowledge "
    "collection. Then the context is given to the model, which writes the answer. "
    "RAG does not guarantee accuracy because documents can be incomplete or "
    "outdated. Step by step, that is the whole process."
)
EVIDENCE = "A language model is an AI system that writes text"


def concept_entries(uncovered: tuple[str, ...] = ()) -> list[dict]:
    return [
        {
            "concept_id": c.id,
            "covered": c.id not in uncovered,
            "evidence_quote": "" if c.id in uncovered else EVIDENCE,
            "missing_detail": "Not taught at all." if c.id in uncovered else "",
        }
        for c in RAG_REQUIREMENTS.concepts
    ]


def rag_result(uncovered=(), failing=(), body=RAG_SENTENCES) -> tuple:
    data = make_result(failing=failing)
    data["concept_checks"] = concept_entries(uncovered)
    return data, lesson_with(body)


def run_with_concepts(monkeypatch, data, lesson):
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))
    return result, mock


def test_all_concepts_covered_leaves_verdict_untouched(monkeypatch):
    data, lesson = rag_result()
    result, mock = run_with_concepts(monkeypatch, data, lesson)
    assert result.overall_pass is True
    assert result.notes == []
    assert len(result.concept_checks) == 8
    assert mock.await_count == 1


def test_missing_concept_forces_completeness_to_fail(monkeypatch):
    data, lesson = rag_result(uncovered=("augmentation",))
    result, _ = run_with_concepts(monkeypatch, data, lesson)

    completeness = result.checks[6]
    assert completeness.passed is False
    assert "Augmentation" in completeness.reason
    assert "Providing the retrieved information as context" in (
        completeness.improvement_suggestion
    )
    assert any("Augmentation" in i.problem for i in completeness.issues)
    assert result.overall_pass is False
    assert any(CRITERIA[6] in r for r in result.failure_reasons)
    assert "Augmentation" in result.improvement_feedback
    assert any("marked failed by code" in n for n in result.notes)
    assert all(c.passed for i, c in enumerate(result.checks) if i != 6)


def test_missing_concept_extends_an_existing_completeness_failure(monkeypatch):
    data, lesson = rag_result(uncovered=("limitations",), failing=(6,))
    result, _ = run_with_concepts(monkeypatch, data, lesson)
    completeness = result.checks[6]
    assert completeness.passed is False
    assert len(completeness.issues) == 2  # the model's issue + the missing concept
    assert "Because." in completeness.reason
    assert "Limitations" in completeness.reason
    assert completeness.improvement_suggestion.startswith("Fix it.")
    assert "Limitations" in completeness.improvement_suggestion


def test_several_missing_concepts_are_all_named(monkeypatch):
    data, lesson = rag_result(uncovered=("why_needed", "language_model"))
    result, _ = run_with_concepts(monkeypatch, data, lesson)
    problems = " ".join(i.problem for i in result.checks[6].issues)
    assert "Why RAG is needed" in problems
    assert "language model (LLM)" in problems


def test_evaluator_marked_missing_but_signal_present_is_still_failed(monkeypatch):
    data, lesson = rag_result(uncovered=("language_model",))
    result, _ = run_with_concepts(monkeypatch, data, lesson)
    assert result.checks[6].passed is False  # the evaluator's word is not overridden


def test_wrong_concept_ids_are_re_asked_then_rejected(monkeypatch):
    data, lesson = rag_result()
    data["concept_checks"].pop()
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    with pytest.raises(EvidenceError, match="concept ids"):
        asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))
    assert mock.await_count == 2


def test_duplicate_concept_id_rejected(monkeypatch):
    data, lesson = rag_result()
    data["concept_checks"][1] = dict(data["concept_checks"][0])
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    with pytest.raises(EvidenceError, match="concept ids"):
        asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))


def test_invented_concept_evidence_quote_rejected(monkeypatch):
    data, lesson = rag_result()
    data["concept_checks"][0]["evidence_quote"] = "A sentence nobody wrote."
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    with pytest.raises(EvidenceError, match="concept 'definition'"):
        asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))


def test_concept_checks_must_be_empty_without_requirements(mock_llm):
    data = make_result()
    data["concept_checks"] = concept_entries()
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(EvidenceError, match="empty list"):
        run(None)


def test_prompt_lists_concepts_or_says_none_apply():
    with_req = evaluator.build_prompt(make_lesson(), RAG_REQUIREMENTS)
    assert "Mandatory concepts" in with_req
    assert "[language_model]" in with_req
    assert "public internet" in with_req
    without = evaluator.build_prompt(make_lesson())
    assert "No mandatory concepts apply" in without


def test_api_schema_requires_concept_checks():
    schema = evaluator.api_schema()
    assert "concept_checks" in schema["properties"]
    assert "concept_checks" in schema["required"]


def test_apply_concept_gaps_without_requirements_is_a_no_op():
    data = make_result()
    result = EvaluationResult.model_validate(data)
    assert apply_concept_gaps(result, None) is result


def test_concept_check_validators():
    with pytest.raises(ValueError, match="needs a quote"):
        ConceptCheck(concept_id="x", covered=True)
    with pytest.raises(ValueError, match="needs detail"):
        ConceptCheck(concept_id="x", covered=False)
    assert ConceptCheck(concept_id="x", covered=True, evidence_quote="q").covered
    assert not ConceptCheck(concept_id="x", covered=False, missing_detail="d").covered


# --- Phase 2: accuracy failures must name the claim and the correction ------------


def accuracy_failure(quote: str, correction: str) -> dict:
    data = make_result(failing=(0,))
    data["checks"][0]["issues"] = [
        {"problem": "Overclaim.", "quote": quote, "term": "", "correction": correction}
    ]
    return data


def test_accuracy_failure_with_quote_and_correction_is_accepted(mock_llm):
    mock_llm.return_value = json.dumps(
        accuracy_failure("Intro to RAG", "RAG does not guarantee accuracy.")
    )
    result = run(None)
    assert result.checks[0].passed is False
    assert result.checks[0].issues[0].correction == "RAG does not guarantee accuracy."


@pytest.mark.parametrize(("quote", "correction"), [("", "Fix."), ("Intro to RAG", "")])
def test_accuracy_failure_without_quote_or_correction_rejected(
    mock_llm, quote, correction
):
    mock_llm.return_value = json.dumps(accuracy_failure(quote, correction))
    with pytest.raises(EvaluationValidationError, match="accuracy issue"):
        run(None)


def test_accuracy_quote_must_exist_in_the_lesson(mock_llm):
    mock_llm.return_value = json.dumps(
        accuracy_failure("RAG guarantees correct answers.", "It does not.")
    )
    with pytest.raises(EvidenceError, match="does not appear"):
        run(None)


def test_non_accuracy_failures_do_not_need_a_correction(mock_llm):
    data = make_result(failing=(2,))
    data["checks"][2]["issues"][0]["correction"] = ""
    mock_llm.return_value = json.dumps(data)
    assert run(None).checks[2].passed is False


def test_accuracy_rubric_lists_the_guardrail_conditions():
    from app.evaluation.rubric import get_item

    text = " ".join(get_item(CRITERIA[0]).pass_when)
    assert "commercial products" in text
    assert "speculative predictions" in text
    assert "guarantees" in text
    assert "training or updating a model" in text
    assert "exact problematic claim" in text


def test_evaluator_instructions_require_exact_claim_and_correction():
    text = evaluator.SYSTEM_INSTRUCTIONS
    assert "exact problematic" in text
    assert '"correction"' in text


# --- Phase 3: precise feedback and debug logging -----------------------------------


def test_feedback_names_exact_text_correction_and_fix(mock_llm):
    data = make_result(failing=(0,))
    data["checks"][0]["issues"] = [
        {
            "problem": "Overclaim.",
            "quote": "Intro to RAG",
            "term": "",
            "correction": "RAG does not guarantee accuracy.",
        }
    ]
    mock_llm.return_value = json.dumps(data)
    feedback = run(None).improvement_feedback
    assert f"{CRITERIA[0]}:" in feedback
    assert '- Overclaim. Text: "Intro to RAG"' in feedback
    assert "Correction: RAG does not guarantee accuracy." in feedback
    assert "Fix: Fix it." in feedback


def test_feedback_is_empty_when_everything_passes(mock_llm):
    assert run(None).improvement_feedback == ""


@pytest.fixture
def events(monkeypatch):
    captured = []
    monkeypatch.setattr(
        evaluator, "log_event", lambda name, **fields: captured.append((name, fields))
    )
    return captured


def test_response_is_logged_for_every_attempt(mock_llm, events):
    run(None)
    names = [name for name, _ in events]
    assert names == ["evaluator_response"]
    assert events[0][1]["attempt"] == 1
    assert "overall_pass" in events[0][1]["raw"]


def test_reask_and_evidence_problems_are_logged(mock_llm, events):
    bad = make_result()
    bad["checks"][0]["evidence_quote"] = "Made up text."
    mock_llm.side_effect = [json.dumps(bad), json.dumps(make_result())]
    run(None)
    names = [name for name, _ in events]
    assert names == [
        "evaluator_response",
        "evaluator_evidence_problems",
        "evaluator_reask",
        "evaluator_response",
    ]
    assert "Made up text." in events[1][1]["invalid"][0]
    assert "Made up text." in events[2][1]["feedback"]


def test_validation_failures_are_logged(mock_llm, events):
    mock_llm.return_value = "not json"
    with pytest.raises(InvalidEvaluationJSONError):
        run(None)
    names = [name for name, _ in events]
    assert names.count("evaluator_validation_failure") == 2
    assert names.count("evaluator_reask") == 1
    failure = next(f for n, f in events if n == "evaluator_validation_failure")
    assert failure["error_type"] == "InvalidEvaluationJSONError"


def test_resolution_notes_are_logged(monkeypatch, events):
    lesson = lesson_with(RAG_BODY)
    bad = result_with_issues([issue("RAG")]).model_dump()
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(bad))
    )
    asyncio.run(evaluate_lesson(lesson))
    notes = [f for n, f in events if n == "evaluator_notes"]
    assert notes
    assert any("Resolved unexplained-term issue" in n for n in notes[0]["notes"])


# --- Fix 2: hints support the verdict; a missing hint asks for confirmation --------

from app.agents.evaluator import find_unconfirmed_concepts  # noqa: E402

DIFFERENT_WORDING = (
    "RAG (Retrieval-Augmented Generation) is a way to help an AI answer questions. "
    "An AI chat program only knows what it studied in the past, so it cannot know "
    "about last week's news or your own company files, and it may guess wrongly. "
    "A large AI text program, the kind that composes sentences, is the answering "
    "part. Before it answers, a search step looks through a library of saved "
    "articles and picks the most useful passages. Those passages are pasted into "
    "the question so the AI can see them. After that, the AI composes a reply "
    "using what it was shown. Step 1 is the search, step 2 is the pasting, step 3 "
    "is the reply. The search can miss things, and the saved articles can be "
    "stale, so answers might still be unreliable."
)
SEMANTIC_QUOTE = (
    "A large AI text program, the kind that composes sentences, is the answering part"
)
FILLER = "First we begin. Then we continue. Next we go on. This has context and write."


def covered_result(quote: str, uncovered: tuple[str, ...] = ()) -> dict:
    data = make_result()
    data["concept_checks"] = [
        {
            "concept_id": c.id,
            "covered": c.id not in uncovered,
            "evidence_quote": "" if c.id in uncovered else quote,
            "missing_detail": "Not taught." if c.id in uncovered else "",
        }
        for c in RAG_REQUIREMENTS.concepts
    ]
    return data


def evaluate_sequence(monkeypatch, lesson, *responses):
    mock = AsyncMock(side_effect=[json.dumps(r) for r in responses])
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))
    return result, mock


def test_correct_wording_without_hints_is_not_failed_outright():
    lesson = lesson_with(DIFFERENT_WORDING)
    unconfirmed = find_unconfirmed_concepts(
        EvaluationResult.model_validate(covered_result(SEMANTIC_QUOTE)),
        lesson,
        RAG_REQUIREMENTS,
    )
    assert [c.id for c in unconfirmed] == ["language_model"]


def test_unconfirmed_concept_triggers_one_confirmation_then_is_accepted(monkeypatch):
    lesson = lesson_with(DIFFERENT_WORDING)
    response = covered_result(SEMANTIC_QUOTE)
    result, mock = evaluate_sequence(monkeypatch, lesson, response, response)

    assert mock.await_count == 2
    assert result.overall_pass is True
    assert all(c.covered for c in result.concept_checks)
    assert any(
        "What a language model (LLM) is" in n and "hints did not confirm" in n
        for n in result.notes
    )
    retry = mock.await_args_list[1].args[0]
    assert "mandatory concept 'What a language model (LLM) is'" in retry
    assert "Re-check it" in retry
    assert "expected wording" not in retry
    assert "hint" not in retry.lower().replace("hints", "")


def test_confirmation_can_flip_the_concept_to_missing(monkeypatch):
    lesson = lesson_with(DIFFERENT_WORDING)
    first = covered_result(SEMANTIC_QUOTE)
    second = covered_result(SEMANTIC_QUOTE, uncovered=("language_model",))
    result, mock = evaluate_sequence(monkeypatch, lesson, first, second)

    assert mock.await_count == 2
    assert result.overall_pass is False
    assert result.checks[6].passed is False
    entry = next(c for c in result.concept_checks if c.concept_id == "language_model")
    assert entry.covered is False
    assert "What a language model (LLM) is" in result.checks[6].reason


def test_invalid_quote_still_uses_the_validation_path(monkeypatch):
    lesson = lesson_with(DIFFERENT_WORDING)
    bad = covered_result("A sentence that is not in the lesson")
    mock = AsyncMock(return_value=json.dumps(bad))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    with pytest.raises(EvidenceError, match="does not appear"):
        asyncio.run(evaluate_lesson(lesson, requirements=RAG_REQUIREMENTS))
    assert mock.await_count == 2


def test_hints_present_means_no_confirmation_question(monkeypatch):
    data, lesson = rag_result()
    result, mock = evaluate_sequence(monkeypatch, lesson, data)
    assert mock.await_count == 1
    assert result.notes == []


def test_filler_with_generic_words_is_unconfirmed_for_every_concept():
    lesson = lesson_with(FILLER)
    result = EvaluationResult.model_validate(covered_result("First we begin"))
    unconfirmed = find_unconfirmed_concepts(result, lesson, RAG_REQUIREMENTS)
    assert [c.id for c in unconfirmed] == [c.id for c in RAG_REQUIREMENTS.concepts]


def test_filler_is_questioned_and_every_acceptance_is_noted(monkeypatch):
    lesson = lesson_with(FILLER)
    response = covered_result("First we begin")
    result, mock = evaluate_sequence(monkeypatch, lesson, response, response)
    assert mock.await_count == 2  # generic words never count as proof
    assert sum("hints did not confirm" in n for n in result.notes) == 8


def test_hint_in_a_different_section_does_not_confirm_the_quoted_one():
    plain = {
        "heading": "A",
        "explanation": "A big AI program writes text.",
        "example": "X",
    }
    other = {
        "heading": "B",
        "explanation": "A language model is mentioned here.",
        "example": "X",
    }
    base = make_lesson().model_dump()
    lesson = LessonContent.model_validate({**base, "sections": [plain, other, other]})

    result = EvaluationResult.model_validate(
        covered_result("A big AI program writes text")
    )
    unconfirmed = [
        c.id for c in find_unconfirmed_concepts(result, lesson, RAG_REQUIREMENTS)
    ]
    assert "language_model" in unconfirmed

    result = EvaluationResult.model_validate(
        covered_result("A language model is mentioned here")
    )
    unconfirmed = [
        c.id for c in find_unconfirmed_concepts(result, lesson, RAG_REQUIREMENTS)
    ]
    assert "language_model" not in unconfirmed


def test_substring_collisions_do_not_confirm_a_concept():
    body = "A willmore bullmarket story. It was authentic, firsthand and inventive."
    lesson = lesson_with(body)
    result = EvaluationResult.model_validate(
        covered_result("A willmore bullmarket story")
    )
    ids = [c.id for c in find_unconfirmed_concepts(result, lesson, RAG_REQUIREMENTS)]
    assert "language_model" in ids
    assert "why_needed" in ids


def test_no_confirmation_without_requirements():
    result = EvaluationResult.model_validate(make_result())
    assert find_unconfirmed_concepts(result, make_lesson(), None) == []


def test_missing_concept_still_fails_completeness_even_if_hints_are_present(
    monkeypatch,
):
    data, lesson = rag_result(uncovered=("language_model",))
    result, _ = run_with_concepts(monkeypatch, data, lesson)
    assert result.checks[6].passed is False
    assert not any("hints did not confirm" in n for n in result.notes)


def test_generator_prompt_does_not_list_hint_keywords():
    from app.agents.generator import build_prompt
    from app.evaluation.topics import RAG_REQUIREMENTS as REQ

    prompt = build_prompt("Introduction to RAG", REQ).lower()
    for hint in ("pasted into", "handed over", "knowledge base", "searches through"):
        assert hint not in prompt
    for word in ("keyword", "signal terms", "expected wording"):
        assert word not in prompt


def test_logged_events_never_contain_prompts(mock_llm, events):
    bad = make_result()
    bad["checks"][0]["evidence_quote"] = "Made up text."
    mock_llm.side_effect = [json.dumps(bad), json.dumps(make_result())]
    run(None)
    allowed = {
        "attempt",
        "raw",
        "feedback",
        "invalid",
        "contradicted",
        "unconfirmed_concepts",
        "error_type",
        "message",
        "notes",
    }
    for _, fields in events:
        assert set(fields) <= allowed
        assert "Evaluate the following lesson" not in json.dumps(fields)
