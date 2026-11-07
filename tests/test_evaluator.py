import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import evaluator
from app.agents.evaluator import (
    EvaluationValidationError,
    InvalidEvaluationJSONError,
    body_text,
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


def test_an_invented_evidence_quote_is_never_evidence_and_never_an_error(mock_llm):
    from app.agents.evaluator import _normalize

    data = make_result()
    data["checks"][0]["evidence_quote"] = "This sentence is not in the lesson."
    mock_llm.return_value = json.dumps(data)

    result = run(None)

    assert mock_llm.await_count == 1  # no re-ask, no error
    quote = result.checks[0].evidence_quote
    assert quote != "This sentence is not in the lesson."
    assert _normalize(quote) in _normalize(render_lesson(make_lesson()))  # real text
    assert any("not in the lesson" in n for n in result.notes)
    assert result.overall_pass is True  # the verdict is the evaluator's judgement


def test_quote_matching_ignores_case_whitespace_curly_quotes_and_ellipsis(mock_llm):
    data = make_result()
    data["checks"][0]["evidence_quote"] = "INTRO   to RAG ... Topic: RAG"
    mock_llm.return_value = json.dumps(data)
    assert run(None).overall_pass is True


def test_a_malformed_response_is_retried_once_then_accepted(mock_llm):
    mock_llm.side_effect = ["this is not json", json.dumps(make_result())]
    result = run(None)
    assert result.overall_pass is True
    assert mock_llm.await_count == 2
    second_prompt = mock_llm.await_args_list[1].args[0]
    assert "It was rejected for these specific problems" in second_prompt
    assert "not valid JSON" in second_prompt
    # The previous response is carried along so unrelated verdicts stay put.
    assert "Your previous response was" in second_prompt
    assert "this is not json" in second_prompt
    assert "Change ONLY what these problems require" in second_prompt


def test_claim_contradicted_by_lesson_is_resolved_without_a_reask(monkeypatch):
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
    mock = AsyncMock(return_value=json.dumps(bad))
    monkeypatch.setattr(evaluator, "generate_text", mock)

    result = asyncio.run(evaluate_lesson(lesson))

    assert mock.await_count == 1  # a re-ask destabilizes verdicts: none is sent
    assert result.overall_pass is True
    assert any("database" in n for n in result.notes)


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

    assert mock.await_count == 1  # resolved in code, no re-ask
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
def test_changed_words_are_never_evidence_but_do_not_end_the_run(monkeypatch, quote):
    data = make_result()
    data["checks"][1]["evidence_quote"] = quote
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(_lesson_with_sentence(SENTENCE)))
    assert mock.await_count == 1
    assert result.checks[1].evidence_quote != quote
    assert "searching in a library" in result.checks[1].evidence_quote  # real text
    assert any(
        "not in the lesson" in n or "corrected to the exact" in n for n in result.notes
    )


def test_a_dissimilar_quote_is_replaced_by_the_closest_real_sentence(monkeypatch):
    data = make_result()
    data["checks"][1]["evidence_quote"] = (
        "RAG helps a computer find stuff in a library."
    )
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    result = asyncio.run(evaluate_lesson(_lesson_with_sentence(SENTENCE)))
    assert "like searching in a library" in result.checks[1].evidence_quote


def test_an_unverifiable_issue_quote_in_other_criteria_keeps_the_claim(mock_llm):
    data = make_result(failing=(1,))
    data["checks"][1]["issues"] = [
        {"problem": "p", "quote": "Not in lesson", "term": "", "correction": ""}
    ]
    mock_llm.return_value = json.dumps(data)
    result = run(None)
    issue = result.checks[1].issues[0]
    assert result.checks[1].passed is False  # the judgement is kept
    assert issue.quote == ""  # but the unverifiable quote is not
    assert any("kept without the quote" in n for n in result.notes)


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


def test_invented_evidence_and_a_provable_contradiction_are_both_handled(monkeypatch):
    lesson = lesson_with(RAG_BODY)
    data = result_with_issues([issue("RAG")]).model_dump()
    data["checks"][0]["evidence_quote"] = "Invented sentence."
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(lesson))
    assert mock.await_count == 1
    assert result.overall_pass is True
    assert any("Resolved unexplained-term issue 'RAG'" in n for n in result.notes)
    assert any("not in the lesson" in n for n in result.notes)


def test_a_jargon_issue_with_an_invented_quote_is_dropped_as_unverifiable(
    monkeypatch,
):
    lesson = lesson_with(RAG_BODY)
    data = result_with_issues([issue("RAG", "Made up quote")]).model_dump()
    monkeypatch.setattr(
        evaluator, "generate_text", AsyncMock(return_value=json.dumps(data))
    )
    result = asyncio.run(evaluate_lesson(lesson))
    assert result.checks[3].passed is True
    notes = " ".join(result.notes)
    assert "dropped an issue whose quote is not in the lesson" in notes
    assert "Resolved unexplained-term issue" not in notes  # it was never verified


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


def test_an_accuracy_issue_quoting_text_that_is_not_there_is_dropped(mock_llm):
    mock_llm.return_value = json.dumps(
        accuracy_failure("RAG guarantees correct answers.", "It does not.")
    )
    result = run(None)
    assert result.checks[0].passed is True
    assert any(
        "dropped an issue whose quote is not in the lesson" in n for n in result.notes
    )


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


def test_reask_and_validation_failures_are_logged(mock_llm, events):
    mock_llm.side_effect = ["not json", json.dumps(make_result())]
    run(None)
    names = [name for name, _ in events]
    assert names == [
        "evaluator_response",
        "evaluator_validation_failure",
        "evaluator_reask",
        "evaluator_response",
    ]
    assert events[1][1]["error_type"] == "InvalidEvaluationJSONError"
    assert "not valid JSON" in events[2][1]["feedback"]


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


def test_logged_events_never_contain_prompts(mock_llm, events):
    mock_llm.side_effect = ["not json", json.dumps(make_result())]
    run(None)
    allowed = {"attempt", "raw", "feedback", "error_type", "message", "notes"}
    for _, fields in events:
        assert set(fields) <= allowed
        assert "Evaluate the following lesson" not in json.dumps(fields)


# --- Final cycle: snapping quotes, negation and acronym recognisers -----------------

from app.agents.evaluator import (  # noqa: E402
    quote_negates_claim,
    sanitize_quotes,
    snap_quote,
)
from app.evaluation.schemas import EvaluationIssue  # noqa: E402

SNAP_TEXT = (
    "It helps to create more accurate responses by using real-time information. "
    "A language model (often shortened to LLM) is an AI system."
)


def test_a_tidied_quote_snaps_to_the_exact_sentence():
    quote = (
        "RAG helps to create more accurate responses by using real-time information."
    )
    assert snap_quote(quote, SNAP_TEXT) == (
        "It helps to create more accurate responses by using real-time information."
    )


def test_a_quote_that_starts_a_longer_sentence_snaps_to_it():
    text = "Retrieval finds the most relevant documents for the question asked."
    assert snap_quote("Retrieval finds the most relevant documents", text) == text


@pytest.mark.parametrize(
    "quote", ["The moon is made of cheese.", "RAG trains the model overnight.", ""]
)
def test_dissimilar_or_empty_quotes_do_not_snap(quote):
    assert snap_quote(quote, SNAP_TEXT) is None


def test_sanitize_corrects_a_near_verbatim_passed_evidence_quote():
    lesson = _lesson_with_sentence("It helps to create more accurate responses.")
    result = EvaluationResult.model_validate(make_result())
    data = result.model_dump()
    data["checks"][0]["evidence_quote"] = "RAG helps to create more accurate responses."
    fixed = sanitize_quotes(EvaluationResult.model_validate(data), lesson)
    assert fixed.checks[0].evidence_quote == (
        "It helps to create more accurate responses."
    )
    assert any("corrected to the exact lesson sentence" in n for n in fixed.notes)


def test_sanitize_leaves_a_result_with_real_quotes_untouched():
    result = EvaluationResult.model_validate(make_result())
    assert sanitize_quotes(result, make_lesson()) is result


def test_sanitize_marks_a_check_passed_when_every_issue_was_unverifiable():
    data = make_result(failing=(0,))
    data["checks"][0]["issues"] = [
        {
            "problem": "p",
            "quote": "Nothing like this appears anywhere",
            "term": "",
            "correction": "c",
        }
    ]
    result = sanitize_quotes(EvaluationResult.model_validate(data), make_lesson())
    assert result.checks[0].passed is True
    assert any("none of its issues could be verified" in n for n in result.notes)
    assert result.overall_pass is True


@pytest.mark.parametrize(
    ("quote", "problem", "reason", "expected"),
    [
        (
            "RAG does not guarantee that every answer will be accurate.",
            "Incorrect claim about guarantees.",
            "The lesson claims that RAG can guarantee accuracy.",
            True,
        ),
        (
            "RAG cannot guarantee correct answers.",
            "Claims a guarantee.",
            "The lesson claims a guarantee.",
            True,
        ),
        (  # a real overclaim: no negation
            "RAG guarantees that every answer will be accurate.",
            "Incorrect claim about guarantees.",
            "The lesson claims that RAG can guarantee accuracy.",
            False,
        ),
        (  # negation, but about something else
            "RAG does not train the model, and it uses real-time information.",
            "Claims real-time information.",
            "The lesson claims real-time information.",
            False,
        ),
        ("", "Claims a guarantee.", "Claims a guarantee.", False),
    ],
)
def test_quote_negates_claim(quote, problem, reason, expected):
    issue = EvaluationIssue(problem=problem, quote=quote, term="", correction="c")
    assert quote_negates_claim(issue, reason) is expected


def test_a_negated_sentence_flagged_as_a_guarantee_is_resolved_in_one_call(
    monkeypatch,
):
    claim = "RAG does not guarantee that every answer will be accurate."
    lesson = _lesson_with_sentence(claim)
    data = make_result(failing=(0,))
    data["checks"][0]["reason"] = "The lesson claims that RAG can guarantee accuracy."
    data["checks"][0]["issues"] = [
        {
            "problem": "Incorrect claim about guarantees.",
            "quote": claim,
            "term": "",
            "correction": "RAG does not guarantee accuracy; it may use outdated data.",
        }
    ]
    mock = AsyncMock(return_value=json.dumps(data))
    monkeypatch.setattr(evaluator, "generate_text", mock)
    result = asyncio.run(evaluate_lesson(lesson))
    assert mock.await_count == 1
    assert result.checks[0].passed is True
    assert any("already negates the claim" in n for n in result.notes)


@pytest.mark.parametrize(
    ("term", "text"),
    [
        ("LLM", "A language model (LLM) is an AI system."),
        ("LLM", "A language model (often shortened to LLM) is an AI system."),
        ("LLM", "A language model, also known as LLM, is an AI system."),
        ("LLM", "Language models, abbreviated as LLM, write text."),
        ("LLM", "A large language model (LLM) is an AI system."),
        ("AI", "Artificial Intelligence (AI) is everywhere."),
        ("language model", "A language model (LLM) is an AI system."),
    ],
)
def test_acronym_introductions_are_recognised_as_definitions(term, text):
    assert find_definition(term, text) is not None


@pytest.mark.parametrize(
    ("term", "text"),
    [
        ("LLM", "LLM is important."),
        ("RAG", "A language model (LLM) is an AI system."),
        ("LLM", "The cat (LLM) sat down."),
    ],
)
def test_bare_mentions_and_unrelated_parentheses_are_not_definitions(term, text):
    assert find_definition(term, text) is None
