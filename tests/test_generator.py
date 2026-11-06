import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from app.agents import generator
from app.agents.generator import (
    InvalidLessonJSONError,
    InvalidTopicError,
    LessonValidationError,
    generate_lesson,
)
from app.evaluation.schemas import LessonContent
from app.llm import LLMError


def valid_lesson() -> dict:
    section = {"heading": "H", "explanation": "E", "example": "X"}
    return {
        "title": "Intro to RAG",
        "topic": "RAG",
        "learner_level": "Beginner",
        "learning_objectives": ["Know what RAG is", "Know why it matters"],
        "sections": [section, section, section],
        "key_takeaways": ["a", "b", "c"],
        "summary": "RAG helps models use fresh facts.",
    }


@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock(return_value=json.dumps(valid_lesson()))
    monkeypatch.setattr(generator, "generate_text", mock)
    return mock


def test_valid_lesson_generated(mock_llm):
    lesson = asyncio.run(generate_lesson("RAG"))
    assert isinstance(lesson, LessonContent)
    assert lesson.title == "Intro to RAG"
    assert len(lesson.sections) == 3


def test_topic_and_instructions_passed(mock_llm):
    asyncio.run(generate_lesson("  Introduction to RAG  "))
    args, kwargs = mock_llm.await_args
    assert "Introduction to RAG" in args[0]
    assert "  Introduction to RAG  " not in args[0]
    assert kwargs["instructions"] == generator.SYSTEM_INSTRUCTIONS


@pytest.mark.parametrize("topic", ["", "   ", "x" * 201])
def test_bad_topic_rejected_without_llm_call(mock_llm, topic):
    with pytest.raises(InvalidTopicError):
        asyncio.run(generate_lesson(topic))
    mock_llm.assert_not_awaited()


def test_invalid_json(mock_llm):
    mock_llm.return_value = "not json at all"
    with pytest.raises(InvalidLessonJSONError):
        asyncio.run(generate_lesson("RAG"))


def test_missing_required_field(mock_llm):
    data = valid_lesson()
    del data["summary"]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(LessonValidationError, match="summary"):
        asyncio.run(generate_lesson("RAG"))


def test_insufficient_sections(mock_llm):
    data = valid_lesson()
    data["sections"] = data["sections"][:2]
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(LessonValidationError):
        asyncio.run(generate_lesson("RAG"))


def test_empty_field_and_extra_field_rejected(mock_llm):
    data = valid_lesson()
    data["title"] = "   "
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(LessonValidationError):
        asyncio.run(generate_lesson("RAG"))

    data = valid_lesson()
    data["unexpected"] = "x"
    mock_llm.return_value = json.dumps(data)
    with pytest.raises(LessonValidationError):
        asyncio.run(generate_lesson("RAG"))


def test_code_fenced_json_accepted(mock_llm):
    mock_llm.return_value = "```json\n" + json.dumps(valid_lesson()) + "\n```"
    assert asyncio.run(generate_lesson("RAG")).topic == "RAG"


def test_llm_error_propagates_unchanged(mock_llm):
    error = LLMError("boom")
    mock_llm.side_effect = error
    with pytest.raises(LLMError) as info:
        asyncio.run(generate_lesson("RAG"))
    assert info.value is error


def test_instructions_include_rubric_and_depth_rules():
    from app.evaluation.rubric import RUBRIC

    text = generator.SYSTEM_INSTRUCTIONS
    assert "{rubric}" not in text
    for item in RUBRIC:
        assert item.name in text
    assert "technical term" in text
    assert "4 to 8 sections" in text


def test_rag_prompt_lists_every_mandatory_concept(mock_llm):
    asyncio.run(generate_lesson("Introduction to RAG"))
    prompt = mock_llm.await_args.args[0]
    assert "mandatory concepts" in prompt
    for name in (
        "Definition of RAG",
        "Why RAG is needed",
        "What a language model (LLM) is",
        "Retrieval",
        "Augmentation",
        "Generation",
        "End-to-end interaction",
        "Limitations",
    ):
        assert name in prompt


def test_topic_without_requirements_has_no_concept_block(mock_llm):
    asyncio.run(generate_lesson("Photosynthesis"))
    assert "mandatory concepts" not in mock_llm.await_args.args[0]


# --- Phase 2: accuracy guardrails --------------------------------------------------


def test_instructions_contain_every_general_guardrail():
    from app.evaluation.topics import GENERAL_GUARDRAILS

    assert "{guardrails}" not in generator.SYSTEM_INSTRUCTIONS
    for rule in GENERAL_GUARDRAILS:
        assert rule in generator.SYSTEM_INSTRUCTIONS


@pytest.mark.parametrize(
    "needle",
    [
        "trains or updates the language model",
        "guarantees correct answers",
        "public internet",
    ],
)
def test_rag_prompt_has_rag_specific_guardrails(mock_llm, needle):
    asyncio.run(generate_lesson("Introduction to RAG"))
    assert needle in mock_llm.await_args.args[0]


def test_non_rag_prompt_has_no_rag_guardrails(mock_llm):
    asyncio.run(generate_lesson("Photosynthesis"))
    prompt = mock_llm.await_args.args[0]
    assert "public internet" not in prompt
    assert "trains or updates" not in prompt


@pytest.mark.parametrize(
    "phrase",
    [
        "commercial products or services",
        "speculative predictions about the future",
        "guarantees or certainty",
        "Do not attribute other abilities",
    ],
)
def test_general_guardrails_cover_the_required_cases(phrase):
    assert phrase in generator.SYSTEM_INSTRUCTIONS


def test_generator_logs_response_and_validation_failure(mock_llm, monkeypatch):
    captured = []
    monkeypatch.setattr(
        generator, "log_event", lambda name, **fields: captured.append((name, fields))
    )
    asyncio.run(generate_lesson("RAG"))
    assert [n for n, _ in captured] == ["generator_response"]

    captured.clear()
    mock_llm.return_value = "not json"
    with pytest.raises(InvalidLessonJSONError):
        asyncio.run(generate_lesson("RAG"))
    assert [n for n, _ in captured] == [
        "generator_response",
        "generator_validation_failure",
    ]
