"""Lesson generator agent: builds the prompt, calls the LLM, validates the result."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.debug_log import log_event
from app.evaluation.rubric import rubric_text
from app.evaluation.schemas import LessonContent
from app.evaluation.topics import (
    TopicRequirements,
    concepts_text,
    get_requirements,
    guardrails_text,
    workflow_text,
)
from app.llm import generate_text
from app.memory.context import generator_memory_block

MAX_TOPIC_LENGTH = 200


class LessonGenerationError(Exception):
    """Base class for lesson generation failures."""


class InvalidTopicError(LessonGenerationError, ValueError):
    """The topic is empty or too long."""


class InvalidLessonJSONError(LessonGenerationError):
    """The model response was not valid JSON."""


class LessonValidationError(LessonGenerationError):
    """The model response was JSON but did not match the LessonContent schema."""


# --- Prompt construction (no validation logic here) -----------------------------

SYSTEM_INSTRUCTIONS = """\
You are an expert teacher who writes lessons for complete beginners.

Audience: a learner with zero technical background, comparable to an Indian \
12th-grade graduate. Use simple, clear English. Avoid jargon; if a technical \
term is unavoidable, explain it in plain words the first time it appears. \
Prefer short sentences and everyday analogies.

Every lesson must cover:
1. What the topic is, including what kind of thing it is and what larger \
system it belongs to or works with (for example, which kind of software or AI \
it is used with).
2. Why it matters.
3. How it works, step by step, including how its parts interact.
4. Practical, concrete examples.

Depth: write as many sections as the topic's required content needs, and no \
more. Never add filler, padding or speculative sections. Each explanation \
should be detailed: about 8 to 12 sentences (roughly 150 to 250 words), \
split into short paragraphs, and must STATE its ideas explicitly in plain \
sentences; naming a step or a term is not teaching it. Each section needs a \
specific, named example, explained in 2 to 4 sentences. Define every \
technical term, acronym and specialized word (even common-sounding ones such \
as "database") in plain words at or before its first use. Make sure every \
learning objective is taught by a section and every key takeaway is supported \
by the sections.

Your lesson will be judged against this rubric, so write to it:
{rubric}

Be accurate. Do not invent facts. Accuracy rules you must follow:
{guardrails}

Output rules: respond with a single JSON object and nothing else. No markdown, \
no code fences, no commentary. The JSON must match the schema you are given \
exactly, with no extra fields.\
""".replace("{rubric}", rubric_text()).replace("{guardrails}", guardrails_text())


def _requirements_block(requirements: TopicRequirements) -> str:
    return (
        "You MUST teach every concept below, and you MUST state every teaching "
        "point in an explicit sentence of its own, at beginner level. Use one "
        "section per concept, or combine closely related concepts in one section; "
        "do not add sections that teach nothing new:\n"
        f"{concepts_text(requirements, with_ids=False)}\n\n"
        "Teach this workflow as numbered steps, and explain how each step leads to "
        "the next (how retrieval, augmentation and generation connect):\n"
        f"{workflow_text(requirements)}\n\n"
        f"{requirements.worked_example} Put it in the example field of the section "
        "that teaches the end-to-end workflow.\n\n"
        "Topic-specific accuracy rules:\n"
        f"{guardrails_text(requirements)}\n\n"
    )


def build_prompt(
    topic: str,
    requirements: TopicRequirements | None = None,
    memory: Mapping[str, Any] | None = None,
) -> str:
    """Build the user prompt: the topic, the curriculum and the JSON schema."""
    schema = json.dumps(LessonContent.model_json_schema(), indent=2)
    curriculum = _requirements_block(requirements) if requirements else ""
    curriculum += generator_memory_block(memory)
    return (
        f"Write a complete beginner-friendly lesson on: {topic}\n\n{curriculum}"
        "Set learner_level to describe a beginner with no technical background. "
        "Include at least 2 learning objectives, one heading, explanation and "
        "example per section, and at least 3 key takeaways.\n\n"
        f"JSON schema to follow:\n{schema}"
    )


# --- Validation (no prompt logic here) -------------------------------------------


def validate_topic(topic: str) -> str:
    """Return the trimmed topic, or raise ``InvalidTopicError``."""
    if not isinstance(topic, str) or not topic.strip():
        raise InvalidTopicError("Topic must be a non-empty string.")
    cleaned = topic.strip()
    if len(cleaned) > MAX_TOPIC_LENGTH:
        raise InvalidTopicError(f"Topic must be at most {MAX_TOPIC_LENGTH} characters.")
    return cleaned


def _strip_code_fence(raw: str) -> str:
    """Remove one surrounding markdown code fence, which models sometimes add."""
    text = raw.strip()
    if text.startswith("```") and text.endswith("```") and len(text) >= 6:
        body = text[3:-3]
        first_line, _, rest = body.partition("\n")
        text = rest if first_line.strip().isalpha() or not first_line.strip() else body
    return text.strip()


def parse_lesson(raw: str) -> LessonContent:
    """Parse and validate raw model output into a ``LessonContent``.

    Raises:
        InvalidLessonJSONError: The text is not valid JSON.
        LessonValidationError: The JSON does not match the schema.
    """
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        raise InvalidLessonJSONError(
            f"Model response is not valid JSON: {exc.msg} (position {exc.pos})."
        ) from exc
    try:
        return LessonContent.model_validate(data)
    except ValidationError as exc:
        raise LessonValidationError(
            f"Lesson failed schema validation with {exc.error_count()} error(s): {exc}"
        ) from exc


# --- Public API --------------------------------------------------------------------


async def generate_lesson(
    topic: str, memory: Mapping[str, Any] | None = None
) -> LessonContent:
    """Generate a validated beginner-friendly lesson for ``topic``.

    ``memory`` is optional feedback from earlier stored runs of this topic. It is
    advice only: the curriculum and rubric above still decide what the lesson needs.

    Raises:
        InvalidTopicError: The topic is empty or too long.
        ConfigurationError: ``OPENAI_API_KEY`` is not configured (from the client).
        LLMError: The OpenAI call failed (propagated unchanged).
        InvalidLessonJSONError: The response was not valid JSON.
        LessonValidationError: The response did not match the schema.
    """
    cleaned = validate_topic(topic)
    raw = await generate_text(
        build_prompt(cleaned, get_requirements(cleaned), memory),
        instructions=SYSTEM_INSTRUCTIONS,
    )
    log_event("generator_response", raw=raw)
    try:
        return parse_lesson(raw)
    except LessonGenerationError as exc:
        log_event(
            "generator_validation_failure",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
