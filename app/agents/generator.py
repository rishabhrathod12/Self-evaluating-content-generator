"""Lesson generator agent: builds the prompt, calls the LLM, validates the result."""

import json

from pydantic import ValidationError

from app.evaluation.rubric import rubric_text
from app.evaluation.schemas import LessonContent
from app.llm import generate_text

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

Depth: write 4 to 6 sections. Each explanation should be about 3 to 5 \
sentences, and each section needs a specific, named example. Define every \
technical term, acronym and specialized word (even common-sounding ones such \
as "database") in plain words at or before its first use. Make sure every \
learning objective is taught by a section and every key takeaway is supported \
by the sections.

Your lesson will be judged against this rubric, so write to it:
{rubric}

Be accurate. Do not invent facts.

Output rules: respond with a single JSON object and nothing else. No markdown, \
no code fences, no commentary. The JSON must match the schema you are given \
exactly, with no extra fields.\
""".replace("{rubric}", rubric_text())


def build_prompt(topic: str) -> str:
    """Build the user prompt for a topic, including the required JSON schema."""
    schema = json.dumps(LessonContent.model_json_schema(), indent=2)
    return (
        f"Write a complete beginner-friendly lesson on: {topic}\n\n"
        "Set learner_level to describe a beginner with no technical background. "
        "Include at least 2 learning objectives, 4 to 6 sections (each with a "
        "heading, explanation and example) and at least 3 key takeaways.\n\n"
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


async def generate_lesson(topic: str) -> LessonContent:
    """Generate a validated beginner-friendly lesson for ``topic``.

    Raises:
        InvalidTopicError: The topic is empty or too long.
        ConfigurationError: ``OPENAI_API_KEY`` is not configured (from the client).
        LLMError: The OpenAI call failed (propagated unchanged).
        InvalidLessonJSONError: The response was not valid JSON.
        LessonValidationError: The response did not match the schema.
    """
    cleaned = validate_topic(topic)
    raw = await generate_text(build_prompt(cleaned), instructions=SYSTEM_INSTRUCTIONS)
    return parse_lesson(raw)
