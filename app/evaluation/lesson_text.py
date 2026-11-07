"""Plain-text rendering and normalization of lessons, shared by the agents."""

import re

from app.evaluation.schemas import LessonContent

_QUOTE_CHARS = str.maketrans(  # curly quotes -> straight quotes
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
    }
)
_LABEL_LINE = re.compile(r"^(?:Title|Topic|Learner level):|^Section \d+:")


def normalize_text(text: str) -> str:
    """Casefold, straighten curly quotes and collapse whitespace."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_CHARS)).strip().casefold()


def render_lesson(lesson: LessonContent) -> str:
    """Render the complete lesson as readable plain text."""
    lines = [
        f"Title: {lesson.title}",
        f"Topic: {lesson.topic}",
        f"Learner level: {lesson.learner_level}",
        "",
        "Learning objectives:",
        *(f"- {item}" for item in lesson.learning_objectives),
    ]
    for number, section in enumerate(lesson.sections, 1):
        lines += [
            "",
            f"Section {number}: {section.heading}",
            section.explanation,
            f"Example: {section.example}",
        ]
    lines += [
        "",
        "Key takeaways:",
        *(f"- {item}" for item in lesson.key_takeaways),
        "",
        f"Summary: {lesson.summary}",
    ]
    return "\n".join(lines)


def body_text(lesson: LessonContent) -> str:
    """The lesson without label lines (title, topic, level, section headings)."""
    lines = render_lesson(lesson).split("\n")
    return "\n".join(line for line in lines if not _LABEL_LINE.match(line))


STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "that",
        "this",
        "with",
        "from",
        "has",
        "have",
        "had",
        "its",
        "but",
        "you",
        "your",
        "can",
        "may",
        "might",
        "will",
        "would",
        "should",
        "could",
        "does",
        "did",
        "not",
        "out",
        "all",
        "any",
        "each",
        "some",
        "more",
        "most",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "these",
        "those",
        "when",
        "what",
        "which",
        "who",
        "how",
        "why",
        "also",
        "only",
        "just",
        "into",
        "over",
        "about",
        "because",
        "while",
        "such",
        "like",
        "very",
        "much",
    }
)


def content_words(text: str) -> set[str]:
    """Lowercase words of three or more letters, without stop words."""
    words = re.findall(r"[a-z]{3,}", normalize_text(text))
    return {w for w in words if w not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, never across line breaks."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]
