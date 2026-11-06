"""Topic-specific learning requirements shared by generator, evaluator, improver.

A topic's requirements list the concepts a lesson must teach. Each concept has
"hints": phrases that usually appear when it is taught. Hints support the
evaluator's verdict (a missing hint prompts one confirmation question) but never
decide it, so a correct explanation in unexpected wording is not rejected.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    """One mandatory concept.

    ``hints`` are specific phrases (lowercase) that usually appear when the concept
    is taught. They only *support* the evaluator's verdict: a missing hint triggers
    a confirmation question, never an automatic failure. A trailing ``*`` makes a
    hint a word-prefix match.
    """

    id: str
    name: str
    description: str
    hints: tuple[str, ...]


@dataclass(frozen=True)
class TopicRequirements:
    """Mandatory concepts and topic-specific accuracy guardrails for a topic."""

    key: str
    concepts: tuple[Concept, ...]
    guardrails: tuple[str, ...]


# Accuracy guardrails that apply to every topic.
GENERAL_GUARDRAILS: tuple[str, ...] = (
    "Do not make claims about real commercial products or services (for example, "
    "do not say a named assistant, search engine, app or company uses the topic).",
    "Do not make speculative predictions about the future unless the topic asks "
    "for them.",
    "Do not claim guarantees or certainty that the topic cannot provide.",
    "Only say what the topic actually does. Do not attribute other abilities to it "
    "(such as learning, remembering users or personalizing).",
)

RAG_REQUIREMENTS = TopicRequirements(
    key="rag",
    concepts=(
        Concept(
            "definition",
            "Definition of RAG",
            "What Retrieval-Augmented Generation is and what the name means.",
            ("retrieval-augmented generation", "retrieval augmented generation"),
        ),
        Concept(
            "why_needed",
            "Why RAG is needed",
            "Language models can lack recent or private information and can "
            "generate unsupported answers.",
            (
                "up-to-date",
                "up to date",
                "out of date",
                "outdated",
                "recent information",
                "latest information",
                "recent events",
                "private data",
                "private information",
                "private documents",
                "company files",
                "make things up",
                "makes things up",
                "made-up",
                "made up",
                "unsupported answer*",
                "wrong answer*",
                "incorrect answer*",
                "hallucinat*",
                "does not know",
                "do not know",
                "doesn't know",
                "cannot know",
                "can't know",
                "only knows what",
                "guess wrongly",
            ),
        ),
        Concept(
            "language_model",
            "What a language model (LLM) is",
            "A beginner-level explanation of a language model (LLM): an AI system "
            "that writes text.",
            ("language model", "language models", "llm", "llms"),
        ),
        Concept(
            "retrieval",
            "Retrieval",
            "Finding relevant information from a prepared knowledge collection.",
            (
                "knowledge collection",
                "knowledge source",
                "knowledge base",
                "document collection",
                "collection of documents",
                "collection of information",
                "database",
                "databases",
                "library of",
                "searches through",
                "search through",
                "looks through",
                "look through",
                "finds relevant",
                "finding relevant",
                "relevant passages",
                "relevant documents",
                "relevant information",
            ),
        ),
        Concept(
            "augmentation",
            "Augmentation",
            "Providing the retrieved information as context to the language model.",
            (
                "as context",
                "retrieved context",
                "retrieved text",
                "retrieved passages",
                "retrieved information to",
                "added to the question",
                "added to the prompt",
                "added to the model",
                "given to the model",
                "provided to the model",
                "passed to the model",
                "pasted into",
                "handed to the model",
                "handed over",
                "adds the retrieved",
                "adds those passages",
                "sent to the model",
                "fed to the model",
            ),
        ),
        Concept(
            "generation",
            "Generation",
            "The language model uses the question and the retrieved context to "
            "write an answer.",
            (
                "model writes",
                "model generates",
                "model creates",
                "model composes",
                "model produces",
                "writes the answer",
                "generates the answer",
                "writes a response",
                "generates a response",
                "creates a response",
                "composes a reply",
                "composes an answer",
                "writes an answer",
                "produces an answer",
                "writes its answer",
                "generates its answer",
            ),
        ),
        Concept(
            "end_to_end",
            "End-to-end interaction",
            "The complete flow: question, retrieval, context handed to the model, "
            "answer written from that context.",
            (
                "step by step",
                "end-to-end",
                "end to end",
                "whole process",
                "full process",
                "complete process",
                "entire process",
                "after retrieving",
                "after the search",
                "after finding",
                "before answering",
                "before it answers",
                "before the model answers",
                "step 1",
                "step one",
                "first step",
                "second step",
            ),
        ),
        Concept(
            "limitations",
            "Limitations",
            "Retrieved information may be incomplete, outdated or incorrect, and "
            "RAG does not guarantee accuracy.",
            (
                "not guarantee",
                "does not guarantee",
                "doesn't guarantee",
                "no guarantee",
                "not guaranteed",
                "incomplete",
                "outdated",
                "incorrect",
                "can be wrong",
                "may be wrong",
                "might be wrong",
                "not always correct",
                "not always right",
                "not always accurate",
                "unreliable",
                "can miss",
                "may miss",
                "might miss",
                "mistakes",
            ),
        ),
    ),
    guardrails=(
        "Do not claim that RAG trains or updates the language model. The model "
        "itself does not change; RAG only supplies information at question time.",
        "Do not claim that RAG guarantees correct answers.",
        "RAG retrieves from an available knowledge source (documents, a database, "
        "company files). It does not necessarily use the public internet.",
    ),
)

_REGISTRY = (RAG_REQUIREMENTS,)


def get_requirements(topic: str) -> TopicRequirements | None:
    """Return the requirements for ``topic``, or ``None`` if it has none."""
    normalized = re.sub(r"[\s\-]+", " ", topic.lower())
    if "retrieval augmented generation" in normalized or re.search(
        r"\brag\b", normalized
    ):
        return RAG_REQUIREMENTS
    return None


def concepts_text(requirements: TopicRequirements) -> str:
    """The mandatory concepts as prompt text."""
    return "\n".join(
        f"- [{c.id}] {c.name}: {c.description}" for c in requirements.concepts
    )


def guardrails_text(requirements: TopicRequirements | None = None) -> str:
    """General (and, if given, topic-specific) guardrails as prompt text."""
    rules = GENERAL_GUARDRAILS + (requirements.guardrails if requirements else ())
    return "\n".join(f"- {rule}" for rule in rules)


def has_hint(concept: Concept, text: str) -> bool:
    """True if any hint of ``concept`` appears in ``text`` as whole words.

    ``text`` should be lowercase with straight quotes. Matching is word-boundary
    based, so "llm" does not match "willmore" and "then" would not match
    "authentic".
    """
    for hint in concept.hints:
        stem = hint.endswith("*")
        body = re.escape(hint.rstrip("*"))
        pattern = rf"(?<![a-z0-9]){body}" + ("" if stem else r"(?![a-z0-9])")
        if re.search(pattern, text):
            return True
    return False
