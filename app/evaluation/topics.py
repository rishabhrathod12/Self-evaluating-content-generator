"""Topic-specific curriculum shared by the generator, evaluator and improver.

A topic's requirements list the *concepts* a lesson must teach. Each concept is
broken into *teaching points*: concrete statements the lesson must make
explicitly. Merely mentioning a term does not satisfy a point.

Each point also carries *anchor groups*: lenient groups of synonyms. They are NOT
used to decide whether a lesson is good. They are only a sanity check on the
quote an evaluator offers as proof, so an unrelated sentence can never count as
evidence. A quote must touch every group of the point it is offered for.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorGroup:
    """Synonyms for one idea a proving quote must touch (``*`` = word prefix)."""

    label: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class TeachingPoint:
    """One statement the lesson must make explicitly."""

    id: str
    text: str
    anchors: tuple[AnchorGroup, ...]


@dataclass(frozen=True)
class Concept:
    """One mandatory concept, taught through one or more teaching points."""

    id: str
    name: str
    description: str
    points: tuple[TeachingPoint, ...]


@dataclass(frozen=True)
class TopicRequirements:
    """The curriculum and accuracy rules for a topic."""

    key: str
    concepts: tuple[Concept, ...]
    guardrails: tuple[str, ...]
    workflow: tuple[str, ...] = ()
    worked_example: str = ""
    # Points that together prove the generic "Covers key points" check: what it
    # is, why it matters, how it works, and a worked example.
    key_point_ids: tuple[str, ...] = ()


def _g(label: str, *terms: str) -> AnchorGroup:
    return AnchorGroup(label, terms)


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

_QUESTION = _g(
    "the user's question",
    "question*",
    "query",
    "queries",
    "prompt*",
    "ask*",
    "request*",
)
_INFO = _g(
    "the retrieved information",
    "retrieved",
    "retriev*",
    "information",
    "passage*",
    "document*",
    "fact*",
    "text",
    "result*",
    "source*",
    "data",
    "found",
    "knowledge",
    "context",
    "content",
    "article*",
    "note*",
)
_MODEL = _g("the language model", "language model*", "llm*", "model*", "ai", "chatbot*")
_WRITES = _g(
    "writing the answer",
    "write*",
    "writing",
    "wrote",
    "generat*",
    "produc*",
    "creat*",
    "compos*",
    "draft*",
    "answer*",
    "respon*",
    "repl*",
)
_USES = _g(
    "the model using the question and the information together",
    "read",
    "reads",
    "reading",
    "uses",
    "using",
    "used",
    "takes",
    "taking",
    "combines",
    "combining",
    "combined",
    "together",
    "both",
    "based on",
    "draws",
    "drawing",
)
_FINDS = _g(
    "finding or searching",
    "retriev*",
    "search*",
    "find*",
    "found",
    "look*",
    "fetch*",
    "match*",
    "pick*",
    "select*",
)

RAG_REQUIREMENTS = TopicRequirements(
    key="rag",
    concepts=(
        Concept(
            "definition",
            "Definition of RAG",
            "What Retrieval-Augmented Generation is and what its name means.",
            (
                TeachingPoint(
                    "definition.name",
                    "RAG stands for Retrieval-Augmented Generation and combines "
                    "retrieving information with generating an answer.",
                    (
                        _g(
                            "the full name",
                            "retrieval-augmented generation",
                            "retrieval augmented generation",
                        ),
                        _g(
                            "what it is or means",
                            "stands for",
                            "means",
                            "is a method",
                            "is a technique",
                            "is an approach",
                            "is a way",
                            "is a system",
                            "is a process",
                            "combines",
                            "combining",
                            "combine",
                        ),
                    ),
                ),
            ),
        ),
        Concept(
            "why_needed",
            "Why RAG is needed",
            "The limits of a language model's built-in knowledge.",
            (
                TeachingPoint(
                    "why.knowledge_gap",
                    "A language model only knows what it learned during training, "
                    "so it can lack recent or private information.",
                    (
                        _MODEL,
                        _g(
                            "missing recent or private information",
                            "lack*",
                            "recent",
                            "latest",
                            "up-to-date",
                            "up to date",
                            "outdated",
                            "out of date",
                            "private",
                            "current",
                            "new",
                            "training",
                            "does not know",
                            "do not know",
                            "doesn't know",
                            "cannot know",
                            "only knows",
                        ),
                    ),
                ),
                TeachingPoint(
                    "why.hallucination",
                    "Without the right information a language model can give "
                    "unsupported or made-up answers (hallucinations).",
                    (
                        _g(
                            "a language model",
                            "language model*",
                            "llm*",
                            "model*",
                            "ai",
                            "chatbot*",
                        ),
                        _g(
                            "unsupported or made-up answers",
                            "make up",
                            "makes up",
                            "made up",
                            "made-up",
                            "make things up",
                            "makes things up",
                            "making things up",
                            "making up",
                            "things up",
                            "fabricat*",
                            "hallucinat*",
                            "invent*",
                            "guess*",
                            "unsupported",
                            "not supported",
                            "inaccurate",
                            "incorrect",
                            "wrong",
                            "false",
                            "untrue",
                            "mistake*",
                            "error*",
                        ),
                    ),
                ),
            ),
        ),
        Concept(
            "language_model",
            "What a language model (LLM) is",
            "What a language model is and what it does, in beginner terms.",
            (
                TeachingPoint(
                    "llm.what_it_is",
                    "A language model (LLM) is an AI system that has learned from "
                    "large amounts of text and writes text, such as answers.",
                    (
                        _g("a language model", "language model*", "llm*"),
                        _g(
                            "what it is: an AI system that learned from text",
                            "ai system",
                            "artificial intelligence",
                            "ai model",
                            "ai program",
                            "learns from",
                            "learned from",
                            "has learned",
                            "trained",
                            "patterns",
                            "predict*",
                        ),
                    ),
                ),
            ),
        ),
        Concept(
            "retrieval",
            "Knowledge source and retrieval",
            "Where the information comes from and how relevant pieces are found.",
            (
                TeachingPoint(
                    "retrieval.source",
                    "The information comes from a prepared knowledge source such "
                    "as a collection of documents, a database or company files.",
                    (
                        _g(
                            "a knowledge source",
                            "knowledge source*",
                            "knowledge base*",
                            "knowledge collection*",
                            "collection*",
                            "document*",
                            "database*",
                            "file*",
                            "librar*",
                            "article*",
                            "archive*",
                            "manual*",
                            "source*",
                            "stored",
                            "store*",
                        ),
                        _g(
                            "where the information comes from",
                            "comes from",
                            "come from",
                            "prepared",
                            "stored",
                            "kept",
                            "drawn from",
                            "taken from",
                            "retrieved from",
                            "found in",
                            "contains",
                            "made up of",
                            "collection of",
                        ),
                    ),
                ),
                TeachingPoint(
                    "retrieval.finding",
                    "Retrieval searches that source and finds the pieces most "
                    "relevant to the user's question.",
                    (
                        _FINDS,
                        _g(
                            "relevant information",
                            "relevant",
                            "related",
                            "useful",
                            "best",
                            "matching",
                            "similar",
                            "right",
                            "specific",
                            "needed",
                            "most",
                        ),
                    ),
                ),
            ),
        ),
        Concept(
            "augmentation",
            "Augmentation",
            "How the retrieved information is added to the question as context.",
            (
                TeachingPoint(
                    "augmentation.added_as_context",
                    "The retrieved information is added to the user's question as "
                    "context before the language model reads it.",
                    (
                        _INFO,
                        _g(
                            "being added as context",
                            "context",
                            "add*",
                            "attach*",
                            "combin*",
                            "paste*",
                            "append*",
                            "insert*",
                            "alongside",
                            "along with",
                            "together",
                            "hand",
                            "handed",
                            "hands",
                            "given to",
                            "provided to",
                            "supplied",
                            "fed",
                            "sent",
                            "help the",
                            "helps the",
                            "use",
                            "uses",
                            "used",
                            "using",
                        ),
                    ),
                ),
            ),
        ),
        Concept(
            "generation",
            "Generation",
            "How the language model uses the question and the context to answer.",
            (
                TeachingPoint(
                    "generation.writes_from_both",
                    "The language model reads the question together with the "
                    "retrieved context and writes the answer from both.",
                    (_MODEL, _QUESTION, _INFO, _USES, _WRITES),
                ),
            ),
        ),
        Concept(
            "end_to_end",
            "End-to-end workflow",
            "The complete sequence from the user's question to the final answer.",
            (
                TeachingPoint(
                    "flow.sequence",
                    "The full sequence, in order: the user asks a question, "
                    "relevant information is retrieved, it is added as context, "
                    "and the language model writes the answer.",
                    (_MODEL, _QUESTION, _FINDS, _WRITES),
                ),
                TeachingPoint(
                    "flow.example",
                    "One practical example follows a single user question through "
                    "the whole process: what is asked, what is retrieved, and the "
                    "answer the model writes.",
                    (
                        _g(
                            "a concrete situation",
                            "example*",
                            "imagine*",
                            "suppose*",
                            "consider*",
                            "picture*",
                            "scenario*",
                            "for instance",
                            "user*",
                            "student*",
                            "customer*",
                            "person",
                            "someone",
                            "you",
                            "asks",
                            "asked",
                            "wants to know",
                            "employee*",
                            "step",
                        ),
                        _MODEL,
                        _QUESTION,
                        _FINDS,
                        _WRITES,
                    ),
                ),
            ),
        ),
        Concept(
            "limitations",
            "Limitations",
            "RAG can still give incomplete or incorrect answers.",
            (
                TeachingPoint(
                    "limits.fallible_sources",
                    "The retrieved information can be incomplete, outdated or "
                    "incorrect.",
                    (
                        _g(
                            "information that can be wrong",
                            "incomplete",
                            "outdated",
                            "out of date",
                            "incorrect",
                            "wrong",
                            "inaccurate",
                            "missing",
                            "stale",
                            "unreliable",
                            "error*",
                            "mistake*",
                            "miss*",
                            "may not",
                            "might not",
                            "not always",
                            "can be",
                        ),
                        _g(
                            "the retrieved information",
                            "retrieved",
                            "retriev*",
                            "information",
                            "document*",
                            "source*",
                            "knowledge",
                            "data",
                            "result*",
                            "passage*",
                        ),
                    ),
                ),
                TeachingPoint(
                    "limits.no_guarantee",
                    "RAG does not guarantee that answers are accurate.",
                    (
                        _g(
                            "no guarantee",
                            "not guarantee*",
                            "no guarantee*",
                            "cannot guarantee",
                            "can't guarantee",
                            "not ensure*",
                            "not always",
                            "not perfect",
                            "not necessarily",
                            "no promise",
                            "cannot promise",
                            "still make",
                            "still be",
                            "still wrong",
                            "may still",
                            "might still",
                            "can still",
                            "not fully",
                            "never perfect",
                            "not completely",
                        ),
                        _g(
                            "accuracy",
                            "accura*",
                            "correct*",
                            "right",
                            "reliab*",
                            "answer*",
                            "wrong",
                            "mistake*",
                            "perfect",
                            "error*",
                        ),
                    ),
                ),
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
    workflow=(
        "The user asks a question.",
        "The knowledge source is searched and the most relevant pieces of "
        "information are retrieved.",
        "The retrieved information is added to the user's question as context.",
        "The language model reads the question together with that context.",
        "The language model writes the answer using both.",
    ),
    key_point_ids=(
        "definition.name",
        "why.knowledge_gap",
        "flow.sequence",
        "flow.example",
    ),
    worked_example=(
        "One practical example that follows a single user question through all "
        "five workflow steps: state the question, show what is retrieved and from "
        "which knowledge source, show how it is added as context, and show the "
        "answer the model writes."
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


def all_points(
    requirements: TopicRequirements,
) -> list[tuple[Concept, TeachingPoint]]:
    """Every (concept, point) pair, in curriculum order."""
    return [(c, p) for c in requirements.concepts for p in c.points]


def find_point(requirements: TopicRequirements, point_id: str) -> TeachingPoint | None:
    """The teaching point with ``point_id``, if any."""
    return next((p for _, p in all_points(requirements) if p.id == point_id), None)


def _term_matches(term: str, text: str) -> bool:
    stem = term.endswith("*")
    body = re.escape(term.rstrip("*"))
    pattern = rf"(?<![a-z0-9]){body}" + ("" if stem else r"(?![a-z0-9])")
    return re.search(pattern, text) is not None


def unmet_groups(point: TeachingPoint, text: str) -> list[AnchorGroup]:
    """The anchor groups of ``point`` that ``text`` does not touch.

    ``text`` must be lowercase with straight quotes (see ``normalize_text``).
    """
    return [
        group
        for group in point.anchors
        if not any(_term_matches(term, text) for term in group.terms)
    ]


def touches_all(point: TeachingPoint, text: str) -> bool:
    """True if ``text`` touches every anchor group of ``point`` (stops at the first
    group it misses). ``text`` must be lowercase with straight quotes."""
    return all(
        any(_term_matches(term, text) for term in group.terms)
        for group in point.anchors
    )


def concepts_text(requirements: TopicRequirements, *, with_ids: bool = True) -> str:
    """The concepts and their teaching points as prompt text."""
    lines = []
    for concept in requirements.concepts:
        lines.append(f"- [{concept.id}] {concept.name}: {concept.description}")
        for point in concept.points:
            label = f"{point.id}: " if with_ids else ""
            lines.append(f"    * {label}{point.text}")
    return "\n".join(lines)


def workflow_text(requirements: TopicRequirements) -> str:
    """The numbered workflow steps as prompt text."""
    return "\n".join(f"{i}. {step}" for i, step in enumerate(requirements.workflow, 1))


def guardrails_text(requirements: TopicRequirements | None = None) -> str:
    """General (and, if given, topic-specific) guardrails as prompt text."""
    rules = GENERAL_GUARDRAILS + (requirements.guardrails if requirements else ())
    return "\n".join(f"- {rule}" for rule in rules)
