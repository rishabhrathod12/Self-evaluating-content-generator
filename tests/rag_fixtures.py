"""Realistic RAG lessons and evaluator responses for offline regression tests."""

import json

from app.evaluation.schemas import CRITERIA, EvaluationResult, LessonContent
from app.evaluation.topics import RAG_REQUIREMENTS, all_points

# One section per concept, worded differently from what the generator prompt says,
# and each genuinely stating its teaching points.
SECTIONS: dict[str, tuple[str, str, str]] = {
    "definition": (
        "What RAG means",
        "RAG stands for Retrieval-Augmented Generation. It is a method that lets "
        "a computer look up information first and then have a language model "
        "write an answer from what was found.",
        "It works like an open-book exam: you look up the facts, then write.",
    ),
    "why_needed": (
        "Why we need it",
        "A language model only knows what it learned during training, so it can "
        "lack recent or private information, such as last week's news or a "
        "company's own files. When it lacks the facts it may make things up, "
        "which are called hallucinations.",
        "Asked about yesterday's sports score, a model may simply guess.",
    ),
    "language_model": (
        "The language model",
        "A language model, or LLM, is an AI system that learned from huge amounts "
        "of text and writes text such as answers.",
        "Think of a well-read writer who answers from memory.",
    ),
    "retrieval": (
        "Knowledge source and retrieval",
        "The information comes from a prepared knowledge source, such as a "
        "collection of documents, a database or company files. Retrieval searches "
        "that source and finds the pieces most relevant to the user's question.",
        "A search over the HR policy documents finds the leave-policy page.",
    ),
    "augmentation": (
        "Adding context",
        "Augmentation means the retrieved information is added to the user's "
        "question as context before the language model reads it.",
        "The leave-policy text is pasted under the employee's question.",
    ),
    "generation": (
        "Writing the answer",
        "The language model reads the question together with the retrieved context "
        "and writes the answer from both.",
        "It reads the question and the policy text, then writes a short reply.",
    ),
    "end_to_end": (
        "The whole workflow",
        "Step by step: first the user asks a question, then relevant information "
        "is retrieved, then it is added as context, and finally the language "
        "model writes the answer.",
        "Suppose a user asks, 'How many leave days do I get?' The system searches "
        "the HR documents and retrieves the leave policy. That text is added to "
        "the question as context. The language model then writes: 'You get 20 "
        "days.'",
    ),
    "limitations": (
        "Limits of RAG",
        "The retrieved information can be incomplete, outdated or incorrect. RAG "
        "does not guarantee that answers are accurate.",
        "If the policy page is old, the answer will be old too.",
    ),
}

# The exact passage in the default lesson that states each teaching point.
EVIDENCE: dict[str, str] = {
    "definition.name": "RAG stands for Retrieval-Augmented Generation",
    "why.knowledge_gap": (
        "A language model only knows what it learned during training, so it can "
        "lack recent or private information"
    ),
    "why.hallucination": "When it lacks the facts it may make things up",
    "llm.what_it_is": (
        "A language model, or LLM, is an AI system that learned from huge amounts "
        "of text and writes text such as answers"
    ),
    "retrieval.source": (
        "The information comes from a prepared knowledge source, such as a "
        "collection of documents, a database or company files"
    ),
    "retrieval.finding": (
        "Retrieval searches that source and finds the pieces most relevant to the "
        "user's question"
    ),
    "augmentation.added_as_context": (
        "Augmentation means the retrieved information is added to the user's "
        "question as context before the language model reads it"
    ),
    "generation.writes_from_both": (
        "The language model reads the question together with the retrieved "
        "context and writes the answer from both"
    ),
    "flow.sequence": (
        "Step by step: first the user asks a question, then relevant information "
        "is retrieved, then it is added as context, and finally the language "
        "model writes the answer"
    ),
    "flow.example": "Suppose a user asks, 'How many leave days do I get?'",
    "limits.fallible_sources": (
        "The retrieved information can be incomplete, outdated or incorrect"
    ),
    "limits.no_guarantee": "RAG does not guarantee that answers are accurate",
}


def rag_lesson(
    omit: tuple[str, ...] = (),
    replace: dict[str, tuple[str, str, str]] | None = None,
    objectives: tuple[str, ...] = (
        "Understand what RAG is and why it is needed.",
        "Follow how a question becomes an answer in RAG.",
    ),
    takeaways: tuple[str, ...] = (
        "RAG looks up information, then a language model writes the answer.",
        "The retrieved text is added to the question as context.",
        "RAG does not guarantee accurate answers.",
    ),
    summary: str = "RAG combines looking up information with writing an answer.",
) -> LessonContent:
    """A full lesson, optionally missing some concepts or with sections replaced."""
    replace = replace or {}
    sections = [
        {"heading": h, "explanation": e, "example": x}
        for concept_id, (h, e, x) in (SECTIONS | replace).items()
        if concept_id not in omit
    ]
    return LessonContent.model_validate(
        {
            "title": "Introduction to RAG",
            "topic": "Introduction to RAG",
            "learner_level": "Beginner (no technical background)",
            "learning_objectives": list(objectives),
            "sections": sections,
            "key_takeaways": list(takeaways),
            "summary": summary,
        }
    )


VAGUE = ("Overview", "The parts of RAG work together.", "X")
NEUTRAL_TAKEAWAYS = ("RAG is useful.", "It has parts.", "It has limits.")
NEUTRAL_SUMMARY = "RAG is a topic with several parts."
# why_needed with the knowledge gap but without any "made-up answers" statement
WEAK_WHY = (
    "Why we need it",
    "A language model only knows what it learned during training, so it can lack "
    "recent or private information, such as last week's news or a company's own "
    "files.",
    "X",
)


def lesson_lacking(*concept_ids: str) -> LessonContent:
    """A lesson in which the given concepts are stated NOWHERE.

    Removing a section is not enough: the workflow summary, takeaways and other
    sections of the full lesson restate neighbouring ideas, so those are made vague
    too. Code decides coverage from the lesson text, so tests need this.
    """
    replace = {"end_to_end": VAGUE}
    if "augmentation" in concept_ids:
        replace["generation"] = (
            "Writing the answer",
            "The language model writes an answer.",
            "X",
        )
    return rag_lesson(
        omit=concept_ids,
        replace=replace,
        takeaways=NEUTRAL_TAKEAWAYS,
        summary=NEUTRAL_SUMMARY,
    )


def lacking_variant(index: int, *concept_ids: str) -> LessonContent:
    """A different lesson lacking the same concepts (only an example changes)."""
    data = lesson_lacking(*concept_ids).model_dump()
    data["sections"][0]["example"] = f"Variant {index} example."
    return LessonContent.model_validate(data)


def variant(index: int, omit: tuple[str, ...] = ()) -> LessonContent:
    """A different lesson that keeps every quoted passage (only an example changes)."""
    heading, explanation, _ = SECTIONS["limitations"]
    return rag_lesson(
        omit=omit,
        replace={"limitations": (heading, explanation, f"Variant {index} example.")},
    )


def _checks(failing: tuple[int, ...] = ()) -> list[dict]:
    checks = []
    for index, name in enumerate(CRITERIA):
        failed = index in failing
        checks.append(
            {
                "criterion": name,
                "passed": not failed,
                "reason": "A specific problem." if failed else "It meets the check.",
                "improvement_suggestion": "Fix it." if failed else "",
                "evidence_quote": "" if failed else "Introduction to RAG",
                "issues": (
                    [
                        {
                            "problem": "A specific problem.",
                            "quote": "Introduction to RAG",
                            "term": "RAG",
                            "correction": "The correct statement.",
                        }
                    ]
                    if failed
                    else []
                ),
            }
        )
    return checks


def eval_data(
    uncovered: tuple[str, ...] = (),
    failing: tuple[int, ...] = (),
    evidence: dict[str, str] | None = None,
) -> dict:
    """An evaluator response with every point judged. ``uncovered`` = point ids."""
    evidence = EVIDENCE | (evidence or {})
    failing_checks = _checks(failing)
    point_checks = [
        (
            {
                "point_id": point.id,
                "covered": False,
                "evidence_quote": "",
                "gap": f"The lesson never states: {point.text}",
            }
            if point.id in uncovered
            else {
                "point_id": point.id,
                "covered": True,
                "evidence_quote": evidence[point.id],
                "gap": "",
            }
        )
        for _, point in all_points(RAG_REQUIREMENTS)
    ]
    return {
        "overall_pass": not failing,
        "checks": failing_checks,
        "failure_reasons": [f"{CRITERIA[i]} failed" for i in failing],
        "improvement_feedback": "Do better." if failing else "",
        "point_checks": point_checks,
        "concept_checks": [],
        "notes": [],
    }


def eval_json(**kwargs) -> str:
    return json.dumps(eval_data(**kwargs))


def concept_points(concept_id: str) -> list[str]:
    """The point ids of one concept."""
    concept = next(c for c in RAG_REQUIREMENTS.concepts if c.id == concept_id)
    return [p.id for p in concept.points]


def as_result(data: dict) -> EvaluationResult:
    return EvaluationResult.model_validate(data)
