import pytest
from app.evaluation.lesson_text import normalize_text
from app.evaluation.topics import (
    GENERAL_GUARDRAILS,
    RAG_REQUIREMENTS,
    all_points,
    concepts_text,
    find_point,
    get_requirements,
    guardrails_text,
    unmet_groups,
    workflow_text,
)

EXPECTED_CONCEPTS = [
    "definition",
    "why_needed",
    "language_model",
    "retrieval",
    "augmentation",
    "generation",
    "end_to_end",
    "limitations",
]


@pytest.mark.parametrize(
    "topic",
    [
        "Introduction to RAG",
        "RAG",
        "rag basics",
        "Retrieval-Augmented Generation",
        "retrieval augmented generation for beginners",
    ],
)
def test_rag_topics_match(topic):
    assert get_requirements(topic) is RAG_REQUIREMENTS


@pytest.mark.parametrize("topic", ["Photosynthesis", "Fragments of history", "rags"])
def test_other_topics_have_no_requirements(topic):
    assert get_requirements(topic) is None


def test_rag_has_the_eight_mandatory_concepts_in_order():
    assert [c.id for c in RAG_REQUIREMENTS.concepts] == EXPECTED_CONCEPTS


def test_every_concept_has_concrete_teaching_points():
    for concept in RAG_REQUIREMENTS.concepts:
        assert concept.name and concept.description
        assert concept.points, concept.id
        for point in concept.points:
            assert point.id.split(".")[0], point.id
            assert len(point.text.split()) >= 8, point.id  # a statement, not a label
            assert point.anchors, point.id


def test_point_ids_are_unique():
    ids = [p.id for _, p in all_points(RAG_REQUIREMENTS)]
    assert len(ids) == len(set(ids)) == 12


def test_the_required_teaching_content_is_in_the_curriculum():
    """Every item the assignment names must be a stated teaching point."""
    text = concepts_text(RAG_REQUIREMENTS).lower()
    required = [
        "what it is",  # not literally; checked via the points below
    ]
    del required
    expectations = {
        "llm.what_it_is": ("language model", "writes text"),
        "retrieval.source": ("knowledge source", "documents"),
        "retrieval.finding": ("searches", "relevant"),
        "augmentation.added_as_context": ("added", "question", "context"),
        "generation.writes_from_both": ("question", "retrieved context", "writes"),
        "flow.sequence": ("asks a question", "retrieved", "added as context"),
        "flow.example": ("single user question", "whole process"),
        "why.knowledge_gap": ("recent or private",),
        "why.hallucination": ("hallucinations",),
        "limits.fallible_sources": ("incomplete", "outdated", "incorrect"),
        "limits.no_guarantee": ("does not guarantee",),
    }
    for point_id, phrases in expectations.items():
        point = find_point(RAG_REQUIREMENTS, point_id)
        assert point is not None, point_id
        for phrase in phrases:
            assert phrase in point.text.lower(), (point_id, phrase)
    assert "[generation]" in text


def test_workflow_and_worked_example_are_defined():
    steps = workflow_text(RAG_REQUIREMENTS)
    assert steps.startswith("1. The user asks a question.")
    assert "5. The language model writes the answer using both." in steps
    assert "single user question" in RAG_REQUIREMENTS.worked_example


def test_concepts_text_lists_points_with_and_without_ids():
    with_ids = concepts_text(RAG_REQUIREMENTS)
    without = concepts_text(RAG_REQUIREMENTS, with_ids=False)
    assert "generation.writes_from_both:" in with_ids
    assert "generation.writes_from_both" not in without
    assert "The language model reads the question together" in without


def test_find_point():
    assert find_point(RAG_REQUIREMENTS, "retrieval.source").id == "retrieval.source"
    assert find_point(RAG_REQUIREMENTS, "nope") is None


def test_guardrails_text_general_and_topic_specific():
    general_only = guardrails_text()
    for rule in GENERAL_GUARDRAILS:
        assert rule in general_only
    assert "trains or updates" not in general_only
    with_topic = guardrails_text(RAG_REQUIREMENTS)
    assert "trains or updates" in with_topic
    assert "guarantees correct answers" in with_topic
    assert "public internet" in with_topic


def test_prompt_text_never_describes_matching_internals():
    text = concepts_text(RAG_REQUIREMENTS).lower()
    for word in ("anchor", "hint", "keyword", "signal"):
        assert word not in text


# --- anchor groups: a sanity check on quotes, never a judgement of the lesson ------


def missing(point_id: str, text: str) -> list[str]:
    point = find_point(RAG_REQUIREMENTS, point_id)
    return [g.label for g in unmet_groups(point, normalize_text(text))]


def test_a_sentence_touching_every_group_has_nothing_missing():
    sentence = (
        "The language model reads the question and the retrieved passages, then "
        "writes the answer."
    )
    assert missing("generation.writes_from_both", sentence) == []


def test_unrelated_sentences_miss_groups():
    assert missing("generation.writes_from_both", "A student writes an essay.") != []
    assert missing("limits.no_guarantee", "The sky is blue.") != []


def test_matching_is_by_whole_word_and_supports_prefix_terms():
    # "llm*" is a prefix term; "ai" is a whole word only.
    assert missing("llm.what_it_is", "An LLM is an AI system trained on text.") == []
    assert missing("llm.what_it_is", "Llms are an AI system trained on text.") == []
    assert missing("why.knowledge_gap", "A model can lack facts.") == []
    assert missing("why.knowledge_gap", "The mail arrives.") != []  # not the word "ai"
    # a bare mention of the term is not a statement of what it is
    assert missing("llm.what_it_is", "An LLM writes text.") != []
