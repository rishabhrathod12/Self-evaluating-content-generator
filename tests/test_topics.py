import pytest
from app.evaluation.topics import (
    GENERAL_GUARDRAILS,
    RAG_REQUIREMENTS,
    concepts_text,
    get_requirements,
    guardrails_text,
)

EXPECTED_IDS = [
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
    assert [c.id for c in RAG_REQUIREMENTS.concepts] == EXPECTED_IDS


def test_concepts_are_well_formed():
    ids = [c.id for c in RAG_REQUIREMENTS.concepts]
    assert len(ids) == len(set(ids))
    for concept in RAG_REQUIREMENTS.concepts:
        assert concept.name and concept.description
        assert len(concept.hints) >= 2
        assert all(h == h.lower() for h in concept.hints)


def test_concepts_text_lists_every_concept():
    text = concepts_text(RAG_REQUIREMENTS)
    for concept in RAG_REQUIREMENTS.concepts:
        assert f"[{concept.id}]" in text
        assert concept.name in text


def test_guardrails_text_general_and_topic_specific():
    general_only = guardrails_text()
    for rule in GENERAL_GUARDRAILS:
        assert rule in general_only
    assert "trains or updates" not in general_only
    with_topic = guardrails_text(RAG_REQUIREMENTS)
    assert "trains or updates" in with_topic
    assert "guarantees correct answers" in with_topic
    assert "public internet" in with_topic


GENERIC_WORDS = "first then next finally context write documents step"


def test_generic_words_are_not_hints_for_any_concept():
    from app.evaluation.topics import has_hint

    for concept in RAG_REQUIREMENTS.concepts:
        assert not has_hint(concept, GENERIC_WORDS), concept.id
        for hint in concept.hints:
            assert hint.rstrip("*") not in {"write", "first", "then", "context", "next"}


def test_hints_do_not_leak_into_generator_or_improver_prompts():
    from app.evaluation.topics import concepts_text

    text = concepts_text(RAG_REQUIREMENTS)
    for hint in ("pasted into", "handed over", "knowledge base", "searches through"):
        assert hint not in text
    for word in ("hint", "keyword", "signal", "expected wording"):
        assert word not in text.lower()


def test_has_hint_uses_word_boundaries():
    from app.evaluation.topics import has_hint

    by_id = {c.id: c for c in RAG_REQUIREMENTS.concepts}
    llm = by_id["language_model"]
    assert has_hint(llm, "the llm answers")
    assert has_hint(llm, "many llms exist")
    assert not has_hint(llm, "willmore bullmarket allmost")
    limits = by_id["limitations"]
    assert has_hint(limits, "it is incomplete.")
    assert not has_hint(limits, "an incompletely filled form")
    assert not has_hint(limits, "incorrectly stored")
    assert has_hint(by_id["why_needed"], "models hallucinate sometimes")  # stem hint
    assert has_hint(by_id["why_needed"], "it cannot know last week's news")
    assert not has_hint(by_id["why_needed"], "he cannot knowingly lie")


def test_substring_collisions_do_not_count():
    from app.evaluation.topics import has_hint

    by_id = {c.id: c for c in RAG_REQUIREMENTS.concepts}
    assert not has_hint(by_id["retrieval"], "databasesx and nodatabase")
    assert not has_hint(by_id["end_to_end"], "a firsthand story, authentic then")
