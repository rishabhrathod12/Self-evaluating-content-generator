import pytest
from app.evaluation.concepts import (
    addressing_text,
    derive_concept_checks,
    quote_unmet,
    requested_point_ids,
    unaddressed_points,
)
from app.evaluation.lesson_text import body_text, content_words, normalize_text
from app.evaluation.schemas import PointCheck
from app.evaluation.topics import RAG_REQUIREMENTS, all_points, find_point

from tests.rag_fixtures import (
    EVIDENCE,
    as_result,
    concept_points,
    eval_data,
    rag_lesson,
)

POINTS = {p.id: p for _, p in all_points(RAG_REQUIREMENTS)}

# Correct sentences taken from real model output, worded unlike our fixtures.
REAL_CORRECT = {
    "definition.name": (
        "Retrieval-Augmented Generation, or RAG, is a method that helps language "
        "models find and use information."
    ),
    "why.knowledge_gap": (
        "Language models sometimes lack recent information or specific details."
    ),
    "why.hallucination": (
        "Without the right facts, a language model may provide inaccurate or "
        "outdated answers."
    ),
    "llm.what_it_is": (
        "A language model (LLM) is an AI system designed to generate text based on "
        "patterns it learns from vast amounts of data."
    ),
    "retrieval.finding": (
        "Retrieval is the process of finding relevant information from a "
        "collection of knowledge, which may include documents, databases, or "
        "other sources."
    ),
    "augmentation.added_as_context": (
        "Augmentation involves using the retrieved information as context for the "
        "language model to generate a response."
    ),
    "generation.writes_from_both": (
        "Generation is the process where the language model takes both the user's "
        "question and the retrieved context to produce a coherent answer."
    ),
    "flow.sequence": (
        "End-to-end interaction refers to the entire sequence of events: asking a "
        "question, retrieving relevant information, providing that context to the "
        "model, and then generating an answer."
    ),
    "limits.no_guarantee": (
        "RAG does not ensure the accuracy of the answers because it depends on the "
        "quality of the information retrieved."
    ),
    # a wording that an earlier synonym list rejected
    "why.hallucination ": (
        "When a language model lacks the facts it may make things up."
    ),
}


@pytest.mark.parametrize("point_id", sorted(REAL_CORRECT))
def test_genuinely_correct_quotes_are_never_rejected(point_id):
    assert quote_unmet(POINTS[point_id.strip()], REAL_CORRECT[point_id]) == []


@pytest.mark.parametrize(
    ("point_id", "quote"),
    [
        ("generation.writes_from_both", "Think of RAG like a student who researches."),
        ("augmentation.added_as_context", "Imagine a newspaper that prints news."),
        ("retrieval.source", "A language model is an AI system that writes text."),
        ("limits.no_guarantee", "Picture a student who has gathered notes."),
        ("flow.sequence", "RAG is a method that helps language models."),
        ("llm.what_it_is", "RAG stands for Retrieval-Augmented Generation."),
    ],
)
def test_irrelevant_quotes_are_rejected_and_the_missing_ideas_named(point_id, quote):
    unmet = quote_unmet(POINTS[point_id], quote)
    assert unmet
    assert all(isinstance(label, str) and label for label in unmet)


def test_keyword_mentions_do_not_prove_a_point():
    # Mentions the words but states nothing about using the question and context.
    assert quote_unmet(
        POINTS["generation.writes_from_both"],
        "Generation is an important part of RAG.",
    )


def test_overlap_path_accepts_unlisted_wording_but_only_when_words_really_match():
    # Different phrasing, but it shares most of the point's own words.
    assert (
        quote_unmet(
            POINTS["retrieval.finding"],
            "Retrieval searches that source and finds the pieces most relevant to "
            "the question.",
        )
        == []
    )
    # Shares almost none of them.
    assert quote_unmet(POINTS["retrieval.finding"], "Cats like warm places.")


def test_content_words_ignore_stopwords_and_short_words():
    assert content_words("The model is on it") == {"model"}
    assert "the" not in content_words("The question")


def test_addressing_text_finds_the_stating_passage_across_sentences():
    body = (
        "Some intro. The model reads the question. It also reads the retrieved "
        "passages. Then it writes an answer."
    )
    found = addressing_text(POINTS["generation.writes_from_both"], body)
    assert found is not None
    assert "question" in found and "answer" in found


def test_addressing_text_none_when_the_point_is_not_stated():
    body = "RAG is a method. It is useful. Language models exist."
    assert addressing_text(POINTS["generation.writes_from_both"], body) is None


def test_addressing_text_never_spans_line_breaks():
    body = (
        "The model reads the question.\nThe retrieved passages exist. Answers follow."
    )
    # question is on line 1; context and writing are on line 2: not one passage
    assert addressing_text(POINTS["generation.writes_from_both"], body) is None


def test_unaddressed_points_only_reports_requested_points():
    body = (
        "RAG does not guarantee that answers are accurate.\n"
        "Retrieval finds relevant documents in the knowledge source."
    )
    missing = unaddressed_points(
        RAG_REQUIREMENTS, body, ["generation.writes_from_both", "limits.no_guarantee"]
    )
    assert [p.id for p in missing] == ["generation.writes_from_both"]


def test_unaddressed_points_for_a_complete_lesson_is_empty():
    ids = [p.id for p in POINTS.values()]
    assert unaddressed_points(RAG_REQUIREMENTS, body_text(rag_lesson()), ids) == []


def test_unaddressed_points_without_requirements():
    assert unaddressed_points(None, "anything", ["x"]) == []


def test_a_lesson_that_only_mentions_the_keywords_leaves_points_unaddressed():
    vague = ("Overview", "The parts work together.", "X")
    keyword_lesson = rag_lesson(
        replace={
            "generation": ("Generation", "Generation is an important step.", "X"),
            "augmentation": ("Augmentation", "Augmentation matters a lot.", "X"),
            "end_to_end": vague,
        },
        takeaways=("RAG is useful.", "It has parts.", "It has limits."),
        summary="RAG is a topic with several parts.",
    )
    ids = ["generation.writes_from_both", "augmentation.added_as_context"]
    missing = unaddressed_points(RAG_REQUIREMENTS, body_text(keyword_lesson), ids)
    assert {p.id for p in missing} == set(ids)


def test_the_lexical_check_is_lenient_by_design():
    """It is a plausibility check, not proof: another passage may touch the same
    vocabulary, so it can under-report. The evaluator's judgement stays the arbiter
    and a revision is never reported as a success on this check alone."""
    lesson = rag_lesson(omit=("generation",))  # the workflow section still says it
    missing = unaddressed_points(
        RAG_REQUIREMENTS, body_text(lesson), ["generation.writes_from_both"]
    )
    assert missing == []


# --- derive_concept_checks ------------------------------------------------------------


def _checks(uncovered=()):
    data = eval_data(uncovered=uncovered)
    return [PointCheck.model_validate(p) for p in data["point_checks"]]


def test_all_points_covered_means_every_concept_is_covered():
    derived = derive_concept_checks(RAG_REQUIREMENTS, _checks())
    assert [c.status for c in derived] == ["covered"] * 8
    assert all(c.covered for c in derived)
    assert all(c.missing_points == [] for c in derived)


def test_status_is_missing_when_no_point_is_covered():
    derived = derive_concept_checks(
        RAG_REQUIREMENTS, _checks(uncovered=concept_points("generation"))
    )
    generation = next(c for c in derived if c.concept_id == "generation")
    assert generation.status == "missing"
    assert generation.missing_points == ["generation.writes_from_both"]
    assert "never states" in generation.gaps[0]
    assert generation.evidence_quotes == []


def test_status_is_weak_when_only_some_points_are_covered():
    derived = derive_concept_checks(
        RAG_REQUIREMENTS, _checks(uncovered=("why.hallucination",))
    )
    why = next(c for c in derived if c.concept_id == "why_needed")
    assert why.status == "weak"
    assert why.missing_points == ["why.hallucination"]
    assert why.evidence_quotes == [EVIDENCE["why.knowledge_gap"]]
    assert not why.covered


def test_a_point_the_evaluator_did_not_judge_counts_as_not_covered():
    only_one = [c for c in _checks() if c.point_id == "definition.name"]
    derived = derive_concept_checks(RAG_REQUIREMENTS, only_one)
    by_id = {c.concept_id: c for c in derived}
    assert by_id["definition"].status == "covered"
    assert by_id["generation"].status == "missing"
    assert by_id["generation"].gaps == ["Not evaluated."]


def test_requested_point_ids_lists_every_open_point():
    data = eval_data(uncovered=("generation.writes_from_both", "why.hallucination"))
    result = as_result(data).model_copy(
        update={
            "concept_checks": derive_concept_checks(
                RAG_REQUIREMENTS,
                [PointCheck.model_validate(p) for p in data["point_checks"]],
            )
        }
    )
    assert sorted(requested_point_ids(result)) == [
        "generation.writes_from_both",
        "why.hallucination",
    ]


def test_normalize_text_straightens_curly_quotes():
    curly = "Ravi\N{RIGHT SINGLE QUOTATION MARK}s  \N{LEFT DOUBLE QUOTATION MARK}Idea"
    assert normalize_text(curly) == "ravi's \"idea"


def test_find_point_helper_is_consistent_with_all_points():
    assert (
        find_point(RAG_REQUIREMENTS, "generation.writes_from_both")
        is POINTS["generation.writes_from_both"]
    )


from app.evaluation.concepts import unsupported_statements  # noqa: E402

# --- unsupported_statements -----------------------------------------------------------

SECTIONS_TEXT = (
    "Retrieval searches a knowledge source. The language model writes the answer "
    "from the question and the retrieved context."
)


def test_statements_whose_words_appear_in_the_sections_are_supported():
    assert (
        unsupported_statements(
            ["Retrieval searches the knowledge source.", "The model writes answers."],
            SECTIONS_TEXT,
        )
        == []
    )


def test_statements_about_something_else_are_flagged():
    flagged = unsupported_statements(
        ["Explain quantum computing hardware.", "Understand retrieval."], SECTIONS_TEXT
    )
    assert flagged == ["Explain quantum computing hardware."]


def test_stemming_lets_inflected_words_match():
    assert (
        unsupported_statements(
            ["Learning about retrieving documents."], "Retrieval of documents."
        )
        == []
    )


def test_empty_statements_are_ignored():
    assert unsupported_statements(["", "the and of"], SECTIONS_TEXT) == []
