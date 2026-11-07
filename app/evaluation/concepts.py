"""Pure logic for concept coverage: status derivation and evidence sanity checks."""

from collections.abc import Iterable
from functools import lru_cache

from app.evaluation.lesson_text import (
    content_words,
    normalize_text,
    split_sentences,
)
from app.evaluation.schemas import ConceptCheck, EvaluationResult, PointCheck
from app.evaluation.topics import (
    TeachingPoint,
    TopicRequirements,
    all_points,
    touches_all,
    unmet_groups,
)

_WINDOW = 6  # a point may be stated across up to this many consecutive sentences
_OVERLAP = 0.7  # share of a point's own words that also proves relevance


def _shares_the_points_words(point: TeachingPoint, text: str) -> bool:
    words = content_words(point.text)
    return bool(words) and len(words & content_words(text)) / len(words) >= _OVERLAP


def quote_unmet(point: TeachingPoint, quote: str) -> list[str]:
    """Labels of the anchor groups ``quote`` misses (empty = relevant).

    A quote is relevant if it touches every anchor group, or if it shares at
    least 70% of the point's own content words. The second path means a correct
    quote is not rejected just because its wording is unlisted, while a sentence
    about a neighbouring point (which shares only some of the words) still is.
    """
    unmet = [g.label for g in unmet_groups(point, normalize_text(quote))]
    if unmet and _shares_the_points_words(point, quote):
        return []
    return unmet


@lru_cache(maxsize=32)
def _windows(text: str) -> tuple[tuple[int, str, str, frozenset[str]], ...]:
    """Every run of up to six sentences per line: (size, text, normalized, words)."""
    windows = []
    for line in text.split("\n"):
        sentences = split_sentences(line)
        for start in range(len(sentences)):
            for size in range(1, min(_WINDOW, len(sentences) - start) + 1):
                window = " ".join(sentences[start : start + size])
                windows.append(
                    (
                        size,
                        window,
                        normalize_text(window),
                        frozenset(content_words(window)),
                    )
                )
    return tuple(windows)


def addressing_text(point: TeachingPoint, text: str) -> str | None:
    """The passage of up to six sentences in ``text`` that best states the point.

    A run of sentences qualifies if it touches every anchor group of the point (or
    shares most of its words). Among the qualifying runs the best is chosen: the one
    sharing the most of the point's own words, and the shortest on a tie, so the
    evidence is the statement itself rather than the first loosely related passage.
    Sentences never span line breaks. This is how code finds the evidence for a
    teaching point, and how a revised lesson is checked.
    """
    wanted = content_words(point.text)
    best: tuple[float, int] | None = None
    best_text: str | None = None
    for size, window, normalized, words in _windows(text):
        shared = len(wanted & words) / len(wanted) if wanted else 0.0
        if not touches_all(point, normalized) and shared < _OVERLAP:
            continue
        score = (round(shared, 6), -size)
        if best is None or score > best:
            best, best_text = score, window
    return best_text


def unaddressed_points(
    requirements: TopicRequirements | None,
    body: str,
    point_ids: Iterable[str],
) -> list[TeachingPoint]:
    """The requested points that no passage of ``body`` plausibly states."""
    if requirements is None:
        return []
    wanted = set(point_ids)
    return [
        point
        for _, point in all_points(requirements)
        if point.id in wanted and addressing_text(point, body) is None
    ]


def derive_concept_checks(
    requirements: TopicRequirements, point_checks: list[PointCheck]
) -> list[ConceptCheck]:
    """Derive each concept's status from the point verdicts.

    A point the evaluator did not judge counts as not covered.
    """
    by_id = {p.point_id: p for p in point_checks}
    derived = []
    for concept in requirements.concepts:
        covered, missing, gaps, quotes = 0, [], [], []
        for point in concept.points:
            check = by_id.get(point.id)
            if check is not None and check.covered:
                covered += 1
                quotes.append(check.evidence_quote)
            else:
                missing.append(point.id)
                gaps.append(check.gap if check else "Not evaluated.")
        if covered == len(concept.points):
            status = "covered"
        elif covered:
            status = "weak"
        else:
            status = "missing"
        derived.append(
            ConceptCheck(
                concept_id=concept.id,
                status=status,
                missing_points=missing,
                gaps=gaps,
                evidence_quotes=quotes,
            )
        )
    return derived


def requested_point_ids(evaluation: EvaluationResult) -> list[str]:
    """Ids of every point an evaluation found not covered (weak or missing)."""
    return [pid for c in evaluation.concept_checks for pid in c.missing_points]


def _stem_present(word: str, body_words: set[str]) -> bool:
    stem = word[:5]
    return any(other.startswith(stem) for other in body_words)


def unsupported_statements(
    statements: Iterable[str], sections_text: str, minimum: float = 0.5
) -> list[str]:
    """Statements (objectives, takeaways) whose words mostly do not appear in the
    sections. A lexical check, used only to avoid overriding an evaluator claim
    that might be right: failing it never fails a lesson by itself.
    """
    body_words = content_words(sections_text)
    unsupported = []
    for statement in statements:
        words = content_words(statement)
        if not words:
            continue
        present = sum(1 for w in words if _stem_present(w, body_words))
        if present / len(words) < minimum:
            unsupported.append(statement)
    return unsupported
