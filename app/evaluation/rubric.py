"""The quality rubric: the single source of truth for the seven checkpoints.

The generator writes to this rubric, the evaluator judges against it and the
improver targets its failed items, so all three share the same definitions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RubricItem:
    """One checkpoint with concrete, checkable pass conditions."""

    name: str
    pass_when: tuple[str, ...]


RUBRIC: tuple[RubricItem, ...] = (
    RubricItem(
        "Accuracy and groundedness",
        (
            "No claim contradicts well-established knowledge.",
            "No invented facts, statistics, names or dates.",
            "Examples agree with the explanations they illustrate.",
            "The lesson correctly says what kind of thing the topic is and what "
            "larger system it belongs to or works with.",
            "You have no external sources: judge only from your own knowledge and "
            "the lesson's internal consistency, and never claim to have verified "
            "anything externally.",
        ),
    ),
    RubricItem(
        "Beginner-friendly language",
        (
            "Short sentences and everyday words suitable for a 12th-grade "
            "graduate with no technical background.",
            "No unexplained abbreviations.",
            "Analogies come from everyday life.",
        ),
    ),
    RubricItem(
        "Teaches by example",
        (
            "Every section has an example.",
            "Each example is concrete and specific (a named situation), not a "
            "generic restatement.",
            "Each example really illustrates that section's idea.",
        ),
    ),
    RubricItem(
        "Clear explanations without unexplained jargon",
        (
            "Consider EVERY technical term, acronym and specialized word that a "
            "non-technical learner might not know, anywhere in the lesson, "
            "including the words that make up an acronym's name.",
            "A term is explained if the lesson defines it in plain words in the "
            "same sentence, the next sentence or the same paragraph as its first "
            "use, or earlier. Search the whole lesson before flagging a term.",
            "Titles, section headings and the topic line are labels, not uses of "
            "a term. Judge first use in the lesson body: the objectives, "
            "explanations, examples, takeaways and summary.",
            "Report ALL unexplained terms, not just one, each with the term "
            "itself and an exact quote of its first use.",
        ),
    ),
    RubricItem(
        "Covers key points",
        (
            "The lesson explicitly says what the topic is.",
            "It explicitly says why it matters.",
            "It explains how it works step by step.",
            "It gives practical examples.",
        ),
    ),
    RubricItem(
        "Coherent teaching flow",
        (
            "Learning objectives, sections, key takeaways and summary agree with "
            "each other.",
            "Each section builds on the previous one.",
            "No contradictions or pointless repetition.",
        ),
    ),
    RubricItem(
        "Overall lesson completeness",
        (
            "Every learning objective is actually taught by some section.",
            "Every key takeaway is supported by the sections.",
            "Each explanation is substantive (roughly 3 or more sentences), not a "
            "one-liner.",
            "The mechanism is explained (how the parts interact), not just named.",
            "If something is missing, name the exact missing piece. Do not say "
            "only that the lesson 'lacks depth'.",
        ),
    ),
)


def get_item(name: str) -> RubricItem:
    """Return the rubric item called ``name``."""
    for item in RUBRIC:
        if item.name == name:
            return item
    raise KeyError(name)


def format_item(item: RubricItem) -> str:
    """Format one item as ``"Name": cond; cond; ...``."""
    return f'"{item.name}": ' + " ".join(item.pass_when)


def rubric_text(names: list[str] | None = None) -> str:
    """Return the rubric (or the named subset) as numbered prompt text."""
    items = RUBRIC if names is None else tuple(get_item(n) for n in names)
    lines = []
    for number, item in enumerate(items, 1):
        lines.append(f'{number}. "{item.name}". Passes only if:')
        lines.extend(f"   - {condition}" for condition in item.pass_when)
    return "\n".join(lines)
