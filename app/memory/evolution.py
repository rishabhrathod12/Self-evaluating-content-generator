"""Controlled self-evolution: learn from repeated failures, but only with approval.

1. ``detect_repeated_patterns`` reads the stored rejection logs of a topic and finds
   failures (a failed rubric criterion, or a concept that stayed missing/weak) that
   occurred in at least ``REPEAT_THRESHOLD`` different runs.
2. ``propose_refinements`` turns each new pattern into a *proposed* instruction and
   stores it as a prompt version. Proposals are built from fixed templates that quote
   the rubric's own pass conditions or the curriculum's own teaching points, so they
   can only ask the agents to be more careful, never less. Each pattern is proposed
   once, ever (a rejected proposal is not proposed again).
3. A person approves or rejects a proposal (``approve`` / ``reject``, or
   ``python -m app.evolve``). Only approved refinements are read by
   ``build_memory_context`` and so used in future runs.

Nothing here touches the seven criteria, evaluator scoring or the retry limit.
"""

from dataclasses import dataclass

from app.evaluation.rubric import get_item
from app.evaluation.topics import get_requirements
from app.memory.repository import MemoryRepository, PromptVersion

REPEAT_THRESHOLD = 2  # runs in which a failure must appear to count as repeated
MAX_NEW_PROPOSALS = 3  # per call, most frequent patterns first
MAX_RUNS_ANALYSED = 200


@dataclass(frozen=True)
class FailurePattern:
    """A failure that repeated across runs of one topic."""

    key: str  # "criterion:<name>" or "concept:<id>"
    description: str
    run_ids: tuple[int, ...]  # distinct runs it appeared in, newest first
    latest_feedback: str  # evaluator feedback from the newest occurrence

    @property
    def count(self) -> int:
        return len(self.run_ids)


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def detect_repeated_patterns(
    repository: MemoryRepository, topic: str, threshold: int = REPEAT_THRESHOLD
) -> list[FailurePattern]:
    """Failures that appeared in at least ``threshold`` different runs of ``topic``.

    Most frequent first. Counts runs, not attempts, so one run that fails the same
    check twice still counts once.
    """
    requirements = get_requirements(topic)
    concept_names = (
        {c.id: c.name for c in requirements.concepts} if requirements else {}
    )
    runs_by_key: dict[str, list[int]] = {}
    descriptions: dict[str, str] = {}
    feedback: dict[str, str] = {}
    runs = repository.list_runs(topic=topic, limit=MAX_RUNS_ANALYSED)  # newest first
    for run in runs:
        for record in run.rejection_log:
            found = {
                f"criterion:{name}": f"Criterion '{name}' failed"
                for name in record.get("failed_criteria", [])
            }
            open_concepts = [
                *record.get("missing_concepts", []),
                *record.get("weak_concepts", []),
            ]
            for cid in open_concepts:
                found[f"concept:{cid}"] = (
                    f"Concept '{concept_names.get(cid, cid)}' was missing or weak"
                )
            for key, text in found.items():
                runs = runs_by_key.setdefault(key, [])
                if run.id not in runs:
                    runs.append(run.id)
                    descriptions[key] = text
                    feedback.setdefault(key, record.get("improvement_feedback", ""))
    patterns = [
        FailurePattern(key, descriptions[key], tuple(runs), _clip(feedback[key]))
        for key, runs in runs_by_key.items()
        if len(runs) >= threshold
    ]
    return sorted(patterns, key=lambda p: (-p.count, p.key))


def build_instruction(topic: str, pattern: FailurePattern) -> str | None:
    """The proposed instruction for a pattern, from a fixed template (or ``None``)."""
    kind, _, name = pattern.key.partition(":")
    if kind == "criterion":
        try:
            item = get_item(name)
        except KeyError:
            return None
        conditions = " ".join(item.pass_when[:4])
        return (
            f"Pay particular attention to the checkpoint '{name}', which earlier "
            f"drafts failed repeatedly. Make sure that: {conditions}"
        )
    requirements = get_requirements(topic)
    concept = (
        next((c for c in requirements.concepts if c.id == name), None)
        if (requirements and kind == "concept")
        else None
    )
    if concept is None:
        return None
    points = " ".join(point.text for point in concept.points)
    return (
        f"Earlier drafts repeatedly left the concept '{concept.name}' missing or only "
        "partly taught. Teach it clearly and state each of these points in a full, "
        f"explicit sentence: {points}"
    )


def propose_refinements(
    repository: MemoryRepository, topic: str, max_new: int = MAX_NEW_PROPOSALS
) -> list[PromptVersion]:
    """Store a proposal for each repeated failure pattern that has none yet.

    Proposals start as ``proposed`` and have no effect until approved. Returns only
    the newly created ones (empty when there is no history or nothing repeated).
    """
    created: list[PromptVersion] = []
    for pattern in detect_repeated_patterns(repository, topic):
        if len(created) >= max_new:
            break
        instruction = build_instruction(topic, pattern)
        if instruction is None:
            continue
        ids = ", ".join(map(str, pattern.run_ids))
        rationale = (
            f"Seen in {pattern.count} runs (ids {ids}). "
            f"Latest evaluator feedback: {pattern.latest_feedback or 'none recorded'}"
        )
        version = repository.add_prompt_version(
            topic, pattern.key, pattern.description, rationale, instruction
        )
        if version is not None:  # None: this pattern already has a proposal
            created.append(version)
    return created


def approve(repository: MemoryRepository, version_id: int) -> PromptVersion:
    """Approve a proposal so that future runs use it."""
    return repository.set_prompt_status(version_id, "approved")


def reject(repository: MemoryRepository, version_id: int) -> PromptVersion:
    """Reject a proposal. It is never used and never proposed again."""
    return repository.set_prompt_status(version_id, "rejected")
