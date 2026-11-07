"""Pydantic schemas for lesson content."""

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _StrictModel(BaseModel):
    """Base model: strict types, no unknown fields, whitespace-trimmed strings."""

    model_config = ConfigDict(strict=True, extra="forbid")


class LessonSection(_StrictModel):
    """One teachable section of a lesson."""

    heading: NonEmptyStr = Field(description="Short, clear section heading.")
    explanation: NonEmptyStr = Field(
        description="Plain-English explanation of the idea for a complete beginner."
    )
    example: NonEmptyStr = Field(
        description="A concrete, practical example that illustrates the explanation."
    )


class LessonContent(_StrictModel):
    """A complete beginner-friendly lesson."""

    title: NonEmptyStr = Field(description="Lesson title.")
    topic: NonEmptyStr = Field(description="The topic the lesson covers.")
    learner_level: NonEmptyStr = Field(
        description="Intended audience, e.g. 'Beginner (no technical background)'."
    )
    learning_objectives: list[NonEmptyStr] = Field(
        min_length=2, description="What the learner will be able to do (at least 2)."
    )
    sections: list[LessonSection] = Field(
        min_length=3, description="Ordered lesson sections (at least 3)."
    )
    key_takeaways: list[NonEmptyStr] = Field(
        min_length=3, description="The most important points to remember (at least 3)."
    )
    summary: NonEmptyStr = Field(description="Short closing summary of the lesson.")


CriterionName = Literal[
    "Accuracy and groundedness",
    "Beginner-friendly language",
    "Teaches by example",
    "Clear explanations without unexplained jargon",
    "Covers key points",
    "Coherent teaching flow",
    "Overall lesson completeness",
]

CRITERIA: tuple[str, ...] = CriterionName.__args__  # type: ignore[attr-defined]


JARGON_CRITERION = CRITERIA[3]
ACCURACY_CRITERION = CRITERIA[0]


class EvaluationIssue(_StrictModel):
    """One specific, evidence-backed problem found by the evaluator."""

    problem: NonEmptyStr = Field(description="What is wrong, specifically.")
    quote: str = Field(
        default="",
        description="Exact text copied from the lesson that shows the problem. "
        "Empty only when the problem is something missing entirely.",
    )
    term: str = Field(
        default="",
        description="For jargon issues only: the exact unexplained term.",
    )
    correction: str = Field(
        default="",
        description="For accuracy issues: what the correct statement is. Required "
        "for accuracy failures.",
    )


class EvaluationCheck(_StrictModel):
    """The independent PASS/FAIL verdict for one quality checkpoint."""

    criterion: CriterionName = Field(description="The checkpoint being judged.")
    passed: bool = Field(description="True for PASS, False for FAIL.")
    reason: NonEmptyStr = Field(description="Specific justification for the verdict.")
    improvement_suggestion: str = Field(
        default="",
        description="Actionable fix. Required when the check fails, may be empty "
        "when it passes.",
    )

    evidence_quote: str = Field(
        default="",
        description="Exact sentence copied from the lesson that supports this "
        "verdict. Required when the check passes.",
    )
    issues: list[EvaluationIssue] = Field(
        default_factory=list,
        description="Every specific problem found for this check, one item each. "
        "Required when the check fails, empty when it passes.",
    )

    @model_validator(mode="after")
    def _failed_check_needs_detail(self) -> Self:
        if not self.passed and not self.improvement_suggestion.strip():
            raise ValueError(
                f"Failed check '{self.criterion}' needs an improvement_suggestion."
            )
        if not self.passed and not self.issues:
            raise ValueError(f"Failed check '{self.criterion}' needs issues.")
        if self.passed and not self.evidence_quote.strip():
            raise ValueError(f"Passed check '{self.criterion}' needs evidence_quote.")
        if not self.passed and self.criterion == JARGON_CRITERION:
            for issue in self.issues:
                if not issue.term.strip() or not issue.quote.strip():
                    raise ValueError(
                        "Each jargon issue needs the exact term and a quote of "
                        "its first use."
                    )
        if not self.passed and self.criterion == ACCURACY_CRITERION:
            for issue in self.issues:
                if not issue.quote.strip() or not issue.correction.strip():
                    raise ValueError(
                        "Each accuracy issue needs the exact problematic "
                        "claim (quote) and the correction."
                    )
        return self


class PointCheck(_StrictModel):
    """Whether the lesson explicitly states one teaching point.

    The model only says ``covered`` and, when not, the ``gap``. Code finds the
    evidence: ``evidence_quote`` is always a passage of the lesson located by code.
    """

    point_id: NonEmptyStr = Field(description="Id of the teaching point.")
    covered: bool = Field(
        description="True only if the lesson states the point explicitly in plain "
        "words. Merely mentioning a term is not enough."
    )
    evidence_quote: str = Field(
        default="",
        description="Set by code: the lesson passage that states the point. Never "
        "written by the model.",
    )
    gap: str = Field(
        default="",
        description="Exactly what the lesson must state but does not. Required "
        "when not covered.",
    )

    @model_validator(mode="after")
    def _gap_when_uncovered(self) -> Self:
        if not self.covered and not self.gap.strip():
            raise ValueError(f"Uncovered point '{self.point_id}' needs a gap.")
        return self


ConceptStatus = Literal["covered", "weak", "missing"]


class ConceptCheck(_StrictModel):
    """A concept's status, derived by code from its point checks.

    ``covered``: every point is explicitly stated. ``weak``: some are. ``missing``:
    none are.
    """

    concept_id: NonEmptyStr
    status: ConceptStatus
    missing_points: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)

    @property
    def covered(self) -> bool:
        """True when every teaching point of the concept is stated."""
        return self.status == "covered"


class EvaluationResult(_StrictModel):
    """The evaluation of one lesson across all seven checkpoints."""

    overall_pass: bool = Field(
        description="True only if every one of the seven checks passed."
    )
    checks: list[EvaluationCheck] = Field(
        min_length=7,
        max_length=7,
        description="Exactly seven checks, one per criterion.",
    )
    failure_reasons: list[NonEmptyStr] = Field(
        description="One reason per failed check. Empty when overall_pass is true."
    )
    improvement_feedback: str = Field(
        description="Combined actionable feedback for revising the lesson. "
        "Required when overall_pass is false, may be empty otherwise."
    )
    point_checks: list[PointCheck] = Field(
        default_factory=list,
        description="One entry per teaching point listed in the prompt, or an "
        "empty list when none are listed.",
    )
    concept_checks: list[ConceptCheck] = Field(
        default_factory=list,
        description="Set by code from point_checks, never by the model.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Set by code, never by the model: records every issue that was "
        "resolved automatically, with the matched definition and the reason.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [check.criterion for check in self.checks]
        if sorted(names) != sorted(CRITERIA):
            raise ValueError(
                "checks must contain each of the seven criteria exactly once."
            )
        all_passed = all(check.passed for check in self.checks)
        if self.overall_pass != all_passed:
            raise ValueError("overall_pass is inconsistent with the individual checks.")
        if self.overall_pass:
            if self.failure_reasons:
                raise ValueError("failure_reasons must be empty when overall_pass.")
        else:
            if not self.failure_reasons:
                raise ValueError("failure_reasons is required when a check fails.")
            if not self.improvement_feedback.strip():
                raise ValueError("improvement_feedback is required on failure.")
        return self


_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"default", "minLength", "maxLength", "minItems", "maxItems"}
)


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return ``model``'s JSON Schema adapted for OpenAI strict structured output.

    Strict mode needs every property listed as required and
    ``additionalProperties: false`` on every object, and does not accept some
    constraint keywords. Those constraints are dropped here only; Pydantic still
    enforces them when the response is validated.
    """

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        converted: dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                # Keys here are field names, not schema keywords: keep them all.
                converted[key] = {name: convert(sub) for name, sub in value.items()}
            elif key not in _UNSUPPORTED_SCHEMA_KEYS:
                converted[key] = convert(value)
        if "properties" in converted:
            converted["required"] = list(converted["properties"])
            converted["additionalProperties"] = False
        return converted

    return convert(model.model_json_schema())
