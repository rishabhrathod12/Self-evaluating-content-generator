"""Pydantic schemas for lesson content."""

from typing import Annotated, Literal, Self

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

    @model_validator(mode="after")
    def _failed_check_needs_suggestion(self) -> Self:
        if not self.passed and not self.improvement_suggestion.strip():
            raise ValueError(
                f"Failed check '{self.criterion}' needs an improvement_suggestion."
            )
        return self


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
