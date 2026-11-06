"""Pydantic schemas for lesson content."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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
