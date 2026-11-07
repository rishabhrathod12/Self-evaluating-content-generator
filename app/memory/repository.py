"""Persistence for finished workflow runs: lessons, evaluations and rejection logs.

The workflow itself never touches the database. Call ``save_run`` with the state
returned by ``run_workflow`` to store it, and use the query methods to retrieve
stored lessons and evaluations later (for example to reuse the latest lesson that
passed for a topic).

What is stored is what the final state holds: the returned lesson with its own
evaluation, the latest lesson and its evaluation when they differ, the rejection log
and the run summary. Intermediate versions that the workflow did not keep (a lesson
that was neither the best nor the latest) are not stored.
"""

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.debug_log import redact
from app.evaluation.schemas import LessonContent
from app.memory.database import (
    MemoryDatabaseError,
    connect,
    init_schema,
    sqlite_path,
)


@dataclass(frozen=True)
class PromptVersion:
    """A stored prompt-refinement proposal (see ``app.memory.evolution``)."""

    id: int
    topic: str
    version: int  # per topic: 1, 2, 3, ...
    pattern_key: str
    failure_pattern: str
    rationale: str
    previous_instruction: str  # the approved instructions when this was proposed
    proposed_instruction: str
    created_at: str
    status: str  # proposed | approved | rejected
    decided_at: str | None


_UNSAFE_PHRASES = (
    "ignore",
    "skip",
    "relax",
    "lenient",
    "weaken",
    "lower the",
    "always pass",
    "do not check",
    "disregard",
)


def check_instruction_is_safe(text: str) -> None:
    """Refuse instruction text that tries to weaken the evaluation standards."""
    lowered = text.lower()
    for phrase in _UNSAFE_PHRASES:
        if phrase in lowered:
            raise ValueError(f"Refinement may not weaken the standards ({phrase!r}).")


@dataclass(frozen=True)
class StoredRun:
    """A stored workflow run."""

    id: int
    topic: str
    status: str
    created_at: str
    maximum_attempts: int
    evaluation_count: int
    improvement_attempts: int
    best_evaluation_number: int
    rejection_reason: str
    error: dict[str, Any] | None
    lesson: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    latest_lesson: dict[str, Any] | None
    latest_evaluation: dict[str, Any] | None
    rejection_log: list[dict[str, Any]]


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _scrub(text: str) -> str:
    """Remove credentials from free text before it is stored."""
    key = get_settings().openai_api_key
    return redact(text, key.get_secret_value() if key else "")


def _passed_checks(evaluation: Mapping[str, Any]) -> int:
    return sum(1 for check in evaluation["checks"] if check["passed"])


class MemoryRepository:
    """Stores and retrieves workflow runs in a SQLite database.

    A new connection is opened per operation, so one repository can be shared freely.
    That is why it needs a real file: ``:memory:`` does not survive a connection.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("The repository needs a database file, not :memory:.")
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            init_schema(connection)

    @classmethod
    def from_settings(cls) -> "MemoryRepository":
        """A repository at the ``DATABASE_URL`` setting."""
        return cls(sqlite_path(get_settings().database_url))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(connect(self.path)) as connection:
                yield connection
        except sqlite3.Error as exc:
            raise MemoryDatabaseError(
                f"SQLite memory at {self.path!r} failed: {exc}"
            ) from exc

    # --- writing --------------------------------------------------------------------

    def save_run(self, state: Mapping[str, Any]) -> int:
        """Store a finished run (the dict returned by ``run_workflow``).

        Returns:
            The new run's id.

        Raises:
            ValueError: The state is not finished (status ``in_progress``).
            MemoryDatabaseError: The database could not be written.
        """
        status = state["status"]
        if status not in ("passed", "rejected", "error"):
            raise ValueError(f"Only finished runs can be stored, got {status!r}.")
        topic = state["topic"]
        with self._connection() as connection, connection:  # one transaction
            cursor = connection.execute(
                "INSERT INTO runs (topic, topic_key, status, created_at, "
                "maximum_attempts, evaluation_count, improvement_attempts, "
                "best_evaluation_number, rejection_reason, error_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    topic,
                    topic.strip().casefold(),
                    status,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    state["maximum_attempts"],
                    state["evaluation_count"],
                    state["improvement_attempts"],
                    state["best_evaluation_number"],
                    _scrub(state["rejection_reason"]),
                    _scrub(_dump(state["error"])) if state["error"] else None,
                ),
            )
            run_id = int(cursor.lastrowid or 0)
            for role, lesson, evaluation in (
                ("returned", state["lesson"], state["evaluation"]),
                ("latest", state["latest_lesson"], state["latest_evaluation"]),
            ):
                if lesson is None:
                    continue
                connection.execute(
                    "INSERT INTO lessons (run_id, role, title, topic, content_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, role, lesson["title"], lesson["topic"], _dump(lesson)),
                )
                if evaluation is None:
                    continue
                points = evaluation.get("point_checks", [])
                concepts = evaluation.get("concept_checks", [])
                connection.execute(
                    "INSERT INTO evaluations (run_id, lesson_role, overall_pass, "
                    "passed_checks, covered_concepts, covered_points, notes_json, "
                    "evaluation_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        role,
                        int(evaluation["overall_pass"]),
                        _passed_checks(evaluation),
                        sum(1 for c in concepts if c["status"] == "covered"),
                        sum(1 for p in points if p["covered"]),
                        _dump(evaluation.get("notes", [])),
                        _dump(evaluation),
                    ),
                )
            for record in state["rejection_log"]:
                connection.execute(
                    "INSERT INTO rejection_log (run_id, attempt_number, "
                    "failed_criteria_json, record_json) VALUES (?, ?, ?, ?)",
                    (
                        run_id,
                        record["attempt_number"],
                        _dump(record["failed_criteria"]),
                        _dump(record),
                    ),
                )
        return run_id

    def delete_run(self, run_id: int) -> bool:
        """Delete a run and everything stored with it. True if it existed."""
        with self._connection() as connection, connection:
            cursor = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0

    # --- reading --------------------------------------------------------------------

    def get_run(self, run_id: int) -> StoredRun | None:
        """The stored run with ``run_id``, or ``None``."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            return self._assemble(connection, row)

    def list_runs(
        self, topic: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[StoredRun]:
        """Stored runs, newest first, optionally for one topic and/or status."""
        clauses, params = [], []
        if topic is not None:
            clauses.append("topic_key = ?")
            params.append(topic.strip().casefold())
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs {where} ORDER BY id DESC LIMIT ?",
                (*params, max(1, limit)),
            ).fetchall()
            return [self._assemble(connection, row) for row in rows]

    def count_runs(self) -> int:
        """How many runs are stored."""
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def latest_passed_lesson(self, topic: str) -> LessonContent | None:
        """The most recent lesson that passed evaluation for ``topic``, if any.

        This is the retrieval entry point for reusing earlier work: a lesson returned
        here was evaluated, and passed, in a run with status ``passed``.
        """
        with self._connection() as connection:
            row = connection.execute(
                "SELECT lessons.content_json FROM runs "
                "JOIN lessons ON lessons.run_id = runs.id "
                "AND lessons.role = 'returned' "
                "WHERE runs.topic_key = ? AND runs.status = 'passed' "
                "ORDER BY runs.id DESC LIMIT 1",
                (topic.strip().casefold(),),
            ).fetchone()
        if row is None:
            return None
        return LessonContent.model_validate(json.loads(row["content_json"]))

    def evaluations_for_topic(
        self, topic: str, only_passed: bool = False, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Stored evaluation summaries for a topic, newest first.

        Each item has ``run_id``, ``lesson_role``, ``overall_pass``, ``passed_checks``,
        ``covered_concepts``, ``covered_points`` and ``notes``.
        """
        passed = "AND evaluations.overall_pass = 1" if only_passed else ""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT evaluations.* FROM evaluations "
                "JOIN runs ON runs.id = evaluations.run_id "
                f"WHERE runs.topic_key = ? {passed} "
                "ORDER BY evaluations.run_id DESC, evaluations.lesson_role LIMIT ?",
                (topic.strip().casefold(), max(1, limit)),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "lesson_role": row["lesson_role"],
                "overall_pass": bool(row["overall_pass"]),
                "passed_checks": row["passed_checks"],
                "covered_concepts": row["covered_concepts"],
                "covered_points": row["covered_points"],
                "notes": json.loads(row["notes_json"]),
            }
            for row in rows
        ]

    # --- prompt versions -------------------------------------------------------------

    def add_prompt_version(
        self,
        topic: str,
        pattern_key: str,
        failure_pattern: str,
        rationale: str,
        proposed_instruction: str,
    ) -> PromptVersion | None:
        """Store a new ``proposed`` refinement; ``None`` if the pattern has one already.

        The version number is the next one for the topic, and the previous instruction
        is the currently approved instruction text (empty if there is none).

        Raises:
            ValueError: The instruction would weaken the evaluation standards.
        """
        check_instruction_is_safe(proposed_instruction)
        key = topic.strip().casefold()
        previous = "\n".join(self.approved_instructions(topic))
        with self._connection() as connection, connection:
            if connection.execute(
                "SELECT 1 FROM prompt_versions WHERE topic_key = ? AND pattern_key = ?",
                (key, pattern_key),
            ).fetchone():
                return None
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM prompt_versions "
                "WHERE topic_key = ?",
                (key,),
            ).fetchone()[0]
            cursor = connection.execute(
                "INSERT INTO prompt_versions (topic, topic_key, version, pattern_key, "
                "failure_pattern, rationale, previous_instruction, "
                "proposed_instruction, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    topic,
                    key,
                    version,
                    pattern_key,
                    failure_pattern,
                    rationale,
                    previous,
                    proposed_instruction,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            new_id = int(cursor.lastrowid or 0)
        found = self.get_prompt_version(new_id)
        assert found is not None
        return found

    def get_prompt_version(self, version_id: int) -> PromptVersion | None:
        """One stored prompt version, or ``None``."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM prompt_versions WHERE id = ?", (version_id,)
            ).fetchone()
        return self._prompt_version(row) if row else None

    def list_prompt_versions(
        self, topic: str | None = None, status: str | None = None
    ) -> list[PromptVersion]:
        """Prompt versions, oldest first, optionally for one topic and/or status."""
        clauses, params = [], []
        if topic is not None:
            clauses.append("topic_key = ?")
            params.append(topic.strip().casefold())
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM prompt_versions {where} ORDER BY id", params
            ).fetchall()
        return [self._prompt_version(row) for row in rows]

    def set_prompt_status(self, version_id: int, status: str) -> PromptVersion:
        """Approve or reject a proposal (a decided proposal cannot be changed).

        Raises:
            ValueError: Unknown id, invalid status, or the version was already decided.
        """
        if status not in ("approved", "rejected"):
            raise ValueError("Status must be 'approved' or 'rejected'.")
        with self._connection() as connection, connection:
            cursor = connection.execute(
                "UPDATE prompt_versions SET status = ?, decided_at = ? "
                "WHERE id = ? AND status = 'proposed'",
                (status, datetime.now(UTC).isoformat(timespec="seconds"), version_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"Prompt version {version_id} does not exist or is already decided."
                )
        found = self.get_prompt_version(version_id)
        assert found is not None
        return found

    def approved_instructions(self, topic: str) -> list[str]:
        """The approved refinement texts for a topic, in version order.

        This is the only way refinements reach the agents: proposed and rejected
        versions are never returned.
        """
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT proposed_instruction FROM prompt_versions "
                "WHERE topic_key = ? AND status = 'approved' ORDER BY version",
                (topic.strip().casefold(),),
            ).fetchall()
        return [row["proposed_instruction"] for row in rows]

    @staticmethod
    def _prompt_version(row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            id=row["id"],
            topic=row["topic"],
            version=row["version"],
            pattern_key=row["pattern_key"],
            failure_pattern=row["failure_pattern"],
            rationale=row["rationale"],
            previous_instruction=row["previous_instruction"],
            proposed_instruction=row["proposed_instruction"],
            created_at=row["created_at"],
            status=row["status"],
            decided_at=row["decided_at"],
        )

    @staticmethod
    def _assemble(connection: sqlite3.Connection, run: sqlite3.Row) -> StoredRun:
        run_id = run["id"]
        lessons = {
            row["role"]: json.loads(row["content_json"])
            for row in connection.execute(
                "SELECT role, content_json FROM lessons WHERE run_id = ?", (run_id,)
            )
        }
        evaluations = {
            row["lesson_role"]: json.loads(row["evaluation_json"])
            for row in connection.execute(
                "SELECT lesson_role, evaluation_json FROM evaluations WHERE run_id = ?",
                (run_id,),
            )
        }
        log = [
            json.loads(row["record_json"])
            for row in connection.execute(
                "SELECT record_json FROM rejection_log WHERE run_id = ? "
                "ORDER BY attempt_number",
                (run_id,),
            )
        ]
        return StoredRun(
            id=run_id,
            topic=run["topic"],
            status=run["status"],
            created_at=run["created_at"],
            maximum_attempts=run["maximum_attempts"],
            evaluation_count=run["evaluation_count"],
            improvement_attempts=run["improvement_attempts"],
            best_evaluation_number=run["best_evaluation_number"],
            rejection_reason=run["rejection_reason"],
            error=json.loads(run["error_json"]) if run["error_json"] else None,
            lesson=lessons.get("returned"),
            evaluation=evaluations.get("returned"),
            latest_lesson=lessons.get("latest"),
            latest_evaluation=evaluations.get("latest"),
            rejection_log=log,
        )
