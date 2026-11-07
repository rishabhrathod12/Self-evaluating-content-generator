"""SQLite connection and schema for the lesson memory (standard library only)."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2
MEMORY_PATH = ":memory:"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('passed', 'rejected', 'error')),
    created_at TEXT NOT NULL,
    maximum_attempts INTEGER NOT NULL,
    evaluation_count INTEGER NOT NULL,
    improvement_attempts INTEGER NOT NULL,
    best_evaluation_number INTEGER NOT NULL,
    rejection_reason TEXT NOT NULL,
    error_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_topic ON runs (topic_key, status, id);

-- The returned lesson (always the best evaluated one when any was evaluated) and,
-- if different, the latest lesson, which may never have been evaluated.
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('returned', 'latest')),
    title TEXT NOT NULL,
    topic TEXT NOT NULL,
    content_json TEXT NOT NULL,
    UNIQUE (run_id, role)
);

-- The evaluation of the lesson with the same role (absent if never evaluated).
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    lesson_role TEXT NOT NULL CHECK (lesson_role IN ('returned', 'latest')),
    overall_pass INTEGER NOT NULL,
    passed_checks INTEGER NOT NULL,
    covered_concepts INTEGER NOT NULL,
    covered_points INTEGER NOT NULL,
    notes_json TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    UNIQUE (run_id, lesson_role)
);

CREATE TABLE IF NOT EXISTS rejection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    failed_criteria_json TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE (run_id, attempt_number)
);

-- Proposed refinements to the generator/improver instructions. Only rows with
-- status 'approved' are ever used. One proposal per (topic, failure pattern), ever,
-- so a rejected proposal is not proposed again.
CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    pattern_key TEXT NOT NULL,
    failure_pattern TEXT NOT NULL,
    rationale TEXT NOT NULL,
    previous_instruction TEXT NOT NULL,
    proposed_instruction TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed'
        CHECK (status IN ('proposed', 'approved', 'rejected')),
    decided_at TEXT,
    UNIQUE (topic_key, version),
    UNIQUE (topic_key, pattern_key)
);
"""


class MemoryDatabaseError(Exception):
    """The SQLite memory could not be opened, read or written."""


def sqlite_path(database_url: str) -> str:
    """The file path (or ``:memory:``) of a ``sqlite:///...`` URL.

    ``sqlite:///./memory.db`` is relative, ``sqlite:////abs/path.db`` is absolute.

    Raises:
        ValueError: The URL is not a ``sqlite:///`` URL.
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix) or len(database_url) == len(prefix):
        raise ValueError(
            f"Only sqlite:/// database URLs are supported, got {database_url!r}."
        )
    return database_url[len(prefix) :]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with rows as mappings and foreign keys enforced."""
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    """Create the tables if they do not exist. Safe to call repeatedly.

    Raises:
        RuntimeError: The database was created by a newer schema version.
    """
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {version} is newer than this code "
            f"({SCHEMA_VERSION})."
        )
    connection.executescript(SCHEMA)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()
