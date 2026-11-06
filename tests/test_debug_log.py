import json
from types import SimpleNamespace

import pytest
from app import debug_log
from pydantic import SecretStr

SECRET = "sk-test-SECRET-1234567890"


def settings(debug: bool, key: str | None = SECRET):
    return SimpleNamespace(debug=debug, openai_api_key=SecretStr(key) if key else None)


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_nothing_is_written_when_debug_is_off(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(False))
    debug_log.log_event("evaluator_response", raw="text")
    assert not debug_log.LOG_PATH.exists()


def test_event_is_appended_as_json_when_debug_is_on(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    debug_log.log_event("evaluator_response", attempt=1, raw='{"a": 1}')
    debug_log.log_event("evaluator_reask", attempt=1, feedback="fix it")
    first, second = records(debug_log.LOG_PATH)
    assert first["event"] == "evaluator_response"
    assert first["attempt"] == 1
    assert first["raw"] == '{"a": 1}'
    assert "time" in first
    assert second["event"] == "evaluator_reask"


def test_configured_key_and_key_like_strings_are_redacted(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    debug_log.log_event(
        "evaluator_response",
        raw=f"the key is {SECRET} and another sk-abcdefghijklmnop here",
        nested={"deep": [f"also {SECRET}"]},
    )
    text = debug_log.LOG_PATH.read_text(encoding="utf-8")
    assert SECRET not in text
    assert "sk-abcdefghijklmnop" not in text
    assert text.count("[REDACTED]") == 3


def test_long_fields_are_truncated(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    debug_log.log_event(
        "evaluator_response", raw="x" * (debug_log.MAX_FIELD_CHARS + 500)
    )
    (record,) = records(debug_log.LOG_PATH)
    assert len(record["raw"]) < debug_log.MAX_FIELD_CHARS + 100
    assert "truncated" in record["raw"]


def test_logging_never_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    monkeypatch.setattr(debug_log, "LOG_PATH", tmp_path)  # a directory, not a file
    debug_log.log_event("evaluator_response", raw="text")  # must not raise

    def broken():
        raise RuntimeError("settings exploded")

    monkeypatch.setattr(debug_log, "get_settings", broken)
    debug_log.log_event("evaluator_response", raw="text")  # must not raise


def test_works_without_an_api_key(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True, key=None))
    debug_log.log_event("evaluator_response", raw="text")
    assert records(debug_log.LOG_PATH)[0]["raw"] == "text"


@pytest.mark.parametrize("secret", ["", "short"])
def test_redact_ignores_empty_or_tiny_secrets(secret):
    assert debug_log.redact("a short word", secret) == "a short word"


# --- Hardening: more credential formats, size cap, warn-once ------------------------

import logging  # noqa: E402

CREDENTIALS = {
    "bearer": "Authorization: Bearer abcdef0123456789abcdef",
    "bearer lowercase": "header bearer eyAbCdEf1234567890xyz",
    "github classic": "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "github oauth": "gho_abcdefghijklmnopqrstuvwxyz0123456789",
    "github fine grained": "github_pat_11ABCDEFG0abcdefghijklmnopqrst",
    "aws access key": "AKIAABCDEFGHIJKLMNOP",
    "google api key": "AIzaSyA-abcdefghijklmnopqrstuvwxyz012345",
    "slack token": "xoxb-1234567890-abcdefghij",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
    "openai project key": "sk-proj-abcdefghijklmnop1234",
    "assignment": "api_key = 'abcd1234efgh5678'",
    "password": "password: hunter2hunter2",
    "db url": "postgresql://admin:SuperSecret99@db.internal:5432/app",
    "private key": (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc123\n-----END RSA PRIVATE KEY-----"
    ),
}


@pytest.mark.parametrize("name", list(CREDENTIALS))
def test_common_credential_formats_are_redacted(name):
    secret_part = {
        "bearer": "abcdef0123456789abcdef",
        "bearer lowercase": "eyAbCdEf1234567890xyz",
        "github classic": "abcdefghijklmnopqrstuvwxyz0123456789",
        "github oauth": "abcdefghijklmnopqrstuvwxyz0123456789",
        "github fine grained": "11ABCDEFG0abcdefghijklmnopqrst",
        "aws access key": "AKIAABCDEFGHIJKLMNOP",
        "google api key": "AIzaSyA-abcdefghijklmnopqrstuvwxyz012345",
        "slack token": "xoxb-1234567890-abcdefghij",
        "jwt": "eyJhbGciOiJIUzI1NiJ9",
        "openai project key": "sk-proj-abcdefghijklmnop1234",
        "assignment": "abcd1234efgh5678",
        "password": "hunter2hunter2",
        "db url": "SuperSecret99",
        "private key": "MIIEabc123",
    }[name]
    out = debug_log.redact(CREDENTIALS[name])
    assert secret_part not in out
    assert "[REDACTED]" in out


@pytest.mark.parametrize(
    "text",
    [
        "Retrieval-Augmented Generation is a technique.",
        "A token is a small piece of text.",
        "The lesson has 5 sections and 3 takeaways.",
        "https://example.com/docs",
        "ask: what is a database?",
    ],
)
def test_ordinary_text_is_not_redacted(text):
    assert debug_log.redact(text) == text


def test_secret_named_environment_values_are_redacted(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    monkeypatch.setenv("MY_SERVICE_TOKEN", "plainvalue-without-prefix-42")
    monkeypatch.setenv("UNRELATED_SETTING", "harmless-value-1234")
    debug_log.log_event(
        "evaluator_response",
        raw="a plainvalue-without-prefix-42 b harmless-value-1234 c",
    )
    text = debug_log.LOG_PATH.read_text(encoding="utf-8")
    assert "plainvalue-without-prefix-42" not in text
    assert "harmless-value-1234" in text  # only secret-named variables count


def test_secret_values_from_the_env_file_are_redacted(monkeypatch, tmp_path):
    env = tmp_path / "test.env"
    env.write_text(
        "MY_DB_PASSWORD=hunter2hunter2\nOPENAI_MODEL=gpt-4o-mini\n# comment\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(debug_log, "ENV_FILE", env)
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    debug_log.log_event("evaluator_response", raw="pw hunter2hunter2 model gpt-4o-mini")
    (record,) = records(debug_log.LOG_PATH)
    assert "hunter2hunter2" not in record["raw"]
    assert "gpt-4o-mini" in record["raw"]  # non-secret settings are untouched


def test_missing_env_file_is_fine(monkeypatch, tmp_path):
    monkeypatch.setattr(debug_log, "ENV_FILE", tmp_path / "does-not-exist.env")
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    debug_log.log_event("evaluator_response", raw="text")
    assert records(debug_log.LOG_PATH)[0]["raw"] == "text"


def test_log_rotates_at_the_size_cap_and_keeps_one_backup(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    monkeypatch.setattr(debug_log, "MAX_LOG_BYTES", 400)
    backup = debug_log.LOG_PATH.with_name(debug_log.LOG_PATH.name + ".1")

    for i in range(4):
        debug_log.log_event("evaluator_response", raw=f"{i} " + "x" * 150)
    assert backup.exists()
    assert debug_log.LOG_PATH.stat().st_size < 400 + 400  # one record may overshoot

    first_backup = backup.read_text(encoding="utf-8")
    for i in range(10):
        debug_log.log_event("evaluator_response", raw=f"later {i} " + "y" * 150)
    assert backup.read_text(encoding="utf-8") != first_backup  # replaced, not kept
    files = sorted(p.name for p in debug_log.LOG_PATH.parent.iterdir())
    assert files == ["debug_log.jsonl", "debug_log.jsonl.1"]  # never more than two


def test_no_rotation_below_the_cap(monkeypatch):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    for i in range(3):
        debug_log.log_event("evaluator_response", raw=f"small {i}")
    assert not debug_log.LOG_PATH.with_name("debug_log.jsonl.1").exists()
    assert len(records(debug_log.LOG_PATH)) == 3


def test_repeated_failures_warn_only_once_and_never_raise(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    monkeypatch.setattr(debug_log, "LOG_PATH", tmp_path)  # a directory: writes fail
    with caplog.at_level(logging.WARNING, logger="app.debug"):
        for _ in range(5):
            debug_log.log_event("evaluator_response", raw="text")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "Debug logging failed" in message
    assert str(tmp_path) not in message  # type only, never paths or values


def test_a_failing_warning_cannot_break_the_run(monkeypatch, tmp_path):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(True))
    monkeypatch.setattr(debug_log, "LOG_PATH", tmp_path)

    def broken_warning(*args, **kwargs):
        raise RuntimeError("logger exploded")

    monkeypatch.setattr(debug_log.logger, "warning", broken_warning)
    debug_log.log_event("evaluator_response", raw="text")  # must not raise


def test_no_warning_when_debug_is_off(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(debug_log, "get_settings", lambda: settings(False))
    monkeypatch.setattr(debug_log, "LOG_PATH", tmp_path)
    with caplog.at_level(logging.WARNING, logger="app.debug"):
        debug_log.log_event("evaluator_response", raw="text")
    assert not caplog.records


def test_gitignore_covers_the_log_its_backup_and_env():
    import pathlib

    lines = {
        line.strip()
        for line in pathlib.Path(".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert ".env" in lines
    assert "outputs/*" in lines  # covers debug_log.jsonl and debug_log.jsonl.1
    assert "!outputs/.gitkeep" in lines
