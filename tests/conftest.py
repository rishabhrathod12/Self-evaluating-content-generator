import pytest
from app import debug_log


@pytest.fixture(autouse=True)
def _isolate_debug_log(tmp_path, monkeypatch):
    """Never let tests write to the real log or read the real .env for secrets."""
    monkeypatch.setattr(debug_log, "LOG_PATH", tmp_path / "debug_log.jsonl")
    monkeypatch.setattr(debug_log, "ENV_FILE", tmp_path / "no.env")
    monkeypatch.setattr(debug_log, "_warned", False)
