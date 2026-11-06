"""Safe debug logging of model responses, re-asks and validation failures.

Off by default. Set ``DEBUG=true`` in ``.env`` to append JSON lines to
``outputs/debug_log.jsonl`` (git-ignored). Prompts are never logged and
credentials are redacted from everything that is. The log can still contain the
lesson topic and the full generated lessons, so treat it as sensitive. It rotates
at ``MAX_LOG_BYTES`` (keeping one backup), and logging can never break a run.
"""

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from typing import Any

from app.config import ENV_FILE, ROOT_DIR, get_settings

logger = logging.getLogger("app.debug")

LOG_PATH = ROOT_DIR / "outputs" / "debug_log.jsonl"
MAX_FIELD_CHARS = 20_000
MAX_LOG_BYTES = 5_000_000  # one backup (".1") is kept, so at most about twice this
_MIN_SECRET_LENGTH = 8
_REDACTED = "[REDACTED]"
_warned = False

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        _REDACTED,
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]{8,}=*"), f"Bearer {_REDACTED}"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), _REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), _REDACTED),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), _REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), _REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), _REDACTED),
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
        _REDACTED,
    ),
    # user:password@host in connection strings
    (re.compile(r"(\b\w+://[^\s:/@]+:)[^\s@/]+@"), rf"\1{_REDACTED}@"),
    # api_key=..., password: ..., token = ...
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret|token|password|passwd|authorization)\w*)"
            r"(\s*[:=]\s*)(?!\[REDACTED\])[\"']?[^\s\"',;]{6,}"
        ),
        rf"\1\2{_REDACTED}",
    ),
)
_SECRET_NAME = re.compile(r"(?i)(key|secret|token|password|passwd|credential)")


def _env_file_secrets() -> set[str]:
    """Values of secret-looking variables in the root .env file."""
    values: set[str] = set()
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and _SECRET_NAME.search(name):
                values.add(value.strip().strip("\"'"))
    except OSError:
        pass
    return values


def _known_secrets(configured: str) -> list[str]:
    """Literal secret values to remove: the API key and secret-named settings."""
    values = {configured, *_env_file_secrets()}
    values.update(v for k, v in os.environ.items() if _SECRET_NAME.search(k))
    return sorted(
        (v for v in values if len(v) >= _MIN_SECRET_LENGTH), key=len, reverse=True
    )


def redact(text: str, secret: str | Iterable[str] = "") -> str:
    """Remove credentials: known secret values and common credential patterns."""
    secrets = [secret] if isinstance(secret, str) else list(secret)
    for value in secrets:
        if len(value) >= _MIN_SECRET_LENGTH:
            text = text.replace(value, _REDACTED)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _clean(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        text = redact(value, secrets)
        if len(text) > MAX_FIELD_CHARS:
            text = text[:MAX_FIELD_CHARS] + f"...[truncated {len(text)} chars]"
        return text
    if isinstance(value, dict):
        return {str(k): _clean(v, secrets) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v, secrets) for v in value]
    return value


def _rotate_if_needed() -> None:
    """Move a full log to ``<name>.1`` (replacing any older backup)."""
    if LOG_PATH.exists() and LOG_PATH.stat().st_size >= MAX_LOG_BYTES:
        os.replace(LOG_PATH, LOG_PATH.with_name(LOG_PATH.name + ".1"))


def _warn_once(exc: Exception) -> None:
    """Report the first logging failure, by type only; stay silent afterwards."""
    global _warned
    if _warned:
        return
    _warned = True
    try:
        logger.warning(
            "Debug logging failed (%s); further failures will not be reported.",
            type(exc).__name__,
        )
    except Exception:
        return


def log_event(event: str, **fields: Any) -> None:
    """Append one debug record when DEBUG is on. Never raises."""
    try:
        settings = get_settings()
        if not settings.debug:
            return
        key = settings.openai_api_key
        secrets = _known_secrets(key.get_secret_value() if key else "")
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            **{name: _clean(value, secrets) for name, value in fields.items()},
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed()
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("%s", event)
    except Exception as exc:
        _warn_once(exc)
