import pytest
from app.config import ConfigurationError, Settings, get_settings
from pydantic import ValidationError

ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "APP_ENV",
    "DEBUG",
    "DATABASE_URL",
    "MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults():
    s = make_settings()
    assert s.openai_api_key is None
    assert s.openai_model == "gpt-4o-mini"
    assert s.app_env == "development"
    assert s.debug is False
    assert s.database_url == "sqlite:///./lesson_memory.db"
    assert s.max_retries == 2


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./other.db")
    monkeypatch.setenv("MAX_RETRIES", "1")
    s = make_settings()
    assert s.openai_model == "custom-model"
    assert s.app_env == "production"
    assert s.debug is True
    assert s.database_url == "sqlite:///./other.db"
    assert s.max_retries == 1


def test_loads_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=from-file\nUNRELATED=ignored\n")
    s = Settings(_env_file=env_file)
    assert s.openai_model == "from-file"


@pytest.mark.parametrize("value", [0, 1, 2])
def test_max_retries_accepts_valid_range(value):
    assert make_settings(max_retries=value).max_retries == value


@pytest.mark.parametrize("value", [-1, 3, 10])
def test_max_retries_rejects_out_of_range(value):
    with pytest.raises(ValidationError):
        make_settings(max_retries=value)


def test_invalid_app_env_rejected():
    with pytest.raises(ValidationError):
        make_settings(app_env="staging")


def test_api_key_never_exposed():
    s = make_settings(openai_api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(s)
    assert "sk-super-secret" not in str(s)
    assert "sk-super-secret" not in s.model_dump_json()
    assert "sk-super-secret" not in str(s.model_dump())


def test_missing_key_does_not_fail_on_construction():
    make_settings()  # must not raise


@pytest.mark.parametrize("value", [None, "", "   "])
def test_require_key_raises_clear_error_when_missing(value):
    s = make_settings(openai_api_key=value)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        s.require_openai_api_key()


def test_require_key_returns_value():
    s = make_settings(openai_api_key="sk-test")
    assert s.require_openai_api_key() == "sk-test"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
