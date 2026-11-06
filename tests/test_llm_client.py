import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from app.config import ConfigurationError, Settings, get_settings
from app.llm import client as llm_client
from app.llm import close_client, generate_text, get_client
from app.llm.client import LLMError
from openai import APIConnectionError

ENV_VARS = ("OPENAI_API_KEY", "OPENAI_MODEL")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Stop the real .env from leaking into tests.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    llm_client._client = None
    yield
    get_settings.cache_clear()
    llm_client._client = None


def fake_client(output_text="hello", error=None):
    create = AsyncMock(return_value=SimpleNamespace(output_text=output_text))
    if error is not None:
        create.side_effect = error
    return MagicMock(responses=SimpleNamespace(create=create), close=AsyncMock())


def install(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_client, "AsyncOpenAI", MagicMock(return_value=client))


def test_missing_key_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        get_client()
    with pytest.raises(ConfigurationError):
        asyncio.run(generate_text("hi"))


def test_uses_configured_model_and_returns_text(monkeypatch):
    client = fake_client("lesson text")
    install(monkeypatch, client)
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")

    assert asyncio.run(generate_text("hi")) == "lesson text"
    assert client.responses.create.await_args.kwargs["model"] == "custom-model"


def test_prompt_and_instructions_passed(monkeypatch):
    client = fake_client()
    install(monkeypatch, client)

    asyncio.run(generate_text("the prompt", instructions="be brief"))

    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["input"] == "the prompt"
    assert kwargs["instructions"] == "be brief"


def test_instructions_omitted_when_not_given(monkeypatch):
    client = fake_client()
    install(monkeypatch, client)

    asyncio.run(generate_text("the prompt"))

    assert "instructions" not in client.responses.create.await_args.kwargs


def test_api_error_wrapped_with_original_chained(monkeypatch):
    original = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    install(monkeypatch, fake_client(error=original))

    with pytest.raises(LLMError) as info:
        asyncio.run(generate_text("secret prompt"))

    assert info.value.__cause__ is original
    assert "secret prompt" not in str(info.value)
    assert "sk-test" not in str(info.value)


def test_client_is_created_once_and_secret_used(monkeypatch):
    client = fake_client()
    install(monkeypatch, client)

    async def run():
        await generate_text("a")
        await generate_text("b")

    asyncio.run(run())

    llm_client.AsyncOpenAI.assert_called_once_with(api_key="sk-test")


def test_close_client_closes_and_resets(monkeypatch):
    client = fake_client()
    install(monkeypatch, client)

    async def run():
        get_client()
        await close_client()
        await close_client()  # idempotent

    asyncio.run(run())

    client.close.assert_awaited_once()
    assert llm_client._client is None
