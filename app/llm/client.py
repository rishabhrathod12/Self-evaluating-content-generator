"""Asynchronous OpenAI client built on the Responses API."""

from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.config import get_settings

_client: AsyncOpenAI | None = None


class LLMError(RuntimeError):
    """Raised when an OpenAI API call fails.

    The original exception is always chained as ``__cause__``.
    """


def get_client() -> AsyncOpenAI:
    """Return the shared ``AsyncOpenAI`` client, creating it on first use.

    Raises:
        ConfigurationError: If ``OPENAI_API_KEY`` is not configured.
    """
    global _client
    if _client is None:
        api_key = get_settings().require_openai_api_key()
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def close_client() -> None:
    """Close the shared client, if one exists. Safe to call repeatedly."""
    global _client
    client, _client = _client, None
    if client is not None:
        await client.close()


async def generate_text(
    prompt: str,
    instructions: str | None = None,
    json_schema: dict[str, Any] | None = None,
    schema_name: str = "response",
    temperature: float | None = None,
) -> str:
    """Generate text with the configured model via ``responses.create()``.

    Args:
        prompt: The input text sent to the model.
        instructions: Optional system-level instructions for the model.
        json_schema: Optional JSON Schema. When given, the API is asked for
            strict structured output matching it (Structured Outputs).
        schema_name: Name for the schema, used only with ``json_schema``.
        temperature: Optional sampling temperature. Judging calls use 0 so the
            same lesson gets the same verdict; omitted when ``None``.

    Returns:
        The model's text output.

    Raises:
        ConfigurationError: If ``OPENAI_API_KEY`` is not configured.
        LLMError: If the OpenAI API call fails. The original error is chained.
    """
    client = get_client()
    request: dict[str, Any] = {
        "model": get_settings().openai_model,
        "input": prompt,
    }
    if instructions is not None:
        request["instructions"] = instructions
    if temperature is not None:
        request["temperature"] = temperature
    if json_schema is not None:
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            }
        }

    try:
        response = await client.responses.create(**request)
    except OpenAIError as exc:
        # Report only the error type: messages can echo request details.
        raise LLMError(f"OpenAI request failed ({type(exc).__name__}).") from exc
    return response.output_text
