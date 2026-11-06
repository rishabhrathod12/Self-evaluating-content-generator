"""LLM integration layer."""

from app.llm.client import LLMError, close_client, generate_text, get_client

__all__ = ["LLMError", "close_client", "generate_text", "get_client"]
