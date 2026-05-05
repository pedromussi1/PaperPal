"""LLM provider abstraction.

Two drivers ship with the project:

- :class:`OllamaProvider` — the local default. Talks to ``ollama serve`` on
  ``localhost:11434`` over its native NDJSON streaming API. Free, offline,
  no API key.
- :class:`GroqProvider` — a hosted free-tier API used for the public demo.
  OpenAI-compatible chat completions, SSE streaming, very fast inference.

The same retrieval pipeline drives both — only the final generation step
differs. Switching is a one-env-var change (``LLM_PROVIDER=ollama|groq``).
This is the substitutability story: the project is local-first by design,
but the architecture admits a cloud fallback for cases where local hardware
isn't available (e.g. the public demo).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

import httpx


class LLMProvider(Protocol):
    """Streams assistant tokens for a (system, user) prompt pair."""

    async def stream(self, system: str, user: str) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


class OllamaProvider:
    """Local Ollama via the native ``/api/chat`` NDJSON streaming endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        request_timeout: float = 120.0,
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=request_timeout)
        self._temperature = temperature
        self._num_ctx = num_ctx

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": self._temperature, "num_ctx": self._num_ctx},
        }
        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("done"):
                    break
                msg = event.get("message") or {}
                content = msg.get("content")
                if content:
                    yield content


class GroqProvider:
    """Groq's OpenAI-compatible ``/v1/chat/completions`` SSE endpoint.

    Free tier (no credit card) gives ~30 RPM and 14,400 RPD with Llama 3.1
    8B Instant — ample for a portfolio demo.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        base_url: str = "https://api.groq.com/openai/v1",
        request_timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            timeout=request_timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._temperature = temperature

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "stream": True,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with self._client.stream(
            "POST", f"{self._base_url}/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                event = json.loads(data)
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
