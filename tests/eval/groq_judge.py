"""Groq-backed DeepEval judge (uses your existing GROQ_API_KEY)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from groq import AsyncGroq, Groq
from groq import RateLimitError


class GroqJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model or os.environ.get(
            "DEEPEVAL_GROQ_MODEL",
            os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
        )
        self._temperature = temperature
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is required for the Groq DeepEval judge.")
        super().__init__(model=self._model)

    def load_model(self) -> Groq:
        return Groq(api_key=self._api_key)

    def get_model_name(self) -> str:
        return f"{self._model} (Groq)"

    def supports_json_mode(self) -> bool:
        return True

    def supports_structured_outputs(self) -> bool:
        return True

    def _completion_kwargs(self, prompt: str, schema: Any = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": 4096,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def generate(self, prompt: str, schema: Any = None) -> str:
        response = self.model.chat.completions.create(
            **self._completion_kwargs(prompt, schema)
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str, schema: Any = None) -> str:
        client = AsyncGroq(api_key=self._api_key)
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    **self._completion_kwargs(prompt, schema)
                )
                return response.choices[0].message.content or ""
            except RateLimitError:
                if attempt == 4:
                    raise
                await asyncio.sleep(3 * (attempt + 1))
        return ""
