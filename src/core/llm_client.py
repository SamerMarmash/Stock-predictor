"""Unified LLM client supporting both OpenAI and Anthropic APIs."""

from __future__ import annotations

import json

import structlog

from src.core.config import get_settings

logger = structlog.get_logger()


class LLMClient:
    """Wraps OpenAI and Anthropic SDKs behind a single interface."""

    def __init__(self):
        self.settings = get_settings()
        self._openai_client = None
        self._anthropic_client = None

    def _get_openai(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def _get_anthropic(self):
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._anthropic_client

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
        use_extended_thinking: bool = False,
        thinking_budget: int = 10000,
    ) -> str:
        """Send a prompt to the configured LLM and return the text response."""
        temp = temperature if temperature is not None else self.settings.llm_temperature

        if self.settings.llm_provider == "anthropic":
            return await self._anthropic_complete(
                system_prompt, user_prompt, temp, max_tokens,
                use_extended_thinking=use_extended_thinking,
                thinking_budget=thinking_budget,
            )
        return await self._openai_complete(system_prompt, user_prompt, temp, max_tokens)

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
        use_extended_thinking: bool = False,
        thinking_budget: int = 10000,
    ) -> dict:
        """Send a prompt expecting a JSON response. Parses and returns as dict."""
        raw = await self.complete(
            system_prompt, user_prompt, temperature, max_tokens,
            use_extended_thinking=use_extended_thinking,
            thinking_budget=thinking_budget,
        )
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    async def _openai_complete(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_openai()
        response = await client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _anthropic_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        use_extended_thinking: bool = False,
        thinking_budget: int = 10000,
    ) -> str:
        client = self._get_anthropic()

        kwargs = {
            "model": self.settings.llm_model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        if use_extended_thinking:
            # Extended thinking requires temperature=1 and uses a budget
            kwargs["temperature"] = 1
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        else:
            kwargs["temperature"] = temperature

        response = await client.messages.create(**kwargs)

        # With extended thinking, response has thinking + text blocks
        # Extract only the text block(s)
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else ""
