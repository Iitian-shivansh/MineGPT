"""
LLM client for MineGPT.

Uses OpenRouter (an OpenAI-compatible API gateway) to generate answers grounded
in the retrieved DGMS context. Configuration is read from environment variables
(a local .env file is loaded automatically):

    OPENROUTER_API_KEY   - your OpenRouter key (required for generation)
    OPENROUTER_MODEL     - model id (default: a free model, see DEFAULT_MODEL)

If no API key is configured, the client is considered unavailable and the RAG
pipeline falls back to showing the retrieved passages (extractive mode).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # load variables from a local .env file if present

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# A capable free model on OpenRouter. Override with OPENROUTER_MODEL in .env.
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


class LLMClient:
    """Thin wrapper around the OpenRouter chat-completions endpoint."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=self.api_key,
                    default_headers={
                        # Optional attribution headers recommended by OpenRouter.
                        "HTTP-Referer": "https://localhost/minegpt",
                        "X-Title": "MineGPT",
                    },
                )
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 700) -> str:
        """Call the chat model and return the assistant's text."""
        if not self.available:
            raise RuntimeError("LLM client is not configured (missing OPENROUTER_API_KEY).")
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
