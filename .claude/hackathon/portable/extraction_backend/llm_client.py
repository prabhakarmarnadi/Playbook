"""
LLM client abstraction — replaces legacy AIFlow gRPC LLM calls.

Legacy pain point: AIFlowClient.execute_generic_openai() routes through a gRPC proxy,
requiring service authentication, network connectivity, and specific proto messages.

V2 approach: Direct LLM calls via OpenAI SDK, Ollama, or LiteLLM.
Supports local-first (Ollama) with fallback to cloud (OpenAI).
"""
import json
import os

from openai import OpenAI, AzureOpenAI

from config import (
    LLM_BACKEND, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)


class LLMClient:
    """Unified LLM client supporting Azure OpenAI, OpenAI, Ollama, and LiteLLM backends."""

    def __init__(self, backend: str | None = None):
        self.backend = backend or LLM_BACKEND

        # Auto-detect Azure if endpoint is set and backend is "openai"
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")

        if self.backend == "openai" and azure_endpoint and azure_key:
            # Azure OpenAI detected — use AzureOpenAI client
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=api_version,
            )
            # Azure uses deployment name, not model name
            self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip('"') or OPENAI_MODEL
        elif self.backend == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        elif self.backend == "ollama":
            self.client = OpenAI(
                base_url=f"{OLLAMA_BASE_URL}/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )
            self.model = OLLAMA_MODEL
        elif self.backend == "litellm":
            # LiteLLM uses the OpenAI-compatible interface
            self.client = OpenAI(
                base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
                api_key=os.getenv("LITELLM_API_KEY", "sk-litellm"),
            )
            self.model = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 2000, json_mode: bool = False) -> str:
        """
        Single-turn completion. Returns raw text response.

        Legacy equivalent: AIFlowClient.execute_generic_openai()
        V2: Direct API call, no gRPC proxy.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete_json(self, prompt: str, system: str = "",
                      temperature: float = 0.0) -> dict:
        """Complete and parse JSON response. Returns dict."""
        raw = self.complete(prompt, system=system, temperature=temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            if "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            return {"raw_response": raw, "parse_error": True}
