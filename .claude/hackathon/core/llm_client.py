"""
LLM client abstraction — replaces legacy AIFlow gRPC LLM calls.

Legacy pain point: AIFlowClient.execute_generic_openai() routes through a gRPC proxy,
requiring service authentication, network connectivity, and specific proto messages.

V2 approach: Direct LLM calls via OpenAI SDK, Ollama, or LiteLLM.
Supports local-first (Ollama) with fallback to cloud (OpenAI).
"""
import json
import os

from config import (
    LLM_BACKEND, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)


def _openai_classes():
    """Lazy import: only loads `openai` SDK when an OpenAI-compatible backend is requested."""
    from openai import OpenAI, AzureOpenAI
    return OpenAI, AzureOpenAI


class LLMClient:
    """Unified LLM client supporting Azure OpenAI, OpenAI, Ollama, LiteLLM, and Gemini (Vertex AI) backends."""

    def __init__(self, backend: str | None = None):
        self.backend = backend or LLM_BACKEND
        self._gemini = None  # populated only when backend=="gemini"

        # Auto-detect Azure if endpoint is set and backend is "openai"
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")

        if self.backend == "openai" and azure_endpoint and azure_key:
            # Azure OpenAI detected — use AzureOpenAI client
            _, AzureOpenAI = _openai_classes()
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=api_version,
            )
            # Azure uses deployment name, not model name
            self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip('"') or OPENAI_MODEL
        elif self.backend == "openai":
            OpenAI, _ = _openai_classes()
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        elif self.backend == "ollama":
            OpenAI, _ = _openai_classes()
            self.client = OpenAI(
                base_url=f"{OLLAMA_BASE_URL}/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )
            self.model = OLLAMA_MODEL
        elif self.backend == "litellm":
            # LiteLLM uses the OpenAI-compatible interface
            OpenAI, _ = _openai_classes()
            self.client = OpenAI(
                base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
                api_key=os.getenv("LITELLM_API_KEY", "sk-litellm"),
            )
            self.model = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
        elif self.backend == "gemini":
            # Gemini via Vertex AI (service-account JSON)
            from core.gemini_client import GeminiClient
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
            self._gemini = GeminiClient(
                project_id=project_id, credentials_path=creds_path,
                model=self.model, location=location,
            )
            self.client = None  # OpenAI client unused for gemini path
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 2000, json_mode: bool = False) -> str:
        """
        Single-turn completion. Returns raw text response.

        Legacy equivalent: AIFlowClient.execute_generic_openai()
        V2: Direct API call, no gRPC proxy.
        """
        if self._gemini is not None:
            return self._gemini.complete(
                prompt, system=system, temperature=temperature,
                max_tokens=max_tokens, json_mode=json_mode,
            )

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


# ────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible chat-completions surface for legacy callers (field_discovery,
# intent_extractor) that talk directly to AzureOpenAI without going through
# LLMClient.complete(). This shim lets those modules work with any backend
# (Gemini, OpenAI, Azure, Ollama, LiteLLM) without rewriting their call sites.
# ────────────────────────────────────────────────────────────────────────────

class _ShimMessage:
    def __init__(self, content: str):
        self.content = content


class _ShimChoice:
    def __init__(self, content: str):
        self.message = _ShimMessage(content)


class _ShimResponse:
    def __init__(self, content: str):
        self.choices = [_ShimChoice(content)]


class _GeminiCompletions:
    """Mimics openai SDK's `client.chat.completions.create(...)` for Gemini."""
    def __init__(self, gemini_client):
        self._g = gemini_client

    def create(self, *, messages, model=None, response_format=None,
                temperature: float = 0.0, max_completion_tokens: int = 2000,
                max_tokens: int | None = None, **_kwargs) -> _ShimResponse:
        system = ""
        user_parts: list[str] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system += (content + "\n")
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"[prior assistant turn]\n{content}")
        prompt = "\n".join(user_parts).strip()
        json_mode = bool(response_format and response_format.get("type") == "json_object")
        n_tokens = max_completion_tokens or max_tokens or 2000
        text = self._g.complete(
            prompt, system=system.strip(),
            temperature=temperature, max_tokens=n_tokens, json_mode=json_mode,
        )
        return _ShimResponse(text)


class _GeminiChat:
    def __init__(self, gemini_client):
        self.completions = _GeminiCompletions(gemini_client)


class _GeminiOpenAICompatClient:
    """Quacks like an OpenAI/AzureOpenAI client. Only the .chat.completions.create
    surface is implemented (which is all field_discovery + intent_extractor use)."""
    def __init__(self, gemini_client):
        self.chat = _GeminiChat(gemini_client)


def llm_configured() -> bool:
    """True if any supported LLM backend is configured via env."""
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        return True
    backend = os.getenv("LLM_BACKEND", "openai").lower()
    if backend == "openai" and os.getenv("OPENAI_API_KEY"):
        return True
    if backend == "ollama":
        return True
    if backend == "litellm":
        return True
    if backend == "gemini" and os.getenv("GOOGLE_CLOUD_PROJECT") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    return False


def make_openai_compatible_client():
    """Return an object exposing .chat.completions.create(...) for any configured
    backend. AzureOpenAI / OpenAI return their native SDK clients; Gemini returns
    a shim. Used by legacy modules (field_discovery, intent_extractor) that talk
    chat.completions directly."""
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    az_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if az_endpoint and az_key:
        _, AzureOpenAI = _openai_classes()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        return AzureOpenAI(azure_endpoint=az_endpoint, api_key=az_key, api_version=api_version)

    backend = os.getenv("LLM_BACKEND", "openai").lower()
    if backend == "gemini":
        from core.gemini_client import GeminiClient
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        gemini = GeminiClient(project_id=project_id, credentials_path=creds_path,
                                model=model, location=location)
        return _GeminiOpenAICompatClient(gemini)
    if backend in ("openai", "litellm", "ollama"):
        # Build via LLMClient and return its underlying SDK client (already OpenAI-shaped).
        return LLMClient(backend=backend).client
    raise RuntimeError(f"No LLM backend configured (LLM_BACKEND={backend!r})")
