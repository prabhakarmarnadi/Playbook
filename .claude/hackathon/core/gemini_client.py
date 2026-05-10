"""Vertex AI Gemini client.

Uses a GCP service-account JSON to mint access tokens, then calls the Vertex AI
generateContent REST endpoint. Provides .complete() / .complete_json() with the
same shape as core.llm_client.LLMClient.
"""
from __future__ import annotations
import json as _json
import logging
import os
from typing import Any, Optional

import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request as _AuthRequest

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GeminiClient:
    """Direct REST client for Vertex AI Gemini models.

    Example:
        client = GeminiClient(
            project_id="fdi-ai-vertex",
            credentials_path="/mnt/data/code/insight/secret.json",
            model="gemini-3.1-flash-lite",
            location="global",
        )
        text = client.complete("Summarize: ...", system="You are concise.")
    """

    def __init__(self, *, project_id: str, credentials_path: str,
                 model: str = "gemini-3.1-flash-lite",
                 location: str = "global"):
        if not project_id:
            raise ValueError("GeminiClient requires project_id")
        if not credentials_path or not os.path.exists(credentials_path):
            raise ValueError(f"GeminiClient requires a readable credentials_path; got {credentials_path!r}")
        self.project_id = project_id
        self.location = location
        self.model = model
        self._creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=_SCOPES
        )
        self._endpoint = (
            f"https://aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{location}/publishers/google/models/{model}:generateContent"
        )

    def _token(self) -> str:
        if not self._creds.valid:
            self._creds.refresh(_AuthRequest())
        return self._creds.token

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 2000, json_mode: bool = False) -> str:
        """Single-turn completion. Returns raw text."""
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        r = requests.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json",
            },
            data=_json.dumps(body),
            timeout=120,
        )
        if r.status_code >= 400:
            logger.error("Vertex AI error %s: %s", r.status_code, r.text[:500])
            r.raise_for_status()
        data = r.json()
        return _extract_text(data)

    def complete_json(self, prompt: str, system: str = "",
                      temperature: float = 0.0) -> dict:
        """Complete and parse JSON. Falls back to extraction from code-fenced output."""
        raw = self.complete(prompt, system=system, temperature=temperature, json_mode=True)
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            if "```json" in raw:
                seg = raw.split("```json", 1)[1].split("```", 1)[0].strip()
                return _json.loads(seg)
            if "```" in raw:
                seg = raw.split("```", 1)[1].split("```", 1)[0].strip()
                return _json.loads(seg)
            return {"raw_response": raw, "parse_error": True}


def _extract_text(payload: dict) -> str:
    """Extract the first text part from a Vertex generateContent response."""
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if "text" in p)
