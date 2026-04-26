"""
gRPC Language Model — Custom dspy.LM that calls an external gRPC service.
=========================================================================

Wraps a prompt-in/text-out gRPC service as a dspy.LM so it can be used as
both the main LM and sub-LM in rlm_v3.py and production_pipeline.py.

Usage:
    from core.grpc_lm import GrpcLM

    # As DSPy LM (for RLM V3)
    lm = GrpcLM(host="my-service:50051")
    dspy.configure(lm=lm)

    # Direct
    result = lm("What fields are in this clause?")

    # With RLMV3
    from core.rlm_v3 import RLMV3, RLMV3Config
    v3 = RLMV3(config=RLMV3Config(), lm=lm, sub_lm=lm)
    fields, meta = v3.discover(label, keywords, chunks)

    # With AsyncLLMClient replacement
    from core.grpc_lm import GrpcAsyncLLMClient
    llm = GrpcAsyncLLMClient(host="my-service:50051")

Adapt the proto stub below to match your service's actual .proto definition.
The only contract we need: send prompt text → receive response text.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import grpc

import dspy

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# gRPC stub — ADAPT THIS to your actual proto
# ══════════════════════════════════════════════════════════════════════════════
#
# If you have generated proto stubs (e.g., my_service_pb2, my_service_pb2_grpc),
# replace this section with your actual imports and adjust _call_grpc() below.
#
# Minimal approach: use grpc.unary_unary channel call with raw bytes.
# This works without needing generated proto stubs at all.

def _make_channel(host: str, secure: bool = False) -> grpc.Channel:
    """Create a gRPC channel to the service."""
    if secure:
        credentials = grpc.ssl_channel_credentials()
        return grpc.secure_channel(host, credentials)
    return grpc.insecure_channel(host)


# ══════════════════════════════════════════════════════════════════════════════
# DSPy LM Wrapper
# ══════════════════════════════════════════════════════════════════════════════


class GrpcLM(dspy.LM):
    """
    A dspy.LM that routes completions through a gRPC service.

    The service is expected to accept a prompt string and return a text response.
    Adapt _call_grpc() to match your service's proto definition.

    Args:
        host: gRPC service address (e.g., "my-service:50051")
        model: Model name string (passed to DSPy for tracking/logging)
        secure: Whether to use TLS channel
        grpc_method: Full gRPC method path (e.g., "/my.package.LLMService/Complete")
        timeout_s: Per-call timeout in seconds
        json_mode: Whether to request JSON responses from the service
        metadata: Extra gRPC metadata to send with each call
    """

    def __init__(
        self,
        host: str | None = None,
        model: str = "grpc/custom",
        secure: bool = False,
        grpc_method: str = "/llm.LLMService/Complete",
        timeout_s: float = 120.0,
        json_mode: bool = False,
        metadata: list[tuple[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        **kwargs,
    ):
        # Initialize dspy.LM base (model string is for logging only)
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            **kwargs,
        )
        self._host = host or os.getenv("GRPC_LLM_HOST", "localhost:50051")
        self._secure = secure
        self._grpc_method = grpc_method
        self._timeout_s = timeout_s
        self._json_mode = json_mode
        self._metadata = metadata or []
        self._channel: grpc.Channel | None = None
        self._call_count = 0
        self._total_latency = 0.0

    @property
    def channel(self) -> grpc.Channel:
        if self._channel is None:
            self._channel = _make_channel(self._host, self._secure)
        return self._channel

    def _call_grpc(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the gRPC service and return the response text.

        ═══════════════════════════════════════════════════════════════
        ADAPT THIS METHOD to your service's proto definition.
        ═══════════════════════════════════════════════════════════════

        Option A: If you have generated stubs:
            from my_service_pb2 import CompletionRequest, CompletionResponse
            from my_service_pb2_grpc import LLMServiceStub
            stub = LLMServiceStub(self.channel)
            resp = stub.Complete(CompletionRequest(prompt=prompt, ...))
            return resp.text

        Option B: Raw unary_unary (no proto stubs needed):
            Uses JSON serialization over gRPC bytes.
        """
        # ── Option B: Generic JSON-over-gRPC (works without proto stubs) ──
        request_payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.kwargs.get("temperature", 0.2)),
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 2000)),
        }
        if self._json_mode or kwargs.get("response_format"):
            request_payload["response_format"] = "json"

        request_bytes = json.dumps(request_payload).encode("utf-8")

        # Make the unary call
        method = self.channel.unary_unary(
            self._grpc_method,
            request_serializer=lambda x: x,      # already bytes
            response_deserializer=lambda x: x,    # raw bytes back
        )

        metadata = list(self._metadata)
        response_bytes = method(
            request_bytes,
            timeout=self._timeout_s,
            metadata=metadata or None,
        )

        # Parse response — adapt if your service returns something other than
        # a JSON object with a "text" or "content" field
        try:
            resp = json.loads(response_bytes.decode("utf-8"))
            return resp.get("text") or resp.get("content") or resp.get("response") or str(resp)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return response_bytes.decode("utf-8", errors="replace")

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Override dspy.LM.forward to route through gRPC instead of litellm."""
        # Flatten messages to a single prompt string
        if messages:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multi-part messages (text + images etc.)
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ) or str(content)
                if role == "system":
                    parts.append(f"[System]\n{content}")
                elif role == "assistant":
                    parts.append(f"[Assistant]\n{content}")
                else:
                    parts.append(content)
            prompt_text = "\n\n".join(parts)
        else:
            prompt_text = prompt or ""

        # Call gRPC with retries
        last_err = None
        for attempt in range(self.num_retries):
            try:
                t0 = time.time()
                response_text = self._call_grpc(prompt_text, **kwargs)
                latency = time.time() - t0
                self._call_count += 1
                self._total_latency += latency
                logger.debug("gRPC LM call #%d: %.1fs", self._call_count, latency)

                # Return in the format dspy.LM expects from its _process_lm_response
                # We create a minimal mock of what litellm returns
                return _MockCompletionResponse(response_text)

            except grpc.RpcError as e:
                last_err = e
                logger.warning(
                    "gRPC call failed (attempt %d/%d): %s",
                    attempt + 1, self.num_retries, e,
                )
                if attempt < self.num_retries - 1:
                    import time as _time
                    _time.sleep(1.0 * (attempt + 1))

        raise ConnectionError(f"gRPC LM failed after {self.num_retries} retries: {last_err}")

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_latency_s": round(self._total_latency, 1),
            "avg_latency_s": round(
                self._total_latency / max(1, self._call_count), 2
            ),
            "host": self._host,
        }


class _MockCompletionResponse:
    """Minimal mock of a litellm completion response for dspy.LM._process_lm_response."""

    def __init__(self, text: str):
        self.choices = [_MockChoice(text)]
        self.usage = _MockUsage(text)
        self.cache_hit = False

    def __getitem__(self, key):
        return getattr(self, key)


class _MockChoice:
    def __init__(self, text: str):
        self.message = _MockMessage(text)
        self.finish_reason = "stop"
        self.text = text


class _MockMessage:
    def __init__(self, text: str):
        self.content = text
        self.role = "assistant"

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


class _MockUsage:
    def __init__(self, text: str):
        self.prompt_tokens = 0
        self.completion_tokens = len(text) // 4
        self.total_tokens = self.completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
# Async wrapper for production_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════


class GrpcAsyncLLMClient:
    """
    Drop-in replacement for AsyncLLMClient that routes through gRPC.

    Has the same interface as production_pipeline.AsyncLLMClient:
        await client.complete_json(prompt, temperature, max_tokens) -> dict | None
        client.stats -> dict

    Usage in production_pipeline.py:
        from core.grpc_lm import GrpcAsyncLLMClient
        llm = GrpcAsyncLLMClient(host="my-service:50051")
        # Use exactly like AsyncLLMClient
    """

    def __init__(
        self,
        host: str | None = None,
        max_concurrency: int = 8,
        grpc_method: str = "/llm.LLMService/Complete",
        timeout_s: float = 120.0,
        secure: bool = False,
        metadata: list[tuple[str, str]] | None = None,
    ):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        self._grpc_lm = GrpcLM(
            host=host, grpc_method=grpc_method,
            timeout_s=timeout_s, secure=secure, metadata=metadata,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self._call_count = 0
        self._total_latency = 0.0

    async def complete_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        retries: int = 3,
        model: str | None = None,
    ) -> dict | None:
        """Async JSON completion via gRPC with semaphore rate-limiting."""
        import asyncio

        async with self._semaphore:
            loop = asyncio.get_event_loop()
            for attempt in range(retries):
                try:
                    t0 = time.time()
                    response_text = await loop.run_in_executor(
                        self._executor,
                        lambda: self._grpc_lm._call_grpc(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format="json",
                        ),
                    )
                    self._call_count += 1
                    self._total_latency += time.time() - t0

                    # Parse JSON
                    return json.loads(response_text)

                except json.JSONDecodeError:
                    logger.warning("gRPC JSON parse error (attempt %d)", attempt + 1)
                except Exception as e:
                    logger.warning("gRPC call error (attempt %d): %s", attempt + 1, e)
                    if attempt < retries - 1:
                        await asyncio.sleep(1.5 * (attempt + 1))
        return None

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_latency_s": round(self._total_latency, 1),
            "avg_latency_s": round(
                self._total_latency / max(1, self._call_count), 2
            ),
            "backend": "grpc",
            "host": self._grpc_lm._host,
        }
