"""
Latent Briefing — KV Cache Compaction for Multi-Agent Memory Sharing
====================================================================

Adapted from Ramp Labs' Latent Briefing (April 2026) which demonstrated:
  - 65% worker-token reduction on LongBench v2
  - +3% accuracy improvement over full-context baseline
  - 1.7s median compaction overhead (20× speedup over naive AM)

Core idea: Instead of passing the orchestrator's full reasoning trajectory
to the worker as text (expensive, noisy), we compact the KV cache using
the worker's own attention patterns to keep only what's relevant to the
current subtask.

Three inference-time modifications to Attention Matching (AM):
  1. Task-guided query vectors — score trajectory positions by how
     strongly the worker's current task attends to them
  2. Shared global mask — aggregate across all heads into one per-position
     score, enabling massive batching (320 serial solves → 2-3 batched ops)
  3. MAD thresholding — median + tau * MAD for adaptive compression

Integration with DSPy RLM:
  - The orchestrator (RLM) builds a trajectory of REPL steps
  - Before each llm_query() worker call, we compact the trajectory's
    KV cache using the current subtask as the scoring query
  - The worker receives a compressed latent "briefing" instead of the
    full text trajectory

References:
  - Ramp Labs: https://x.com/RampLabs/status/2042660310851449223
  - Attention Matching: https://arxiv.org/abs/2602.16284
  - Recursive Language Models: https://arxiv.org/abs/2512.24601
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """Configuration for KV cache compaction."""
    # MAD threshold multiplier. Higher = more aggressive compression.
    # Ramp Labs findings:
    #   - Longer docs: lower tau (~1.0) preserves broader evidence
    #   - Harder tasks: higher tau (~2.0) filters speculative reasoning noise
    #   - Default moderate: 1.5
    tau: float = 1.5

    # Minimum retention rate — never drop below this fraction of positions
    min_retention: float = 0.1

    # Maximum retention rate — always drop at least this many positions
    max_retention: float = 0.9

    # Whether to use task-guided queries (True) or self-attention (False)
    task_guided: bool = True

    # Number of layers to sample for scoring (None = all layers)
    # Sampling reduces compute; Ramp Labs used ~25% of layers
    layer_sample_frac: float = 0.25

    # Device for computation
    device: str = "cuda"


@dataclass
class CompactionStats:
    """Stats from a single compaction operation."""
    original_length: int = 0
    compacted_length: int = 0
    retention_rate: float = 1.0
    compaction_time_s: float = 0.0
    tau_used: float = 0.0
    tokens_saved: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_length == 0:
            return 0.0
        return 1.0 - (self.compacted_length / self.original_length)


@dataclass
class LatentBriefingState:
    """Persistent compaction state across RLM iterations."""
    trajectory_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    trajectory_token_ids: Optional[torch.Tensor] = None
    cumulative_stats: list = field(default_factory=list)
    total_tokens_saved: int = 0
    total_compaction_time: float = 0.0

    @property
    def n_compactions(self) -> int:
        return len(self.cumulative_stats)

    @property
    def avg_retention(self) -> float:
        if not self.cumulative_stats:
            return 1.0
        return sum(s.retention_rate for s in self.cumulative_stats) / len(self.cumulative_stats)


class LatentBriefingEngine:
    """
    KV cache compaction engine for multi-agent memory sharing.

    Uses Attention Matching with task-guided scoring to selectively retain
    the most relevant positions from the orchestrator's trajectory KV cache.
    """

    def __init__(self, model, tokenizer, config: CompactionConfig | None = None):
        """
        Args:
            model: HuggingFace model with attention output access
            tokenizer: Corresponding tokenizer
            config: Compaction configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CompactionConfig()
        self.device = self.config.device

        # Model architecture info
        self.n_layers = model.config.num_hidden_layers
        self.n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

    def compute_importance_scores(
        self,
        trajectory_kv: tuple[torch.Tensor, torch.Tensor],
        task_query_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score each trajectory position by how strongly the task query attends to it.

        This is the core of Latent Briefing's task-guided scoring:
        instead of using the trajectory's self-attention (what pays attention
        to what within the trajectory), we use the task query as the scoring
        signal (what would the WORKER attend to for THIS specific subtask).

        Args:
            trajectory_kv: (keys, values) each shape [n_layers, n_kv_heads, seq_len, head_dim]
            task_query_ids: Token IDs for the current worker task prompt

        Returns:
            Per-position importance scores, shape [seq_len]
        """
        keys, values = trajectory_kv
        seq_len = keys.shape[2]

        # Select a subset of layers for efficiency
        n_sample = max(1, int(self.n_layers * self.config.layer_sample_frac))
        layer_indices = torch.linspace(0, self.n_layers - 1, n_sample).long()

        # Forward pass the task query to get its query vectors
        with torch.no_grad():
            task_outputs = self.model(
                task_query_ids.unsqueeze(0).to(self.device),
                output_attentions=False,
                use_cache=True,
            )
            task_kv = task_outputs.past_key_values

        # Score trajectory positions using task query vectors
        # For each sampled layer, compute attention scores between
        # task queries and trajectory keys
        all_scores = torch.zeros(seq_len, device=self.device)

        for layer_idx in layer_indices:
            # Task query vectors for this layer: [1, n_kv_heads, task_len, head_dim]
            task_k = task_kv[layer_idx][0]  # We want Q, but KV cache stores K,V
            # Trajectory keys for this layer: [1, n_kv_heads, seq_len, head_dim]
            traj_k = keys[layer_idx].unsqueeze(0) if keys[layer_idx].dim() == 3 else keys[layer_idx]

            # Compute cross-attention scores: how much does the task attend to each position?
            # Use the last few task tokens as the query (most relevant to the subtask)
            # Score = softmax(Q_task @ K_trajectory^T / sqrt(d))
            scale = self.head_dim ** 0.5

            # Use task keys as proxy for queries (in KV cache, Q is not stored)
            # This is the task-guided scoring from Ramp Labs
            attn_scores = torch.matmul(
                task_k[:, :, -1:, :],  # Last task position as query [1, heads, 1, dim]
                traj_k.transpose(-2, -1)  # [1, heads, dim, seq_len]
            ) / scale  # [1, heads, 1, seq_len]

            # Aggregate across heads (shared global mask — modification #2)
            # Mean across heads, squeeze to [seq_len]
            layer_scores = attn_scores.squeeze(0).mean(dim=0).squeeze(0)  # [seq_len]
            all_scores += layer_scores

        # Average across sampled layers
        all_scores /= len(layer_indices)

        return all_scores

    def mad_threshold(self, scores: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute MAD-normalized threshold for adaptive compression.

        Modification #3 from Ramp Labs: Use median + tau * MAD instead of
        fixed top-k. This adapts to the actual score distribution, meaning:
        - If most positions are equally important → light compression
        - If a few positions dominate → aggressive compression

        Args:
            scores: Per-position importance scores [seq_len]
            tau: Threshold multiplier (higher = keep fewer positions)

        Returns:
            Boolean mask of positions to keep [seq_len]
        """
        median = scores.median()
        mad = (scores - median).abs().median()

        # Handle degenerate case where MAD is 0 (all scores identical)
        if mad < 1e-8:
            # Keep everything — can't distinguish important from unimportant
            return torch.ones_like(scores, dtype=torch.bool)

        threshold = median + tau * mad
        keep_mask = scores >= threshold

        # Enforce retention bounds
        n_total = scores.shape[0]
        n_kept = keep_mask.sum().item()
        min_keep = max(1, int(n_total * self.config.min_retention))
        max_keep = int(n_total * self.config.max_retention)

        if n_kept < min_keep:
            # Keep top-min_keep positions
            _, top_indices = scores.topk(min_keep)
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
            keep_mask[top_indices] = True
        elif n_kept > max_keep:
            # Keep only top-max_keep positions
            _, top_indices = scores.topk(max_keep)
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
            keep_mask[top_indices] = True

        return keep_mask

    @torch.no_grad()
    def compact(
        self,
        trajectory_kv: tuple[torch.Tensor, torch.Tensor],
        task_query_ids: torch.Tensor,
        tau: float | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], CompactionStats]:
        """
        Compact the trajectory KV cache for a specific worker task.

        This is the main entry point. Given the orchestrator's accumulated
        KV cache and the current worker's task prompt, produces a compacted
        KV cache containing only the positions most relevant to the task.

        Args:
            trajectory_kv: (keys, values) from orchestrator trajectory
            task_query_ids: Token IDs for the worker's current subtask
            tau: Override compaction aggressiveness (None = use config)

        Returns:
            (compacted_kv, stats): Compacted KV cache and statistics
        """
        start_time = time.time()
        tau = tau if tau is not None else self.config.tau

        keys, values = trajectory_kv
        original_len = keys.shape[2] if keys.dim() == 4 else keys.shape[1]

        # Step 1: Score each trajectory position using task-guided queries
        scores = self.compute_importance_scores(trajectory_kv, task_query_ids)

        # Step 2: Apply MAD thresholding for adaptive compression
        keep_mask = self.mad_threshold(scores, tau)

        # Step 3: Apply mask to KV cache (shared global mask across all heads)
        compacted_keys = keys[:, :, keep_mask, :] if keys.dim() == 4 else keys[:, keep_mask, :]
        compacted_values = values[:, :, keep_mask, :] if values.dim() == 4 else values[:, keep_mask, :]

        compacted_len = keep_mask.sum().item()
        elapsed = time.time() - start_time

        stats = CompactionStats(
            original_length=original_len,
            compacted_length=compacted_len,
            retention_rate=compacted_len / max(1, original_len),
            compaction_time_s=elapsed,
            tau_used=tau,
            tokens_saved=original_len - compacted_len,
        )

        logger.info(
            f"Latent Briefing: {original_len} → {compacted_len} positions "
            f"({stats.compression_ratio:.1%} compressed, {elapsed:.2f}s, τ={tau})"
        )

        return (compacted_keys, compacted_values), stats


class TextLatentBriefing:
    """
    Text-level Latent Briefing for API-based models (no KV cache access).

    Emulates the core Latent Briefing insights at the text layer:
    1. Task-guided extraction: Score trajectory segments by relevance to
       the current subtask
    2. Attention-inspired compression: Use embedding similarity as a proxy
       for attention scores
    3. Adaptive thresholding: Keep more context for easy tasks, less for hard

    This is the practical variant for our DSPy RLM setup where the orchestrator
    uses Azure OpenAI (API-only, no KV access) but we still want the benefits
    of selective memory sharing.
    """

    def __init__(self, embed_model=None, config: CompactionConfig | None = None):
        """
        Args:
            embed_model: SentenceTransformer model for computing relevance scores
            config: Compaction configuration
        """
        self.config = config or CompactionConfig(tau=1.5)
        self.embed_model = embed_model
        self._trajectory_segments: list[dict] = []  # {text, embedding, metadata}
        self._total_tokens_saved = 0

    def _get_embedder(self):
        if self.embed_model is None:
            # Try to reuse the shared singleton from rlm_v3 to avoid loading a duplicate model
            try:
                from core.rlm_v3 import _get_shared_embedder
                self.embed_model = _get_shared_embedder()
            except (ImportError, Exception):
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embed_model

    def add_trajectory_segment(
        self,
        text: str,
        metadata: dict | None = None,
        segment_type: str = "observation",  # "observation", "reasoning", "tool_output", "worker_reply"
    ):
        """
        Append a segment to the trajectory memory.

        Each REPL iteration in the RLM produces segments:
        - Code the orchestrator wrote
        - Output from REPL execution
        - llm_query() results
        - Intermediate reasoning

        Args:
            text: The segment content
            metadata: Optional metadata (iteration number, tool used, etc.)
            segment_type: Category of this segment
        """
        embedder = self._get_embedder()
        embedding = embedder.encode(text, normalize_embeddings=True)

        self._trajectory_segments.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "type": segment_type,
            "token_count": len(text.split()),  # Rough token estimate
        })

    def compact_for_task(
        self,
        task_prompt: str,
        tau: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, CompactionStats]:
        """
        Produce a compacted text briefing for a specific worker task.

        Mirrors the KV cache compaction flow, but operates on text segments:
        1. Embed the task prompt (analogous to task-guided query vectors)
        2. Score each trajectory segment by cosine similarity to the task
        3. Apply MAD thresholding (adaptive compression)
        4. Concatenate surviving segments in order

        Args:
            task_prompt: The worker's current subtask description/query
            tau: Override compaction aggressiveness (None = use config)
            max_tokens: Hard cap on output token count

        Returns:
            (briefing_text, stats): Compacted text and statistics
        """
        start_time = time.time()
        tau = tau if tau is not None else self.config.tau

        if not self._trajectory_segments:
            return "", CompactionStats()

        embedder = self._get_embedder()

        # Step 1: Task-guided scoring (analogous to task query vectors)
        task_embedding = embedder.encode(task_prompt, normalize_embeddings=True)
        task_tensor = torch.tensor(task_embedding, dtype=torch.float32)

        # Step 2: Compute relevance scores for each segment
        scores = []
        for seg in self._trajectory_segments:
            seg_tensor = torch.tensor(seg["embedding"], dtype=torch.float32)
            score = F.cosine_similarity(task_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)).item()

            # Boost scores for certain segment types (analogous to positional bias)
            type_boost = {
                "worker_reply": 0.1,    # Prior worker answers are usually relevant
                "tool_output": 0.05,    # Tool outputs contain evidence
                "reasoning": 0.0,       # Reasoning may or may not be relevant
                "observation": -0.05,   # Raw observations are often noisy
            }
            score += type_boost.get(seg["type"], 0.0)
            scores.append(score)

        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Step 3: MAD thresholding (modification #3)
        median = scores_tensor.median()
        mad = (scores_tensor - median).abs().median()

        if mad < 1e-8:
            # All scores similar — keep everything
            keep_mask = torch.ones(len(scores), dtype=torch.bool)
        else:
            threshold = median + tau * mad
            keep_mask = scores_tensor >= threshold

        # Enforce retention bounds
        n_total = len(scores)
        n_kept = keep_mask.sum().item()
        min_keep = max(1, int(n_total * self.config.min_retention))
        max_keep = int(n_total * self.config.max_retention)

        if n_kept < min_keep:
            _, top_indices = scores_tensor.topk(min_keep)
            keep_mask = torch.zeros(n_total, dtype=torch.bool)
            keep_mask[top_indices] = True
        elif n_kept > max_keep:
            _, top_indices = scores_tensor.topk(max_keep)
            keep_mask = torch.zeros(n_total, dtype=torch.bool)
            keep_mask[top_indices] = True

        # Step 4: Build compacted briefing (preserve order)
        retained_segments = []
        total_original_tokens = 0
        total_kept_tokens = 0

        for i, seg in enumerate(self._trajectory_segments):
            total_original_tokens += seg["token_count"]
            if keep_mask[i]:
                retained_segments.append(seg)
                total_kept_tokens += seg["token_count"]

        # Apply hard token cap if specified
        if max_tokens and total_kept_tokens > max_tokens:
            # Trim from the least relevant retained segments
            retained_scores = [(scores[i], i, seg) for i, seg in enumerate(self._trajectory_segments) if keep_mask[i]]
            retained_scores.sort(key=lambda x: x[0], reverse=True)

            budget = max_tokens
            final_segments = []
            for score, orig_idx, seg in retained_scores:
                if budget <= 0:
                    break
                final_segments.append((orig_idx, seg))
                budget -= seg["token_count"]

            # Re-sort by original position to preserve trajectory order
            final_segments.sort(key=lambda x: x[0])
            retained_segments = [seg for _, seg in final_segments]
            total_kept_tokens = sum(seg["token_count"] for seg in retained_segments)

        # Compose the briefing text
        briefing_parts = []
        for seg in retained_segments:
            prefix = f"[{seg['type'].upper()}]" if seg["type"] != "observation" else ""
            text = seg["text"].strip()
            if prefix:
                briefing_parts.append(f"{prefix} {text}")
            else:
                briefing_parts.append(text)

        briefing_text = "\n---\n".join(briefing_parts)

        elapsed = time.time() - start_time
        tokens_saved = total_original_tokens - total_kept_tokens
        self._total_tokens_saved += tokens_saved

        stats = CompactionStats(
            original_length=total_original_tokens,
            compacted_length=total_kept_tokens,
            retention_rate=total_kept_tokens / max(1, total_original_tokens),
            compaction_time_s=elapsed,
            tau_used=tau,
            tokens_saved=tokens_saved,
        )

        logger.info(
            f"Text Latent Briefing: {total_original_tokens} → {total_kept_tokens} tokens "
            f"({stats.compression_ratio:.1%} compressed, {elapsed:.3f}s, τ={tau}, "
            f"{n_kept}/{n_total} segments retained)"
        )

        return briefing_text, stats

    def reset(self):
        """Clear trajectory memory for a new task."""
        self._trajectory_segments = []
        self._total_tokens_saved = 0

    @property
    def total_tokens_saved(self) -> int:
        return self._total_tokens_saved

    @property
    def trajectory_length(self) -> int:
        return len(self._trajectory_segments)

    def get_full_trajectory_text(self) -> str:
        """Get the full uncompressed trajectory (for comparison/debugging)."""
        return "\n---\n".join(seg["text"] for seg in self._trajectory_segments)
