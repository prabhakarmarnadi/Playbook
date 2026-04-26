"""
XGBoost integration layer for the Clustering V2 pipeline.
==========================================================

Provides learned models that replace heuristic scoring/classification logic
in 5 integration points:

  1. ExtractionConfidenceModel  — replaces _confidence_score() heuristic
  2. AssignmentClassifier       — replaces fixed cosine thresholds in cluster_assigner
  3. RiskClassifier             — replaces keyword matching in risk_scorer
  4. FieldImportanceScorer      — replaces confidence-only pruning in rlm_v3
  5. IntentDeduplicator         — replaces fixed 0.82 embedding threshold

All models:
  - Gracefully fall back to the original heuristic if no trained model exists
  - Serialize to / load from a single directory (xgboost_models/)
  - Expose .predict() for inference and .train() for retraining from feedback
  - Are thread-safe (XGBoost is read-only after load)

Usage:
    from core.xgb_models import XGBModelRegistry
    registry = XGBModelRegistry("data/xgboost_models")
    registry.load_all()  # loads whatever .json model files exist

    # In production_pipeline._confidence_score():
    conf = registry.extraction_confidence.predict(features)

    # In cluster_assigner.assign_document():
    assignment = registry.assignment.predict(features)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed — all XGB models will use heuristic fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extraction_features(
    value: str,
    source_text: str,
    field_type: str = "text",
    agreement_type: str = "",
    chunk_length: int = 0,
) -> np.ndarray:
    """Build feature vector for extraction confidence prediction.

    Returns 12-dim float32 array:
      [verbatim_match, partial_match, token_overlap_ratio, value_len,
       has_digits, is_json_like, value_to_chunk_ratio, chunk_length,
       field_type_numeric, field_type_date, field_type_currency, field_type_text]
    """
    v_lower = str(value).lower().strip()
    chunk_lower = source_text.lower()

    verbatim = 1.0 if v_lower in chunk_lower else 0.0
    v_tokens = set(v_lower.split())
    c_tokens = set(chunk_lower.split())
    overlap = len(v_tokens & c_tokens) / max(len(v_tokens), 1)
    partial = 1.0 if not verbatim and overlap > 0.5 else 0.0

    v_len = len(str(value))
    has_digits = 1.0 if any(c.isdigit() for c in str(value)) else 0.0
    is_json = 1.0 if str(value).startswith(("{", "[")) else 0.0
    ratio = v_len / max(len(source_text), 1)
    c_len = chunk_length or len(source_text)

    # One-hot field type
    ft = field_type.lower()
    ft_numeric = 1.0 if ft in ("number", "numeric", "integer", "float", "amount") else 0.0
    ft_date = 1.0 if ft in ("date", "datetime", "timestamp") else 0.0
    ft_currency = 1.0 if ft in ("currency", "money", "dollar") else 0.0
    ft_text = 1.0 if ft_numeric == 0 and ft_date == 0 and ft_currency == 0 else 0.0

    return np.array([
        verbatim, partial, overlap, v_len, has_digits, is_json,
        ratio, c_len, ft_numeric, ft_date, ft_currency, ft_text,
    ], dtype=np.float32)


def assignment_features(
    cosine_to_nearest: float,
    cosine_gap: float,
    cluster_size: int,
    cluster_cohesion: float,
    centroid_age_days: float = 0.0,
    novel_buffer_fill: float = 0.0,
) -> np.ndarray:
    """Build feature vector for cluster assignment classification.

    Returns 6-dim float32 array.
    """
    return np.array([
        cosine_to_nearest, cosine_gap, cluster_size, cluster_cohesion,
        centroid_age_days, novel_buffer_fill,
    ], dtype=np.float32)


def risk_features(
    clause_text: str,
    clause_label: str = "",
    description: str = "",
) -> np.ndarray:
    """Build feature vector for risk classification.

    Returns 14-dim float32 array:
      [7 category keyword counts] + [4 severity keyword counts] +
      [clause_length, num_legal_citations, keyword_density]
    """
    from core.risk_scorer import _CATEGORY_KEYWORDS, _SEVERITY_KEYWORDS

    text = f"{clause_label} {description} {clause_text}".lower()
    words = text.split()
    n_words = max(len(words), 1)

    cat_counts = []
    total_kw = 0
    for cat in ["liability", "ip", "termination", "compliance",
                "financial", "confidentiality", "operational"]:
        hits = sum(1 for kw in _CATEGORY_KEYWORDS[cat] if kw in text)
        cat_counts.append(float(hits))
        total_kw += hits

    sev_counts = []
    for sev in ["critical", "high", "medium", "low"]:
        hits = sum(1 for kw in _SEVERITY_KEYWORDS[sev] if kw in text)
        sev_counts.append(float(hits))

    clause_len = float(len(text))
    # Legal citation heuristic: count "Section X", "Article X", "§"
    citations = sum(1 for w in words if w in ("section", "article", "§", "clause"))
    kw_density = total_kw / n_words

    return np.array(cat_counts + sev_counts + [clause_len, citations, kw_density],
                    dtype=np.float32)


def field_importance_features(
    confidence_avg: float,
    confidence_std: float,
    extraction_rate: float,
    feedback_corrections: int = 0,
    field_age_days: float = 0.0,
    source_is_grounded: bool = True,
) -> np.ndarray:
    """Build feature vector for field importance scoring.

    Returns 6-dim float32 array.
    """
    return np.array([
        confidence_avg, confidence_std, extraction_rate,
        float(feedback_corrections), field_age_days,
        1.0 if source_is_grounded else 0.0,
    ], dtype=np.float32)


def intent_dedup_features(
    name_edit_distance: float,
    description_jaccard: float,
    embedding_cosine: float,
    frequency_ratio: float = 1.0,
    cluster_overlap: float = 0.0,
) -> np.ndarray:
    """Build feature vector for intent pair deduplication.

    Returns 5-dim float32 array.
    """
    return np.array([
        name_edit_distance, description_jaccard, embedding_cosine,
        frequency_ratio, cluster_overlap,
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# HEURISTIC FALLBACKS (copied from existing pipeline for graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════════

def _heuristic_confidence(value: str, source_text: str) -> float:
    """Original heuristic from production_pipeline._confidence_score."""
    v_str = str(value)
    chunk_lower = source_text.lower()
    v_lower = v_str.lower().strip()
    conf = 0.5
    if v_lower in chunk_lower:
        conf += 0.3
    elif len(v_lower) > 5 and any(w in chunk_lower for w in v_lower.split()[:3]):
        conf += 0.15
    if len(v_str) > 3:
        conf += 0.05
    if any(c.isdigit() for c in v_str):
        conf += 0.05
    if len(v_str) > 50:
        conf -= 0.1
    if v_str.startswith("{") or v_str.startswith("["):
        conf -= 0.15
    return max(0.1, min(1.0, round(conf, 3)))


def _heuristic_assignment(cosine_to_nearest: float,
                          high_thresh: float = 0.85,
                          tent_thresh: float = 0.60) -> str:
    """Original heuristic from cluster_assigner."""
    if cosine_to_nearest >= high_thresh:
        return "high_confidence"
    elif cosine_to_nearest >= tent_thresh:
        return "tentative"
    return "novel"


# ═══════════════════════════════════════════════════════════════════════════════
# BASE MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class XGBModelBase:
    """Thin wrapper around a single XGBoost model with save/load + heuristic fallback."""

    def __init__(self, name: str, model_dir: Path, n_features: int,
                 objective: str = "binary:logistic", n_classes: int = 2):
        self.name = name
        self.model_dir = model_dir
        self.n_features = n_features
        self.objective = objective
        self.n_classes = n_classes
        self._model: Optional[xgb.Booster] = None  # type: ignore[union-attr]
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and _XGB_AVAILABLE

    def model_path(self) -> Path:
        return self.model_dir / f"{self.name}.json"

    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        if not _XGB_AVAILABLE:
            return False
        p = self.model_path()
        if not p.exists():
            logger.info(f"XGB model '{self.name}' not found at {p} — using heuristic")
            return False
        try:
            self._model = xgb.Booster()
            self._model.load_model(str(p))
            self._ready = True
            logger.info(f"Loaded XGB model '{self.name}' from {p}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load XGB model '{self.name}': {e}")
            self._ready = False
            return False

    def save(self):
        """Save trained model to disk."""
        if self._model is None:
            raise RuntimeError(f"No trained model for '{self.name}'")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self.model_path()))
        logger.info(f"Saved XGB model '{self.name}' to {self.model_path()}")

    def train(self, X: np.ndarray, y: np.ndarray,
              params: dict | None = None, num_rounds: int = 200,
              early_stopping: int = 20, eval_fraction: float = 0.15):
        """Train / retrain the model from labeled data."""
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost is not installed")

        n = len(X)
        split = int(n * (1 - eval_fraction))
        idx = np.random.permutation(n)
        X_train, y_train = X[idx[:split]], y[idx[:split]]
        X_eval, y_eval = X[idx[split:]], y[idx[split:]]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        default_params = {
            "objective": self.objective,
            "eval_metric": "logloss" if "logistic" in self.objective else "rmse",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "seed": 42,
        }
        if self.objective == "multi:softprob":
            default_params["num_class"] = self.n_classes
            default_params["eval_metric"] = "mlogloss"
        if params:
            default_params.update(params)

        self._model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(deval, "eval")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        self._ready = True
        logger.info(
            f"Trained XGB model '{self.name}' on {split} samples "
            f"(eval={n - split}), best_iteration={self._model.best_iteration}"
        )

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw prediction from XGBoost. Returns probabilities or scores."""
        if not self.ready or self._model is None:
            raise RuntimeError(f"Model '{self.name}' not ready — call load() or train() first")
        dmat = xgb.DMatrix(X)
        return self._model.predict(dmat)

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Feature importance scores from the trained model."""
        if not self.ready or self._model is None:
            return {}
        return self._model.get_score(importance_type=importance_type)


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIZED MODEL WRAPPERS (with domain-specific predict + fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractionConfidenceModel(XGBModelBase):
    """Predicts extraction confidence score [0, 1] from extraction features."""

    def __init__(self, model_dir: Path):
        super().__init__("extraction_confidence", model_dir, n_features=12,
                         objective="reg:squarederror")

    def predict(self, value: str, source_text: str,
                field_type: str = "text", **kwargs) -> float:
        """Return confidence score. Falls back to heuristic if model not loaded."""
        if not self.ready:
            return _heuristic_confidence(value, source_text)
        feats = extraction_features(value, source_text, field_type, **kwargs)
        score = float(self.predict_raw(feats.reshape(1, -1))[0])
        return max(0.05, min(1.0, round(score, 3)))

    def predict_batch(self, values: list[str], source_texts: list[str],
                      field_types: list[str] | None = None) -> list[float]:
        """Batch prediction for efficiency."""
        if not self.ready:
            return [_heuristic_confidence(v, s) for v, s in zip(values, source_texts)]
        ft = field_types or ["text"] * len(values)
        X = np.stack([extraction_features(v, s, t) for v, s, t in zip(values, source_texts, ft)])
        scores = self.predict_raw(X)
        return [max(0.05, min(1.0, round(float(s), 3))) for s in scores]


class AssignmentClassifier(XGBModelBase):
    """Predicts assignment type (high_confidence / tentative / novel) from assignment features."""

    LABELS = ["high_confidence", "tentative", "novel"]

    def __init__(self, model_dir: Path):
        super().__init__("assignment_classifier", model_dir, n_features=6,
                         objective="multi:softprob", n_classes=3)

    def predict(self, cosine_to_nearest: float, cosine_gap: float = 0.0,
                cluster_size: int = 50, cluster_cohesion: float = 0.8,
                **kwargs) -> dict:
        """Return assignment type + probabilities. Falls back to cosine thresholds."""
        if not self.ready:
            label = _heuristic_assignment(cosine_to_nearest)
            return {"label": label, "probabilities": {}, "source": "heuristic"}

        feats = assignment_features(cosine_to_nearest, cosine_gap,
                                    cluster_size, cluster_cohesion, **kwargs)
        probs = self.predict_raw(feats.reshape(1, -1))[0]
        prob_dict = {lbl: float(p) for lbl, p in zip(self.LABELS, probs)}
        label = self.LABELS[int(np.argmax(probs))]
        return {"label": label, "probabilities": prob_dict, "source": "xgboost"}


RISK_CATEGORIES_LIST = [
    "liability", "ip", "termination", "compliance",
    "financial", "confidentiality", "operational",
]


class RiskClassifier(XGBModelBase):
    """Multi-label risk classification: for each category, predicts severity probability."""

    def __init__(self, model_dir: Path):
        # One model outputs 7 probabilities (one per category present/absent)
        super().__init__("risk_classifier", model_dir, n_features=14,
                         objective="multi:softprob", n_classes=7)

    def predict(self, clause_text: str, clause_label: str = "",
                description: str = "") -> list[dict]:
        """Return list of {category, severity_weight, rationale}.
        Falls back to keyword heuristic.
        """
        if not self.ready:
            return []  # caller should fall back to existing heuristic

        feats = risk_features(clause_text, clause_label, description)
        probs = self.predict_raw(feats.reshape(1, -1))[0]

        results = []
        for i, cat in enumerate(RISK_CATEGORIES_LIST):
            if probs[i] > 0.3:  # threshold for category presence
                results.append({
                    "category": cat,
                    "probability": float(probs[i]),
                    "source": "xgboost",
                })

        if not results:
            results.append({
                "category": "operational",
                "probability": float(probs[6]),
                "source": "xgboost",
            })

        return sorted(results, key=lambda r: r["probability"], reverse=True)


class FieldImportanceScorer(XGBModelBase):
    """Scores field importance for pruning decisions in RLM state management."""

    def __init__(self, model_dir: Path):
        super().__init__("field_importance", model_dir, n_features=6,
                         objective="reg:squarederror")

    def predict(self, confidence_avg: float, confidence_std: float = 0.0,
                extraction_rate: float = 1.0, **kwargs) -> float:
        """Return importance score [0, 1]. Falls back to confidence_avg."""
        if not self.ready:
            return confidence_avg  # heuristic: importance == confidence
        feats = field_importance_features(
            confidence_avg, confidence_std, extraction_rate, **kwargs)
        return float(np.clip(self.predict_raw(feats.reshape(1, -1))[0], 0, 1))


class IntentDeduplicator(XGBModelBase):
    """Predicts whether two intents should be merged (binary classifier)."""

    def __init__(self, model_dir: Path):
        super().__init__("intent_dedup", model_dir, n_features=5,
                         objective="binary:logistic")

    def predict(self, name_edit_distance: float, description_jaccard: float,
                embedding_cosine: float, **kwargs) -> dict:
        """Return {should_merge: bool, confidence: float}. Falls back to fixed threshold."""
        if not self.ready:
            return {
                "should_merge": embedding_cosine > 0.82,
                "confidence": embedding_cosine,
                "source": "heuristic",
            }
        feats = intent_dedup_features(
            name_edit_distance, description_jaccard, embedding_cosine, **kwargs)
        prob = float(self.predict_raw(feats.reshape(1, -1))[0])
        return {
            "should_merge": prob > 0.5,
            "confidence": prob,
            "source": "xgboost",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — single entry point for all XGB models
# ═══════════════════════════════════════════════════════════════════════════════

class XGBModelRegistry:
    """Central registry that manages all XGBoost models for the pipeline.

    Usage:
        registry = XGBModelRegistry("data/xgboost_models")
        registry.load_all()

        # Use individual models:
        conf = registry.extraction_confidence.predict(value, source_text)
        assign = registry.assignment.predict(cosine_sim, gap, ...)
        risks = registry.risk.predict(clause_text, label, desc)
        importance = registry.field_importance.predict(avg_conf, std, rate)
        merge = registry.intent_dedup.predict(edit_dist, jaccard, cosine)
    """

    def __init__(self, model_dir: str | Path = "data/xgboost_models"):
        self.model_dir = Path(model_dir)
        self.extraction_confidence = ExtractionConfidenceModel(self.model_dir)
        self.assignment = AssignmentClassifier(self.model_dir)
        self.risk = RiskClassifier(self.model_dir)
        self.field_importance = FieldImportanceScorer(self.model_dir)
        self.intent_dedup = IntentDeduplicator(self.model_dir)
        self._models = [
            self.extraction_confidence,
            self.assignment,
            self.risk,
            self.field_importance,
            self.intent_dedup,
        ]

    def load_all(self) -> dict[str, bool]:
        """Load all models. Returns {model_name: loaded_successfully}."""
        return {m.name: m.load() for m in self._models}

    def status(self) -> dict[str, str]:
        """Summary of model readiness."""
        return {m.name: "ready" if m.ready else "heuristic_fallback" for m in self._models}

    def all_ready(self) -> bool:
        return all(m.ready for m in self._models)
