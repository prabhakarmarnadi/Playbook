"""
Regression tracker — compare current metrics against a saved baseline.

Produces a Markdown table with [Metric, Current, Baseline, Delta, Status].
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Metrics where LOWER is better (all others: higher is better)
LOWER_IS_BETTER = {
    "clustering.davies_bouldin_index",
    "clustering.chunk_outlier_pct",
    "clustering.clause_outlier_pct",
    "clustering.clauses_no_embedding_pct",
    "clustering.largest_cluster_pct",
    "clustering.cluster_size_cv",
    "system.low_confidence_extraction_pct",
    "judge.hallucination_rate",
    "judge.faithfulness_below_3_pct",
    "judge.actionability_below_3_pct",
}

# Thresholds for ❌ status: if current is worse BY this relative amount, it's a regression
DEFAULT_REGRESSION_THRESHOLD = 0.05  # 5% relative degradation

# Hard thresholds: if metric crosses this, it's ❌ regardless of baseline
HARD_THRESHOLDS = {
    "clustering.chunk_outlier_pct": 20.0,       # more than 20% outliers = bad
    "clustering.clause_outlier_pct": 50.0,       # more than 50% untyped = bad
    "clustering.silhouette_score": 0.05,         # below 0.05 = no structure
    "clustering.davies_bouldin_index": 5.0,      # above 5.0 = garbage
    "clustering.adjusted_score": 0.05,           # below 0.05 = pipeline broken
    "health.score": 0.50,                        # below 0.50 = pipeline broken
    "fields.f1": 0.1,                            # below 10% F1 = broken
}


@dataclass
class RegressionTracker:
    """Compare current vs baseline metrics, produce regression report."""

    baseline_path: Path | str | None = None
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD
    _baseline: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.baseline_path:
            p = Path(self.baseline_path)
            if p.exists():
                with open(p) as f:
                    self._baseline = json.load(f)
                logger.info(f"Loaded baseline with {len(self._baseline)} metrics")
            else:
                logger.warning(f"Baseline file not found: {p}")

    def save_baseline(self, metrics: dict[str, Any], path: Path | str):
        """Save current metrics as the new baseline."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Only save numeric values
        baseline = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        with open(p, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Saved baseline with {len(baseline)} metrics to {p}")

    def compare(self, current: dict[str, Any]) -> list[dict]:
        """
        Compare current metrics to baseline.

        Returns list of {metric, current, baseline, delta, delta_pct, status, direction}.
        """
        rows = []
        all_keys = sorted(
            set(current.keys()) | set(self._baseline.keys())
        )

        for key in all_keys:
            cur = current.get(key)
            base = self._baseline.get(key)

            if cur is None or not isinstance(cur, (int, float)):
                continue

            row = {
                "metric": key,
                "current": cur,
                "baseline": base,
                "delta": None,
                "delta_pct": None,
                "status": "✅",
                "direction": "↑" if key not in LOWER_IS_BETTER else "↓",
            }

            if base is not None and isinstance(base, (int, float)):
                delta = cur - base
                row["delta"] = round(delta, 4)
                if base != 0:
                    row["delta_pct"] = round(delta / abs(base) * 100, 2)
                else:
                    row["delta_pct"] = 0.0

                # Determine regression
                lower_better = key in LOWER_IS_BETTER
                if lower_better:
                    regressed = delta > 0 and (
                        abs(delta) > abs(base) * self.regression_threshold
                        if base != 0 else delta > 0
                    )
                else:
                    regressed = delta < 0 and (
                        abs(delta) > abs(base) * self.regression_threshold
                        if base != 0 else delta < 0
                    )

                if regressed:
                    row["status"] = "❌"
            else:
                row["status"] = "🆕"  # new metric, no baseline

            # Hard threshold check
            if key in HARD_THRESHOLDS:
                threshold = HARD_THRESHOLDS[key]
                lower_better = key in LOWER_IS_BETTER
                if lower_better and cur > threshold:
                    row["status"] = "❌"
                elif not lower_better and cur < threshold:
                    row["status"] = "❌"

            rows.append(row)

        return rows

    def format_markdown(self, rows: list[dict]) -> str:
        """Format comparison rows as a Markdown table."""
        lines = []
        lines.append("## Evaluation Report")
        lines.append("")

        # Summary counts
        n_pass = sum(1 for r in rows if r["status"] == "✅")
        n_fail = sum(1 for r in rows if r["status"] == "❌")
        n_new = sum(1 for r in rows if r["status"] == "🆕")
        lines.append(
            f"**Summary:** {n_pass} ✅  {n_fail} ❌  {n_new} 🆕  "
            f"({len(rows)} total metrics)"
        )
        lines.append("")

        # Group by component
        groups: dict[str, list[dict]] = {}
        for row in rows:
            component = row["metric"].split(".")[0]
            groups.setdefault(component, []).append(row)

        for component, group_rows in groups.items():
            lines.append(f"### {component.upper()}")
            lines.append("")
            lines.append(
                "| Metric | Current | Baseline | Delta | Δ% | Status |"
            )
            lines.append(
                "|--------|---------|----------|-------|-----|--------|"
            )

            for r in group_rows:
                metric_short = r["metric"].split(".", 1)[1]
                cur_str = self._fmt(r["current"])
                base_str = self._fmt(r["baseline"]) if r["baseline"] is not None else "—"
                delta_str = (
                    f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
                )
                dpct_str = (
                    f"{r['delta_pct']:+.1f}%" if r["delta_pct"] is not None else "—"
                )
                lines.append(
                    f"| {metric_short} | {cur_str} | {base_str} | "
                    f"{delta_str} | {dpct_str} | {r['status']} |"
                )

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"
        return str(v)
