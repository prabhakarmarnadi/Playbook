"""
Risk Scoring Layer
==================
Classifies clause types and intents into risk categories with severity weights.
Produces per-agreement risk profiles and portfolio-level risk heatmaps.

Risk Categories:
  - liability       (indemnification, damages, limitation of liability)
  - ip              (IP ownership, licensing, work-for-hire, assignment)
  - termination     (term, renewal, early termination, cure periods)
  - compliance      (governing law, jurisdiction, regulatory, audit)
  - financial       (payment, pricing, penalties, late fees)
  - confidentiality (NDA, trade secrets, data protection, GDPR)
  - operational     (SLA, warranties, representations, force majeure)

Severity Levels: critical (1.0), high (0.75), medium (0.5), low (0.25)
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import duckdb
import numpy as np

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

RISK_CATEGORIES = {
    "liability":       {"color": "#f85149", "icon": "⚠"},
    "ip":              {"color": "#bc8cff", "icon": "🔒"},
    "termination":     {"color": "#d29922", "icon": "⏹"},
    "compliance":      {"color": "#58a6ff", "icon": "📋"},
    "financial":       {"color": "#3fb950", "icon": "💰"},
    "confidentiality": {"color": "#f778ba", "icon": "🔐"},
    "operational":     {"color": "#79c0ff", "icon": "⚙"},
}

SEVERITY_WEIGHTS = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}

# ── Keyword heuristics for fast pre-classification ──
_CATEGORY_KEYWORDS = {
    "liability": [
        "indemnif", "liability", "damages", "limitation of liability",
        "hold harmless", "negligence", "consequential", "loss", "claim",
    ],
    "ip": [
        "intellectual property", "patent", "copyright", "trademark",
        "license", "work for hire", "assignment of rights", "ownership",
        "proprietary", "invention",
    ],
    "termination": [
        "terminat", "expir", "renewal", "cure period", "notice period",
        "early termination", "cancellation", "wind-down", "survival",
    ],
    "compliance": [
        "governing law", "jurisdiction", "arbitration", "regulatory",
        "audit", "compliance", "applicable law", "dispute resolution",
        "amendment", "waiver",
    ],
    "financial": [
        "payment", "price", "fee", "penalty", "invoice", "tax",
        "late fee", "interest", "compensation", "reimburs",
    ],
    "confidentiality": [
        "confidential", "non-disclosure", "nda", "trade secret",
        "data protection", "privacy", "gdpr", "personal data",
        "information security",
    ],
    "operational": [
        "warranty", "representation", "sla", "service level",
        "force majeure", "insurance", "performance", "deliverable",
        "acceptance", "maintenance",
    ],
}

_SEVERITY_KEYWORDS = {
    "critical": [
        "unlimited liability", "uncapped", "sole remedy", "exclusive",
        "irrevocable", "perpetual license", "assignment without consent",
        "automatic renewal", "mandatory arbitration",
    ],
    "high": [
        "indemnif", "consequential damages", "liquidated damages",
        "termination for convenience", "change of control",
        "non-compete", "exclusivity",
    ],
    "medium": [
        "limitation", "cap", "notice", "cure period", "mutual",
        "reasonable efforts", "material breach",
    ],
    "low": [
        "boilerplate", "entire agreement", "counterpart", "headings",
        "severability", "further assurance",
    ],
}

RISK_SCORING_PROMPT = """You are a legal risk analyst. Classify the following clause type into risk categories.

Clause Type: {clause_label}
Description: {clause_description}
Sample Text (truncated):
{sample_text}

Respond in JSON with this exact structure:
{{
  "categories": [
    {{
      "category": "<one of: liability, ip, termination, compliance, financial, confidentiality, operational>",
      "severity": "<one of: critical, high, medium, low>",
      "rationale": "<one sentence explaining why>"
    }}
  ]
}}

Rules:
- A clause type can belong to 1-3 categories (e.g., an indemnification clause is both "liability" and "financial").
- Always pick the MOST SPECIFIC severity. Uncapped indemnification = critical. Standard mutual NDA = low.
- If the clause is purely procedural (headings, counterparts), return categories: [{{"category": "operational", "severity": "low", "rationale": "Procedural boilerplate"}}]
"""


@dataclass
class RiskScore:
    category: str
    severity: str
    severity_weight: float
    rationale: str


@dataclass
class ClusterRisk:
    cluster_id: str
    cluster_label: str
    risk_scores: list[RiskScore]
    max_severity: float
    primary_category: str


@dataclass
class AgreementRisk:
    agreement_id: str
    filename: str
    total_risk_score: float
    category_scores: dict[str, float]
    high_risk_clauses: list[dict]
    risk_profile: str  # "high-risk" / "moderate" / "low-risk"


class RiskScorer:
    """Classifies clause types and intents into risk categories with severity weights."""

    def __init__(self, db_path: str, llm: Optional[LLMClient] = None):
        self.db = duckdb.connect(db_path, read_only=False)
        self.llm = llm
        self._ensure_tables()

    def _ensure_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS cluster_risks (
                cluster_id VARCHAR PRIMARY KEY,
                categories JSON,
                max_severity DOUBLE,
                primary_category VARCHAR,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS agreement_risks (
                agreement_id VARCHAR PRIMARY KEY,
                total_risk_score DOUBLE,
                category_scores JSON,
                high_risk_count INTEGER,
                risk_profile VARCHAR,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def score_all_clusters(self, use_llm: bool = True) -> list[ClusterRisk]:
        """Score all clause types by risk category and severity."""
        clusters = self.db.execute("""
            SELECT c.cluster_id, c.label, c.description,
                   (SELECT chunk_text FROM chunks ch
                    JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
                    WHERE ca.cluster_id = c.cluster_id
                    LIMIT 1) as sample_text
            FROM clusters c
            ORDER BY c.label
        """).fetchall()

        results = []
        for cluster_id, label, description, sample_text in clusters:
            if use_llm and self.llm:
                risk = self._score_cluster_llm(cluster_id, label, description, sample_text or "")
            else:
                risk = self._score_cluster_heuristic(cluster_id, label, description, sample_text or "")

            # Persist
            categories_json = json.dumps([
                {"category": rs.category, "severity": rs.severity,
                 "severity_weight": rs.severity_weight, "rationale": rs.rationale}
                for rs in risk.risk_scores
            ])
            self.db.execute("""
                INSERT OR REPLACE INTO cluster_risks (cluster_id, categories, max_severity, primary_category)
                VALUES (?, ?, ?, ?)
            """, [risk.cluster_id, categories_json, risk.max_severity, risk.primary_category])

            results.append(risk)
            logger.info(f"  [{risk.primary_category}:{risk.max_severity:.2f}] {label}")

        logger.info(f"Scored {len(results)} clusters for risk")
        return results

    def _score_cluster_heuristic(self, cluster_id: str, label: str, description: str,
                                  sample_text: str) -> ClusterRisk:
        """Fast keyword-based risk classification (no LLM needed)."""
        text = f"{label} {description} {sample_text}".lower()
        scores = []

        for category, keywords in _CATEGORY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits == 0:
                continue

            # Determine severity from keyword matches
            severity = "low"
            for sev in ["critical", "high", "medium"]:
                if any(kw in text for kw in _SEVERITY_KEYWORDS[sev]):
                    severity = sev
                    break

            scores.append(RiskScore(
                category=category,
                severity=severity,
                severity_weight=SEVERITY_WEIGHTS[severity],
                rationale=f"Matched {hits} keywords for {category}",
            ))

        if not scores:
            scores.append(RiskScore(
                category="operational", severity="low",
                severity_weight=0.25, rationale="No specific risk signals detected",
            ))

        max_sev = max(s.severity_weight for s in scores)
        primary = max(scores, key=lambda s: s.severity_weight).category

        return ClusterRisk(
            cluster_id=cluster_id, cluster_label=label,
            risk_scores=scores, max_severity=max_sev, primary_category=primary,
        )

    def _score_cluster_llm(self, cluster_id: str, label: str, description: str,
                            sample_text: str) -> ClusterRisk:
        """LLM-based risk classification with keyword fallback."""
        prompt = RISK_SCORING_PROMPT.format(
            clause_label=label,
            clause_description=description or "No description available",
            sample_text=(sample_text or "")[:1500],
        )
        try:
            result = self.llm.complete_json(prompt, system="You are a legal risk analyst.")
            categories = result.get("categories", [])
            scores = []
            for cat in categories:
                category = cat.get("category", "operational")
                severity = cat.get("severity", "low")
                if category not in RISK_CATEGORIES:
                    category = "operational"
                if severity not in SEVERITY_WEIGHTS:
                    severity = "low"
                scores.append(RiskScore(
                    category=category,
                    severity=severity,
                    severity_weight=SEVERITY_WEIGHTS[severity],
                    rationale=cat.get("rationale", ""),
                ))
            if not scores:
                return self._score_cluster_heuristic(cluster_id, label, description, sample_text)

            max_sev = max(s.severity_weight for s in scores)
            primary = max(scores, key=lambda s: s.severity_weight).category
            return ClusterRisk(
                cluster_id=cluster_id, cluster_label=label,
                risk_scores=scores, max_severity=max_sev, primary_category=primary,
            )
        except Exception as e:
            logger.warning(f"LLM risk scoring failed for {label}: {e}, falling back to heuristic")
            return self._score_cluster_heuristic(cluster_id, label, description, sample_text)

    def score_all_agreements(self) -> list[AgreementRisk]:
        """Compute per-agreement risk profiles from cluster risk scores."""
        # Load cluster risks
        cluster_risks = {}
        rows = self.db.execute("SELECT cluster_id, categories, max_severity, primary_category FROM cluster_risks").fetchall()
        for cid, cats_json, max_sev, primary in rows:
            cats = json.loads(cats_json) if isinstance(cats_json, str) else cats_json
            cluster_risks[cid] = {
                "categories": cats,
                "max_severity": max_sev,
                "primary_category": primary,
            }

        # Get agreement → clause types mapping
        agreements = self.db.execute("""
            SELECT a.agreement_id, a.filename,
                   COALESCE(cl.clause_type_id, ca.cluster_id) as cluster_id,
                   c.label as cluster_label
            FROM agreements a
            JOIN clauses cl ON a.agreement_id = cl.agreement_id
            LEFT JOIN chunks ch ON cl.clause_id = ch.clause_id
            LEFT JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            LEFT JOIN clusters c ON COALESCE(cl.clause_type_id, ca.cluster_id) = c.cluster_id
            WHERE COALESCE(cl.clause_type_id, ca.cluster_id) IS NOT NULL
        """).fetchall()

        # Group by agreement
        agr_clusters: dict[str, dict] = {}
        for agr_id, filename, cluster_id, cluster_label in agreements:
            if agr_id not in agr_clusters:
                agr_clusters[agr_id] = {"filename": filename, "clusters": set()}
            if cluster_id:
                agr_clusters[agr_id]["clusters"].add((cluster_id, cluster_label or ""))

        results = []
        for agr_id, info in agr_clusters.items():
            category_scores = {cat: 0.0 for cat in RISK_CATEGORIES}
            high_risk_clauses = []

            for cluster_id, cluster_label in info["clusters"]:
                risk = cluster_risks.get(cluster_id)
                if not risk:
                    continue
                for cat_info in risk["categories"]:
                    cat = cat_info["category"]
                    weight = cat_info.get("severity_weight", SEVERITY_WEIGHTS.get(cat_info["severity"], 0.25))
                    category_scores[cat] = max(category_scores[cat], weight)

                if risk["max_severity"] >= 0.75:
                    high_risk_clauses.append({
                        "cluster_id": cluster_id,
                        "cluster_label": cluster_label,
                        "severity": risk["max_severity"],
                        "category": risk["primary_category"],
                    })

            total = sum(category_scores.values())
            n_cats = sum(1 for v in category_scores.values() if v > 0)
            avg_risk = total / max(n_cats, 1)

            if avg_risk >= 0.7 or len(high_risk_clauses) >= 3:
                profile = "high-risk"
            elif avg_risk >= 0.4 or len(high_risk_clauses) >= 1:
                profile = "moderate"
            else:
                profile = "low-risk"

            agr_risk = AgreementRisk(
                agreement_id=agr_id,
                filename=info["filename"],
                total_risk_score=round(total, 3),
                category_scores={k: round(v, 3) for k, v in category_scores.items()},
                high_risk_clauses=high_risk_clauses,
                risk_profile=profile,
            )

            self.db.execute("""
                INSERT OR REPLACE INTO agreement_risks
                (agreement_id, total_risk_score, category_scores, high_risk_count, risk_profile)
                VALUES (?, ?, ?, ?, ?)
            """, [agr_id, total, json.dumps(agr_risk.category_scores),
                  len(high_risk_clauses), profile])

            results.append(agr_risk)

        # Summary
        profiles = {}
        for r in results:
            profiles[r.risk_profile] = profiles.get(r.risk_profile, 0) + 1
        logger.info(f"Agreement risk profiles: {profiles}")
        return results

    def get_portfolio_heatmap(self) -> dict:
        """Returns portfolio-level risk heatmap data."""
        cluster_risks = self.db.execute("""
            SELECT cr.cluster_id, c.label, cr.categories, cr.max_severity, cr.primary_category
            FROM cluster_risks cr
            JOIN clusters c ON cr.cluster_id = c.cluster_id
            ORDER BY cr.max_severity DESC
        """).fetchall()

        agr_risks = self.db.execute("""
            SELECT ar.agreement_id, a.filename, ar.total_risk_score,
                   ar.category_scores, ar.risk_profile
            FROM agreement_risks ar
            JOIN agreements a ON ar.agreement_id = a.agreement_id
            ORDER BY ar.total_risk_score DESC
        """).fetchall()

        category_summary = {cat: {"count": 0, "max_severity": 0, "clusters": []}
                           for cat in RISK_CATEGORIES}

        for cid, label, cats_json, max_sev, primary in cluster_risks:
            cats = json.loads(cats_json) if isinstance(cats_json, str) else cats_json
            for cat_info in cats:
                cat = cat_info["category"]
                if cat in category_summary:
                    category_summary[cat]["count"] += 1
                    category_summary[cat]["max_severity"] = max(
                        category_summary[cat]["max_severity"], cat_info.get("severity_weight", 0))
                    category_summary[cat]["clusters"].append({
                        "cluster_id": cid, "label": label,
                        "severity": cat_info["severity"], "rationale": cat_info.get("rationale", ""),
                    })

        profiles = {"high-risk": 0, "moderate": 0, "low-risk": 0}
        for _, _, _, _, profile in agr_risks:
            profiles[profile] = profiles.get(profile, 0) + 1

        return {
            "category_summary": category_summary,
            "agreement_profiles": profiles,
            "total_agreements": len(agr_risks),
            "top_risk_agreements": [
                {"agreement_id": aid, "filename": fn, "score": round(score, 2),
                 "profile": prof, "categories": json.loads(cats) if isinstance(cats, str) else cats}
                for aid, fn, score, cats, prof in agr_risks[:20]
            ],
        }

    def summary(self) -> str:
        """Print risk scoring summary."""
        cluster_count = self.db.execute("SELECT COUNT(*) FROM cluster_risks").fetchone()[0]
        agr_count = self.db.execute("SELECT COUNT(*) FROM agreement_risks").fetchone()[0]

        if cluster_count == 0:
            return "No risk scores computed yet."

        cats = self.db.execute("""
            SELECT primary_category, COUNT(*), AVG(max_severity)
            FROM cluster_risks GROUP BY primary_category ORDER BY AVG(max_severity) DESC
        """).fetchall()

        profiles = self.db.execute("""
            SELECT risk_profile, COUNT(*)
            FROM agreement_risks GROUP BY risk_profile
        """).fetchall()

        lines = [f"Risk Scoring Summary ({cluster_count} clusters, {agr_count} agreements)"]
        lines.append("─" * 60)
        lines.append(f"{'Category':<18} {'Clusters':>8} {'Avg Severity':>14}")
        lines.append("─" * 60)
        for cat, count, avg in cats:
            icon = RISK_CATEGORIES.get(cat, {}).get("icon", "")
            lines.append(f"{icon} {cat:<15} {count:>8} {avg:>13.2f}")
        lines.append("")
        lines.append("Agreement Risk Profiles:")
        for profile, count in profiles:
            lines.append(f"  {profile}: {count}")
        return "\n".join(lines)

    def close(self):
        self.db.close()
