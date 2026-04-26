"""
Field registry — V2 port of legacy FieldRegistry + PrecomputedEmbeddings.

Legacy architecture:
  - 222 predefined field JSON configs loaded from disk
  - 500+ business field names from gliner_field_types.json
  - Each field has key_terms embedded at startup via AIFlow gRPC
  - Document chunks are scored against 720+ field embeddings (cosine similarity)
  - Top-scoring chunks per field are concatenated into a "field summary"
  - Field summary is embedded and sent to AIDB for clustering

V2 approach:
  - Static field taxonomy with key_terms inlined (no JSON file dependency)
  - Embed field key_terms with Nomic/MiniLM at startup (local, no gRPC)
  - Score document chunks → field_name → build field-aware summaries
  - Summaries go into macro clustering instead of raw text truncation

This gives macro clustering domain-aware signal: a document that scores high
on "liability_cap", "indemnification", "warranty_disclaimer" looks very different
from one scoring high on "lease_term", "rent_amount", "tenant" — even if both
start with similar boilerplate.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FieldDef:
    """Lightweight field definition with key terms for embedding."""
    name: str
    category: str
    key_terms: list[str] = field(default_factory=list)
    extraction_type: str = "string"
    regex_patterns: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)


# ── Static field taxonomy (ported from legacy 222 predefined + top business) ──
# Grouped by contract domain so scores reveal agreement type.

FIELD_TAXONOMY: list[FieldDef] = [
    # ── Contract Legal / General ──────────────────────────────────────────
    FieldDef("termination_for_convenience", "contract_legal",
             ["termination for convenience", "terminate without cause", "may terminate at any time",
              "upon written notice", "termination without reason"]),
    FieldDef("termination_for_cause", "contract_legal",
             ["termination for cause", "material breach", "cure period", "default",
              "breach of agreement", "failure to perform"]),
    FieldDef("notice_period", "contract_legal",
             ["notice period", "days prior written notice", "advance notice",
              "notice of termination", "calendar days notice"]),
    FieldDef("renewal_term", "contract_legal",
             ["renewal term", "automatic renewal", "auto-renewal", "successive periods",
              "renew for additional", "renewal period"]),
    FieldDef("governing_law", "contract_legal",
             ["governing law", "governed by", "laws of the state", "jurisdiction",
              "applicable law", "construed in accordance"]),
    FieldDef("dispute_resolution", "contract_legal",
             ["dispute resolution", "arbitration", "mediation", "binding arbitration",
              "dispute settlement", "resolve disputes"]),
    FieldDef("assignment_rights", "contract_legal",
             ["assignment", "assign this agreement", "transfer rights", "assignable",
              "consent to assignment", "anti-assignment"]),
    FieldDef("change_of_control", "contract_legal",
             ["change of control", "merger", "acquisition", "change in ownership",
              "sale of substantially all assets"]),
    FieldDef("entire_agreement", "contract_legal",
             ["entire agreement", "supersedes all prior", "complete agreement",
              "no other agreements", "integration clause"]),
    FieldDef("amendments", "contract_legal",
             ["amendment", "modification", "written amendment", "amended by mutual",
              "no oral modifications"]),
    FieldDef("severability", "contract_legal",
             ["severability", "severable", "unenforceable provision", "remaining provisions",
              "invalid or unenforceable"]),
    FieldDef("force_majeure", "contract_legal",
             ["force majeure", "act of god", "natural disaster", "beyond reasonable control",
              "pandemic", "unforeseeable circumstances"]),
    FieldDef("waiver", "contract_legal",
             ["waiver", "failure to enforce", "waive any right", "no waiver",
              "strict performance"]),
    FieldDef("counterparts", "contract_legal",
             ["counterparts", "executed in counterparts", "separate counterparts",
              "electronic signature", "facsimile"]),

    # ── Liability / Indemnification ───────────────────────────────────────
    FieldDef("cap_on_liability", "liability",
             ["cap on liability", "limitation of liability", "aggregate liability",
              "maximum liability", "total liability shall not exceed",
              "consequential damages", "indirect damages"]),
    FieldDef("indemnification", "liability",
             ["indemnification", "indemnify", "hold harmless", "defend and indemnify",
              "indemnified party", "indemnifying party", "third party claims"]),
    FieldDef("warranty_disclaimer", "liability",
             ["warranty disclaimer", "as is", "without warranty", "merchantability",
              "fitness for a particular purpose", "no warranties"]),
    FieldDef("representations_and_warranties", "liability",
             ["representations and warranties", "represents and warrants", "duly organized",
              "authority to execute", "no conflicts", "valid and binding"]),
    FieldDef("insurance_requirements", "liability",
             ["insurance", "commercial general liability", "workers compensation",
              "professional liability", "errors and omissions", "policy limits"]),

    # ── Confidentiality / NDA ─────────────────────────────────────────────
    FieldDef("confidential_information", "nda_confidentiality",
             ["confidential information", "proprietary information", "trade secrets",
              "non-public information", "disclosed by"]),
    FieldDef("non_disclosure", "nda_confidentiality",
             ["non-disclosure", "shall not disclose", "keep confidential",
              "duty of confidentiality", "confidentiality obligations"]),
    FieldDef("confidentiality_term", "nda_confidentiality",
             ["confidentiality term", "confidentiality period", "survival of confidentiality",
              "years after termination", "obligations shall survive"]),
    FieldDef("exclusions_from_confidentiality", "nda_confidentiality",
             ["exclusions", "publicly available", "independently developed",
              "rightfully received", "required by law", "court order"]),
    FieldDef("return_of_materials", "nda_confidentiality",
             ["return of materials", "return or destroy", "destruction of confidential",
              "certify destruction", "return all copies"]),
    FieldDef("non_compete", "nda_confidentiality",
             ["non-compete", "competitive restriction", "restricted activity",
              "shall not compete", "non-competition"]),
    FieldDef("non_solicitation", "nda_confidentiality",
             ["non-solicitation", "shall not solicit", "hiring restriction",
              "employee solicitation", "customer solicitation"]),

    # ── Financial / Payment ───────────────────────────────────────────────
    FieldDef("payment_terms", "financial",
             ["payment terms", "net 30", "net 60", "due upon receipt", "invoice",
              "payment schedule", "payment due date"]),
    FieldDef("payment_frequency", "financial",
             ["payment frequency", "monthly", "quarterly", "annually", "semi-annually",
              "bi-weekly", "installment"]),
    FieldDef("late_payment", "financial",
             ["late payment", "late fee", "interest on overdue", "past due",
              "penalty for late payment", "default rate"]),
    FieldDef("base_fees", "financial",
             ["base fee", "license fee", "subscription fee", "service fee",
              "annual fee", "setup fee", "one-time fee"]),
    FieldDef("price_adjustment", "financial",
             ["price adjustment", "price increase", "annual increase", "CPI adjustment",
              "cost of living", "rate escalation"]),
    FieldDef("expense_reimbursement", "financial",
             ["expense reimbursement", "out-of-pocket expenses", "travel expenses",
              "reasonable expenses", "pre-approved expenses"]),
    FieldDef("taxes", "financial",
             ["taxes", "sales tax", "withholding tax", "tax obligations",
              "exclusive of taxes", "responsible for taxes"]),

    # ── Intellectual Property ─────────────────────────────────────────────
    FieldDef("ip_ownership", "intellectual_property",
             ["intellectual property ownership", "work product", "work for hire",
              "all rights title and interest", "IP assignment", "ownership of inventions"]),
    FieldDef("license_grant", "intellectual_property",
             ["license grant", "hereby grants", "non-exclusive license", "exclusive license",
              "sublicensable", "royalty-free license", "perpetual license"]),
    FieldDef("ip_indemnification", "intellectual_property",
             ["IP indemnification", "infringement claim", "intellectual property infringement",
              "patent infringement", "copyright infringement"]),

    # ── Employment ────────────────────────────────────────────────────────
    FieldDef("compensation", "employment",
             ["base salary", "annual salary", "compensation", "base pay",
              "remuneration", "gross salary"]),
    FieldDef("bonus", "employment",
             ["bonus", "incentive compensation", "performance bonus", "signing bonus",
              "annual bonus", "target bonus"]),
    FieldDef("benefits", "employment",
             ["benefits", "health insurance", "dental", "vision", "401k",
              "retirement plan", "paid time off", "vacation"]),
    FieldDef("equity_grant", "employment",
             ["stock options", "equity grant", "restricted stock", "vesting schedule",
              "RSU", "option pool", "exercise price"]),
    FieldDef("severance", "employment",
             ["severance", "severance payment", "severance package",
              "separation agreement", "garden leave"]),
    FieldDef("probation_period", "employment",
             ["probation period", "trial period", "introductory period",
              "probationary employment"]),
    FieldDef("invention_assignment", "employment",
             ["invention assignment", "employee inventions", "prior inventions",
              "company inventions", "assign all rights"]),

    # ── Real Estate / Lease ───────────────────────────────────────────────
    FieldDef("lease_term", "real_estate",
             ["lease term", "term of lease", "commencement date", "expiration date",
              "initial term", "lease period"]),
    FieldDef("rent_amount", "real_estate",
             ["rent", "base rent", "monthly rent", "annual rent", "rental payment",
              "rent per square foot"]),
    FieldDef("security_deposit", "real_estate",
             ["security deposit", "deposit amount", "refundable deposit",
              "return of deposit", "deposit shall be"]),
    FieldDef("maintenance_responsibility", "real_estate",
             ["maintenance", "repairs", "landlord responsibility", "tenant responsibility",
              "common area maintenance", "CAM charges"]),
    FieldDef("lease_renewal", "real_estate",
             ["lease renewal", "option to renew", "renewal option", "extend the term",
              "right of first refusal"]),
    FieldDef("permitted_use", "real_estate",
             ["permitted use", "premises shall be used", "zoning", "use restriction",
              "business purpose"]),

    # ── SaaS / Software / MSA ─────────────────────────────────────────────
    FieldDef("sla", "software_it",
             ["service level agreement", "SLA", "uptime guarantee", "availability",
              "99.9%", "service credits", "downtime"]),
    FieldDef("data_protection", "software_it",
             ["data protection", "data processing", "GDPR", "personal data",
              "data controller", "data processor", "DPA"]),
    FieldDef("scope_of_services", "software_it",
             ["scope of services", "statement of work", "SOW", "deliverables",
              "milestones", "acceptance criteria"]),
    FieldDef("support_services", "software_it",
             ["support services", "technical support", "maintenance", "bug fixes",
              "response time", "support hours", "help desk"]),
    FieldDef("data_ownership", "software_it",
             ["customer data", "data ownership", "data portability", "data export",
              "data return", "data deletion upon termination"]),
    FieldDef("license_restrictions", "software_it",
             ["license restrictions", "reverse engineering", "decompilation",
              "modification", "sublicense", "prohibited use"]),
    FieldDef("audit_rights", "software_it",
             ["audit rights", "right to audit", "inspect records", "books and records",
              "compliance audit", "third party auditor"]),

    # ── Shipping / Logistics ──────────────────────────────────────────────
    FieldDef("delivery_terms", "shipping_logistics",
             ["delivery", "FOB", "shipping terms", "delivery date", "transit",
              "risk of loss", "title transfer"]),
    FieldDef("acceptance_testing", "shipping_logistics",
             ["acceptance testing", "inspection period", "acceptance criteria",
              "rejection", "notice of defect", "cure period"]),

    # ── Healthcare ────────────────────────────────────────────────────────
    FieldDef("hipaa_compliance", "healthcare",
             ["HIPAA", "protected health information", "PHI", "business associate",
              "BAA", "privacy rule", "security rule"]),
    FieldDef("regulatory_compliance", "healthcare",
             ["FDA", "regulatory compliance", "clinical trial", "GMP",
              "good manufacturing practice", "regulatory approval"]),
]


class FieldRegistry:
    """
    Load field taxonomy and precompute field key-term embeddings.

    Mirrors legacy PrecomputedEmbeddings but runs locally with Nomic/MiniLM.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        taxonomy: list[FieldDef] | None = None,
    ):
        """
        Args:
            embed_fn: Function that embeds a list of strings → (N, dim) array.
                      e.g. embedder.embed or nomic_embedder.embed_for_clustering.
            taxonomy: Field definitions. Defaults to FIELD_TAXONOMY.
        """
        self.fields = taxonomy or FIELD_TAXONOMY
        self.embed_fn = embed_fn
        self._field_embeddings: np.ndarray | None = None  # (n_fields, dim)
        self._field_names: list[str] = [f.name for f in self.fields]
        self._field_categories: list[str] = [f.category for f in self.fields]

    def precompute_embeddings(self) -> np.ndarray:
        """
        Embed all field key_terms. Each field's key_terms are joined into a
        single string and embedded once. Returns (n_fields, dim) array.

        Legacy: PrecomputedEmbeddings.compute_field_name_embeddings() via AIFlow.
        V2: Single batch call to local embedder.
        """
        if self._field_embeddings is not None:
            return self._field_embeddings

        texts = [" ".join(f.key_terms) for f in self.fields]
        self._field_embeddings = self.embed_fn(texts)
        logger.info(
            f"Precomputed {len(self.fields)} field embeddings "
            f"({self._field_embeddings.shape[1]}-dim)"
        )
        return self._field_embeddings

    @property
    def field_names(self) -> list[str]:
        return self._field_names

    @property
    def field_categories(self) -> list[str]:
        return self._field_categories

    def score_chunks(
        self,
        chunk_embeddings: np.ndarray,
        top_k: int = 3,
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Score all chunks against all field embeddings (cosine similarity).
        Returns {field_name: [(chunk_idx, score), ...]} sorted by score desc.

        Legacy: FieldExtractor.extract_field_from_chunks() with cosine similarity.
        V2: Vectorized matrix multiply, no loops.

        Args:
            chunk_embeddings: (n_chunks, dim) normalized embeddings.
            top_k: Number of top-scoring chunks per field.
        """
        if self._field_embeddings is None:
            self.precompute_embeddings()

        # Normalize
        fe = self._field_embeddings
        fe_norm = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-9)
        ce_norm = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-9)

        # (n_fields, n_chunks) similarity matrix
        sim = fe_norm @ ce_norm.T

        result: dict[str, list[tuple[int, float]]] = {}
        for i, field_name in enumerate(self._field_names):
            scores = sim[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            result[field_name] = [
                (int(idx), float(scores[idx])) for idx in top_indices
            ]
        return result

    def build_field_summary(
        self,
        chunk_texts: list[str],
        chunk_embeddings: np.ndarray,
        top_k: int = 2,
        min_score: float = 0.25,
        max_summary_chars: int = 3000,
    ) -> str:
        """
        Build a field-aware summary for a single document.

        For each field, take the top-K scoring chunks. Deduplicate chunk indices,
        concatenate with field labels as headers. This produces a summary that
        foregrounds the document's key contractual features.

        Legacy: process_field_summary_to_aidb() → AIFlow.call_field_summarization()
                (1 LLM call per doc, expensive).
        V2: Pure vector similarity, 0 LLM calls.

        Args:
            chunk_texts: The document's chunk texts.
            chunk_embeddings: (n_chunks, dim) embeddings for these chunks.
            top_k: Chunks per field to consider.
            min_score: Minimum cosine similarity to include a chunk.
            max_summary_chars: Truncate summary to this length.
        """
        scored = self.score_chunks(chunk_embeddings, top_k=top_k)

        # Collect (field_category, field_name, chunk_idx, score) tuples
        entries: list[tuple[str, str, int, float]] = []
        for i, field_def in enumerate(self.fields):
            for chunk_idx, score in scored[field_def.name]:
                if score >= min_score:
                    entries.append((field_def.category, field_def.name, chunk_idx, score))

        # Sort by score descending, deduplicate chunk indices
        entries.sort(key=lambda x: x[3], reverse=True)
        seen_chunks: set[int] = set()
        summary_parts: list[str] = []
        total_chars = 0

        for category, field_name, chunk_idx, score in entries:
            if chunk_idx in seen_chunks:
                continue
            seen_chunks.add(chunk_idx)

            text = chunk_texts[chunk_idx].strip()
            label = field_name.replace("_", " ").title()
            part = f"[{label}] {text}"

            if total_chars + len(part) > max_summary_chars:
                break
            summary_parts.append(part)
            total_chars += len(part)

        return "\n\n".join(summary_parts)

    def get_field_profile(
        self,
        chunk_embeddings: np.ndarray,
        top_k: int = 2,
        min_score: float = 0.25,
    ) -> dict[str, float]:
        """
        Return a category → max_score profile for a document.
        Useful for understanding what domain a document belongs to.

        E.g.: {"employment": 0.72, "financial": 0.65, "nda_confidentiality": 0.58, ...}
        """
        scored = self.score_chunks(chunk_embeddings, top_k=top_k)
        category_scores: dict[str, float] = {}

        for field_def in self.fields:
            top_score = max((s for _, s in scored[field_def.name]), default=0.0)
            if top_score >= min_score:
                cat = field_def.category
                category_scores[cat] = max(category_scores.get(cat, 0.0), top_score)

        return dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def from_legacy_configs(
        cls,
        embed_fn,
        legacy_dir: str | None = None,
    ) -> "FieldRegistry":
        """
        Create a FieldRegistry populated from the apr-fields-processor data directory.

        Loads 221 predefined + 500+ business fields and converts them to FieldDef
        objects, giving the registry 720+ fields instead of the default 55.

        Args:
            embed_fn: Embedding function (list[str] → ndarray).
            legacy_dir: Path to apr-fields-processor root. Auto-detected if None.
        """
        from core.field_features import load_predefined_fields, load_business_fields, \
            _DEFAULT_LEGACY_DIR, _PREDEFINED_DIR, _BUSINESS_FILE
        from pathlib import Path

        root = Path(legacy_dir) if legacy_dir else _DEFAULT_LEGACY_DIR
        predefined = load_predefined_fields(root / _PREDEFINED_DIR)
        business = load_business_fields(root / _BUSINESS_FILE)

        taxonomy = []
        seen = set()
        for rc in predefined + business:
            if rc.name in seen:
                continue
            seen.add(rc.name)
            taxonomy.append(FieldDef(
                name=rc.name,
                category=rc.taxonomy_category,
                key_terms=rc.key_terms[:10],
                extraction_type=rc.extraction_type,
                regex_patterns=rc.regex_patterns[:5],
                examples=[ex[:200] for ex in rc.examples[:3]],
                negative_examples=[ne[:200] for ne in rc.negative_examples[:3]],
            ))

        logger.info(f"FieldRegistry.from_legacy_configs: {len(taxonomy)} fields "
                     f"({len(predefined)} predefined, {len(business)} business)")
        return cls(embed_fn=embed_fn, taxonomy=taxonomy)
