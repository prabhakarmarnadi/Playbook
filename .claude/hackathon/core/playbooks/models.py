"""Pydantic v2 mirrors of playbook DuckDB tables."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    approval_required = "approval_required"
    blocker = "blocker"


class AppliesTo(str, Enum):
    document = "document"
    domain = "domain"
    cluster = "cluster"
    field = "field"
    composite = "composite"
    cross_field = "cross_field"


class RuleStatus(str, Enum):
    draft = "draft"
    active = "active"
    retired = "retired"


class Outcome(str, Enum):
    pass_ = "pass"
    fail = "fail"
    na = "n/a"
    needs_human = "needs_human"


class Playbook(BaseModel):
    playbook_id: str
    name: str
    owner_org: Optional[str] = None
    description: Optional[str] = None
    source_file: Optional[str] = None
    status: str = "draft"
    version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Rule(BaseModel):
    rule_id: str
    playbook_id: str
    title: str
    description: Optional[str] = None
    applies_to: AppliesTo
    predicate: Optional[dict[str, Any]] = None
    nl_assertion: Optional[str] = None
    reference_text: Optional[str] = None
    similarity_threshold: Optional[float] = None
    answer_type: Optional[str] = None
    answer_options: Optional[list[str]] = None
    severity: Severity
    escalation_owner: Optional[str] = None
    preferred_language: Optional[str] = None
    fallback_language: Optional[list[str]] = None
    walkaway_language: Optional[str] = None
    rationale: Optional[str] = None
    tags: Optional[list[str]] = None
    source_provenance: Optional[dict[str, Any]] = None
    status: RuleStatus = RuleStatus.draft


class Binding(BaseModel):
    binding_id: str
    rule_id: str
    entity_kind: str   # domain | cluster | field | composite
    entity_id: str
    label_text: Optional[str] = None
    confidence: float = 1.0


class Finding(BaseModel):
    eval_id: str
    rule_id: str
    agreement_id: str
    run_id: str
    outcome: Outcome
    severity: Severity
    deviation: float = 0.0
    answer_value: Optional[dict[str, Any]] = None
    evidence: Optional[list[dict[str, Any]]] = None
    rationale: Optional[str] = None
    evaluator_used: Optional[dict[str, bool]] = None
