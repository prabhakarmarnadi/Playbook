"""
Validation script for the playbook subsystem (mirror of validate_pipeline.py).
Each test_* function asserts and prints [PASS]; main() runs them and reports.

Usage:  poetry run python scripts/validate_playbook.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_package_importable():
    """Validate: core.playbooks package exists and imports cleanly."""
    import core.playbooks  # noqa: F401
    print("  [PASS] Package import")


def test_store_schema_idempotent():
    """Validate: PlaybookStore creates 5 tables and is safe to re-init."""
    from core.playbooks.store import PlaybookStore
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s1 = PlaybookStore(tf.name)
        s2 = PlaybookStore(tf.name)  # second init must not fail
        tables = {r[0] for r in s1.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        for required in ("playbooks", "playbook_rules", "rule_bindings",
                          "rule_revisions", "rule_evaluations"):
            assert required in tables, f"missing table: {required}"
        s1.close(); s2.close()
    print("  [PASS] Schema idempotent (5 tables)")


def test_store_crud_roundtrip():
    """Validate: insert + fetch a playbook and a rule round-trips fields."""
    from core.playbooks.store import PlaybookStore
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="Demo", owner_org="Acme")
        rid = s.create_rule(playbook_id=pid, title="LoL cap",
                             applies_to="field", severity="warn",
                             predicate={"op": "field.gte", "args": ["cap", 250000]})
        rules = s.list_rules(pid)
        assert len(rules) == 1 and rules[0]["rule_id"] == rid
        assert rules[0]["predicate"]["op"] == "field.gte"
        s.close()
    print("  [PASS] CRUD round-trip")


def test_models_roundtrip():
    """Validate: Pydantic models round-trip a Rule with predicate JSON."""
    from core.playbooks.models import Rule, Severity, AppliesTo
    r = Rule(
        rule_id="r1", playbook_id="p1", title="cap",
        applies_to=AppliesTo.field, severity=Severity.warn,
        predicate={"op": "field.gte", "args": ["cap", 250000]},
    )
    j = r.model_dump_json()
    r2 = Rule.model_validate_json(j)
    assert r2.predicate == r.predicate
    assert r2.severity == Severity.warn
    print("  [PASS] Pydantic round-trip")


def test_predicate_dsl_examples():
    """Validate: each spec §5 example evaluates correctly."""
    from core.playbooks.predicates.evaluator import evaluate

    # Walmart-style OR
    ctx = {"fields": {"Indemnification_Standard_SP": "yes"}}
    p = {"op": "or", "args": [
        {"op": "field.exists", "args": ["Indemnification_Standard_SP"]},
        {"op": "field.exists", "args": ["Indemnification_Standard_SS"]},
    ]}
    assert evaluate(p, ctx) is True

    # Cloudera-style regex
    ctx = {"fields": {"late_payment_penalty": "1.5%"}}
    p = {"op": "field.matches",
         "args": ["late_payment_penalty", "regex", r"\d+(\.\d+)?%"]}
    assert evaluate(p, ctx) is True

    # AI Playbook AND/OR with numeric gte
    ctx = {"fields": {"liability_cap_amount": 300000, "liability_cap_ratio_fees": 0.5}}
    p = {"op": "and", "args": [
        {"op": "field.exists", "args": ["liability_cap_amount"]},
        {"op": "or", "args": [
            {"op": "field.gte", "args": ["liability_cap_amount", 250000]},
            {"op": "field.gte", "args": ["liability_cap_ratio_fees", 1.0]},
        ]},
    ]}
    assert evaluate(p, ctx) is True

    # if_then with violated consequent
    ctx = {"fields": {"governing_law": "New York", "arbitration_venue": "Delaware"}}
    p = {"op": "if_then", "args": [
        {"op": "field.eq", "args": ["governing_law", "New York"]},
        {"op": "field.eq", "args": ["arbitration_venue", "New York"]},
    ]}
    assert evaluate(p, ctx) is False

    # any_of over clauses
    ctx = {"clauses": [{"label": "Indemnification"}, {"label": "Limitation of Liability"}]}
    p = {"op": "any_of", "args": [
        {"op": "clause.classified_as", "args": ["Indemnification"]},
    ]}
    assert evaluate(p, ctx) is True

    print("  [PASS] DSL examples (5 sub-cases)")


FIX = Path(__file__).parent.parent / "tests_fixtures" / "playbooks"


def test_walmart_logic_parser():
    """Validate: 'IF X_SP IS SET --> Answer = Standard' parses to predicate JSON."""
    from core.playbooks.importers.walmart_logic import parse
    src = "1) IF X_Std_SP IS SET --> Answer = Standard\n2) IF X_NonStd_SP IS SET --> Answer = Non-Standard"
    rules = parse(src)
    assert len(rules) >= 2
    std = next(r for r in rules if r["answer"] == "Standard")
    assert std["predicate"]["op"] == "field.exists"
    assert std["predicate"]["args"] == ["X_Std_SP"]
    print("  [PASS] Walmart logic parser")


def test_walmart_logic_paren_precedence():
    """Validate: (A OR B) AND C parses as and(or(A,B), C), not or(A, and(B,C))."""
    from core.playbooks.importers.walmart_logic import parse
    rules = parse("1) IF (A_SP OR B_SS) AND C_SP IS SET --> Answer = Standard")
    assert len(rules) == 1
    pred = rules[0]["predicate"]
    assert pred["op"] == "and", f"top-level op should be 'and', got {pred['op']}"
    assert len(pred["args"]) == 2
    inner_or = pred["args"][0]
    assert inner_or["op"] == "or" and {a["args"][0] for a in inner_or["args"]} == {"A_SP", "B_SS"}
    assert pred["args"][1]["op"] == "field.exists" and pred["args"][1]["args"] == ["C_SP"]
    print("  [PASS] Walmart logic paren precedence")


def test_tabular_importer_cloudera():
    """Validate: tabular importer ingests Cloudera xlsx and produces ≥10 rules."""
    if not (FIX / "cloudera.xlsx").exists():
        print("  [SKIP] cloudera fixture missing"); return
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.importers.tabular import import_xlsx
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = import_xlsx(s, str(FIX / "cloudera.xlsx"), name="Cloudera")
        rules = s.list_rules(pid)
        assert len(rules) >= 10, f"too few rules: {len(rules)}"
        # round-trip provenance
        prov = rules[0]["source_provenance"]
        assert prov and "row" in prov
        s.close()
    print(f"  [PASS] Cloudera importer ({len(rules)} rules)")


def test_narrative_importer_docusign():
    """Validate: narrative importer ingests Docusign MSA sheet without crashing."""
    src = FIX / "docusign.xlsx"
    if not src.exists():
        print("  [SKIP] docusign fixture missing"); return
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.importers.narrative import import_xlsx
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = import_xlsx(s, str(src), name="Docusign")
        rules = s.list_rules(pid)
        assert len(rules) >= 5, f"too few rules: {len(rules)}"
        # at least one with escalation_owner set
        assert any(r.get("escalation_owner") for r in rules)
        s.close()
    print(f"  [PASS] Docusign narrative importer ({len(rules)} rules)")


def test_desirable_importer_ai_playbook():
    """Validate: desirable/undesirable docx importer extracts rule + ref + walkaway."""
    src = FIX / "sample_ai.docx"
    if not src.exists():
        print("  [SKIP] sample_ai fixture missing"); return
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.importers.desirable import import_docx
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = import_docx(s, str(src), name="Sample AI")
        rules = s.list_rules(pid)
        assert len(rules) >= 3
        assert any(r.get("reference_text") for r in rules), \
            "expected at least one rule with reference_text from Desirable column"
        # Note: the Sample AI Playbook fixture's Undesirable column is empty across all rows,
        # so walkaway_language is intentionally None. Importer correctly preserves None.
        assert all(r.get("walkaway_language") is None for r in rules), \
            "fixture's Undesirable column is empty; walkaway_language should be None"
        s.close()
    print(f"  [PASS] AI Playbook docx importer ({len(rules)} rules)")


def test_miner_proposes_candidates():
    """Validate: miner emits ≥1 coverage and ≥1 distribution candidate from a fake corpus."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.miner import mine_candidates

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        # synthetic corpus: 20 MSAs, 19 of which contain the "Indemnification" cluster
        corpus = {
            "domain_clusters": {
                "msa": {"Indemnification": 19, "Other": 5},
            },
            "field_values": {
                "msa::Limitation of Liability::cap_amount":
                    [250_000] * 18 + [None, 500_000],
            },
        }
        pid = s.create_playbook(name="mined")
        cands = mine_candidates(s, pid, corpus, n_total_per_domain=20)
        coverage = [c for c in cands if c["kind"] == "coverage"]
        distribution = [c for c in cands if c["kind"] == "distribution"]
        assert coverage, "no coverage candidates"
        assert distribution, "no distribution candidates"
        # all should be inserted as draft rules
        rules = s.list_rules(pid)
        assert all(r["status"] == "draft" for r in rules)
        s.close()
    print(f"  [PASS] Miner produced {len(cands)} candidates")


def test_aligner_predicate_only():
    """Validate: aligner runs a predicate-only rule and writes a Finding."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(
            playbook_id=pid, title="cap >= 250k",
            applies_to="field", severity="warn",
            predicate={"op": "field.gte", "args": ["cap", 250000]},
            status="active",
        )
        ctx = {"agreement_id": "a1", "fields": {"cap": 100000}}
        findings = align(s, pid, ctx)
        assert len(findings) == 1
        assert findings[0]["outcome"] == "fail"
        # second run identical → determinism
        findings2 = align(s, pid, ctx)
        assert findings2[0]["outcome"] == findings[0]["outcome"]
        s.close()
    print("  [PASS] Aligner predicate-only + determinism")


def test_aligner_combiner_severity():
    """Validate: when predicate fails AND nl_judge stub passes, outcome is fail (combiner rule)."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(
            playbook_id=pid, title="combo",
            applies_to="field", severity="approval_required",
            predicate={"op": "field.gte", "args": ["cap", 999999]},  # fails
            nl_assertion="liability cap is reasonable",  # stub will say pass
            status="active",
        )
        ctx = {"agreement_id": "a1", "fields": {"cap": 100000},
                "nl_judge": lambda assertion, evidence: ("pass", "stubbed")}
        findings = align(s, pid, ctx)
        assert findings[0]["outcome"] == "fail"  # predicate fail dominates
        assert findings[0]["severity"] == "approval_required"
        s.close()
    print("  [PASS] Aligner combiner")


def test_aligner_isolates_callable_exceptions():
    """Validate: a crashing embed/nl_judge does not kill align(); rule degrades to needs_human."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    def boom_embed(_text):
        raise RuntimeError("embed crashed")

    def boom_judge(_assertion, _evidence):
        raise RuntimeError("judge crashed")

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="x")
        # Rule that exercises BOTH similarity (with reference_text) AND nl_judge.
        # No predicate → NL gate is open, so both evaluators are invoked and BOTH crash.
        s.create_rule(playbook_id=pid, title="brittle", applies_to="cluster",
                       severity="warn",
                       reference_text="standard text",
                       nl_assertion="assertion",
                       similarity_threshold=0.15,
                       status="active")
        ctx = {
            "agreement_id": "a1",
            "clauses": [{"id": "c1", "text": "some clause text"}],
            "embed": boom_embed,
            "nl_judge": boom_judge,
        }
        findings = align(s, pid, ctx)
        assert len(findings) == 1
        # Both evaluators returned None → no outcomes → combiner returns needs_human
        assert findings[0]["outcome"] == "needs_human", \
            f"expected needs_human, got {findings[0]['outcome']}"
        s.close()
    print("  [PASS] Aligner isolates callable exceptions")


def test_aligner_applies_to_domain_dispatch():
    """Rule with applies_to=domain bound to MSA only fires when ctx.domain.id matches."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(playbook_id=pid, title="msa-must-have-indem",
                             applies_to="domain", severity="warn",
                             predicate={"op": "any_of", "args": [
                                 {"op": "clause.classified_as", "args": ["Indemnification"]}
                             ]},
                             status="active")
        s.add_binding(rule_id=rid, entity_kind="domain", entity_id="dom_msa",
                       label_text="MSA", confidence=1.0)

        # Match: agreement is in MSA domain → predicate evaluates → fail (no Indemnification clause)
        ctx_match = {"agreement_id": "a1", "domain": {"id": "dom_msa", "name": "MSA"},
                      "clauses": [{"label": "Other"}]}
        findings = align(s, pid, ctx_match)
        assert findings[0]["outcome"] == "fail", f"expected fail, got {findings[0]['outcome']}"

        # Mismatch: agreement is in NDA domain → rule scoped out → n/a
        ctx_other = {"agreement_id": "a2", "domain": {"id": "dom_nda", "name": "NDA"},
                      "clauses": [{"label": "Other"}]}
        findings = align(s, pid, ctx_other)
        assert findings[0]["outcome"] == "n/a", f"expected n/a, got {findings[0]['outcome']}"
        s.close()
    print("  [PASS] Aligner applies_to=domain dispatch")


def test_aligner_applies_to_cluster_filters_clauses():
    """Rule with applies_to=cluster bound to cluster_X only sees clauses with that cluster_id."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(playbook_id=pid, title="indem-clause-present",
                             applies_to="cluster", severity="warn",
                             predicate={"op": "any_of", "args": [
                                 {"op": "clause.classified_as", "args": ["Indemnification"]}
                             ]},
                             status="active")
        s.add_binding(rule_id=rid, entity_kind="cluster", entity_id="clu_indem",
                       label_text="Indemnification", confidence=1.0)

        ctx = {"agreement_id": "a1",
                "clauses": [
                    {"cluster_id": "clu_indem", "label": "Indemnification", "text": "..."},
                    {"cluster_id": "clu_other", "label": "Termination",   "text": "..."},
                ]}
        findings = align(s, pid, ctx)
        # Only the Indemnification clause is in scope → predicate passes
        assert findings[0]["outcome"] == "pass", f"expected pass, got {findings[0]['outcome']}"
        s.close()
    print("  [PASS] Aligner applies_to=cluster filters clauses")


def test_aligner_unresolved_binding_is_na():
    """Rule with only label:* bindings (unresolved soft binding) reports n/a."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(playbook_id=pid, title="unresolved",
                             applies_to="cluster", severity="warn",
                             predicate={"op": "field.gte", "args": ["x", 0]},
                             status="active")
        s.add_binding(rule_id=rid, entity_kind="cluster", entity_id="label:Indemnification",
                       label_text="Indemnification", confidence=0.5)
        ctx = {"agreement_id": "a1", "fields": {"x": 999}, "clauses": []}
        findings = align(s, pid, ctx)
        assert findings[0]["outcome"] == "n/a", f"expected n/a, got {findings[0]['outcome']}"
        s.close()
    print("  [PASS] Aligner unresolved binding is n/a")


def test_ui_modules_import():
    """Validate: UI modules import without errors (manual visual check still required)."""
    import importlib
    for mod in ("ui.playbook_authoring_page",
                "ui.playbook_mining_page",
                "ui.playbook_findings_page"):
        importlib.import_module(mod)
    print("  [PASS] UI modules importable")


def test_benchmark_migration():
    """Validate: migration script lifts existing playbook_benchmarks rows into a playbook."""
    import os
    import tempfile
    import duckdb
    # Use mkstemp then unlink so duckdb can create the file fresh (NamedTemporaryFile
    # pre-creates an empty file that duckdb rejects as "not a valid DuckDB database file").
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(db_path)
    try:
        # seed legacy benchmark
        c = duckdb.connect(db_path)
        c.execute("""CREATE TABLE IF NOT EXISTS playbook_benchmarks (
            cluster_id VARCHAR, cluster_label VARCHAR, benchmark_clause_id VARCHAR,
            benchmark_text TEXT, source VARCHAR, mean_deviation DOUBLE,
            std_deviation DOUBLE, n_clauses INTEGER)""")
        c.execute("""INSERT INTO playbook_benchmarks VALUES
                     ('c1','Indemnification','b1','standard text','centroid',0.1,0.05,42)""")
        c.close()

        from scripts.migrate_benchmarks_to_playbook import migrate
        pid = migrate(db_path, name="Auto-mined baseline")

        from core.playbooks.store import PlaybookStore
        s = PlaybookStore(db_path)
        rules = s.list_rules(pid)
        assert len(rules) == 1
        assert rules[0]["reference_text"] == "standard text"
        s.close()
    finally:
        for ext in ("", ".wal"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)
    print("  [PASS] Benchmark migration")


def test_agreement_ctx_recovers_field_names():
    """integration helper joins extractions × field_definitions to expose human field names."""
    import os
    import tempfile
    from core.store import ClusteringStore
    from core.playbooks.integration import agreement_ctx

    # Use mkstemp+unlink so DuckDB can create the file fresh (NamedTemporaryFile
    # pre-creates an empty file that DuckDB rejects as "not a valid DuckDB database file").
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(db_path)
    try:
        cs = ClusteringStore(db_path)
        # Seed minimal fixture: domain → cluster → field → extraction → clause
        cs.conn.execute("INSERT INTO domains (domain_id, label) VALUES ('d1', 'MSA')")
        cs.conn.execute(
            "INSERT INTO agreements (agreement_id, filename, domain_id) VALUES ('a1', '/x', 'd1')"
        )
        cs.conn.execute(
            "INSERT INTO clusters (cluster_id, domain_id, label) VALUES ('c1', 'd1', 'Indemnification')"
        )
        cs.conn.execute(
            "INSERT INTO field_definitions (field_id, cluster_id, name, field_type) VALUES (?, ?, ?, ?)",
            ['f1', 'c1', 'liability_cap_amount', 'number']
        )
        cs.conn.execute(
            "INSERT INTO extractions (extraction_id, agreement_id, field_id, value) VALUES (?, ?, ?, ?)",
            ['e1', 'a1', 'f1', '250000']
        )
        cs.conn.execute(
            "INSERT INTO clauses (clause_id, agreement_id, clause_type_id, full_text) VALUES (?, ?, ?, ?)",
            ['cl1', 'a1', 'c1', 'Indemnification clause text...']
        )

        ctx = agreement_ctx(cs, 'a1')
        assert ctx["agreement_id"] == "a1"
        assert ctx["domain"] == {"id": "d1", "name": "MSA"}
        assert "liability_cap_amount" in ctx["fields"], f"got {list(ctx['fields'].keys())}"
        assert ctx["fields"]["liability_cap_amount"] == "250000"
        assert len(ctx["clauses"]) == 1
        assert ctx["clauses"][0]["cluster_id"] == "c1"
        if hasattr(cs, "close"):
            cs.close()
    finally:
        for ext in ("", ".wal"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)
    print("  [PASS] agreement_ctx recovers field names")


CHECKS = [
    ("package_importable",        test_package_importable),
    ("store_schema_idempotent",   test_store_schema_idempotent),
    ("store_crud_roundtrip",      test_store_crud_roundtrip),
    ("models_roundtrip",          test_models_roundtrip),
    ("predicate_dsl_examples",    test_predicate_dsl_examples),
    ("walmart_logic_parser",           test_walmart_logic_parser),
    ("walmart_logic_paren_precedence", test_walmart_logic_paren_precedence),
    ("tabular_importer_cloudera",      test_tabular_importer_cloudera),
    ("narrative_importer_docusign",    test_narrative_importer_docusign),
    ("desirable_importer_ai_playbook", test_desirable_importer_ai_playbook),
    ("miner_proposes_candidates",      test_miner_proposes_candidates),
    ("aligner_predicate_only",         test_aligner_predicate_only),
    ("aligner_combiner_severity",      test_aligner_combiner_severity),
    ("aligner_isolates_callable_exceptions", test_aligner_isolates_callable_exceptions),
    ("aligner_applies_to_domain_dispatch",   test_aligner_applies_to_domain_dispatch),
    ("aligner_applies_to_cluster_filters_clauses", test_aligner_applies_to_cluster_filters_clauses),
    ("aligner_unresolved_binding_is_na",     test_aligner_unresolved_binding_is_na),
    ("ui_modules_import",              test_ui_modules_import),
    ("benchmark_migration",            test_benchmark_migration),
    ("agreement_ctx_recovers_field_names", test_agreement_ctx_recovers_field_names),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run only checks whose name contains this substring")
    args = parser.parse_args()

    failed = 0
    for name, fn in CHECKS:
        if args.only and args.only not in name:
            continue
        print(f"\n— {name} —")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}\n{len(CHECKS)-failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
