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


CHECKS = [
    ("package_importable", test_package_importable),
    ("store_schema_idempotent", test_store_schema_idempotent),
    ("store_crud_roundtrip",    test_store_crud_roundtrip),
    ("models_roundtrip",        test_models_roundtrip),
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
