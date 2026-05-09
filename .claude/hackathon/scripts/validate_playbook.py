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


CHECKS = [
    ("package_importable", test_package_importable),
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
