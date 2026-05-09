"""Playbook importers. Auto-dispatch by file extension + header detection."""
from pathlib import Path
from ..store import PlaybookStore


def import_file(store: PlaybookStore, path: str, **kwargs) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".xlsx":
        from . import tabular
        # try tabular first; if it produces 0 rules, retry as narrative
        pid = tabular.import_xlsx(store, path, **kwargs)
        rules = store.list_rules(pid)
        if rules:
            return pid
        # narrative fallback (Docusign) — imported lazily so Tasks 5 works before Task 6 lands
        from . import narrative  # noqa: PLC0415
        return narrative.import_xlsx(store, path, **kwargs)
    if ext == ".docx":
        from . import desirable  # noqa: PLC0415
        return desirable.import_docx(store, path, **kwargs)
    raise ValueError(f"unsupported playbook file extension: {ext}")
