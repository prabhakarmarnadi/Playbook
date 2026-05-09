import re
from . import predicate


def _val(args, ctx):
    name = args[0]
    return ctx.get("fields", {}).get(name)


@predicate("field.exists")
def op_exists(args, ctx, _ev):
    return _val(args, ctx) is not None


@predicate("field.eq")
def op_eq(args, ctx, _ev):
    return _val(args, ctx) == args[1]


@predicate("field.in")
def op_in(args, ctx, _ev):
    return _val(args, ctx) in args[1]


@predicate("field.gte")
def op_gte(args, ctx, _ev):
    v = _val(args, ctx)
    try:
        return v is not None and float(v) >= float(args[1])
    except (TypeError, ValueError):
        return False


@predicate("field.lte")
def op_lte(args, ctx, _ev):
    v = _val(args, ctx)
    try:
        return v is not None and float(v) <= float(args[1])
    except (TypeError, ValueError):
        return False


@predicate("field.between")
def op_between(args, ctx, _ev):
    v = _val(args, ctx)
    try:
        return v is not None and float(args[1]) <= float(v) <= float(args[2])
    except (TypeError, ValueError):
        return False


@predicate("field.matches")
def op_matches(args, ctx, _ev):
    """args = [field_name, "regex"|"glob", pattern]"""
    v = _val(args, ctx)
    if v is None:
        return False
    kind, pattern = args[1], args[2]
    if kind == "regex":
        return re.search(pattern, str(v)) is not None
    if kind == "glob":
        import fnmatch
        return fnmatch.fnmatch(str(v), pattern)
    raise ValueError(f"unknown match kind: {kind}")


@predicate("field.value")
def op_value(args, ctx, _ev):
    return _val(args, ctx)


@predicate("field.semantic_match")
def op_sem(args, ctx, _ev):
    """args = [field_name, target_text, threshold]
    Requires ctx['embed'] callable (text -> vector). Returns False if absent."""
    v = _val(args, ctx)
    embed = ctx.get("embed")
    if v is None or embed is None:
        return False
    import numpy as np
    a = np.asarray(embed(str(v)))
    b = np.asarray(embed(args[1]))
    if a.size == 0 or b.size == 0:
        return False
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return cos >= float(args[2])
