from . import predicate


@predicate("clause.classified_as")
def op_classified_as(args, ctx, _ev):
    target = args[0]
    for c in ctx.get("clauses", []):
        if c.get("label") == target:
            return True
    return False


@predicate("clause.contains")
def op_contains(args, ctx, _ev):
    needle = args[0].lower()
    for c in ctx.get("clauses", []):
        if needle in (c.get("text") or "").lower():
            return True
    return False


@predicate("clause.absent")
def op_absent(args, ctx, _ev):
    target = args[0]
    return not any(c.get("label") == target for c in ctx.get("clauses", []))


@predicate("clause.similar_to")
def op_similar(args, ctx, _ev):
    """args = [reference_text, threshold]. Uses ctx['embed']."""
    embed = ctx.get("embed")
    if embed is None:
        return False
    import numpy as np
    ref = np.asarray(embed(args[0]))
    thr = float(args[1])
    for c in ctx.get("clauses", []):
        text = c.get("text")
        if not text:
            continue
        v = np.asarray(embed(text))
        if v.size == 0 or ref.size == 0:
            continue
        cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
        if cos >= thr:
            return True
    return False
