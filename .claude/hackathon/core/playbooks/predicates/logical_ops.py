from . import predicate


@predicate("and")
def op_and(args, ctx, ev):
    return all(ev(a, ctx) for a in args)


@predicate("or")
def op_or(args, ctx, ev):
    return any(ev(a, ctx) for a in args)


@predicate("not")
def op_not(args, ctx, ev):
    return not ev(args[0], ctx)


@predicate("if_then")
def op_if_then(args, ctx, ev):
    return (not ev(args[0], ctx)) or ev(args[1], ctx)


@predicate("any_of")
def op_any_of(args, ctx, ev):
    return any(ev(a, ctx) for a in args)


@predicate("all_of")
def op_all_of(args, ctx, ev):
    return all(ev(a, ctx) for a in args)


@predicate("count")
def op_count(args, ctx, ev):
    """args = [sub_predicate, op, n]   e.g. [..., '>=', 2]"""
    sub, cmp_op, n = args
    items = ctx.get("clauses", [])
    hits = sum(1 for it in items if ev(sub, {**ctx, "clauses": [it]}))
    return {">=": hits >= n, ">": hits > n, "<=": hits <= n,
            "<": hits < n, "==": hits == n}[cmp_op]
