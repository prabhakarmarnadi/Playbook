from . import predicate


@predicate("domain.is")
def op_domain_is(args, ctx, _ev):
    return (ctx.get("domain") or {}).get("name") == args[0]


@predicate("doc.party_role.is")
def op_role_is(args, ctx, _ev):
    return (ctx.get("doc") or {}).get("party_role") == args[0]


@predicate("doc.value.gte")
def op_value_gte(args, ctx, _ev):
    v = (ctx.get("doc") or {}).get("value")
    try:
        return v is not None and float(v) >= float(args[0])
    except (TypeError, ValueError):
        return False
