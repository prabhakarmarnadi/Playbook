"""
Composite field engine — NEW V2 capability (doesn't exist in V1).

Implements a safe DSL for computing derived values from extracted fields.
Uses Python AST parsing to support arithmetic expressions with:
  - Field references (variable names)
  - Constants (numbers)
  - Operations: +, -, *, /
  - DAG-based evaluation order (composites can reference other composites)
  - Cycle detection to prevent infinite loops

Example: TCV = unit_price * quantity * term_years
         cost_per_month = TCV / (term_years * 12)

When user corrects unit_price, both TCV and cost_per_month cascade automatically
with zero LLM calls.
"""
import ast
import logging
import operator
from collections import defaultdict

logger = logging.getLogger(__name__)

# Only allow safe arithmetic operations
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def parse_and_eval(expression: str, field_values: dict[str, float | int | None]) -> float | None:
    """
    Safely evaluate an arithmetic expression with field value substitution.

    Args:
        expression: e.g., "unit_price * quantity * term_years"
        field_values: e.g., {"unit_price": 50000, "quantity": 100, "term_years": 3}

    Returns:
        Computed value, or None if any referenced field is None.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body, field_values)
    except Exception as e:
        logger.warning(f"Failed to evaluate '{expression}': {e}")
        return None


def _eval_node(node: ast.AST, values: dict) -> float | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    if isinstance(node, ast.Name):
        val = values.get(node.id)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if left is None or right is None:
            return None
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_node(node.operand, values)
        return -operand if operand is not None else None

    # Parenthesized expressions are just nested nodes — already handled
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def get_dependencies(expression: str) -> list[str]:
    """Extract all variable names referenced in an expression."""
    try:
        tree = ast.parse(expression, mode="eval")
        return sorted(set(
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        ))
    except SyntaxError:
        return []


def detect_cycles(composites: dict[str, str]) -> list[str]:
    """
    Check for circular dependencies in composite definitions.
    composites = {name: expression}
    Returns list of error messages for any cycles found.
    """
    deps = {name: get_dependencies(expr) for name, expr in composites.items()}
    visited: set[str] = set()
    in_stack: set[str] = set()
    errors: list[str] = []

    def dfs(node: str):
        if node in in_stack:
            errors.append(f"Cycle detected involving '{node}'")
            return
        if node in visited:
            return
        in_stack.add(node)
        for dep in deps.get(node, []):
            if dep in composites:
                dfs(dep)
        in_stack.discard(node)
        visited.add(node)

    for name in composites:
        dfs(name)
    return errors


def topological_order(composites: dict[str, str]) -> list[str]:
    """Return composite names in evaluation order (dependencies first)."""
    deps = {name: [d for d in get_dependencies(expr) if d in composites]
            for name, expr in composites.items()}
    visited: set[str] = set()
    order: list[str] = []

    def dfs(node: str):
        if node in visited:
            return
        visited.add(node)
        for dep in deps.get(node, []):
            dfs(dep)
        order.append(node)

    for name in composites:
        dfs(name)
    return order


def evaluate_all_composites(
    composites: dict[str, str],
    base_values: dict[str, float | int | None],
) -> dict[str, float | None]:
    """
    Evaluate all composites in topological order, propagating intermediate results.

    Args:
        composites: {composite_name: expression}
        base_values: {field_name: value} for base (non-composite) fields

    Returns:
        {composite_name: computed_value}
    """
    errors = detect_cycles(composites)
    if errors:
        logger.warning(f"Composite cycle errors: {errors}")
        return {name: None for name in composites}

    eval_order = topological_order(composites)
    all_values = dict(base_values)
    results = {}

    for name in eval_order:
        expr = composites[name]
        value = parse_and_eval(expr, all_values)
        results[name] = value
        all_values[name] = value  # Available for downstream composites

    return results


def compute_cascade(
    composites: dict[str, str],
    base_values: dict[str, float | int | None],
    changed_field: str,
    old_value: float | None,
    new_value: float | None,
) -> list[dict]:
    """
    Compute the cascade effect of changing one base field value.
    Returns list of {name, old_value, new_value, expression} for each affected composite.
    Used for the cascade animation in the UI (Act 5).
    """
    # Compute with old values
    old_base = dict(base_values)
    old_base[changed_field] = old_value
    old_results = evaluate_all_composites(composites, old_base)

    # Compute with new values
    new_base = dict(base_values)
    new_base[changed_field] = new_value
    new_results = evaluate_all_composites(composites, new_base)

    # Find affected composites
    cascade = []
    eval_order = topological_order(composites)
    for name in eval_order:
        deps = get_dependencies(composites[name])
        # Check if this composite depends (directly or indirectly) on the changed field
        is_affected = changed_field in deps or any(
            c in deps for c in [n for n in cascade]
        )
        old_val = old_results.get(name)
        new_val = new_results.get(name)
        if is_affected or old_val != new_val:
            cascade.append({
                "name": name,
                "expression": composites[name],
                "old_value": old_val,
                "new_value": new_val,
            })

    return cascade
