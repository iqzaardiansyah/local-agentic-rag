"""
AST-safe expression evaluation for csv_query.

No raw eval/exec. Only a closed set of pandas DataFrame/Series operations
and comparison/filter syntax is allowed. Free / local.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# Binary / unary operators allowed in expressions.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

# Attribute methods allowed on DataFrame / Series / Index objects.
_ALLOWED_METHODS: Set[str] = {
    # selection / shape
    "head", "tail", "sample", "iloc", "loc", "at", "iat",
    "filter", "drop", "dropna", "drop_duplicates", "fillna", "ffill", "bfill",
    "isin", "between", "where", "mask", "query",
    # aggregations
    "mean", "sum", "count", "min", "max", "median", "std", "var",
    "nunique", "unique", "value_counts", "nlargest", "nsmallest",
    "idxmin", "idxmax", "quantile", "describe",
    "abs", "round", "cumsum", "cummax", "cummin", "cumprod",
    "any", "all",
    # grouping / reshape
    "groupby", "agg", "aggregate", "pivot_table", "crosstab",
    "sort_values", "sort_index", "reset_index", "set_index",
    "rename", "melt", "explode",
    # string helpers (via .str)
    "str", "dt", "cat",
    # misc
    "astype", "apply", "map", "replace", "clip", "rank",
    "isnull", "notnull", "isna", "notna",
    "to_string", "to_list", "tolist", "item", "get",
    "sum", "mean",
}

# Attributes allowed as data (not callables) on DataFrame-like objects.
_ALLOWED_ATTRS: Set[str] = {
    "columns", "index", "shape", "size", "dtypes", "values", "T",
    "empty", "ndim",
}

_FORBIDDEN_NAMES: Set[str] = {
    "eval", "exec", "compile", "open", "input", "print", "breakpoint",
    "getattr", "setattr", "delattr", "hasattr", "globals", "locals",
    "vars", "dir", "type", "object", "super", "classmethod", "staticmethod",
    "property", "vars", "memoryview", "bytearray", "bytes",
    "import", "importlib", "reload", "exit", "quit",
    "__import__", "eval", "exec",
}


class SafeExpressionError(ValueError):
    """Raised when an expression is rejected by the AST safety filter."""


def _is_forbidden_name(name: str) -> bool:
    if name in _FORBIDDEN_NAMES:
        return True
    if name.startswith("__"):
        return True
    return False


def _check_node(node: ast.AST, depth: int = 0) -> None:
    if depth > 40:
        raise SafeExpressionError("Expression too deeply nested.")

    # ast.parse(..., mode="eval") produces ast.Expression
    if isinstance(node, ast.Expression):
        _check_node(node.body, depth + 1)
        return

    if isinstance(node, ast.Module):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise SafeExpressionError("Only a single expression is allowed.")
        _check_node(node.body[0], depth + 1)
        return

    if isinstance(node, ast.Expr):
        _check_node(node.value, depth + 1)
        return

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise SafeExpressionError("Unsupported boolean operator.")
        for v in node.values:
            _check_node(v, depth + 1)
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise SafeExpressionError(f"Operator not allowed: {type(node.op).__name__}")
        _check_node(node.left, depth + 1)
        _check_node(node.right, depth + 1)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise SafeExpressionError(f"Unary operator not allowed: {type(node.op).__name__}")
        _check_node(node.operand, depth + 1)
        return

    if isinstance(node, ast.Compare):
        if not node.ops or not any(type(op) in _CMP_OPS for op in node.ops):
            raise SafeExpressionError("Comparison operator not allowed.")
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                raise SafeExpressionError(f"Comparison not allowed: {type(op).__name__}")
        _check_node(node.left, depth + 1)
        for c in node.comparators:
            _check_node(c, depth + 1)
        return

    if isinstance(node, ast.Attribute):
        if _is_forbidden_name(node.attr):
            raise SafeExpressionError(f"Attribute not allowed: {node.attr}")
        # Only allow known pandas methods/attrs, or .str/.dt chains
        if node.attr not in _ALLOWED_METHODS and node.attr not in _ALLOWED_ATTRS:
            # Allow single-letter accessors already in methods; block unknown
            raise SafeExpressionError(
                f"Attribute '{node.attr}' is not allowed. "
                f"Use one of: {', '.join(sorted(list(_ALLOWED_METHODS | _ALLOWED_ATTRS)[:30]))}…"
            )
        _check_node(node.value, depth + 1)
        return

    if isinstance(node, ast.Call):
        _check_node(node.func, depth + 1)
        for a in node.args:
            _check_node(a, depth + 1)
        for kw in node.keywords:
            if kw.arg is None:
                raise SafeExpressionError("**kwargs not allowed in expressions.")
            _check_node(kw.value, depth + 1)
        return

    if isinstance(node, ast.Subscript):
        _check_node(node.value, depth + 1)
        _check_node(node.slice, depth + 1)
        return

    if isinstance(node, ast.Slice):
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                _check_node(part, depth + 1)
        return

    if isinstance(node, ast.Index):  # pragma: no cover - py<3.9
        _check_node(node.value, depth + 1)
        return

    if isinstance(node, ast.Name):
        if _is_forbidden_name(node.id):
            raise SafeExpressionError(f"Name not allowed: {node.id}")
        return

    if isinstance(node, (ast.Constant, ast.Load, ast.Store)):
        return

    # Tuples / lists for isin([a,b]) and multi-col access
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _check_node(elt, depth + 1)
        return

    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                _check_node(k, depth + 1)
            _check_node(v, depth + 1)
        return

    if isinstance(node, ast.Set):
        for elt in node.elts:
            _check_node(elt, depth + 1)
        return

    if isinstance(node, ast.IfExp):
        _check_node(node.test, depth + 1)
        _check_node(node.body, depth + 1)
        _check_node(node.orelse, depth + 1)
        return

    # Explicitly reject dangerous / unused forms
    raise SafeExpressionError(
        f"Expression node not allowed: {type(node).__name__}"
    )


def safe_eval_expression(
    expression: str,
    df: pd.DataFrame,
    extra_locals: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Evaluate a restricted pandas expression against `df`.

    Allowed names: df, pd, and anything in extra_locals.
    Allowed constructs: comparisons, bool ops, arithmetic, subscripts,
    and a closed set of DataFrame/Series methods/attributes.
    """
    if not expression or not expression.strip():
        raise SafeExpressionError("Empty expression.")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise SafeExpressionError(f"Invalid syntax: {e}") from e

    _check_node(tree)

    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "True": True,
        "False": False,
        "None": None,
    }

    env: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": df,
        "pd": pd,
    }
    # Expose column names as locals so `df[sales > 100]` works like pandas.query
    for col in df.columns:
        # Avoid overwriting df/pd or forbidden names
        if col in env or _is_forbidden_name(str(col)):
            continue
        try:
            env[str(col)] = df[col]
        except Exception:
            continue
    if extra_locals:
        env.update(extra_locals)

    code = compile(tree, "<csv_query>", "eval")
    return eval(code, env, {})  # noqa: S307 — AST-filtered, closed env


def looks_like_filter(expression: str) -> bool:
    """Heuristic: comparison-only expressions are usually row filters."""
    expr = expression.strip()
    if not expr:
        return False
    # df.query style without df. prefix
    if expr.startswith("df.") or expr.startswith("df["):
        return False
    tokens = ("==", "!=", ">", "<", ">=", "<=", " in ", " not in ", " and ", " or ")
    return any(t in expr for t in tokens)


def evaluate_csv_expression(expression: str, df: pd.DataFrame) -> Any:
    """
    Public entry: prefer df.query for plain filters; otherwise AST-safe eval.
    """
    expr = (expression or "").strip()
    if not expr:
        raise SafeExpressionError("Empty expression.")

    # Normalize leading "df.query('...')" / bare filter
    if expr.startswith("df.query(") and expr.endswith(")"):
        inner = expr[len("df.query(") : -1].strip().strip("'\"")
        return df.query(inner)

    if looks_like_filter(expr):
        try:
            return df.query(expr)
        except Exception:
            # Fall through to AST-safe eval (handles df['col'] > 3)
            pass

    return safe_eval_expression(expr, df)
