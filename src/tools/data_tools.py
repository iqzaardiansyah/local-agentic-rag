"""Free local CSV / tabular data analysis tools for the agent sandbox."""

import os
from typing import List

import pandas as pd
from langchain_core.tools import tool

from src.tools.coding_tools import WORKSPACE_DIR, get_safe_workspace_path, resolve_readable_path


def _resolve_data_path(path: str) -> str:
    """Resolve a data file path: workspace first, then project data/, then project root."""
    candidates = []
    if not os.path.isabs(path):
        candidates.append(os.path.join(WORKSPACE_DIR, path))
        candidates.append(os.path.join(os.path.dirname(WORKSPACE_DIR), "data", path))
        candidates.append(os.path.join(os.path.dirname(WORKSPACE_DIR), path))
    else:
        candidates.append(path)

    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fall back to sandbox-safe path (may not exist — caller reports error)
    try:
        return get_safe_workspace_path(path)
    except Exception:
        return resolve_readable_path(path)


def _load_df(path: str) -> pd.DataFrame:
    resolved = _resolve_data_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Data file not found: {path}")
    ext = os.path.splitext(resolved)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(resolved)
    return pd.read_csv(resolved)


@tool
def analyze_csv_summary(path: str) -> str:
    """
    Load a CSV/Excel file and return a free local summary: shape, dtypes, nulls,
    numeric describe stats, and a preview of the first rows.
    Args:
        path: CSV/Excel path relative to workspace, data/, or project root.
    """
    try:
        df = _load_df(path)
        lines = [
            f"File: {path}",
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            "",
            "Dtypes:",
            df.dtypes.astype(str).to_string(),
            "",
            "Null counts:",
            df.isnull().sum().to_string(),
            "",
            "Numeric describe:",
            df.describe(include="number").to_string() if not df.select_dtypes("number").empty else "(no numeric columns)",
            "",
            "Preview (first 5 rows):",
            df.head(5).to_string(),
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing CSV: {e}"


@tool
def csv_query(path: str, expression: str) -> str:
    """
    Run a pandas query expression against a CSV/Excel file and print the result.
    Example expression: "age > 30 and city == 'NYC'" or "df.groupby('city')['sales'].mean()".
    The DataFrame is available as `df`. Always return a printable result.
    Args:
        path: CSV/Excel path.
        expression: Python expression using pandas DataFrame `df`.
    """
    try:
        df = _load_df(path)
        result = eval(expression, {"df": df, "pd": pd, "__builtins__": {}})  # noqa: S307 — sandboxed limited builtins
        if isinstance(result, pd.DataFrame):
            if len(result) > 50:
                return f"Result truncated to 50 of {len(result)} rows:\n{result.head(50).to_string()}"
            return result.to_string()
        return str(result)
    except Exception as e:
        return f"Error running CSV query: {e}"


@tool
def csv_groupby(path: str, group_col: str, agg_col: str, agg: str = "mean") -> str:
    """
    Group a CSV by one column and aggregate another (mean/sum/count/min/max).
    Args:
        path: CSV/Excel path.
        group_col: Column to group by.
        agg_col: Column to aggregate.
        agg: Aggregation function name (default mean).
    """
    try:
        df = _load_df(path)
        if group_col not in df.columns or agg_col not in df.columns:
            return f"Error: columns must exist. Available: {list(df.columns)}"
        if agg not in ("mean", "sum", "count", "min", "max", "median", "std"):
            return "Error: agg must be one of mean/sum/count/min/max/median/std."
        if agg == "count":
            out = df.groupby(group_col)[agg_col].count()
        else:
            out = getattr(df.groupby(group_col)[agg_col], agg)()
        return out.reset_index().to_string()
    except Exception as e:
        return f"Error in groupby: {e}"


@tool
def save_csv_chart(path: str, chart_type: str, x: str, y: str, output_name: str = "chart.png") -> str:
    """
    Create a free local matplotlib chart from a CSV and save it into ./workspace.
    Chart types: line, bar, scatter, hist.
    Args:
        path: CSV/Excel path.
        chart_type: line | bar | scatter | hist
        x: Column for X axis (for hist, the value column).
        y: Column for Y axis (ignored for hist).
        output_name: Output image filename inside workspace (default chart.png).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _load_df(path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ct = chart_type.lower().strip()
        if ct == "line":
            ax.plot(df[x], df[y] if y in df.columns else None)
        elif ct == "bar":
            ax.bar(df[x], df[y])
        elif ct == "scatter":
            ax.scatter(df[x], df[y])
        elif ct == "hist":
            ax.hist(df[x].dropna(), bins=min(30, max(5, df[x].nunique() or 10)))
        else:
            return "Error: chart_type must be line, bar, scatter, or hist."

        ax.set_xlabel(x)
        if ct != "hist":
            ax.set_ylabel(y)
        ax.set_title(f"{chart_type.title()} of {x}" + (f" vs {y}" if ct != "hist" else ""))
        fig.tight_layout()

        safe_name = os.path.basename(output_name)
        if not safe_name.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
            safe_name += ".png"
        out_path = get_safe_workspace_path(safe_name)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return f"Chart saved to workspace/{safe_name}"
    except Exception as e:
        return f"Error creating chart: {e}"


@tool
def list_data_files() -> str:
    """List available CSV/Excel/JSON data files in data/ and workspace/."""
    roots = [
        os.path.join(os.path.dirname(WORKSPACE_DIR), "data"),
        WORKSPACE_DIR,
    ]
    found: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".csv", ".xlsx", ".xls", ".json", ".parquet")):
                    rel = os.path.relpath(os.path.join(dirpath, f), os.path.dirname(WORKSPACE_DIR))
                    found.append(rel.replace("\\", "/"))
    if not found:
        return "No data files found in data/ or workspace/."
    return "Available data files:\n" + "\n".join(found[:50])
