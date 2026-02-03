"""
Stage 4 — Reporting.

Produces a human-readable summary at each pipeline stage and writes the
final cleaned + transformed output to disk as CSV files.

Outputs written:
  data/cleaned.csv              — post-cleaning, pre-transform snapshot
  data/X_transformed.csv        — final feature matrix
  data/y_target.csv             — target vector
  data/pipeline_report.txt      — full text report
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPORT_DIR = Path("data")


def _section(title: str) -> str:
    width = 60
    return f"\n{'=' * width}\n  {title}\n{'=' * width}\n"


def _df_summary(df: pd.DataFrame) -> str:
    lines = [
        f"  Shape        : {df.shape}",
        f"  Columns      : {list(df.columns)}",
        f"  Dtypes       :\n{df.dtypes.to_string()}",
        f"  Missing vals :\n{df.isna().sum().to_string()}",
        "",
        "  Value counts for first 5 columns:",
    ]
    for col in df.columns[:5]:
        lines.append(f"    [{col}]\n{df[col].value_counts().head(5).to_string()}\n")
    return "\n".join(lines)


def report(raw: pd.DataFrame,
           cleaned: pd.DataFrame,
           X: pd.DataFrame,
           y: pd.Series,
           clean_stats: dict | None = None) -> str:
    """
    Build, print, and persist a full pipeline report.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []

    # --- Raw input ---
    parts.append(_section("1. RAW INPUT"))
    parts.append(_df_summary(raw))

    # --- Cleaning stats ---
    parts.append(_section("2. CLEANING"))
    if clean_stats:
        for k, v in clean_stats.items():
            parts.append(f"  {k}: {v}")
        parts.append("")
    parts.append(_df_summary(cleaned))

    # --- Transformation ---
    parts.append(_section("3. TRANSFORMATION"))
    parts.append(f"  Feature matrix shape : {X.shape}")
    parts.append(f"  Features             : {list(X.columns)}")
    parts.append(f"  Target distribution  :\n{y.value_counts().to_string()}")
    parts.append(f"  Target class balance : {y.mean():.3f} positive rate")
    parts.append("")
    parts.append("  Feature statistics:")
    parts.append(X.describe().to_string())

    full_report = "\n".join(parts)

    # --- Persist outputs ---
    cleaned.to_csv(REPORT_DIR / "cleaned.csv", index=False)
    X.to_csv(REPORT_DIR / "X_transformed.csv", index=False)
    y.rename("mental_health_condition").to_csv(REPORT_DIR / "y_target.csv", index=False)

    report_path = REPORT_DIR / "pipeline_report.txt"
    report_path.write_text(full_report)

    print(f"\n[reporter] Outputs written to {REPORT_DIR}/")
    print(f"           cleaned.csv, X_transformed.csv, y_target.csv, pipeline_report.txt")

    return full_report
