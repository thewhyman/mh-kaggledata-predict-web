"""
Stage 3 — Transform.

Sub-steps:
  1. Separate target (`mental_health_condition`) from features
  2. Label-encode binary Yes/No columns  →  0 / 1
  3. One-hot encode remaining categoricals (Gender, Country, work_interferes,
     No_employees) with drop='first' to avoid the dummy-variable trap
  4. Min-max scale the Age column to [0, 1]
  5. Engineer two interaction features:
       - `remote_x_family`          : remote_work AND family_history (both 1)
       - `age_treatment_interaction`: Age_scaled * treatment  (continuous × binary)
  6. Encode the target as 0 / 1
  7. Return (X, y) as numpy-compatible DataFrames plus the fitted transformers
     (so they can be persisted for inference)

Design notes:
  - All transformers are fit on the full dataset here because this script is
    explicitly a *batch preprocessing* step meant to produce a single cleaned
    file.  In a production serving pipeline you would fit on train only and
    transform train+val+test separately.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "treatment"

BINARY_COLS = [
    "self_employed", "family_history",
    "remote_work", "mental_health_interview",
    "tech_company", "obs_consequence",
]

CATEGORICAL_COLS = [
    "Gender", "Country", "work_interferes", "No_employees", "mental_vs_physical",
    "benefits", "care_options", "wellness_program", "seek_help", "anonymity",
    "leave", "mental_health_consequence", "phys_health_consequence",
    "coworkers", "supervisor", "phys_health_interview",
]


def _encode_binary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Map Yes/No → 1/0 for the given columns (in-place copy)."""
    mapping = {"Yes": 1, "No": 0}
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(int)
    return df


def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full feature transformation.

    Returns
    -------
    X : pd.DataFrame   — transformed feature matrix
    y : pd.Series      — binary target (0 / 1)
    meta : dict        — fitted scaler + column lists (for serialisation)
    """
    df = df.copy()

    # 1. Split target
    y = (df.pop(TARGET_COL) == "Yes").astype(int)

    # 2. Binary encode Yes/No columns
    df = _encode_binary(df, BINARY_COLS)

    # 3. One-hot encode high-cardinality categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)

    # 4. Scale Age to [0, 1]
    scaler = MinMaxScaler()
    df["Age"] = scaler.fit_transform(df[["Age"]])

    # 5. Feature engineering
    df["remote_x_family"] = df["remote_work"] * df["family_history"]

    # treatment is now target, so we don't use it for feature engineering
    # df["age_treatment_interaction"] = df["Age"] * df["treatment"]

    # 6. Collect metadata
    meta = {
        "scaler": scaler,
        "feature_columns": list(df.columns),
        "binary_cols": BINARY_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }

    print(f"[transformer] Feature matrix shape: {df.shape}")
    print(f"[transformer] Target distribution:\n{y.value_counts().to_string()}")
    print(f"[transformer] Engineered features: remote_x_family")

    df.attrs["transform_meta"] = meta
    return df, y, meta
