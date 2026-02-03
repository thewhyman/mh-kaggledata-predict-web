"""
Stage 1 — Load & Validate.

Responsibilities:
  - Read CSV from disk
  - Enforce expected column presence (fail fast if columns are missing)
  - Provide an initial shape / dtype snapshot for the reporter
"""

import pandas as pd

EXPECTED_COLUMNS = {
    "Age", "Gender", "Country", "self_employed", "family_history",
    "treatment", "work_interferes", "No_employees", "remote_work",
    "mental_health_interview", "mental_vs_physical",
    "tech_company", "benefits", "care_options", "wellness_program",
    "seek_help", "anonymity", "leave", "mental_health_consequence",
    "phys_health_consequence", "coworkers", "supervisor",
    "phys_health_interview", "obs_consequence",
}

# Columns we actually need after renaming; everything else is dropped.
KEEP_COLUMNS = list(EXPECTED_COLUMNS)

# Maps raw Kaggle column names → what the rest of the pipeline expects.
RENAME_MAP = {
    "work_interfere":    "work_interferes",
    "no_employees":      "No_employees",
}


def load(path: str) -> pd.DataFrame:
    """Read CSV, rename & select columns, validate schema."""
    df = pd.read_csv(path, dtype=str)          # everything as str first; cleaner casts later

    # Strip BOM / leading-trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename Kaggle originals → pipeline names (no-op if already correct)
    df.rename(columns=RENAME_MAP, inplace=True)

    # Drop any columns not in our pipeline schema
    extra = set(df.columns) - EXPECTED_COLUMNS
    if extra:
        df.drop(columns=list(extra), inplace=True)
        print(f"[loader] Dropped {len(extra)} extra columns: {sorted(extra)}")

    # Final schema check
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[loader] CSV is missing required columns: {missing_cols}"
        )

    print(f"[loader] Loaded {len(df)} rows x {len(df.columns)} columns from '{path}'")
    return df
