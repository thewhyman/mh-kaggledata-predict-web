"""
Stage 2 — Clean.

Sub-steps executed in order:
  1. Strip whitespace from all string cells
  2. Normalise Gender to a canonical set  {Male, Female, Other}
  3. Normalise Country to canonical names  (typo map + case folding)
  4. Parse & validate Age  (drop Excel-date strings, clamp numeric range 10-90)
  5. Impute missing values column-by-column using sensible defaults
  6. Drop exact duplicate rows

Nothing here touches the target column (`mental_health_condition`) in a way
that would leak information — imputation there is majority-class only, which
is intentional and documented.
"""

import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Canonical mappings
# ---------------------------------------------------------------------------
GENDER_MAP = {
    "male": "Male", "m": "Male",
    "female": "Female", "f": "Female",
    "other": "Other", "non-binary": "Other",
    "prefer not to say": "Other",
}

COUNTRY_MAP = {
    "us": "United States", "u.s.": "United States", "united states": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "united kingdom": "United Kingdom",
    "canada": "Canada",
    "germany": "Germany",
    "france": "France",
    "india": "India",
    "australia": "Australia", "austalia": "Australia",   # known typo
    "brazil": "Brazil",
    "japan": "Japan",
    "netherlands": "Netherlands",
    "spain": "Spain",
    "ireland": "Ireland",
    "singapore": "Singapore",
    "sweden": "Sweden",
    "norway": "Norway",
    "poland": "Poland", "polland": "Poland",            # known typo
}

# Normalise the real Kaggle employee-bucket strings into the 4 buckets
# the transformer expects.
EMPLOYEES_BUCKET_MAP = {
    "1-5":            "1-5",
    "6-25":           "6-25",
    "26-100":         "26-100",
    "100-500":        "100+",
    "500-1000":       "100+",
    "more than 1000": "100+",
}

# Columns where mode imputation is appropriate (low-cardinality categoricals)
MODE_IMPUTE_COLS = [
    "Gender", "Country", "self_employed", "family_history",
    "treatment", "work_interferes", "No_employees",
    "remote_work", "mental_health_interview", "mental_vs_physical",
    "tech_company", "benefits", "care_options", "wellness_program",
    "seek_help", "anonymity", "leave", "mental_health_consequence",
    "phys_health_consequence", "coworkers", "supervisor",
    "phys_health_interview", "obs_consequence",
]

VALID_AGE_RANGE = (10, 90)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_excel_date(val: str) -> bool:
    """Detect strings like '25-Mar', '30-Apr' that Excel auto-converts."""
    return bool(re.match(r"^\d{1,2}-[A-Za-z]{3}$", val.strip()))


def _parse_age(val) -> float:
    """Return numeric age or NaN if unparseable / out of range."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if _is_excel_date(s):
        return np.nan                          # can't recover original intent
    try:
        age = float(s)
    except (ValueError, TypeError):
        return np.nan
    if age < VALID_AGE_RANGE[0] or age > VALID_AGE_RANGE[1]:
        return np.nan
    return round(age)


def _normalise_gender(val) -> str | float:
    if pd.isna(val):
        return np.nan
    key = str(val).strip().lower()
    return GENDER_MAP.get(key, "Other")


def _normalise_country(val) -> str | float:
    if pd.isna(val):
        return np.nan
    key = str(val).strip().lower()
    return COUNTRY_MAP.get(key, str(val).strip().title())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps; return a new DataFrame (original untouched)."""
    df = df.copy()
    stats: dict = {}

    # 1. Global whitespace strip on object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # 2. Gender
    before_gender_na = df["Gender"].isna().sum()
    df["Gender"] = df["Gender"].map(_normalise_gender)
    stats["gender_na_before"] = int(before_gender_na)

    # 3. Country
    before_country_na = df["Country"].isna().sum()
    df["Country"] = df["Country"].map(_normalise_country)
    stats["country_na_before"] = int(before_country_na)

    # 3b. Collapse "Maybe" / "Don't know" → "No" in binary Yes/No columns
    #     (the real Kaggle survey uses these extra values)
    BINARY_COLS_TO_COLLAPSE = [
        "self_employed", "family_history", "treatment",
        "remote_work", "mental_health_interview",
    ]
    for col in BINARY_COLS_TO_COLLAPSE:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: "No" if pd.notna(v) and str(v).strip().lower() in
                          ("maybe", "don't know", "don't know") else v
            )

    # 3c. Normalise No_employees buckets (real Kaggle has "100-500", "More than 1000", etc.)
    df["No_employees"] = df["No_employees"].apply(
        lambda v: EMPLOYEES_BUCKET_MAP.get(str(v).strip().lower(), v) if pd.notna(v) else v
    )

    # 4. Age
    before_age_na = df["Age"].isna().sum()
    df["Age"] = df["Age"].map(_parse_age)
    stats["age_na_after_parse"] = int(df["Age"].isna().sum())
    stats["age_rows_invalidated"] = stats["age_na_after_parse"] - int(before_age_na)

    # 5. Impute missing values
    # Age: median (robust to skew)
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    stats["age_imputed_with"] = float(age_median)

    # Categorical columns: mode
    for col in MODE_IMPUTE_COLS:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                n_filled = df[col].isna().sum()
                df[col] = df[col].fillna(mode_val.iloc[0])
                stats[f"{col}_mode_imputed"] = int(n_filled)
                stats[f"{col}_mode"] = mode_val.iloc[0]

    # 6. Drop duplicates
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicates_removed"] = before_dedup - len(df)

    # Print a short summary
    print(f"[cleaner] Age rows invalidated (Excel dates / out-of-range): "
          f"{stats['age_rows_invalidated']}")
    print(f"[cleaner] Age imputed with median={stats['age_imputed_with']}")
    print(f"[cleaner] Duplicates removed: {stats['duplicates_removed']}")
    print(f"[cleaner] Rows after cleaning: {len(df)}")

    # Attach stats as a DataFrame attribute for the reporter
    df.attrs["clean_stats"] = stats
    return df
