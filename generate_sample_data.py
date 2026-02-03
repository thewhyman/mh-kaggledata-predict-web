"""
Generates a synthetic dataset that mirrors the schema and common data-quality
issues found in public mental-health survey datasets (e.g. the Kaggle
"Mental Health in Tech Survey").

Noise injected on purpose so the cleaning pipeline has real work to do:
  - Ages that are Excel-date strings ("25-Mar"), floats, negatives, or > 100
  - Inconsistent gender strings ("male", "Male", "M", "MALE", " male ", etc.)
  - Free-text country names with typos and extra whitespace
  - Missing values scattered across every column at realistic rates
  - Duplicate rows (exact and near-duplicates differing only in whitespace)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 1200


# ---------------------------------------------------------------------------
# Raw value pools
# ---------------------------------------------------------------------------
GENDERS = ["Male", "Female", "male", "MALE", "female", "FEMALE", "M", "F",
           "Other", "other", "non-binary", "Non-Binary", "prefer not to say"]

COUNTRIES = [
    "United States", "US", "U.S.", " United States ", "united states",
    "Canada", "canada", " Canada",
    "United Kingdom", "UK", "U.K.", "united kingdom",
    "Germany", "germany", "France", "france",
    "India", "india", "Australia", "australia",
    "Brazil", "brazil", "Japan", "japan",
    "Netherlands", "Spain", "Ireland", "ireland",
    "Singapore", "sweden", "Sweden", "Norway",
    "Poland", "Polland",            # intentional typo
    "Austalia",                      # intentional typo
]

WORK_INTERFERES = ["Never", "Rarely", "Sometimes", "Often", "Maybe", np.nan]
EMPLOYEES_BUCKETS = ["1-5", "6-25", "26-100", "100+"]
TREATMENT_SOUGHT = ["Yes", "No"]
FAMILY_HISTORY   = ["Yes", "No"]
SELF_EMPLOYED    = ["Yes", "No", np.nan]       # sparse column

# Additional categorical pools
BENEFITS = ["Yes", "No", "Don't know"]
CARE_OPTIONS = ["Yes", "No", "Not sure"]
WELLNESS_PROGRAM = ["Yes", "No", "Don't know"]
SEEK_HELP = ["Yes", "No", "Don't know"]
ANONYMITY = ["Yes", "No", "Don't know"]
LEAVE = ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
MENTAL_HEALTH_CONSEQUENCE = ["Yes", "No", "Maybe"]
PHYS_HEALTH_CONSEQUENCE = ["Yes", "No", "Maybe"]
COWORKERS = ["Yes", "No", "Some of them"]
SUPERVISOR = ["Yes", "No", "Some of them"]
MENTAL_HEALTH_INTERVIEW = ["Yes", "No", "Maybe"]
PHYS_HEALTH_INTERVIEW = ["Yes", "No", "Maybe"]
MENTAL_VS_PHYSICAL = ["Yes", "No", "Don't know"]
OBS_CONSEQUENCE = ["Yes", "No"]
TECH_COMPANY = ["Yes", "No"]


def _noisy_age() -> object:
    """Return an age value with occasional injected noise."""
    roll = RNG.random()
    if roll < 0.70:
        return int(RNG.integers(18, 65))
    if roll < 0.78:
        return float(RNG.integers(18, 65)) + RNG.random()   # float age
    if roll < 0.82:
        return RNG.choice(["25-Mar", "30-Apr", "22-Jun", "18-Sep"])  # Excel dates
    if roll < 0.86:
        return int(RNG.integers(-5, 0))                      # negative
    if roll < 0.90:
        return int(RNG.integers(100, 200))                   # absurd
    return np.nan                                            # missing


def _pick(pool, p_nan: float = 0.0):
    val = RNG.choice([x for x in pool if x is not np.nan])
    return np.nan if RNG.random() < p_nan else val


def _build_row() -> dict:
    return {
        "Age":                        _noisy_age(),
        "Gender":                     _pick(GENDERS, p_nan=0.03),
        "Country":                    _pick(COUNTRIES, p_nan=0.05),
        "self_employed":              _pick(SELF_EMPLOYED, p_nan=0.18),
        "family_history":             _pick(FAMILY_HISTORY, p_nan=0.04),
        "treatment":                  _pick(TREATMENT_SOUGHT, p_nan=0.06),
        "work_interferes":            _pick(WORK_INTERFERES, p_nan=0.12),
        "No_employees":               _pick(EMPLOYEES_BUCKETS, p_nan=0.07),
        "remote_work":                _pick(["Yes", "No"], p_nan=0.05),
        "tech_company":               _pick(TECH_COMPANY, p_nan=0.05),
        "benefits":                   _pick(BENEFITS, p_nan=0.10),
        "care_options":               _pick(CARE_OPTIONS, p_nan=0.10),
        "wellness_program":           _pick(WELLNESS_PROGRAM, p_nan=0.10),
        "seek_help":                  _pick(SEEK_HELP, p_nan=0.10),
        "anonymity":                  _pick(ANONYMITY, p_nan=0.10),
        "leave":                      _pick(LEAVE, p_nan=0.10),
        "mental_health_consequence":  _pick(MENTAL_HEALTH_CONSEQUENCE, p_nan=0.10),
        "phys_health_consequence":    _pick(PHYS_HEALTH_CONSEQUENCE, p_nan=0.10),
        "coworkers":                  _pick(COWORKERS, p_nan=0.10),
        "supervisor":                 _pick(SUPERVISOR, p_nan=0.10),
        "mental_health_interview":    _pick(MENTAL_HEALTH_INTERVIEW, p_nan=0.08),
        "phys_health_interview":      _pick(PHYS_HEALTH_INTERVIEW, p_nan=0.08),
        "mental_vs_physical":         _pick(MENTAL_VS_PHYSICAL, p_nan=0.08),
        "obs_consequence":            _pick(OBS_CONSEQUENCE, p_nan=0.08),
        "mental_health_condition":    _pick(["Yes", "No"], p_nan=0.10),
    }


def generate(path: str = "data/mental_health_raw.csv", n: int = N_ROWS) -> pd.DataFrame:
    rows = [_build_row() for _ in range(n)]
    df = pd.DataFrame(rows)

    # Inject ~40 exact duplicate rows at random positions
    dup_indices = RNG.choice(len(df), size=40, replace=False)
    duplicates = df.iloc[dup_indices].copy()

    # Inject ~20 near-duplicates (trailing whitespace on Gender)
    near_dup_indices = RNG.choice(len(df), size=20, replace=False)
    near_dups = df.iloc[near_dup_indices].copy()
    near_dups["Gender"] = near_dups["Gender"].apply(
        lambda g: f"{g}  " if pd.notna(g) else g
    )

    df = pd.concat([df, duplicates, near_dups], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    path_obj = __import__("pathlib").Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generate] Wrote {len(df)} rows -> {path}")
    return df


if __name__ == "__main__":
    generate()
