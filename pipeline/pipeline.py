"""
Pipeline orchestrator.

Wires loader → cleaner → transformer → reporter in sequence.
Each stage is isolated; the orchestrator owns the data hand-offs and
error handling so individual modules stay focused.
"""

from pipeline.loader      import load
from pipeline.cleaner     import clean
from pipeline.transformer import transform
from pipeline.reporter    import report


def run(input_path: str = "data/mental_health_raw.csv") -> dict:
    """
    Execute the full preprocessing pipeline.

    Parameters
    ----------
    input_path : str
        Path to the raw CSV produced by generate_sample_data.py

    Returns
    -------
    dict with keys: raw, cleaned, X, y, meta, report_text
    """
    print("\n" + "─" * 60)
    print("  MENTAL HEALTH DATASET PREPROCESSING PIPELINE")
    print("─" * 60 + "\n")

    # Stage 1 — Load
    print("── Stage 1: Load & Validate ──")
    raw = load(input_path)

    # Stage 2 — Clean
    print("\n── Stage 2: Clean ──")
    cleaned = clean(raw)
    clean_stats = cleaned.attrs.get("clean_stats", {})

    # Stage 3 — Transform
    print("\n── Stage 3: Transform ──")
    X, y, meta = transform(cleaned)

    # Stage 4 — Report
    print("\n── Stage 4: Report ──")
    report_text = report(raw, cleaned, X, y, clean_stats)

    print("\n" + "─" * 60)
    print("  PIPELINE COMPLETE")
    print("─" * 60 + "\n")

    return {
        "raw":         raw,
        "cleaned":     cleaned,
        "X":           X,
        "y":           y,
        "meta":        meta,
        "report_text": report_text,
    }
