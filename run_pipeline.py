"""
CLI entry point.

Usage:
    python run_pipeline.py                        # uses default paths
    python run_pipeline.py --input path/to/raw.csv
    python run_pipeline.py --generate             # regenerate sample data first
"""

import argparse
import sys
import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Run the mental-health dataset preprocessing pipeline."
    )
    parser.add_argument(
        "--input", default="data/mental_health_raw.csv",
        help="Path to the raw CSV input file."
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Regenerate the sample dataset before running the pipeline."
    )
    args = parser.parse_args()

    if args.generate:
        from generate_sample_data import generate
        generate(path=args.input)

    from pipeline.pipeline import run
    try:
        result = run(input_path=args.input)
        
        # Save pipeline artifacts for inference
        pipeline_artifacts = {
            "clean_stats": result["cleaned"].attrs.get("clean_stats", {}),
            "transform_meta": result["meta"]
        }
        joblib.dump(pipeline_artifacts, "pipeline_artifacts.joblib")
        print("\n[INFO] Saved 'pipeline_artifacts.joblib' for inference.")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input}")
        print("        Run with --generate to create sample data first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
