
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load artifacts
try:
    PIPELINE = joblib.load("pipeline_artifacts.joblib")
    MODEL_DATA = joblib.load("model.pkl")
    MODEL = MODEL_DATA["model"]
    # Overwrite FEATURE_COLS with the ones the model actually wants
    FEATURE_COLS = MODEL_DATA["feature_names"]
    
    # Extract specific components
    CLEAN_STATS = PIPELINE.get("clean_stats", {})
    SCALER = PIPELINE.get("transform_meta", {}).get("scaler")
    
    if not FEATURE_COLS:
        raise ValueError("Feature columns not found in model artifacts.")
    print(f"[INFO] Artifacts loaded. Model expects {len(FEATURE_COLS)} features.")
except Exception as e:
    print(f"[ERROR] Failed to load artifacts: {e}")
    PIPELINE = {}
    MODEL = None
    CLEAN_STATS = {}
    SCALER = None
    FEATURE_COLS = []


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Replicate the pipeline preprocessing for a single input dictionary.
    
    Steps:
    1. Create DataFrame
    2. Impute missing values using CLEAN_STATS
    3. Feature Engineering / Transformation (manual to match pipeline)
    4. Reindex to match training schema
    """
    df = pd.DataFrame([data])
    
    # --- 1. Imputation (Cleaning) ---
    # Age
    age_median = CLEAN_STATS.get("age_imputed_with", 31.0)
    try:
        df["Age"] = pd.to_numeric(df["Age"], errors='coerce').fillna(age_median)
    except KeyError:
        df["Age"] = age_median
        
    # Categorical Modes
    # List of cols we expect to clean
    cols_to_clean = [
        "Gender", "Country", "self_employed", "family_history",
        "work_interferes", "No_employees", "remote_work", 
        "mental_health_interview", "mental_vs_physical",
        "tech_company", "benefits", "care_options", "wellness_program",
        "seek_help", "anonymity", "leave", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor",
        "phys_health_interview", "obs_consequence"
    ]
    
    for col in cols_to_clean:
        if col not in df.columns or pd.isna(df[col].iloc[0]) or str(df[col].iloc[0]).strip() == "":
            # Impute with mode if available, else a sensible default or just "No" / "Don't know"
            mode_val = CLEAN_STATS.get(f"{col}_mode")
            if mode_val:
                df[col] = mode_val
            else:
                # Fallback defaults if mode missing for some reason
                if col == "Gender": df[col] = "Male"
                elif col == "Country": df[col] = "United States"
                else: df[col] = "No"

    # Normalize Gender/Country manually or rely on robustness? 
    # The pipeline uses map lookups. We should probably replicate minimal normalization logic.
    # For a demo, let's assume valid inputs from the form, except for quick normalization.
    # A robust production app would import the cleaning functions directly.
    # Given the constraint, we'll assume the form sends clean-ish data or raw strings that 
    # match the pipeline's categories sufficiently after one-hot encoding (mismatches -> 0).
    
    # --- 2. Transformation ---
    
    # Binary Encoding (Yes/No -> 1/0)
    # Pipeline list: self_employed, family_history, remote_work, mental_health_interview, tech_company, obs_consequence
    binary_cols = [
        "self_employed", "family_history", "remote_work", 
        "mental_health_interview", "tech_company", "obs_consequence"
    ]
    for col in binary_cols:
        val = str(df[col].iloc[0]).strip()
        df[col] = 1 if val.lower() == "yes" else 0
        
    # Dummies / One-Hot
    # We do this by creating the columns directly.
    # Or we can use get_dummies and then reindex.
    cat_cols = [
        "Gender", "Country", "work_interferes", "No_employees", "mental_vs_physical",
        "benefits", "care_options", "wellness_program", "seek_help", "anonymity",
        "leave", "mental_health_consequence", "phys_health_consequence",
        "coworkers", "supervisor", "phys_health_interview"
    ]
    
    # NOTE: The pipeline cleaning normalises Gender to Male/Female/Other and Country names.
    # If we don't normalize here, 'male' might not match 'Gender_Male'.
    # Simplified normalization for the critical ones:
    g = str(df["Gender"].iloc[0]).lower()
    if g in ['male', 'm', 'man']: df["Gender"] = "Male"
    elif g in ['female', 'f', 'woman']: df["Gender"] = "Female"
    else: df["Gender"] = "Other"
    
    c = str(df["Country"].iloc[0]).lower()
    if c in ['united states', 'usa', 'us', 'u.s.']: df["Country"] = "United States"
    elif c in ['united kingdom', 'uk', 'u.k.']: df["Country"] = "United Kingdom"
    # (Leaving others as is, hoping form sends correct values)
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # Scale Age
    if SCALER:
        df["Age"] = SCALER.transform(df[["Age"]])
        
    # Engineered Features
    # remote_x_family = remote_work * family_history
    # Make sure cols exist (they are binary, so they should remain after get_dummies unless passed in binary list?)
    # Binary encoded cols remain as columns in df.
    try:
        if "remote_work" in df.columns and "family_history" in df.columns:
            df["remote_x_family"] = df["remote_work"] * df["family_history"]
        else:
             df["remote_x_family"] = 0
    except Exception:
        df["remote_x_family"] = 0
        
    # Reindex to match model features
    # This aligns columns and fills missing dummy cols with 0
    df_final = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df_final


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.form.to_dict()
        X_input = preprocess_input(data)
        
        # Predict
        # 1 = Yes (needs treatment), 0 = No
        prediction = MODEL.predict(X_input)[0]
        prob =0 
        if hasattr(MODEL, "predict_proba"):
            prob = MODEL.predict_proba(X_input)[0][1]
            
        result = "Yes" if prediction == 1 else "No"
        
        return jsonify({
            "treatment_needed": result,
            "probability": f"{prob:.2f}",
            "input_summary": data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
