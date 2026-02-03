"""
Train and evaluate classifiers to predict mental_health_condition.

Reads the preprocessed X / y CSVs produced by the pipeline, then:
  1. Drops leaky features (treatment, mental_health_interview,
     age_treatment_interaction) — these are effects of having a condition,
     not causes, so including them would inflate metrics artificially.
  2. Splits 80/20 train/test (stratified).
  3. Trains 3 models with 5-fold stratified cross-validation.
  4. Picks the best model by mean CV F1.
  5. Evaluates on the held-out test set: accuracy, precision, recall, F1,
     confusion matrix.
  6. Prints the top-15 feature importances from the best model.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import joblib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
X_PATH = "data/X_transformed.csv"
Y_PATH = "data/y_target.csv"

# These features are causally downstream of having a mental health condition.
# Keeping them would be data leakage.
LEAKY_FEATURES = ["mental_health_interview"]

TEST_SIZE   = 0.2
RANDOM_SEED = 42
CV_FOLDS    = 5

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_SEED
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _section(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}\n")


def _get_feature_importances(model, feature_names: list[str]) -> pd.Series:
    """Extract importances regardless of model type."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)
    # Logistic Regression: use absolute coefficients
    coefs = np.abs(model.coef_[0])
    return pd.Series(coefs, index=feature_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load ---
    _section("Loading data")
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).squeeze("columns")   # DataFrame → Series

    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Target  : {y.value_counts().to_dict()}")

    # --- Drop leaky features ---
    dropped = [c for c in LEAKY_FEATURES if c in X.columns]
    X = X.drop(columns=dropped)
    print(f"\n  Dropped leaky features: {dropped}")
    print(f"  X shape after drop    : {X.shape}")

    feature_names = list(X.columns)

    # --- Train / test split ---
    _section("Train / Test Split (80 / 20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Train : {X_train.shape[0]} rows   |  Test : {X_test.shape[0]} rows")
    print(f"  Train target dist : {y_train.value_counts().to_dict()}")
    print(f"  Test  target dist : {y_test.value_counts().to_dict()}")

    # --- Cross-validation ---
    _section("5-Fold Stratified Cross-Validation (scoring = F1)")
    cv_results = {}
    for name, model in MODELS.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=CV_FOLDS, scoring="f1"
        )
        cv_results[name] = scores
        print(f"  {name:25s}  mean F1 = {scores.mean():.4f}  "
              f"(+/- {scores.std():.4f})  folds: {np.round(scores, 3)}")

    best_name = max(cv_results, key=lambda n: cv_results[n].mean())
    print(f"\n  Best model by CV F1: {best_name}")

    # --- Train best model on full training set ---
    _section(f"Training best model: {best_name}")
    best_model = MODELS[best_name]
    best_model.fit(X_train, y_train)
    
    # Save model and feature names
    joblib.dump({
        "model": best_model,
        "feature_names": X_train.columns.tolist()
    }, "model.pkl")
    print(f"\n[INFO] Saved model and features to 'model.pkl'")

    # --- Test-set evaluation ---
    _section("Test-Set Evaluation")
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred,
                                target_names=["No condition", "Has condition"]))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                    Pred No   Pred Yes")
    print(f"    Actual No       {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"    Actual Yes      {cm[1][0]:>6}    {cm[1][1]:>6}")

    # --- Feature importances ---
    _section(f"Top 15 Feature Importances ({best_name})")
    importances = _get_feature_importances(best_model, feature_names)
    top15 = importances.sort_values(ascending=False).head(15)
    max_val = top15.iloc[0]
    for feat, val in top15.items():
        bar = "█" * int((val / max_val) * 40)
        print(f"  {feat:40s} {val:.4f}  {bar}")


if __name__ == "__main__":
    main()
