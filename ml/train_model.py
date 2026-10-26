"""
Train a RandomForest diabetes-risk classifier on the Pima Indians Diabetes dataset.

Steps
-----
1. Load ml/data/diabetes.csv (768 rows, 8 features + binary `Outcome`).
2. Preprocessing: in Glucose, BloodPressure, SkinThickness, Insulin and BMI a value of 0
   is physiologically impossible and encodes "missing" -> replaced with NaN.
3. Stratified 80/20 train/test split (random_state=42). The test set is NEVER used for
   fitting, imputation statistics, or hyperparameter selection.
4. Pipeline = SimpleImputer(median) -> RandomForestClassifier(class_weight="balanced").
   The imputer lives inside the pipeline, so medians are learned from training data only
   (and inside each CV fold during the grid search).
5. Hyperparameters are chosen with 5-fold stratified CV on the TRAINING set (ROC-AUC).
6. The final model is evaluated once on the held-out test set. Only those measured numbers
   are written to ml/artifacts/metrics.json.

Usage:  python ml/train_model.py
Output: ml/artifacts/diabetes_rf.joblib, ml/artifacts/metrics.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "diabetes_rf.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")
    df = df.drop_duplicates().reset_index(drop=True)
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    return df[FEATURES], df[TARGET].astype(int)


def main() -> None:
    X, y = load_data()
    print(f"Loaded {len(X)} rows | positives: {int(y.sum())} ({y.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    param_grid = {
        "rf__n_estimators": [200, 400],
        "rf__max_depth": [4, 6, 8, None],
        "rf__min_samples_leaf": [1, 3, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    model: Pipeline = search.best_estimator_
    print("Best params:", search.best_params_, f"| CV ROC-AUC: {search.best_score_:.3f}")

    # ---- Held-out evaluation (the only source of reported metrics) ----
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test_metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  test {k:<9}: {v:.3f}")
    print("  confusion  :", test_metrics["confusion_matrix"])

    importances = dict(
        sorted(
            zip(FEATURES, model.named_steps["rf"].feature_importances_.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "zero_means_missing": ZERO_MEANS_MISSING,
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Pima Indians Diabetes (768 rows, public)",
        "n_rows_used": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "best_params": search.best_params_,
        "cv_roc_auc_train_only": search.best_score_,
        "test_metrics_held_out": test_metrics,
        "feature_importances": importances,
        "sklearn_version": sklearn.__version__,
        "note": "Metrics were measured on a held-out test split never used for fitting or tuning. "
        "Demonstration only - not medical advice.",
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
