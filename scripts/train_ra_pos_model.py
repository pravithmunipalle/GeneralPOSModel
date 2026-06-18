#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "enrollment",
    "number_of_arms",
    "Active Comparator Arm Number",
    "Placebo Comparator Arm Number",
    "Experimental Arm Number",
    "study_design_info/masking_num",
    "population_mtx_naive",
    "population_mtx_ir",
    "population_dmard_ir",
    "population_biologic_ir",
    "population_tnf_ir",
    "endpoint_acr20",
    "endpoint_acr50",
    "endpoint_acr70",
    "endpoint_das28",
    "endpoint_cdai",
    "endpoint_sdai",
    "endpoint_haq_di",
    "background_mtx",
    "rescue_allowed",
    "combination_therapy",
]
CATEGORICAL = [
    "phase",
    "study_design_info/allocation",
    "oversight_info/has_dmc",
    "study_design_info/intervention_model",
    "study_design_info/primary_purpose",
    "sponsors/lead_sponsor/agency_class",
]


def coerce(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in NUMERIC:
        result[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    for column in CATEGORICAL:
        result[column] = frame[column].fillna("Unknown").astype(str) if column in frame else "Unknown"
    return result


def make_model(method: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    if method == "forest":
        base = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced", random_state=42)
    else:
        base = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, l2_regularization=0.08, random_state=42)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return Pipeline([("features", preprocessor), ("model", calibrated)])


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else 0.5,
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RA-only calibrated POS model from the RA candidate dataset.")
    parser.add_argument("--data", default="data/ra_pos/ra_trials.csv")
    parser.add_argument("--out", default="models/ra_pos_model.joblib")
    parser.add_argument("--method", choices=["hgb", "forest"], default="forest")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[df["trialbench_outcome_label"].notna()].copy()
    y = df["trialbench_outcome_label"].astype(int)
    x = coerce(df)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.28, random_state=42, stratify=stratify)
    model = make_model(args.method)
    model.fit(x_train, y_train)
    metrics = {"test": evaluate(model, x_test, y_test), "train_rows": int(len(x_train)), "test_rows": int(len(x_test))}

    artifact = {
        "model": model,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "metrics": metrics,
        "label_note": "TrialBench outcome labels are used as a proxy until endpoint-success labels are curated.",
        "examples": df.head(500).to_dict(orient="records"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    print(json.dumps({"model": args.out, "rows": int(len(df)), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
