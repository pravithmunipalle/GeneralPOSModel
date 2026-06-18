from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureSpec, coerce_features


def make_pipeline(spec: FeatureSpec, task_type: str) -> Pipeline:
    transformers = []
    if spec.numeric:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                spec.numeric,
            )
        )
    if spec.categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=3),
                spec.categorical,
            )
        )
    transformers.append(("text", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2), "trial_text_blob"))

    estimator: Any
    if task_type == "regression":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)

    return Pipeline([("features", ColumnTransformer(transformers, remainder="drop")), ("model", estimator)])


def task_type_from_target(y: pd.Series, num_classes: int | None) -> str:
    if num_classes and num_classes > 1:
        return "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    return "classification" if y.nunique(dropna=True) <= 20 else "regression"


def evaluate(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, task_type: str) -> dict[str, float]:
    prediction = pipeline.predict(x)
    if task_type == "regression":
        return {
            "mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(mean_squared_error(y, prediction, squared=False)),
        }

    metrics = {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }
    if hasattr(pipeline, "predict_proba") and y.nunique(dropna=True) == 2:
        proba = pipeline.predict_proba(x)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def save_artifact(
    path: Path,
    pipeline: Pipeline,
    spec: FeatureSpec,
    task: str,
    phase: str,
    task_type: str,
    classes: list[Any],
    metrics: dict[str, dict[str, float]],
    examples: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "spec": asdict(spec),
        "task": task,
        "phase": phase,
        "task_type": task_type,
        "classes": classes,
        "metrics": metrics,
        "examples": examples.head(300).to_dict(orient="records"),
    }
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_from_record(artifact: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    spec = FeatureSpec(**artifact["spec"])
    frame = coerce_features(pd.DataFrame([record]), spec)
    pipeline = artifact["pipeline"]
    task_type = artifact["task_type"]

    if task_type == "regression":
        value = float(pipeline.predict(frame)[0])
        return {"prediction": value, "label": f"{value:.2f}", "scores": []}

    classes = [str(item) for item in artifact.get("classes", [])]
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(frame)[0], dtype=float)
        best_index = int(np.argmax(probabilities))
        scores = [
            {"class": classes[index] if index < len(classes) else str(index), "probability": float(prob)}
            for index, prob in enumerate(probabilities)
        ]
        return {"prediction": scores[best_index]["class"], "label": scores[best_index]["class"], "scores": scores}

    label = str(pipeline.predict(frame)[0])
    return {"prediction": label, "label": label, "scores": []}


def record_from_query(artifact: dict[str, Any], query: str) -> tuple[dict[str, Any], str]:
    spec = FeatureSpec(**artifact["spec"])
    normalized = query.strip().lower()
    examples = artifact.get("examples", [])

    for example in examples:
        haystack = " ".join(str(value).lower() for value in example.values())
        if normalized and normalized in haystack:
            return example, "matched TrialBench-like reference row"

    record = {column: "" for column in spec.categorical + spec.text}
    record.update({column: np.nan for column in spec.numeric})
    for column in spec.text:
        record[column] = query
    if spec.categorical:
        record[spec.categorical[0]] = "Unknown"
    return record, "query-only estimate"
