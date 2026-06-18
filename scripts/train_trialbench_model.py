#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from trialfusion_light.features import coerce_features, find_target_column, infer_feature_spec, normalize_phase
from trialfusion_light.model import evaluate, make_pipeline, save_artifact, task_type_from_target


TASK_FOLDERS = {
    "outcome": ("trial-approval-forecasting", "outcome"),
    "mortality_rate": ("mortality-event-prediction", "mortality_rate"),
    "mortality_rate_yn": ("mortality-event-prediction", "mortality_rate_yn"),
    "serious_adverse_rate": ("serious-adverse-event-forecasting", "serious_adverse_rate"),
    "serious_adverse_rate_yn": ("serious-adverse-event-forecasting", "serious_adverse_rate_yn"),
    "patient_dropout_rate": ("patient-dropout-event-forecasting", "patient_dropout_rate"),
    "patient_dropout_rate_yn": ("patient-dropout-event-forecasting", "patient_dropout_rate_yn"),
    "duration": ("trial-duration-forecasting", "duration"),
    "failure_reason": ("trial-failure-reason-identification", "failure_reason"),
    "dose": ("drug-dose-prediction", "dose"),
    "dose_cls": ("drug-dose-prediction", "Avg"),
}


def phase_folder(phase: str) -> str:
    compact = phase.replace(" ", "")
    if compact in {"Phase1", "Phase2", "Phase3", "Phase4"}:
        return compact
    if compact in {"1", "2", "3", "4"}:
        return f"Phase{compact}"
    return phase


def download_label_file(folder: str, phase: str, name: str, output: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/ML2Health/ML2ClinicalTrials/main/"
        f"Trialbench/data/{folder}/{phase}/{name}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output.write_bytes(response.content)


def load_csv_task(task: str, phase: str, download: bool):
    if task not in TASK_FOLDERS:
        raise ValueError(f"No CSV fallback is configured for task {task!r}.")

    folder, default_target = TASK_FOLDERS[task]
    phases = ["Phase1", "Phase2", "Phase3", "Phase4"] if phase.lower() == "all" else [phase_folder(phase)]

    train_parts = []
    test_parts = []
    for current_phase in phases:
        phase_dir = Path("data") / folder / current_phase
        for name in ("train_y.csv", "test_y.csv"):
            target_path = phase_dir / name
            if download and not target_path.exists():
                print(f"Downloading missing label file {folder}/{current_phase}/{name} from GitHub.")
                download_label_file(folder, current_phase, name, target_path)

        train_x = pd.read_csv(phase_dir / "train_x.csv")
        train_y = pd.read_csv(phase_dir / "train_y.csv")
        test_x = pd.read_csv(phase_dir / "test_x.csv")
        test_y = pd.read_csv(phase_dir / "test_y.csv")

        if "Unnamed: 0" in train_x.columns:
            train_x = train_x.rename(columns={"Unnamed: 0": "nctid"})
            test_x = test_x.rename(columns={"Unnamed: 0": "nctid"})

        target = default_target if default_target in train_y.columns else train_y.columns[-1]
        train_x[target] = train_y[target].values
        test_x[target] = test_y[target].values
        train_parts.append(train_x)
        test_parts.append(test_x)

    train_full = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df, valid_df = train_test_split(
        train_full,
        test_size=0.2,
        random_state=42,
        stratify=train_full[default_target] if default_target in train_full and train_full[default_target].nunique() <= 20 else None,
    )
    target = default_target if default_target in train_full.columns else train_full.columns[-1]
    num_classes = int(train_full[target].nunique(dropna=True)) if train_full[target].nunique(dropna=True) <= 50 else None
    return train_df, valid_df, test_df, num_classes, None


def load_trialbench(task: str, phase: str, data_dir: str, download: bool):
    import trialbench

    if data_dir != "data":
        print(
            "Note: the installed trialbench package reads task datasets from ./data. "
            f"Ignoring --data-dir={data_dir!r} for compatibility."
        )
    if download:
        print("TrialBench will download the requested task dataset if it is not already in ./data.")

    try:
        return trialbench.function.load_data(task, normalize_phase(phase), data_format="df")
    except (FileNotFoundError, KeyError) as exc:
        print(f"TrialBench package loader could not read this task directly ({exc}). Falling back to CSV loader.")
        return load_csv_task(task, phase, download)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight TrialFusion-style model on TrialBench.")
    parser.add_argument("--task", default="outcome", help="TrialBench task, e.g. outcome, mortality_rate_yn, dose_cls.")
    parser.add_argument("--phase", default="All", help="All, Phase 1, Phase 2, Phase 3, or Phase 4.")
    parser.add_argument("--target-column", default=None, help="Override target column if auto-detection is wrong.")
    parser.add_argument("--data-dir", default="data", help="TrialBench currently reads task datasets from ./data.")
    parser.add_argument("--model-out", default="models/trialfusion_light.joblib", help="Output model artifact path.")
    parser.add_argument("--download", action="store_true", help="Ask trialbench to download datasets before loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, valid_df, test_df, num_classes, _tabular_input_dim = load_trialbench(
        args.task, args.phase, args.data_dir, args.download
    )

    train_df = pd.DataFrame(train_df).reset_index(drop=True)
    valid_df = pd.DataFrame(valid_df).reset_index(drop=True)
    test_df = pd.DataFrame(test_df).reset_index(drop=True)

    known_target = TASK_FOLDERS.get(args.task, (None, None))[1]
    preferred_target = args.target_column or (known_target if known_target in train_df.columns else None)
    target = find_target_column(train_df, preferred_target)
    spec = infer_feature_spec(train_df, target)
    task_type = task_type_from_target(train_df[target], num_classes)

    x_train = coerce_features(train_df, spec)
    y_train = train_df[target]
    x_valid = coerce_features(valid_df, spec)
    y_valid = valid_df[target]
    x_test = coerce_features(test_df, spec)
    y_test = test_df[target]

    pipeline = make_pipeline(spec, task_type)
    pipeline.fit(x_train, y_train)

    metrics = {
        "valid": evaluate(pipeline, x_valid, y_valid, task_type),
        "test": evaluate(pipeline, x_test, y_test, task_type),
    }
    classes = list(getattr(pipeline.named_steps["model"], "classes_", []))

    examples = pd.concat([test_df, valid_df], ignore_index=True)
    save_artifact(
        Path(args.model_out),
        pipeline,
        spec,
        args.task,
        args.phase,
        task_type,
        classes,
        metrics,
        examples,
    )

    print(json.dumps({"model": args.model_out, "target": target, "task_type": task_type, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
