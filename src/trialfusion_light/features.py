from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


ID_HINTS = ("id", "nct", "url", "link", "path", "file")
TARGET_HINTS = (
    "label",
    "target",
    "y",
    "outcome",
    "status",
    "failure_reason",
    "dose",
    "duration",
    "mortality",
    "serious_adverse",
    "dropout",
)
TEXT_HINTS = (
    "title",
    "summary",
    "description",
    "criteria",
    "condition",
    "intervention",
    "drug",
    "mesh",
    "brief",
    "official",
    "keyword",
    "phase",
    "sponsor",
)


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    numeric: list[str]
    categorical: list[str]
    text: list[str]
    dropped: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric + self.categorical + self.text


def normalize_phase(phase: str | None) -> str | None:
    if not phase or phase.lower() == "all":
        return None
    return phase


def find_target_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Target column '{preferred}' was not found. Available columns: {list(df.columns)}")
        return preferred

    lower_map = {column.lower(): column for column in df.columns}
    exact = ("label", "target", "y")
    for name in exact:
        if name in lower_map:
            return lower_map[name]

    candidates: list[tuple[int, str]] = []
    for column in df.columns:
        lower = column.lower()
        if any(hint in lower for hint in TARGET_HINTS):
            unique_count = df[column].nunique(dropna=True)
            score = 0 if unique_count <= 20 else 1
            candidates.append((score, column))

    if candidates:
        return sorted(candidates)[0][1]

    compact_columns = [column for column in df.columns if df[column].nunique(dropna=True) <= 20]
    if compact_columns:
        return compact_columns[-1]

    return df.columns[-1]


def infer_feature_spec(df: pd.DataFrame, target: str) -> FeatureSpec:
    numeric: list[str] = []
    categorical: list[str] = []
    text: list[str] = []
    dropped: list[str] = []

    for column in df.columns:
        if column == target:
            continue
        lower = column.lower()
        if any(hint in lower for hint in ID_HINTS):
            dropped.append(column)
            continue

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            continue

        as_text = series.fillna("").astype(str)
        avg_len = as_text.map(len).mean()
        unique_ratio = as_text.nunique(dropna=True) / max(len(as_text), 1)
        if avg_len > 35 or any(hint in lower for hint in TEXT_HINTS):
            text.append(column)
        elif unique_ratio < 0.35:
            categorical.append(column)
        else:
            text.append(column)

    return FeatureSpec(target=target, numeric=numeric, categorical=categorical, text=text, dropped=dropped)


def join_text_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[available].fillna("").astype(str).agg(" ".join, axis=1)


def coerce_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in spec.numeric:
        result[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.NA
    for column in spec.categorical + spec.text:
        result[column] = frame[column].fillna("").astype(str) if column in frame else ""
    result["trial_text_blob"] = join_text_columns(result, spec.text)
    return result
