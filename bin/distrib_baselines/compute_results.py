#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score


ORIGINAL_MODEL = "original"
LABEL_ORIGINAL = 1
LABEL_MODEL = 0

ID_COLUMNS = {
    "orig_id",
    "vendor_id",
    "session_id",
    "conversation_id",
    "audio_path",
    "speakers",
    "relationship",
    "relationship_detail",
    "extraction_status",
}

STATUS_SUBSTRINGS = (
    "status",
    "source",
)


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature CSV: {path}")
    return pd.read_csv(path)


def feature_path(results_root: Path, model: str, split: str, subset: str) -> Path:
    return results_root / model / split / subset / "distrib_baselines_features.csv"


def output_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "distrib_baselines_feature_scores.csv"


def summary_path(results_root: Path, model: str) -> Path:
    return results_root / model / "distrib_baselines_summary.csv"


def is_candidate_feature(column: str) -> bool:
    if column in ID_COLUMNS:
        return False
    lower = column.lower()
    return not any(part in lower for part in STATUS_SUBSTRINGS)


def numeric_feature_columns(frames: Iterable[pd.DataFrame]) -> List[str]:
    common: Optional[set[str]] = None
    for frame in frames:
        cols = set(frame.columns)
        common = cols if common is None else common & cols
    if not common:
        return []

    features: List[str] = []
    for column in sorted(common):
        if not is_candidate_feature(column):
            continue
        usable = False
        for frame in frames:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                usable = True
                break
        if usable:
            features.append(column)
    return features


def finite_series(frame: pd.DataFrame, feature: str) -> pd.Series:
    values = pd.to_numeric(frame[feature], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def train_feature_lda(
    original_dev: pd.DataFrame,
    model_dev: pd.DataFrame,
    feature: str,
) -> Optional[Tuple[LinearDiscriminantAnalysis, float, float]]:
    x_original = finite_series(original_dev, feature).dropna().to_numpy(dtype=float)
    x_model = finite_series(model_dev, feature).dropna().to_numpy(dtype=float)

    if len(x_original) < 2 or len(x_model) < 2:
        return None

    x_train = np.concatenate([x_original, x_model]).reshape(-1, 1)
    y_train = np.concatenate(
        [
            np.full(len(x_original), LABEL_ORIGINAL),
            np.full(len(x_model), LABEL_MODEL),
        ]
    )

    if np.nanstd(x_train) == 0:
        return None

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    return lda, float(np.nanmean(x_original)), float(np.nanmean(x_model))


def probability_original(lda: LinearDiscriminantAnalysis, values: pd.Series) -> np.ndarray:
    x = values.to_numpy(dtype=float).reshape(-1, 1)
    probs = lda.predict_proba(x)
    original_idx = int(np.where(lda.classes_ == LABEL_ORIGINAL)[0][0])
    return probs[:, original_idx]


def score_model_utterances(
    model_test: pd.DataFrame,
    feature_models: Dict[str, Tuple[LinearDiscriminantAnalysis, float, float]],
) -> pd.DataFrame:
    out = pd.DataFrame()
    for column in ["orig_id", "audio_path"]:
        if column in model_test.columns:
            out[column] = model_test[column]

    per_feature_score_cols: List[str] = []
    for feature, (lda, original_mean, model_mean) in feature_models.items():
        values = finite_series(model_test, feature)
        valid = values.notna()

        score_col = f"{feature}__naturalness_score"
        distance_col = f"{feature}__distance_to_dev_original_mean"
        model_distance_col = f"{feature}__distance_to_dev_model_mean"

        out[score_col] = np.nan
        if valid.any():
            out.loc[valid, score_col] = probability_original(lda, values.loc[valid])
        out[distance_col] = values - original_mean
        out[model_distance_col] = values - model_mean
        per_feature_score_cols.append(score_col)

    if per_feature_score_cols:
        out["average_naturalness_score"] = out[per_feature_score_cols].mean(axis=1, skipna=True)
    else:
        out["average_naturalness_score"] = np.nan

    return out


def evaluate_feature(
    original_test: pd.DataFrame,
    model_test: pd.DataFrame,
    feature: str,
    lda: LinearDiscriminantAnalysis,
) -> dict:
    x_original = finite_series(original_test, feature)
    x_model = finite_series(model_test, feature)

    original_valid = x_original.notna()
    model_valid = x_model.notna()

    y_true = np.concatenate(
        [
            np.full(int(original_valid.sum()), LABEL_ORIGINAL),
            np.full(int(model_valid.sum()), LABEL_MODEL),
        ]
    )
    x_values = np.concatenate(
        [
            x_original.loc[original_valid].to_numpy(dtype=float),
            x_model.loc[model_valid].to_numpy(dtype=float),
        ]
    )

    if len(np.unique(y_true)) < 2 or len(x_values) == 0:
        return {
            "feature": feature,
            "accuracy": np.nan,
            "auroc": np.nan,
            "n_original_test": int(original_valid.sum()),
            "n_model_test": int(model_valid.sum()),
        }

    x_matrix = x_values.reshape(-1, 1)
    y_pred = lda.predict(x_matrix)
    p_original = probability_original(lda, pd.Series(x_values))

    try:
        auroc = roc_auc_score(y_true, p_original)
    except ValueError:
        auroc = np.nan

    return {
        "feature": feature,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(auroc),
        "n_original_test": int(original_valid.sum()),
        "n_model_test": int(model_valid.sum()),
    }


def process_subset(results_root: Path, model: str, subset: str) -> Tuple[pd.DataFrame, Path]:
    original_dev = read_features(feature_path(results_root, ORIGINAL_MODEL, "dev", subset))
    model_dev = read_features(feature_path(results_root, model, "dev", subset))
    original_test = read_features(feature_path(results_root, ORIGINAL_MODEL, "test", subset))
    model_test = read_features(feature_path(results_root, model, "test", subset))

    features = numeric_feature_columns([original_dev, model_dev, original_test, model_test])
    feature_models: Dict[str, Tuple[LinearDiscriminantAnalysis, float, float]] = {}
    skipped = []

    for feature in features:
        trained = train_feature_lda(original_dev, model_dev, feature)
        if trained is None:
            skipped.append(feature)
            continue
        feature_models[feature] = trained

    scored = score_model_utterances(model_test, feature_models)
    out_path = output_path(results_root, model, subset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)

    summary_rows = []
    for feature, (lda, original_mean, model_mean) in feature_models.items():
        row = evaluate_feature(original_test, model_test, feature, lda)
        row.update(
            {
                "subset": subset,
                "model": model,
                "dev_original_mean": original_mean,
                "dev_model_mean": model_mean,
                "dev_mean_difference_original_minus_model": original_mean - model_mean,
                "output_csv": str(out_path),
            }
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if skipped:
        skipped_summary = pd.DataFrame(
            [
                {
                    "subset": subset,
                    "model": model,
                    "feature": feature,
                    "accuracy": np.nan,
                    "auroc": np.nan,
                    "n_original_test": 0,
                    "n_model_test": 0,
                    "dev_original_mean": np.nan,
                    "dev_model_mean": np.nan,
                    "dev_mean_difference_original_minus_model": np.nan,
                    "output_csv": str(out_path),
                    "skip_reason": "insufficient_nonconstant_dev_values",
                }
                for feature in skipped
            ]
        )
        summary = pd.concat([summary, skipped_summary], ignore_index=True)

    return summary, out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subsets", nargs="+", default=["improvised", "naturalistic"])
    args = parser.parse_args()

    all_summaries = []
    for subset in args.subsets:
        print(f"Scoring explainable-feature baselines for model={args.model} subset={subset}")
        summary, out_path = process_subset(args.results_root, args.model, subset)
        print(f"Wrote utterance feature scores: {out_path}")
        all_summaries.append(summary)

    full_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    summary_csv = summary_path(args.results_root, args.model)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    full_summary.to_csv(summary_csv, index=False)

    if not full_summary.empty:
        printable = full_summary[
            ["subset", "feature", "accuracy", "auroc", "n_original_test", "n_model_test"]
        ].sort_values(["subset", "feature"])
        print(printable.to_string(index=False))
    print(f"Wrote summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
