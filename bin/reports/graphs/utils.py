#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ORIGINAL = "original"
SUBSETS = ["improvised", "naturalistic"]
EXCLUDED_REPORT_METRICS = {"question_end_time"}
RELATIONSHIP_ORDER = [
    "coworkers",
    "dating/spouse/romantic_partner",
    "familiar-generic",
    "family-generic",
    "friends",
    "stranger",
]
DIALECT_LABELS = [
    "East Asia",
    "English",
    "Germanic",
    "Irish",
    "North America",
    "Northern Irish",
    "Oceania",
    "Other",
    "Romance",
    "Scottish",
    "Semitic",
    "Slavic",
    "South African",
    "Southeast Asia",
    "South Asia",
    "Welsh",
]
DIALECT_PROFILE_GROUPS = [
    ("East Asia", ["East Asia"]),
    ("British Isles", ["English", "Welsh", "Scottish", "Irish", "Northern Irish"]),
    ("Germanic", ["Germanic"]),
    ("North America", ["North America"]),
    ("Oceania", ["Oceania"]),
    ("Romance", ["Romance"]),
    ("Semitic", ["Semitic"]),
    ("Slavic", ["Slavic"]),
    ("South African", ["South African"]),
    ("Southeast Asia", ["Southeast Asia"]),
    ("South Asia", ["South Asia"]),
]
DIALECT_PROFILE_LABELS = [label for label, _ in DIALECT_PROFILE_GROUPS]


sns.set_theme(style="whitegrid", context="talk")


def add_common_graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ignore-features", nargs="*", default=[])


def all_systems(models: Iterable[str]) -> list[str]:
    systems = [ORIGINAL]
    for model in models:
        if model not in systems:
            systems.append(model)
    return systems


def system_palette(systems: list[str]) -> dict[str, str]:
    palette = sns.color_palette("tab10", n_colors=max(3, len(systems)))
    return {system: palette[idx] for idx, system in enumerate(systems)}


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return clean[:180] or "graph"


def result_file(results_root: Path, model: str, *parts: str) -> Path:
    path = results_root / model
    for part in parts:
        path /= part
    return path


def humanize_original_label(text: object) -> object:
    if not isinstance(text, str):
        return text
    return text.replace("ORIGINAL", "HUMAN").replace("Original", "Human").replace("original", "human")


def humanize_figure_text(fig) -> None:
    for artist in fig.findobj(lambda item: hasattr(item, "get_text") and hasattr(item, "set_text")):
        artist.set_text(humanize_original_label(artist.get_text()))


def save_figure(fig, output_path: Path, *, dpi: int = 180, bbox_inches: str = "tight") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    humanize_figure_text(fig)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Wrote {output_path}")


def remove_axis_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def available_base_metric_columns(results_root: Path, systems: list[str]) -> list[str]:
    columns = set()
    for subset in SUBSETS:
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "base_metrics.csv"))
            if frame is None:
                continue
            columns.update(
                column
                for column in frame.columns
                if column in {"CER", "WER", "UTMOS", "latency", "interrupted", "interruptions"}
                or column.startswith("CER_")
                or column.startswith("WER_")
            )

    ordered: list[str] = []
    for prefix in ["CER", "WER"]:
        specific = sorted(column for column in columns if column.startswith(f"{prefix}_"))
        if specific:
            ordered.extend(specific)
        elif prefix in columns:
            ordered.append(prefix)
    for metric in ["UTMOS", "latency", "interrupted", "interruptions"]:
        if metric in columns:
            ordered.append(metric)
    return ordered


def load_base_metric_values(results_root: Path, systems: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    metrics = list(metrics)
    for subset in SUBSETS:
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "base_metrics.csv"))
            if frame is None:
                continue
            for metric in metrics:
                if metric not in frame.columns:
                    continue
                values = numeric(frame[metric]).dropna()
                if values.empty:
                    continue
                display = metric
                if metric.startswith("WER"):
                    values = values * 100.0
                    display = metric.replace("_", " ") + " (%)"
                elif metric.startswith("CER"):
                    values = values * 100.0
                    display = metric.replace("_", " ") + " (%)"
                elif metric == "interrupted":
                    values = values / 1000.0
                    display = "interrupted time (s)"
                rows.append(pd.DataFrame({"value": values, "system": system, "subset": subset, "metric": display}))
    if not rows:
        return pd.DataFrame(columns=["value", "system", "subset", "metric"])
    return pd.concat(rows, ignore_index=True)


def remove_iqr_outliers(data: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    kept = []
    for _, group in data.groupby(group_cols, dropna=False):
        values = group[value_col].dropna()
        if len(values) < 4:
            kept.append(group)
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            kept.append(group)
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        kept.append(group[group[value_col].between(lower, upper) | group[value_col].isna()])
    return pd.concat(kept, ignore_index=True) if kept else data.iloc[0:0].copy()


def load_naturalness_values(results_root: Path, systems: list[str]) -> pd.DataFrame:
    frames = []
    for subset in SUBSETS:
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "naturalness_scores_normalized.csv"))
            if frame is None or "naturalness_logit" not in frame.columns:
                continue
            values = numeric(frame["naturalness_logit"]).dropna()
            if not values.empty:
                frames.append(pd.DataFrame({"naturalness_logit": values, "system": system, "subset": subset}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_naturalness_relationship_values(results_root: Path, systems: list[str]) -> pd.DataFrame:
    frames = []
    subset = "naturalistic"
    for system in systems:
        scores = safe_read_csv(result_file(results_root, system, "test", subset, "naturalness_scores_normalized.csv"))
        inference = safe_read_csv(result_file(results_root, system, "test", subset, "naturalness_inference_input.csv"))
        if scores is None or inference is None:
            continue
        if "naturalness_logit" not in scores.columns or "rel_detail" not in inference.columns:
            continue
        joined = scores.copy().reset_index(drop=True)
        rels = inference.copy().reset_index(drop=True)
        if "row_idx" in joined.columns and "row_idx" in rels.columns:
            joined["_row_idx"] = numeric(joined["row_idx"])
            rels["_row_idx"] = numeric(rels["row_idx"])
            joined = joined.merge(rels[["_row_idx", "rel_detail"]], on="_row_idx", how="left")
        else:
            joined["rel_detail"] = rels["rel_detail"].reindex(joined.index).values
        joined["relationship"] = joined["rel_detail"].astype(str).str.strip()
        joined["naturalness_logit"] = numeric(joined["naturalness_logit"])
        joined = joined[joined["relationship"].isin(RELATIONSHIP_ORDER)].dropna(subset=["naturalness_logit"])
        if not joined.empty:
            frames.append(pd.DataFrame({"naturalness_logit": joined["naturalness_logit"], "relationship": joined["relationship"], "system": system}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_score_vector(value) -> Optional[list[float]]:
    if isinstance(value, list):
        parsed = value
    else:
        if pd.isna(value):
            return None
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(DIALECT_LABELS):
        return None
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def grouped_dialect_scores(vector: list[float]) -> dict[str, float]:
    by_label = dict(zip(DIALECT_LABELS, vector))
    return {
        group: float(sum(by_label.get(label, 0.0) for label in labels))
        for group, labels in DIALECT_PROFILE_GROUPS
    }


def load_dialect_score_values(results_root: Path, systems: list[str]) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "dialect_id.csv"))
            if frame is None or "answer_dialect_score" not in frame.columns:
                continue
            for vector in frame["answer_dialect_score"].map(parse_score_vector).dropna():
                rows.extend(
                    {"subset": subset, "system": system, "dialect": dialect, "score": score}
                    for dialect, score in grouped_dialect_scores(vector).items()
                )
    return pd.DataFrame(rows)


def load_dialect_change_values(results_root: Path, systems: list[str]) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "dialect_id.csv"))
            if frame is None or not {"dialect", "answer_dialect"}.issubset(frame.columns):
                continue
            valid = frame.copy()
            valid["dialect"] = valid["dialect"].astype(str).str.strip()
            valid["answer_dialect"] = valid["answer_dialect"].astype(str).str.strip()
            valid = valid[valid["dialect"].isin(DIALECT_LABELS) & valid["answer_dialect"].isin(DIALECT_LABELS)]
            if valid.empty:
                continue
            changed = (valid["dialect"] != valid["answer_dialect"]).astype(float)
            rows.append(pd.DataFrame({"changed": changed, "system": system, "subset": subset}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_avd_values(results_root: Path, systems: list[str]) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "SER_AVD.csv"))
            if frame is None:
                continue
            for emotion in ["arousal", "dominance", "valence"]:
                question_col = f"question_{emotion}"
                answer_col = f"answer_{emotion}"
                if question_col not in frame.columns or answer_col not in frame.columns:
                    continue
                pairs = pd.DataFrame(
                    {
                        "question": (numeric(frame[question_col]) * 2.0 - 1.0).clip(-1.0, 1.0),
                        "answer": (numeric(frame[answer_col]) * 2.0 - 1.0).clip(-1.0, 1.0),
                        "system": system,
                        "subset": subset,
                        "emotion": emotion,
                    }
                ).dropna(subset=["question", "answer"])
                if not pairs.empty:
                    rows.append(pairs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def correlation_label(data: pd.DataFrame) -> float:
    if len(data) < 2:
        return np.nan
    x = data["question"].astype(float).to_numpy()
    y = data["answer"].astype(float).to_numpy()
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_stance_values(results_root: Path, systems: list[str]) -> pd.DataFrame:
    def stance_polarity(score: float) -> Optional[str]:
        if pd.isna(score) or score == 0:
            return None
        return "positive" if score > 0 else "negative"

    rows = []
    for system in systems:
        frame = safe_read_csv(result_file(results_root, system, "test", "improvised", "merged_stances.csv"))
        if frame is None or not {"question_index", "score_original", "score_llm"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["question_index"] = numeric(frame["question_index"])
        frame["score_original"] = numeric(frame["score_original"])
        frame["score_llm"] = numeric(frame["score_llm"])
        frame = frame.dropna(subset=["question_index"])
        frame["original_polarity"] = frame["score_original"].map(stance_polarity)
        frame["system_polarity"] = frame["score_llm"].map(stance_polarity)
        label_columns = [column for column in ["target_category", "stance_related_categories", "stance_question"] if column in frame.columns]
        if label_columns:
            frame["stance_label"] = ""
            for column in label_columns:
                values = frame[column].fillna("").astype(str).str.strip().str.replace("|", " / ", regex=False)
                frame["stance_label"] = frame["stance_label"].where(frame["stance_label"].astype(str).str.strip() != "", values)
        else:
            frame["stance_label"] = ""
        frame = frame.dropna(subset=["original_polarity", "system_polarity"])
        if not frame.empty:
            rows.append(frame[["question_index", "stance_label", "original_polarity", "system_polarity"]].assign(system=system))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_feature_values(results_root: Path, systems: list[str], ignored: Iterable[str], *, max_features: int = 18) -> pd.DataFrame:
    ignored = set(ignored) | EXCLUDED_REPORT_METRICS
    feature_counts: dict[str, int] = {}
    summaries = []
    for system in systems:
        summary = safe_read_csv(result_file(results_root, system, "distrib_baselines_summary.csv"))
        if summary is None or "feature" not in summary.columns:
            continue
        summaries.append(summary)
        for feature in summary["feature"].dropna().astype(str):
            if feature not in ignored:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    if not feature_counts and summaries:
        for system in systems:
            for subset in SUBSETS:
                frame = safe_read_csv(result_file(results_root, system, "test", subset, "distrib_baselines_features.csv"))
                if frame is None:
                    continue
                for feature in frame.columns:
                    if feature not in ignored and pd.api.types.is_numeric_dtype(frame[feature]):
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    features = sorted(feature_counts, key=lambda feature: (-feature_counts[feature], feature))[:max_features]
    rows = []
    for subset in SUBSETS:
        feature_frames = {}
        mins: dict[str, float] = {}
        maxes: dict[str, float] = {}
        for system in systems:
            frame = safe_read_csv(result_file(results_root, system, "test", subset, "distrib_baselines_features.csv"))
            if frame is None:
                continue
            feature_frames[system] = frame
            for feature in features:
                if feature not in frame.columns:
                    continue
                values = numeric(frame[feature]).dropna()
                if values.empty:
                    continue
                mins[feature] = min(mins.get(feature, values.min()), values.min())
                maxes[feature] = max(maxes.get(feature, values.max()), values.max())
        for system, frame in feature_frames.items():
            for feature in features:
                if feature not in frame.columns or feature not in mins:
                    continue
                values = numeric(frame[feature]).dropna()
                if values.empty:
                    continue
                denom = maxes[feature] - mins[feature]
                normalized = pd.Series(np.full(len(values), 0.5), index=values.index) if denom == 0 or pd.isna(denom) else (values - mins[feature]) / denom
                rows.append(pd.DataFrame({"value": normalized, "system": system, "subset": subset, "feature": feature}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
