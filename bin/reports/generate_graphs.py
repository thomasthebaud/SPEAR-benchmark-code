#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ORIGINAL = "original"
SUBSETS = ["improvised", "naturalistic"]
DATASET_PALETTE = {"original": "#4C72B0", "model": "#DD8452"}
EXCLUDED_REPORT_METRICS = {"question_end_time"}
REPORTED_EXPLAINABLE_FEATURES = {"total_duration_s", "voiced_duration_s", "voiced_ratio"}
TURNTAKING_SURPRISAL_FILE = "turntaking.group4-dualturn-full-all6-fvad256.csv"
TURNTAKING_SURPRISAL_METRICS = [
    "mean_nll",
    "tail_nll",
    "dialog_nll",
    "naturalness_score",
]
EXPLAINABLE_FEATURE_GROUPS = {
    "pitch": {
        "title": "Pitch-Based Explainable Feature Distributions",
        "features": {
            "f0_mean_raw",
            "f0_median_raw",
            "f0_std_raw",
            "f0_min_raw",
            "f0_max_raw",
            "f0_range_raw",
            "f0_p10",
            "f0_p90",
            "f0_range_p10_p90",
            "f0_mean_p10_p90",
            "f0_std_p10_p90",
            "f0_p25",
            "f0_p75",
            "f0_range_p25_p75",
            "f0_mean_p25_p75",
            "f0_std_p25_p75",
        },
        "prefixes": ("f0_",),
    },
    "lexical": {
        "title": "Lexical Explainable Feature Distributions",
        "features": {
            "total_words",
            "unique_words",
            "mean_asr_confidence",
            "low_conf_flag",
            "content_word_count",
            "function_word_count",
            "lexical_density",
            "ttr",
            "mattr_small",
            "mattr_large",
            "mattr_ratio",
            "mtld",
            "hapax_ratio",
            "lexical_entropy",
            "backchannel_ratio",
            "discourse_marker_ratio",
        },
        "prefixes": ("lexical_",),
    },
    "temporal": {
        "title": "Temporal Explainable Feature Distributions",
        "features": {
            "total_duration_s",
            "voiced_duration_s",
            "voiced_ratio",
            "n_voiced_frames",
            "speech_active_time_s",
            "pause_count",
            "pause_total_duration_s",
            "pause_mean_duration_s",
            "pause_ratio",
            "speech_rate_wps",
            "speech_rate_wpm",
            "articulation_rate_wps",
            "articulation_rate_wpm",
        },
        "prefixes": ("temporal_",),
    },
}
NON_EXPLAINABLE_COLUMNS = {
    "orig_id",
    "vendor_id",
    "session_id",
    "conversation_id",
    "audio_path",
    "speakers",
    "relationship",
    "relationship_detail",
    "total_duration",
    "question_end_time",
    "extraction_status",
    "f0_status",
    "lexical_status",
    "lexical_status_reason",
    "temporal_alignment_source",
    "temporal_status",
    "answered_speaker",
}
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
    ("British Isles", ["English", "Welsh", "Scottish","Irish", "Northern Irish"]),
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


def grouped_dialect_scores(vector: list[float]) -> dict[str, float]:
    by_label = dict(zip(DIALECT_LABELS, vector))
    return {
        group: float(sum(by_label.get(label, 0.0) for label in labels))
        for group, labels in DIALECT_PROFILE_GROUPS
    }

sns.set_theme(style="whitegrid", context="talk")


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
    return clean[:180] or "feature"


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


def is_reported_explainable_feature(feature: str) -> bool:
    return feature in REPORTED_EXPLAINABLE_FEATURES or feature.startswith("f0_p")


def explainable_feature_group(feature: str) -> Optional[str]:
    for group_name, spec in EXPLAINABLE_FEATURE_GROUPS.items():
        if feature in spec["features"] or any(feature.startswith(prefix) for prefix in spec["prefixes"]):
            return group_name
    return None


def explainable_feature_file(results_root: Path, model: str, subset: str) -> Path:
    return result_file(results_root, model, "test", subset, "distrib_baselines_features_normalized.csv")


def turntaking_surprisal_file(results_root: Path, model: str, subset: str) -> Path:
    return result_file(results_root, model, "test", subset, TURNTAKING_SURPRISAL_FILE)


def explainable_question_feature_file(results_root: Path, subset: str) -> Path:
    return result_file(results_root, ORIGINAL, "test", subset, "distrib_baselines_features_normalized_q.csv")


F0_BOX_FEATURES = [
    "f0_min_raw",
    "f0_p10",
    "f0_p25",
    "f0_median_raw",
    "f0_p75",
    "f0_p90",
    "f0_max_raw",
]
F0_BOX_LABELS = {
    "f0_min_raw": "min",
    "f0_p10": "10%",
    "f0_p25": "25%",
    "f0_median_raw": "50%",
    "f0_p75": "75%",
    "f0_p90": "90%",
    "f0_max_raw": "max",
}


def available_explainable_features(results_root: Path, model: str, subset: str) -> List[str]:
    features = set()
    for label_model in [ORIGINAL, model]:
        frame = safe_read_csv(explainable_feature_file(results_root, label_model, subset))
        if frame is None:
            continue
        features.update(
            str(column)
            for column in frame.columns
            if is_reported_explainable_feature(str(column)) and str(column) not in EXCLUDED_REPORT_METRICS
        )
    return sorted(features)


def numeric_explainable_columns(frame: pd.DataFrame) -> set[str]:
    features = set()
    for column in frame.columns:
        feature = str(column)
        if feature in EXCLUDED_REPORT_METRICS or feature in NON_EXPLAINABLE_COLUMNS:
            continue
        if numeric(frame[column]).notna().any():
            features.add(feature)
    return features


def available_all_explainable_features(results_root: Path, model: str, subset: str) -> List[str]:
    feature_sets = []
    for label_model in [ORIGINAL, model]:
        frame = safe_read_csv(explainable_feature_file(results_root, label_model, subset))
        if frame is None:
            continue
        feature_sets.append(numeric_explainable_columns(frame))
    if not feature_sets:
        return []
    return sorted(set.intersection(*feature_sets))


def load_feature_values(results_root: Path, model: str, subset: str, feature: str) -> Optional[pd.DataFrame]:
    frames = []
    for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
        path = explainable_feature_file(results_root, label_model, subset)
        frame = safe_read_csv(path)
        if frame is None or feature not in frame.columns:
            continue
        values = numeric(frame[feature]).dropna()
        if values.empty:
            continue
        frames.append(pd.DataFrame({"value": values, "dataset": label, "subset": subset}))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def available_base_metric_columns(results_root: Path, model: str) -> List[str]:
    columns = set()
    for subset in SUBSETS:
        for label_model in [ORIGINAL, model]:
            frame = safe_read_csv(result_file(results_root, label_model, "test", subset, "base_metrics.csv"))
            if frame is None:
                continue
            columns.update(
                column
                for column in frame.columns
                if column in {"CER", "WER", "UTMOS", "latency", "interrupted"}
                or column.startswith("CER_")
                or column.startswith("WER_")
            )
    ordered = []
    for prefix in ["CER", "WER"]:
        specific = sorted(column for column in columns if column.startswith(f"{prefix}_"))
        if specific:
            ordered.extend(specific)
        elif prefix in columns:
            ordered.append(prefix)
    for metric in ["UTMOS", "latency", "interrupted"]:
        if metric in columns:
            ordered.append(metric)
    return ordered


def load_base_metric_values(results_root: Path, model: str, metric: str) -> Optional[pd.DataFrame]:
    frames = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(result_file(results_root, label_model, "test", subset, "base_metrics.csv"))
            if frame is None or metric not in frame.columns:
                continue
            values = numeric(frame[metric]).dropna()
            if values.empty:
                continue
            display_metric = "interrupted time (s)" if metric == "interrupted" else metric
            if metric == "interrupted":
                values = values / 1000.0
            frames.append(pd.DataFrame({"value": values, "dataset": label, "subset": subset, "metric": display_metric}))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_asr_error_values(results_root: Path, model: str, error_type: str) -> Optional[pd.DataFrame]:
    frames = []
    available_columns = set(available_base_metric_columns(results_root, model))
    columns = sorted(column for column in available_columns if column.startswith(f"{error_type}_"))
    if not columns and error_type in available_columns:
        columns = [error_type]

    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(result_file(results_root, label_model, "test", subset, "base_metrics.csv"))
            if frame is None:
                continue
            for column in columns:
                if column not in frame.columns:
                    continue
                values = numeric(frame[column]).dropna() * 100.0
                if values.empty:
                    continue
                asr_system = "default" if column == error_type else column.split("_", 1)[1]
                frames.append(pd.DataFrame({"value": values, "dataset": label, "subset": subset, "metric": error_type, "asr_system": asr_system}))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def remove_iqr_outliers(data: pd.DataFrame, group_cols: List[str], value_col: str = "value") -> pd.DataFrame:
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
    if not kept:
        return data.iloc[0:0].copy()
    filtered = pd.concat(kept, ignore_index=True)
    removed = len(data) - len(filtered)
    if removed:
        print(f"Removed {removed} basic metric outlier rows before plotting.")
    return filtered


def graph_asr_error_histogram(results_root: Path, model: str, error_type: str, output_path: Path) -> None:
    data = load_asr_error_values(results_root, model, error_type)
    if data is None or data.empty:
        return
    data = remove_iqr_outliers(data, ["subset", "metric", "asr_system"])
    if data.empty:
        return
    panels = []
    for subset in SUBSETS:
        subset_data = data[data["subset"] == subset]
        for asr_system in sorted(subset_data["asr_system"].dropna().astype(str).unique()):
            panels.append((subset, asr_system))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, max(4.0, 3.2 * len(panels))), sharex=True, squeeze=False)
    for ax, (subset, asr_system) in zip(axes[:, 0], panels):
        sub = data[(data["subset"] == subset) & (data["asr_system"].astype(str) == asr_system)]
        sns.histplot(
            data=sub,
            x="value",
            hue="dataset",
            hue_order=["original", "model"],
            bins=30,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.30,
            kde=True,
            palette=DATASET_PALETTE,
            ax=ax,
        )
        ax.set_title(subset.title())
        ax.set_xlabel(f"{error_type} (%)")
        ax.set_ylabel(asr_system)
    fig.suptitle(f"{error_type}: Original vs Model Across ASR Systems", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def graph_basic_metric_histograms(results_root: Path, model: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for error_type in ["CER", "WER"]:
        graph_asr_error_histogram(results_root, model, error_type, output_dir / f"{error_type}.png")

    for metric in [metric for metric in available_base_metric_columns(results_root, model) if not metric.startswith(("CER", "WER"))]:
        data = load_base_metric_values(results_root, model, metric)
        if data is None or data.empty:
            continue
        data = remove_iqr_outliers(data, ["subset", "metric"])
        if data.empty:
            continue
        subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
        fig, axes = plt.subplots(len(subsets), 1, figsize=(10, 5 * len(subsets)), sharex=True, sharey=True, squeeze=False)
        for ax, subset in zip(axes[:, 0], subsets):
            sub = data[data["subset"] == subset]
            sns.histplot(
                data=sub,
                x="value",
                hue="dataset",
                bins=30,
                stat="density",
                common_norm=False,
                element="step",
                fill=True,
                alpha=0.35,
                kde=True,
                palette=DATASET_PALETTE,
                ax=ax,
            )
            ax.set_title(subset.title())
            ax.set_xlabel(data["metric"].iloc[0] if "metric" in data else metric)
            ax.set_ylabel("Density")
        fig.suptitle(f"{metric}: Original vs Model", y=1.03)
        fig.tight_layout()
        out_path = output_dir / f"{sanitize_filename(metric)}.png"
        humanize_figure_text(fig)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote basic metric histograms to {output_dir}")


def graph_basic_metrics_histograms(results_root: Path, model: str, output_path: Path) -> None:
    frames = []
    plot_order = []
    for error_type in ["CER", "WER"]:
        data = load_asr_error_values(results_root, model, error_type)
        if data is not None:
            data = data.copy()
            data["plot_metric"] = data["metric"].astype(str) + " (%): " + data["asr_system"].astype(str)
            frames.append(data)
            for asr_system in sorted(data["asr_system"].dropna().astype(str).unique()):
                plot_order.append(f"{error_type} (%): {asr_system}")
    for metric in [metric for metric in available_base_metric_columns(results_root, model) if not metric.startswith(("CER", "WER"))]:
        data = load_base_metric_values(results_root, model, metric)
        if data is not None:
            data = data.copy()
            plot_metric = str(data["metric"].iloc[0]) if "metric" in data and not data.empty else metric
            data["plot_metric"] = plot_metric
            frames.append(data)
            plot_order.append(plot_metric)
    if not frames:
        print("[WARN] No base metrics found; skipping basic_metrics.png.")
        return
    data = pd.concat(frames, ignore_index=True)
    group_cols = ["subset", "plot_metric"]
    data = remove_iqr_outliers(data, group_cols)
    if data.empty:
        print("[WARN] All base metric rows were filtered as outliers; skipping basic_metrics.png.")
        return
    available_subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    plot_metrics = [metric for metric in plot_order if metric in set(data["plot_metric"])]
    fig, axes = plt.subplots(len(plot_metrics), len(available_subsets), figsize=(8.5 * len(available_subsets), 3.4 * len(plot_metrics)), squeeze=False)
    for row_idx, plot_metric in enumerate(plot_metrics):
        for col_idx, subset in enumerate(available_subsets):
            ax = axes[row_idx, col_idx]
            sub = data[(data["subset"] == subset) & (data["plot_metric"] == plot_metric)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sns.histplot(
                data=sub,
                x="value",
                hue="dataset",
                hue_order=["original", "model"],
                bins=30,
                stat="density",
                common_norm=False,
                element="step",
                fill=True,
                alpha=0.30,
                kde=True,
                palette=DATASET_PALETTE,
                ax=ax,
            )
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.set_title(subset.title() if row_idx == 0 else "")
            if " (%): " in plot_metric:
                metric_label, system_label = plot_metric.split(" (%): ", 1)
                ax.set_xlabel(f"{metric_label} (%)")
                ax.set_ylabel(system_label if col_idx == 0 else "")
            else:
                ax.set_xlabel(plot_metric)
                ax.set_ylabel("Density")
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

    legend_handles = [Patch(facecolor=DATASET_PALETTE[label], label=label) for label in ["original", "model"]]
    fig.legend(legend_handles, ["original", "model"], title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Intelligibility and Interruptions histograms: Original vs Model", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_path)

def f0_boxplot_rows(frame: pd.DataFrame, label: str, subset: str) -> list[pd.DataFrame]:
    rows = []
    for feature in F0_BOX_FEATURES:
        if feature not in frame.columns or feature in EXCLUDED_REPORT_METRICS:
            continue
        values = numeric(frame[feature]).dropna()
        if values.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "label": label,
                    "subset": subset,
                    "feature": feature,
                    "feature_label": F0_BOX_LABELS.get(feature, feature),
                    "value": values.to_numpy(dtype=float),
                }
            )
        )
    return rows


def load_f0_question_answer_values(results_root: Path, model: str) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        question = safe_read_csv(explainable_question_feature_file(results_root, subset))
        if question is not None:
            rows.extend(f0_boxplot_rows(question, "questions", subset))

        original = safe_read_csv(explainable_feature_file(results_root, ORIGINAL, subset))
        if original is not None:
            rows.extend(f0_boxplot_rows(original, "original answers", subset))

        model_frame = safe_read_csv(explainable_feature_file(results_root, model, subset))
        if model_frame is not None:
            rows.extend(f0_boxplot_rows(model_frame, "model answers", subset))

    if not rows:
        return pd.DataFrame(columns=["label", "subset", "feature", "feature_label", "value"])
    return pd.concat(rows, ignore_index=True)


def graph_f0_question_answer_boxplot(results_root: Path, model: str, output_path: Path) -> None:
    data = load_f0_question_answer_values(results_root, model)
    if data.empty:
        print("[WARN] Missing f0/voiced-ratio explainable rows; skipping f0_question_answer_boxplot.png.")
        return

    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    feature_order = [F0_BOX_LABELS[feature] for feature in F0_BOX_FEATURES if F0_BOX_LABELS[feature] in set(data["feature_label"])]
    label_order = [label for label in ["questions", "original answers", "model answers"] if label in set(data["label"])]
    palette = {"questions": "#6B6B6B", "original answers": DATASET_PALETTE["original"], "model answers": DATASET_PALETTE["model"]}

    fig, axes = plt.subplots(1, len(subsets), figsize=(9.2 * len(subsets), 6.2), sharey=False, squeeze=False)
    legend_handles = None
    legend_labels = None
    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset]
        sns.boxplot(
            data=sub,
            x="feature_label",
            y="value",
            hue="label",
            order=feature_order,
            hue_order=label_order,
            palette=palette,
            showfliers=False,
            linewidth=1.0,
            width=0.78,
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles and legend_handles is None:
            legend_handles, legend_labels = handles, labels
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xlabel("f0 feature / voiced ratio")
        ax.set_ylabel("Normalized value")
        ax.grid(True, axis="y", alpha=0.35)
        ax.tick_params(axis="x", rotation=20)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Audio", loc="upper center", ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.03), frameon=True)
    fig.suptitle("Questions, Original Answers, and Model Answers: f0 Profile and Voiced Ratio", y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)


def graph_feature_histograms(results_root: Path, model: str, output_dir: Path) -> None:
    features = sorted({feature for subset in SUBSETS for feature in available_explainable_features(results_root, model, subset)})
    if not features:
        print("[WARN] No normalized explainable features found; skipping per-feature histograms.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        subset_frames: Dict[str, pd.DataFrame] = {}
        for subset in SUBSETS:
            values = load_feature_values(results_root, model, subset, feature)
            if values is not None:
                subset_frames[subset] = values

        if not subset_frames:
            continue

        n = len(subset_frames)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(4, 4 * n)), squeeze=False)
        for ax, (subset, values) in zip(axes[:, 0], subset_frames.items()):
            sns.histplot(
                data=values,
                x="value",
                hue="dataset",
                bins=40,
                stat="density",
                common_norm=False,
                element="step",
                fill=True,
                alpha=0.35,
                ax=ax,
            )
            ax.set_title(f"{feature} - {subset}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Density")
        fig.tight_layout()
        out_path = output_dir / f"{sanitize_filename(feature)}.png"
        humanize_figure_text(fig)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote per-feature histograms to {output_dir}")


def graph_stances(results_root: Path, model: str, output_path: Path) -> None:
    def stance_polarity(score: float) -> Optional[str]:
        if pd.isna(score) or score == 0:
            return None
        return "positive" if score > 0 else "negative"

    def label_for_group(qidx: int, group: pd.DataFrame) -> str:
        for column in ["target_category", "stance_related_categories", "stance_question"]:
            if column not in group:
                continue
            values = group[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            if not values.empty:
                return values.iloc[0].replace("|", " / ")
        return f"Q{qidx}"

    frame = safe_read_csv(result_file(results_root, model, "test", "improvised", "merged_stances.csv"))
    if frame is None or not {"question_index", "score_original", "score_llm"}.issubset(frame.columns):
        print("[WARN] Missing STANCE merged scores; skipping stances.png.")
        return

    frame = frame.copy()
    frame["question_index"] = numeric(frame["question_index"])
    frame["score_original"] = numeric(frame["score_original"])
    frame["score_llm"] = numeric(frame["score_llm"])
    frame = frame.dropna(subset=["question_index"])
    frame["question_index"] = frame["question_index"].astype(int)
    frame["original_polarity"] = frame["score_original"].map(stance_polarity)
    frame["model_polarity"] = frame["score_llm"].map(stance_polarity)
    questions = sorted(frame["question_index"].unique())
    stance_labels = {qidx: label_for_group(qidx, group) for qidx, group in frame.groupby("question_index", sort=True)}
    polarity_labels = ["negative", "positive"]

    ncols = min(5, max(1, len(questions)))
    nrows = int(np.ceil(len(questions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.5 * nrows), sharex=False, sharey=False, squeeze=False)
    axes_flat = axes.ravel()
    for ax, qidx in zip(axes_flat, questions):
        sub = frame[frame["question_index"] == qidx].dropna(subset=["original_polarity", "model_polarity"])
        counts = pd.crosstab(sub["original_polarity"], sub["model_polarity"]).reindex(index=polarity_labels, columns=polarity_labels, fill_value=0)
        sns.heatmap(
            counts,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            ax=ax,
        )
        ax.set_title(stance_labels.get(qidx, f"Q{qidx}"))
        ax.set_xlabel("Model")
        ax.set_ylabel("Original")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
    for ax in axes_flat[len(questions):]:
        ax.set_axis_off()
    fig.suptitle("STANCE Polarity Confusion Matrices: Original vs Model", y=1.02)
    fig.text(0.5, 0.01, "Neutral scores (0) are excluded from the positive/negative polarity counts.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save_figure(fig, output_path)


def parse_score_vector(value) -> Optional[List[float]]:
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


def load_dialect_frame(results_root: Path, model: str, subset: str) -> Optional[pd.DataFrame]:
    frame = safe_read_csv(result_file(results_root, model, "test", subset, "dialect_id.csv"))
    if frame is None:
        return None
    required = {"dialect", "answer_dialect"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["dialect"] = frame["dialect"].astype(str).str.strip()
    frame["answer_dialect"] = frame["answer_dialect"].astype(str).str.strip()
    frame = frame[frame["dialect"].isin(DIALECT_LABELS) & frame["answer_dialect"].isin(DIALECT_LABELS)]
    return frame


def graph_dialect_confusion(results_root: Path, model: str, output_path: Path) -> None:
    subset_frames = {subset: load_dialect_frame(results_root, model, subset) for subset in SUBSETS}
    subset_frames = {subset: frame for subset, frame in subset_frames.items() if frame is not None and not frame.empty}
    if not subset_frames:
        print("[WARN] Missing dialect ID rows; skipping dialect_confusion.png.")
        return

    fig, axes = plt.subplots(1, len(subset_frames), figsize=(8 * len(subset_frames), 7.2), squeeze=False)
    for ax, (subset, frame) in zip(axes[0], subset_frames.items()):
        counts = pd.crosstab(frame["dialect"], frame["answer_dialect"]).reindex(index=DIALECT_LABELS, columns=DIALECT_LABELS, fill_value=0)
        vmax = max(1, int(counts.to_numpy().max()))
        sns.heatmap(
            counts,
            annot=True,
            fmt="d",
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=vmax),
            mask=counts.eq(0),
            cbar=len(subset_frames) == 1,
            linewidths=0.25,
            linecolor="white",
            square=True,
            annot_kws={"fontsize": 7},
            ax=ax,
        )
        ax.set_title(subset.title())
        ax.set_xlabel("Answer dialect")
        ax.set_ylabel("Question dialect")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle("Dialect Changes from Question to Answer", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def graph_dialect_scores(results_root: Path, model: str, output_path: Path) -> None:
    rows = []
    for subset in SUBSETS:
        frame = safe_read_csv(result_file(results_root, model, "test", subset, "dialect_id.csv"))
        if frame is None or not {"dialect_score", "answer_dialect_score"}.issubset(frame.columns):
            continue
        for score_column, field in [("dialect_score", "question"), ("answer_dialect_score", "answer")]:
            for vector in frame[score_column].map(parse_score_vector).dropna():
                rows.extend(
                    {"subset": subset, "field": field, "dialect": dialect, "score": score}
                    for dialect, score in grouped_dialect_scores(vector).items()
                )
    if not rows:
        print("[WARN] Missing dialect score vectors; skipping dialect_scores.png.")
        return

    data = pd.DataFrame(rows)
    data["score"] = numeric(data["score"])
    data = data.dropna(subset=["score"])
    if data.empty:
        print("[WARN] Dialect score vectors do not contain numeric values; skipping dialect_scores.png.")
        return

    score_floor = 1e-8
    data["score_plot"] = data["score"].clip(lower=score_floor)
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    if not subsets:
        print("[WARN] No subset dialect score vectors found; skipping dialect_scores.png.")
        return

    angles = np.linspace(0, 2 * np.pi, len(DIALECT_PROFILE_LABELS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    palette = {"question": "#4C72B0", "answer": "#DD8452"}
    fig, axes = plt.subplots(1, len(subsets), figsize=(8.5 * len(subsets), 8.5), subplot_kw={"projection": "polar"}, squeeze=False)
    legend_handles = []
    legend_labels = []

    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset]
        profile = (
            sub.groupby(["field", "dialect"], dropna=False)["score_plot"]
            .median()
            .unstack("dialect")
            .reindex(index=["question", "answer"], columns=DIALECT_PROFILE_LABELS)
        )
        for field in ["question", "answer"]:
            if field not in profile.index:
                continue
            values = pd.to_numeric(profile.loc[field], errors="coerce").fillna(score_floor).clip(lower=score_floor).to_numpy(dtype=float)
            closed_values = np.concatenate([values, values[:1]])
            line, = ax.plot(closed_angles, closed_values, color=palette[field], linewidth=2.0, label=field)
            ax.fill(closed_angles, closed_values, color=palette[field], alpha=0.14)
            if field not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(field)

        ax.set_title(subset.title(), pad=24)
        ax.set_yscale("log")
        ax.set_ylim(score_floor, 1.0)
        ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1.0])
        ax.set_yticklabels(["1e-8", "1e-6", "1e-4", "1e-2", "1"], fontsize=8)
        ax.set_xticks(angles)
        ax.set_xticklabels(DIALECT_PROFILE_LABELS, fontsize=8)
        ax.tick_params(axis="x", pad=8)
        ax.grid(True, alpha=0.45)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Field", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Dialect Score Profiles: Question vs Answer", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_path)


def load_emotion_scatter_values(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    rows = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(result_file(results_root, label_model, "test", subset, "SER_AVD.csv"))
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
                        "dataset": label,
                        "subset": subset,
                        "emotion": emotion,
                    }
                ).dropna(subset=["question", "answer"])
                if not pairs.empty:
                    rows.append(pairs)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def draw_correlation_lines(ax, data: pd.DataFrame, palette: Dict[str, str]) -> None:
    label_x = {"original": 0.58, "model": -0.92}
    label_offset = {"original": 0.08, "model": -0.08}
    for dataset in ["original", "model"]:
        sub = data[data["dataset"] == dataset].dropna(subset=["question", "answer"])
        if len(sub) < 2:
            continue
        x = sub["question"].astype(float).to_numpy()
        y = sub["answer"].astype(float).to_numpy()
        x_std = np.nanstd(x)
        y_std = np.nanstd(y)
        rho = np.nan if x_std == 0 or y_std == 0 else float(np.corrcoef(x, y)[0, 1])
        rho_text = "rho=nan" if np.isnan(rho) else f"rho={rho:.2f}"
        color = palette.get(dataset, "#333333")
        if x_std > 0:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.array([-1.0, 1.0])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle=":", linewidth=2.0, alpha=0.95)
            text_x = label_x.get(dataset, 0.6)
            text_y = float(np.clip(slope * text_x + intercept + label_offset.get(dataset, 0.0), -0.94, 0.94))
            ax.text(
                text_x,
                text_y,
                rho_text,
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )
        else:
            ax.text(
                0.05,
                0.9 if dataset == "original" else 0.8,
                rho_text,
                transform=ax.transAxes,
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="top",
            )


def graph_emotion_scatter(results_root: Path, model: str, output_path: Path) -> None:
    data = load_emotion_scatter_values(results_root, model)
    if data is None or data.empty:
        print("[WARN] Missing SER_AVD rows; skipping emotion_scatter.png.")
        return

    emotions = ["arousal", "dominance", "valence"]
    emotion_titles = {"arousal": "Arousal", "dominance": "Dominance", "valence": "Valence"}
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    if not subsets:
        print("[WARN] No subset data found in SER_AVD rows; skipping emotion_scatter.png.")
        return

    fig, axes = plt.subplots(len(subsets), len(emotions), figsize=(17, 10), sharex=True, sharey=True, squeeze=False)
    legend_handles = None
    legend_labels = None
    for row_idx, subset in enumerate(subsets):
        for col_idx, emotion in enumerate(emotions):
            ax = axes[row_idx, col_idx]
            sub = data[(data["subset"] == subset) & (data["emotion"] == emotion)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sns.scatterplot(
                data=sub,
                x="question",
                y="answer",
                hue="dataset",
                hue_order=["original", "model"],
                palette=DATASET_PALETTE,
                alpha=0.55,
                s=24,
                edgecolor="none",
                ax=ax,
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles and legend_handles is None:
                legend_handles, legend_labels = handles, labels
            remove_axis_legend(ax)
            ax.plot([-1, 1], [-1, 1], color="#666666", linestyle="--", linewidth=1.0, alpha=0.55)
            draw_correlation_lines(ax, sub, DATASET_PALETTE)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(emotion_titles[emotion] if row_idx == 0 else "")
            ax.set_xlabel(f"Question {emotion_titles[emotion]}" if row_idx == len(subsets) - 1 else "")
            ax.set_ylabel(f"{subset.title()}\nAnswer {emotion_titles[emotion]}" if col_idx == 0 else "")
            ax.tick_params(axis="both", labelsize=9)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.015), frameon=True)
    fig.suptitle("Question vs Answer VoxProfile Emotion Scores", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output_path)


def load_naturalness_relationship_values(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    frames = []
    subset = "naturalistic"
    for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
        scores = safe_read_csv(result_file(results_root, label_model, "test", subset, "naturalness_scores_normalized.csv"))
        inference = safe_read_csv(result_file(results_root, label_model, "test", subset, "naturalness_inference_input.csv"))
        if scores is None or inference is None:
            continue
        if "naturalness_logit" not in scores.columns or "rel_detail" not in inference.columns:
            print(f"[WARN] Missing naturalness_logit or rel_detail for {label_model}/{subset}; skipping relationship violin rows.")
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
        joined = joined[joined["relationship"].isin(RELATIONSHIP_ORDER)]
        joined = joined.dropna(subset=["naturalness_logit"])
        if joined.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "naturalness_logit": joined["naturalness_logit"],
                    "relationship": joined["relationship"],
                    "dataset": label,
                }
            )
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def graph_emotional_naturalness_by_relationship(results_root: Path, model: str, output_path: Path) -> None:
    data = load_naturalness_relationship_values(results_root, model)
    if data is None or data.empty:
        print("[WARN] Missing naturalistic naturalness relationship rows; skipping emo_naturalness_by_relationship.png.")
        return

    dataset_order = ["original", "model"]
    relationship_order = [relationship for relationship in RELATIONSHIP_ORDER if relationship in set(data["relationship"])]
    fig_height = max(5.0, 1.05 * len(relationship_order) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    sns.violinplot(
        data=data,
        x="naturalness_logit",
        y="relationship",
        hue="dataset",
        order=relationship_order,
        hue_order=dataset_order,
        palette=DATASET_PALETTE,
        inner="quartile",
        cut=0,
        linewidth=1.1,
        orient="h",
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="naturalness_logit",
        y="relationship",
        hue="dataset",
        order=relationship_order,
        hue_order=dataset_order,
        dodge=True,
        palette={label: "#202020" for label in dataset_order},
        alpha=0.14,
        size=1.8,
        jitter=0.18,
        orient="h",
        legend=False,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    deduped = dict(zip(labels, handles))
    ax.legend(
        [deduped[label] for label in dataset_order if label in deduped],
        [label for label in dataset_order if label in deduped],
        title="Dataset",
        loc="upper right",
        frameon=True,
    )
    ax.set_title("Naturalistic Emotional Naturalness by Relationship")
    ax.set_xlabel("Normalized emotional naturalness logit")
    ax.set_ylabel("Relationship")
    fig.tight_layout()
    save_figure(fig, output_path)


def graph_emotional_naturalness(results_root: Path, model: str, output_path: Path) -> None:
    subset_frames = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(result_file(results_root, label_model, "test", subset, "naturalness_scores_normalized.csv"))
            if frame is None or "naturalness_logit" not in frame.columns:
                continue
            values = numeric(frame["naturalness_logit"]).dropna()
            if values.empty:
                continue
            subset_frames.append(pd.DataFrame({"naturalness_logit": values, "dataset": label, "subset": subset}))

    if not subset_frames:
        print("[WARN] Missing emotional naturalness scores; skipping emo_naturalness.png.")
        return

    data = pd.concat(subset_frames, ignore_index=True)
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    fig, axes = plt.subplots(len(subsets), 1, figsize=(11, max(4, 4 * len(subsets))), squeeze=False)
    for ax, subset in zip(axes[:, 0], subsets):
        sub = data[data["subset"] == subset]
        sns.histplot(
            data=sub,
            x="naturalness_logit",
            hue="dataset",
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(f"Normalized Emotional Naturalness Logits - {subset}")
        ax.set_xlabel("normalized naturalness_logit")
        ax.set_ylabel("Density")
    fig.tight_layout()
    save_figure(fig, output_path)


def significance_stars(pvalue) -> str:
    try:
        pvalue = float(pvalue)
    except Exception:
        return ""
    if np.isnan(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""




def cluster_feature_labels(results_root: Path, model: str, subset: str, threshold: str = "0p8") -> Dict[str, str]:
    groups = safe_read_csv(result_file(results_root, model, "test", subset, f"correlation_feature_groups_rho{threshold}.csv"))
    if groups is None or not {"cluster_feature", "features"}.issubset(groups.columns):
        return {}
    labels = {}
    for _, row in groups.dropna(subset=["cluster_feature"]).iterrows():
        feature = str(row.get("cluster_feature", ""))
        source_features = str(row.get("features", "")).strip()
        if feature.startswith("corr_cluster_") and source_features:
            labels[feature] = source_features
    return labels


def load_cluster_feature_values(results_root: Path, model: str, general: bool) -> Optional[pd.DataFrame]:
    frames = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            path = result_file(results_root, label_model, "test", subset, "correlation_cluster_features_rho0p8.csv")
            frame = safe_read_csv(path)
            if frame is None:
                continue
            label_map = cluster_feature_labels(results_root, label_model, subset) if not general else {}
            if general:
                features = [column for column in frame.columns if column.startswith("general_explainable_feature_")]
            else:
                features = [column for column in frame.columns if column.startswith("corr_cluster_")]
            for feature in features:
                if feature in EXCLUDED_REPORT_METRICS:
                    continue
                values = numeric(frame[feature]).dropna()
                if values.empty:
                    continue
                frames.append(pd.DataFrame({"value": values, "dataset": label, "subset": subset, "feature": feature, "feature_label": label_map.get(feature, feature)}))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def graph_cluster_feature_violins(results_root: Path, model: str, output_path: Path) -> None:
    data = load_cluster_feature_values(results_root, model, general=False)
    if data is None or data.empty:
        print("[WARN] Missing correlation cluster feature values; skipping cluster_explainables.png.")
        return
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    max_features = max(data[data["subset"] == subset]["feature_label"].nunique() for subset in subsets)
    fig_height = max(5, 0.38 * max_features + 2.8)
    fig, axes = plt.subplots(1, len(subsets), figsize=(9 * len(subsets), fig_height), sharex=True, squeeze=False)
    legend_handles = None
    legend_labels = None
    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset].copy()
        feature_order = sub.groupby("feature_label")["value"].median().sort_values(ascending=False).index.tolist()
        sns.violinplot(
            data=sub,
            x="value",
            y="feature_label",
            hue="dataset",
            hue_order=["original", "model"],
            order=feature_order,
            palette=DATASET_PALETTE,
            orient="h",
            inner="quartile",
            cut=0,
            linewidth=0.8,
            density_norm="width",
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles and legend_handles is None:
            legend_handles, legend_labels = handles, labels
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xlabel("PCA cluster feature value")
        ax.set_ylabel("Cluster feature" if ax is axes[0, 0] else "")
        if ax is not axes[0, 0]:
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Correlation Cluster Explainable Features", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output_path)


def graph_general_explainable_histogram(results_root: Path, model: str, output_path: Path) -> None:
    data = load_cluster_feature_values(results_root, model, general=True)
    if data is None or data.empty:
        print("[WARN] Missing general explainable feature values; skipping general_explainable.png.")
        return
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    fig, axes = plt.subplots(len(subsets), 1, figsize=(11, max(4, 4 * len(subsets))), sharex=True, squeeze=False)
    for ax, subset in zip(axes[:, 0], subsets):
        sub = data[data["subset"] == subset]
        sns.histplot(
            data=sub,
            x="value",
            hue="dataset",
            hue_order=["original", "model"],
            palette=DATASET_PALETTE,
            bins=40,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(subset.title())
        ax.set_xlabel("General explainable feature value")
        ax.set_ylabel("Density")
    fig.suptitle("General Explainable Feature", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)

def load_turntaking_surprisal_values(results_root: Path, model: str) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(turntaking_surprisal_file(results_root, label_model, subset))
            if frame is None:
                continue
            for metric in TURNTAKING_SURPRISAL_METRICS:
                if metric not in frame.columns:
                    continue
                values = numeric(frame[metric]).dropna()
                if values.empty:
                    continue
                rows.append(pd.DataFrame({"subset": subset, "dataset": label, "metric": metric, "value": values}))
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    naturalness_mask = data["metric"] == "naturalness_score"
    data.loc[naturalness_mask, "value"] = -1.0 * data.loc[naturalness_mask, "value"]
    return data


def graph_turntaking_surprisal(results_root: Path, model: str, output_path: Path) -> None:
    data = load_turntaking_surprisal_values(results_root, model)
    if data.empty:
        print(f"[WARN] No turn-taking surprisal values found; skipping {output_path.name}.")
        return
    metrics = [metric for metric in TURNTAKING_SURPRISAL_METRICS if metric in set(data["metric"])]
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    fig_height = max(2.8 + 0.55 * len(metrics), 4.2)
    fig, axes = plt.subplots(1, len(subsets), figsize=(8.5 * len(subsets), fig_height), sharex=False, squeeze=False)
    legend_handles = None
    legend_labels = None
    for col_idx, subset in enumerate(subsets):
        ax = axes[0, col_idx]
        sub = data[data["subset"] == subset]
        if sub.empty:
            ax.axis("off")
            continue
        sns.violinplot(
            data=sub,
            x="value",
            y="metric",
            hue="dataset",
            order=metrics,
            hue_order=["original", "model"],
            palette=DATASET_PALETTE,
            orient="h",
            inner="quartile",
            cut=0,
            linewidth=0.8,
            density_norm="width",
            ax=ax,
        )
        if legend_handles is None:
            legend_handles = [Patch(facecolor=DATASET_PALETTE[label], label=label) for label in ["original", "model"]]
            legend_labels = ["original", "model"]
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xlabel("Metric value")
        ax.set_ylabel("Metric" if col_idx == 0 else "")
        if col_idx != 0:
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Turn-Taking Surprisal Metrics: Original vs Model", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output_path)


def graph_explainable_scatter(
    results_root: Path,
    model: str,
    output_path: Path,
    *,
    feature_group: Optional[str] = None,
    title: str = "Explainable Feature Distributions: Original vs Model",
) -> None:
    report_dir = output_path.parents[1]
    metrics_by_subset: Dict[str, pd.DataFrame] = {}
    for subset in SUBSETS:
        metrics = safe_read_csv(report_dir / f"metrics-{subset}.csv")
        if metrics is not None and {"metric", "p_value"}.issubset(metrics.columns):
            metrics_by_subset[subset] = metrics

    rows = []
    for subset in SUBSETS:
        original = safe_read_csv(explainable_feature_file(results_root, ORIGINAL, subset))
        model_frame = safe_read_csv(explainable_feature_file(results_root, model, subset))
        if original is None or model_frame is None:
            continue

        features = available_all_explainable_features(results_root, model, subset)
        if feature_group is not None:
            features = [feature for feature in features if explainable_feature_group(feature) == feature_group]

        pvalues = {}
        if subset in metrics_by_subset:
            metric_frame = metrics_by_subset[subset]
            explainable = metric_frame[metric_frame.get("section", "") == "Explainable Features"] if "section" in metric_frame else metric_frame
            pvalues = dict(zip(explainable["metric"].astype(str), pd.to_numeric(explainable["p_value"], errors="coerce")))

        for feature in features:
            if feature not in original.columns or feature not in model_frame.columns:
                continue
            original_values = numeric(original[feature]).dropna()
            model_values = numeric(model_frame[feature]).dropna()
            combined = pd.concat([original_values, model_values], ignore_index=True).dropna()
            if combined.empty:
                continue
            value_min = combined.min()
            value_max = combined.max()
            denom = value_max - value_min
            pvalue = pvalues.get(feature, np.nan)
            label_feature = f"{feature}{significance_stars(pvalue)}"
            for label, values in [("original", original_values), ("model", model_values)]:
                if values.empty:
                    continue
                if denom == 0 or pd.isna(denom):
                    normalized = pd.Series(np.full(len(values), 0.5), index=values.index)
                else:
                    normalized = (values - value_min) / denom
                rows.append(
                    pd.DataFrame(
                        {
                            "feature": feature,
                            "subset": subset,
                            "dataset": label,
                            "value": normalized,
                            "feature_label": label_feature,
                            "p_value": pvalue,
                        }
                    )
                )

    if not rows:
        print(f"[WARN] No explainable feature values found; skipping {output_path.name}.")
        return

    data = pd.concat(rows, ignore_index=True)
    available_subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    if not available_subsets:
        print(f"[WARN] No subset data found; skipping {output_path.name}.")
        return

    max_features = max(data[data["subset"] == subset]["feature_label"].nunique() for subset in available_subsets)
    fig_height = max(8, 0.34 * max_features + 2.8)
    fig, axes = plt.subplots(1, len(available_subsets), figsize=(9 * len(available_subsets), fig_height), sharex=True, squeeze=False)

    legend_handles = None
    legend_labels = None
    for ax, subset in zip(axes[0], available_subsets):
        sub = data[data["subset"] == subset].copy()
        feature_order = (
            sub.groupby("feature_label")["value"]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        pvalues_by_label = (
            sub.groupby("feature_label")["p_value"]
            .first()
            .apply(lambda value: pd.to_numeric(value, errors="coerce"))
            .to_dict()
        )
        for y_idx, feature_label in enumerate(feature_order):
            pvalue = pvalues_by_label.get(feature_label, np.nan)
            if pd.isna(pvalue):
                continue
            if pvalue < 0.001:
                ax.axhspan(y_idx - 0.5, y_idx + 0.5, color="#d62728", alpha=0.14, zorder=0)
            elif pvalue < 0.05:
                ax.axhspan(y_idx - 0.5, y_idx + 0.5, color="#ff7f0e", alpha=0.12, zorder=0)

        sns.violinplot(
            data=sub,
            x="value",
            y="feature_label",
            hue="dataset",
            order=feature_order,
            orient="h",
            split=False,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            density_norm="width",
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles and legend_handles is None:
            legend_handles, legend_labels = handles, labels
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xlabel("Feature value normalized to [0, 1]")
        ax.set_ylabel("Feature" if ax is axes[0, 0] else "")
        if ax is not axes[0, 0]:
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlim(-0.02, 1.02)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, y=1.035)
    fig.text(0.5, 0.005, "Significance: * p<0.05, ** p<0.01, *** p<0.001; orange background p<0.05, red background p<0.001", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save_figure(fig, output_path)


def graph_explainable_feature_groups(results_root: Path, model: str, graphs_dir: Path) -> None:
    for group_name, spec in EXPLAINABLE_FEATURE_GROUPS.items():
        graph_explainable_scatter(
            results_root,
            model,
            graphs_dir / f"explainables_{group_name}.png",
            feature_group=group_name,
            title=f"{spec['title']}: Original vs Model",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-features", nargs="*", default=[])
    args = parser.parse_args()
    EXCLUDED_REPORT_METRICS.update(args.ignore_features)

    graphs_dir = args.output_dir / "graphs"
    graph_basic_metric_histograms(args.results_root, args.model, graphs_dir / "basic_metric_graphs")
    graph_basic_metrics_histograms(args.results_root, args.model, graphs_dir / "basic_metrics.png")
    graph_turntaking_surprisal(args.results_root, args.model, graphs_dir / "turntaking_surprisal.png")
    graph_feature_histograms(args.results_root, args.model, graphs_dir / "feat_graphs")
    graph_stances(args.results_root, args.model, graphs_dir / "stances.png")
    graph_emotional_naturalness(args.results_root, args.model, graphs_dir / "emo_naturalness.png")
    graph_emotional_naturalness_by_relationship(args.results_root, args.model, graphs_dir / "emo_naturalness_by_relationship.png")
    graph_emotion_scatter(args.results_root, args.model, graphs_dir / "emotion_scatter.png")
    graph_dialect_confusion(args.results_root, args.model, graphs_dir / "dialect_confusion.png")
    graph_dialect_scores(args.results_root, args.model, graphs_dir / "dialect_scores.png")
    graph_f0_question_answer_boxplot(args.results_root, args.model, graphs_dir / "f0_question_answer_boxplot.png")
    graph_explainable_feature_groups(args.results_root, args.model, graphs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
