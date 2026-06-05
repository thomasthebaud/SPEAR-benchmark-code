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


def save_figure(fig, output_path: Path, *, dpi: int = 180, bbox_inches: str = "tight") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Wrote {output_path}")


def remove_axis_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def load_feature_values(results_root: Path, model: str, subset: str, feature: str) -> Optional[pd.DataFrame]:
    frames = []
    for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
        path = result_file(results_root, label_model, "test", subset, "distrib_baselines_features.csv")
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
            frames.append(pd.DataFrame({"value": values, "dataset": label, "subset": subset, "metric": metric}))
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
                values = numeric(frame[column]).dropna()
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
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    fig, axes = plt.subplots(len(subsets), 1, figsize=(11, max(4.5, 4.5 * len(subsets))), sharex=True, squeeze=False)
    for ax, subset in zip(axes[:, 0], subsets):
        sub = data[data["subset"] == subset]
        sns.violinplot(
            data=sub,
            x="value",
            y="asr_system",
            hue="dataset",
            hue_order=["original", "model"],
            palette=DATASET_PALETTE,
            orient="h",
            inner="quartile",
            cut=0,
            linewidth=0.8,
            density_norm="width",
            ax=ax,
        )
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xlabel(error_type)
        ax.set_ylabel("ASR system")
    legend_handles = [Patch(facecolor=DATASET_PALETTE[label], label=label) for label in ["original", "model"]]
    fig.legend(legend_handles, ["original", "model"], title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(f"{error_type}: Original vs Model Across ASR Systems", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_path)


def graph_basic_metric_histograms(results_root: Path, model: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for error_type in ["CER", "WER"]:
        graph_asr_error_histogram(results_root, model, error_type, output_dir / f"{error_type}.png")

    for metric in [metric for metric in available_base_metric_columns(results_root, model) if not metric.startswith(("CER", "WER")) and metric != "interruptions"]:
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
                palette=DATASET_PALETTE,
                ax=ax,
            )
            ax.set_title(subset.title())
            ax.set_xlabel(metric)
            ax.set_ylabel("Density")
        fig.suptitle(f"{metric}: Original vs Model", y=1.03)
        fig.tight_layout()
        out_path = output_dir / f"{sanitize_filename(metric)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote basic metric histograms to {output_dir}")


def graph_basic_metrics_violin(results_root: Path, model: str, output_path: Path) -> None:
    frames = []
    metric_order = []
    for error_type in ["CER", "WER"]:
        data = load_asr_error_values(results_root, model, error_type)
        if data is not None:
            frames.append(data)
            metric_order.append(error_type)
    for metric in [metric for metric in available_base_metric_columns(results_root, model) if not metric.startswith(("CER", "WER")) and metric != "interruptions"]:
        data = load_base_metric_values(results_root, model, metric)
        if data is not None:
            frames.append(data)
            metric_order.append(metric)
    if not frames:
        print("[WARN] No base metrics found; skipping basic_metrics.png.")
        return
    data = pd.concat(frames, ignore_index=True)
    group_cols = ["subset", "metric"] + (["asr_system"] if "asr_system" in data.columns else [])
    data = remove_iqr_outliers(data, group_cols)
    if data.empty:
        print("[WARN] All base metric rows were filtered as outliers; skipping basic_metrics.png.")
        return
    available_subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    metrics = [metric for metric in metric_order if metric in set(data["metric"])]
    fig, axes = plt.subplots(len(metrics), len(available_subsets), figsize=(8.5 * len(available_subsets), 3.8 * len(metrics)), squeeze=False)
    for row_idx, metric in enumerate(metrics):
        for col_idx, subset in enumerate(available_subsets):
            ax = axes[row_idx, col_idx]
            sub = data[(data["subset"] == subset) & (data["metric"] == metric)]
            if sub.empty:
                ax.set_axis_off()
                continue
            if metric in {"CER", "WER"} and "asr_system" in sub.columns:
                sns.violinplot(
                    data=sub,
                    x="value",
                    y="asr_system",
                    hue="dataset",
                    hue_order=["original", "model"],
                    palette=DATASET_PALETTE,
                    orient="h",
                    inner="quartile",
                    cut=0,
                    linewidth=0.8,
                    density_norm="width",
                    ax=ax,
                )
                ax.set_ylabel(metric if col_idx == 0 else "")
            else:
                sns.violinplot(
                    data=sub,
                    x="value",
                    y="dataset",
                    hue="dataset",
                    hue_order=["original", "model"],
                    palette=DATASET_PALETTE,
                    orient="h",
                    inner="quartile",
                    cut=0,
                    linewidth=0.8,
                    density_norm="width",
                    ax=ax,
                )
                ax.set_ylabel(metric if col_idx == 0 else "")
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)
            remove_axis_legend(ax)
            ax.set_title(subset.title() if row_idx == 0 else "")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

    legend_handles = [Patch(facecolor=DATASET_PALETTE[label], label=label) for label in ["original", "model"]]
    fig.legend(legend_handles, ["original", "model"], title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Basic Metric Distributions: Original vs Model", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_path)

def graph_feature_histograms(results_root: Path, model: str, output_dir: Path) -> None:
    summary = safe_read_csv(result_file(results_root, model, "distrib_baselines_summary.csv"))
    if summary is None or "feature" not in summary.columns:
        print("[WARN] No explainable feature summary found; skipping per-feature histograms.")
        return

    features = sorted(
        str(feature)
        for feature in summary["feature"].dropna().unique()
        if str(feature) not in EXCLUDED_REPORT_METRICS
    )
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
                    for dialect, score in zip(DIALECT_LABELS, vector)
                )
    if not rows:
        print("[WARN] Missing dialect score vectors; skipping dialect_scores.png.")
        return

    data = pd.DataFrame(rows)
    data["score"] = numeric(data["score"])
    data = data.dropna(subset=["score"])
    positive_scores = data.loc[data["score"] > 0, "score"]
    if positive_scores.empty:
        print("[WARN] Dialect score vectors do not contain positive values; skipping dialect_scores.png.")
        return
    score_floor = float(positive_scores.min()) / 10.0
    data["score_plot"] = data["score"].where(data["score"] > 0, score_floor)
    subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    fig, axes = plt.subplots(1, len(subsets), figsize=(9 * len(subsets), 7.5), sharex=True, squeeze=False)
    legend_handles = None
    legend_labels = None
    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset]
        sns.boxplot(
            data=sub,
            x="score_plot",
            y="dialect",
            hue="field",
            order=DIALECT_LABELS,
            hue_order=["question", "answer"],
            orient="h",
            showfliers=False,
            linewidth=0.8,
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles and legend_handles is None:
            legend_handles, legend_labels = handles, labels
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xscale("log")
        # ax.set_xlim(score_floor, 1.0)
        ax.set_xlabel("Dialect score (log scale)")
        ax.set_ylabel("Dialect" if ax is axes[0, 0] else "")
        if ax is not axes[0, 0]:
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="Field", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=True)
    fig.suptitle("Dialect Score Distributions: Question vs Answer", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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
        scores = safe_read_csv(result_file(results_root, label_model, "test", subset, "naturalness_scores.csv"))
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
    ax.set_xlabel("Emotional naturalness logit")
    ax.set_ylabel("Relationship")
    fig.tight_layout()
    save_figure(fig, output_path)


def graph_emotional_naturalness(results_root: Path, model: str, output_path: Path) -> None:
    subset_frames = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(result_file(results_root, label_model, "test", subset, "naturalness_scores.csv"))
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
        ax.set_title(f"Emotional Naturalness Logits - {subset}")
        ax.set_xlabel("naturalness_logit")
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

def graph_explainable_scatter(results_root: Path, model: str, output_path: Path) -> None:
    summary = safe_read_csv(result_file(results_root, model, "distrib_baselines_summary.csv"))
    if summary is None or "feature" not in summary.columns:
        print("[WARN] Missing explainable feature summary; skipping explainables.png.")
        return

    report_dir = output_path.parents[1]
    metrics_by_subset: Dict[str, pd.DataFrame] = {}
    for subset in SUBSETS:
        metrics = safe_read_csv(report_dir / f"metrics-{subset}.csv")
        if metrics is not None and {"metric", "p_value"}.issubset(metrics.columns):
            metrics_by_subset[subset] = metrics

    rows = []
    for subset in SUBSETS:
        original = safe_read_csv(result_file(results_root, ORIGINAL, "test", subset, "distrib_baselines_features.csv"))
        model_frame = safe_read_csv(result_file(results_root, model, "test", subset, "distrib_baselines_features.csv"))
        if original is None or model_frame is None:
            continue

        if "subset" in summary.columns:
            features = sorted(str(feature) for feature in summary.loc[summary["subset"] == subset, "feature"].dropna().unique())
        else:
            features = sorted(
        str(feature)
        for feature in summary["feature"].dropna().unique()
        if str(feature) not in EXCLUDED_REPORT_METRICS
    )

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
        print("[WARN] No explainable feature values found; skipping explainables.png.")
        return

    data = pd.concat(rows, ignore_index=True)
    available_subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    if not available_subsets:
        print("[WARN] No subset data found; skipping explainables.png.")
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
    fig.suptitle("Explainable Feature Distributions: Original vs Model", y=1.035)
    fig.text(0.5, 0.005, "Significance: * p<0.05, ** p<0.01, *** p<0.001; orange background p<0.05, red background p<0.001", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save_figure(fig, output_path)


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
    graph_basic_metrics_violin(args.results_root, args.model, graphs_dir / "basic_metrics.png")
    graph_feature_histograms(args.results_root, args.model, graphs_dir / "feat_graphs")
    graph_stances(args.results_root, args.model, graphs_dir / "stances.png")
    graph_emotional_naturalness(args.results_root, args.model, graphs_dir / "emo_naturalness.png")
    graph_emotional_naturalness_by_relationship(args.results_root, args.model, graphs_dir / "emo_naturalness_by_relationship.png")
    graph_emotion_scatter(args.results_root, args.model, graphs_dir / "emotion_scatter.png")
    graph_dialect_confusion(args.results_root, args.model, graphs_dir / "dialect_confusion.png")
    graph_dialect_scores(args.results_root, args.model, graphs_dir / "dialect_scores.png")
    graph_explainable_scatter(args.results_root, args.model, graphs_dir / "explainables.png")
    graph_cluster_feature_violins(args.results_root, args.model, graphs_dir / "cluster_explainables.png")
    graph_general_explainable_histogram(args.results_root, args.model, graphs_dir / "general_explainable.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
