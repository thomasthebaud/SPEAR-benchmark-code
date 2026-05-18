#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ORIGINAL = "original"
SUBSETS = ["improvised", "naturalistic"]
DATASET_PALETTE = {"original": "#4C72B0", "model": "#DD8452"}

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


def feature_csv(results_root: Path, model: str, split: str, subset: str) -> Path:
    return results_root / model / split / subset / "distrib_baselines_features.csv"


def naturalness_csv(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "naturalness_scores.csv"


def feature_summary_csv(results_root: Path, model: str) -> Path:
    return results_root / model / "distrib_baselines_summary.csv"


def merged_stances_csv(results_root: Path, model: str, subset: str = "improvised") -> Path:
    return results_root / model / "test" / subset / "merged_stances.csv"


def load_feature_values(results_root: Path, model: str, subset: str, feature: str) -> Optional[pd.DataFrame]:
    frames = []
    for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
        path = feature_csv(results_root, label_model, "test", subset)
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



def base_metrics_csv(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "base_metrics.csv"


def load_base_metric_values(results_root: Path, model: str, metric: str) -> Optional[pd.DataFrame]:
    frames = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(base_metrics_csv(results_root, label_model, subset))
            if frame is None or metric not in frame.columns:
                continue
            values = numeric(frame[metric]).dropna()
            if values.empty:
                continue
            frames.append(pd.DataFrame({"value": values, "dataset": label, "subset": subset, "metric": metric}))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def add_normalized_values(data: pd.DataFrame, group_cols: List[str], value_col: str = "value") -> pd.DataFrame:
    data = data.copy()
    data["normalized_value"] = np.nan
    for _, idx in data.groupby(group_cols).groups.items():
        values = data.loc[idx, value_col].astype(float)
        value_min = values.min()
        value_max = values.max()
        denom = value_max - value_min
        if pd.isna(denom) or denom == 0:
            data.loc[idx, "normalized_value"] = 0.5
        else:
            data.loc[idx, "normalized_value"] = (values - value_min) / denom
    return data


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



def graph_basic_metric_histograms(results_root: Path, model: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["CER", "WER", "UTMOS", "latency"]:
        data = load_base_metric_values(results_root, model, metric)
        if data is None or data.empty:
            continue
        data = remove_iqr_outliers(data, ["subset", "metric"])
        if data.empty:
            continue
        subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
        fig, axes = plt.subplots(1, len(subsets), figsize=(6.5 * len(subsets), 4.8), sharex=True, sharey=True, squeeze=False)
        for ax, subset in zip(axes[0], subsets):
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
                ax=ax,
            )
            ax.set_title(subset.title())
            ax.set_xlabel(metric)
            ax.set_ylabel("Density" if ax is axes[0, 0] else "")
        fig.suptitle(f"{metric}: Original vs Model", y=1.03)
        fig.tight_layout()
        out_path = output_dir / f"{sanitize_filename(metric)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote basic metric histograms to {output_dir}")


def graph_basic_metrics_violin(results_root: Path, model: str, output_path: Path) -> None:
    frames = []
    for metric in ["CER", "WER", "UTMOS", "latency"]:
        data = load_base_metric_values(results_root, model, metric)
        if data is not None:
            frames.append(data)
    if not frames:
        print("[WARN] No base metrics found; skipping basic_metrics.png.")
        return
    data = pd.concat(frames, ignore_index=True)
    data = remove_iqr_outliers(data, ["subset", "metric"])
    if data.empty:
        print("[WARN] All base metric rows were filtered as outliers; skipping basic_metrics.png.")
        return
    available_subsets = [subset for subset in SUBSETS if subset in set(data["subset"])]
    metrics = [metric for metric in ["CER", "WER", "UTMOS", "latency"] if metric in set(data["metric"])]
    fig, axes = plt.subplots(len(metrics), len(available_subsets), figsize=(8 * len(available_subsets), 3.2 * len(metrics)), squeeze=False)
    for row_idx, metric in enumerate(metrics):
        for col_idx, subset in enumerate(available_subsets):
            ax = axes[row_idx, col_idx]
            sub = data[(data["subset"] == subset) & (data["metric"] == metric)]
            if sub.empty:
                ax.set_axis_off()
                continue
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
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            if row_idx == 0:
                ax.set_title(subset.title())
            else:
                ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel(metric if col_idx == 0 else "")
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)

    legend_handles = [Patch(facecolor=DATASET_PALETTE[label], label=label) for label in ["original", "model"]]
    fig.legend(legend_handles, ["original", "model"], title="Dataset", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Basic Metric Distributions: Original vs Model", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")

def graph_feature_histograms(results_root: Path, model: str, output_dir: Path) -> None:
    summary = safe_read_csv(feature_summary_csv(results_root, model))
    if summary is None or "feature" not in summary.columns:
        print("[WARN] No explainable feature summary found; skipping per-feature histograms.")
        return

    features = sorted(str(feature) for feature in summary["feature"].dropna().unique())
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
    frame = safe_read_csv(merged_stances_csv(results_root, model, "improvised"))
    if frame is None or not {"question_index", "score_original", "score_llm"}.issubset(frame.columns):
        print("[WARN] Missing STANCE merged scores; skipping stances.png.")
        return

    frame = frame.copy()
    frame["score_original"] = numeric(frame["score_original"])
    frame["score_llm"] = numeric(frame["score_llm"])
    questions = list(range(10))

    fig, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, qidx in zip(axes_flat, questions):
        sub = frame[frame["question_index"] == qidx]
        plot_rows = []
        if not sub.empty:
            plot_rows.append(pd.DataFrame({"score": sub["score_original"], "dataset": "original"}))
            plot_rows.append(pd.DataFrame({"score": sub["score_llm"], "dataset": "model"}))
        if plot_rows:
            plot_data = pd.concat(plot_rows, ignore_index=True).dropna(subset=["score"])
            if not plot_data.empty:
                sns.histplot(
                    data=plot_data,
                    x="score",
                    hue="dataset",
                    bins=np.arange(-2.5, 3.6, 1.0),
                    stat="probability",
                    multiple="dodge",
                    shrink=0.8,
                    common_norm=False,
                    ax=ax,
                )
        ax.set_title(f"Q{qidx}")
        ax.set_xlabel("STANCE score")
        ax.set_ylabel("Probability")
    fig.suptitle("STANCE Score Distributions: Original vs Model", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def graph_emotional_naturalness(results_root: Path, model: str, output_path: Path) -> None:
    subset_frames = []
    for subset in SUBSETS:
        for label_model, label in [(ORIGINAL, "original"), (model, "model")]:
            frame = safe_read_csv(naturalness_csv(results_root, label_model, subset))
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


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


def graph_explainable_scatter(results_root: Path, model: str, output_path: Path) -> None:
    summary = safe_read_csv(feature_summary_csv(results_root, model))
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
        original = safe_read_csv(feature_csv(results_root, ORIGINAL, "test", subset))
        model_frame = safe_read_csv(feature_csv(results_root, model, "test", subset))
        if original is None or model_frame is None:
            continue

        if "subset" in summary.columns:
            features = sorted(str(feature) for feature in summary.loc[summary["subset"] == subset, "feature"].dropna().unique())
        else:
            features = sorted(str(feature) for feature in summary["feature"].dropna().unique())

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
        if ax.get_legend() is not None:
            ax.get_legend().remove()
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    graphs_dir = args.output_dir / "graphs"
    graph_basic_metric_histograms(args.results_root, args.model, graphs_dir / "basic_metric_graphs")
    graph_basic_metrics_violin(args.results_root, args.model, graphs_dir / "basic_metrics.png")
    graph_feature_histograms(args.results_root, args.model, graphs_dir / "feat_graphs")
    graph_stances(args.results_root, args.model, graphs_dir / "stances.png")
    graph_emotional_naturalness(args.results_root, args.model, graphs_dir / "emo_naturalness.png")
    graph_explainable_scatter(args.results_root, args.model, graphs_dir / "explainables.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
