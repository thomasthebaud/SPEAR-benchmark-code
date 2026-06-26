#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import add_common_graph_args, all_systems, numeric, remove_axis_legend, result_file, safe_read_csv, save_figure, system_palette


ORIGINAL = "original"
F0_PROFILE_FEATURES = [
    "f0_min_raw",
    "f0_p10",
    "f0_p25",
    "f0_median_raw",
    "f0_p75",
    "f0_p90",
    "f0_max_raw",
]
F0_PROFILE_LABELS = ["min", "10%", "25%", "50%", "75%", "90%", "max"]


def normalized_feature_file(results_root: Path, system: str, subset: str, *, questions: bool = False) -> Path:
    filename = "distrib_baselines_features_normalized_q.csv" if questions else "distrib_baselines_features_normalized.csv"
    return result_file(results_root, system, "test", subset, filename)


def raw_feature_file(results_root: Path, system: str, subset: str) -> Path:
    return result_file(results_root, system, "test", subset, "distrib_baselines_features.csv")


def normalize_raw_f0_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "f0_mean_raw" not in frame.columns:
        return frame
    denominator = numeric(frame["f0_mean_raw"]).dropna().mean()
    if pd.isna(denominator) or denominator == 0:
        return frame
    normalized = frame.copy()
    for feature in F0_PROFILE_FEATURES:
        if feature in normalized.columns:
            normalized[feature] = numeric(normalized[feature]) / denominator
    return normalized


def read_normalized_or_raw_f0(results_root: Path, system: str, subset: str, *, questions: bool = False, raw_fallback: bool = False) -> pd.DataFrame | None:
    frame = safe_read_csv(normalized_feature_file(results_root, system, subset, questions=questions))
    if frame is not None:
        return frame
    if not raw_fallback:
        return None
    raw = safe_read_csv(raw_feature_file(results_root, system, subset))
    return normalize_raw_f0_frame(raw) if raw is not None else None


def profile_value_rows(frame: pd.DataFrame, label: str, subset: str) -> list[pd.DataFrame]:
    rows = []
    for idx, feature in enumerate(F0_PROFILE_FEATURES):
        if feature not in frame.columns:
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
                    "feature_label": F0_PROFILE_LABELS[idx],
                    "x": idx,
                    "value": values.to_numpy(dtype=float),
                }
            )
        )
    return rows


def load_f0_profiles(results_root: Path, systems: list[str]) -> pd.DataFrame:
    rows = []
    combined_frames: dict[str, list[pd.DataFrame]] = {}
    model_systems = [system for system in systems if system != ORIGINAL]
    for subset in ["improvised", "naturalistic"]:
        original_answers = read_normalized_or_raw_f0(results_root, ORIGINAL, subset, raw_fallback=True)
        if original_answers is not None:
            rows.extend(profile_value_rows(original_answers, "human", subset))
            combined_frames.setdefault("human", []).append(original_answers)

        original_questions = read_normalized_or_raw_f0(results_root, ORIGINAL, subset, questions=True)
        if original_questions is not None:
            rows.extend(profile_value_rows(original_questions, "human", subset))
            combined_frames.setdefault("human", []).append(original_questions)

        for system in model_systems:
            frame = read_normalized_or_raw_f0(results_root, system, subset)
            if frame is not None:
                rows.extend(profile_value_rows(frame, system, subset))
                combined_frames.setdefault(system, []).append(frame)

    for label, frames in combined_frames.items():
        if frames:
            rows.extend(profile_value_rows(pd.concat(frames, ignore_index=True), label, "combined"))
    if not rows:
        return pd.DataFrame(columns=["label", "subset", "feature", "feature_label", "x", "value"])
    return pd.concat(rows, ignore_index=True)


def companion_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def save_boxplot_figure(
    data: pd.DataFrame,
    subsets: list[str],
    labels: list[str],
    colors: dict[str, str],
    output_path: Path,
    *,
    title: str | None = None,
    legend_inside: bool = False,
    title_as_subplot: bool = False,
) -> None:
    if not subsets:
        return
    fig_height = 5.7 if len(subsets) == 1 else 6.4
    fig, axes = plt.subplots(1, len(subsets), figsize=(8.5 * len(subsets), fig_height), sharey=True, squeeze=False)
    available_features = set(data["feature"])
    feature_order = [label for feature, label in zip(F0_PROFILE_FEATURES, F0_PROFILE_LABELS) if feature in available_features]
    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset]
        if sub.empty:
            continue
        sns.boxplot(
            data=sub,
            x="feature_label",
            y="value",
            hue="label",
            order=feature_order,
            hue_order=labels,
            palette=colors,
            showfliers=False,
            linewidth=1.0,
            width=0.78,
            ax=ax,
        )
        subplot_title = subset.title()
        if title_as_subplot and title and len(subsets) == 1:
            subplot_title = title
        ax.set_title(subplot_title)
        ax.set_xlabel("f0 feature")
        ax.grid(True, axis="y", alpha=0.35)
        remove_axis_legend(ax)
    axes[0, 0].set_ylabel("Normalized f0 value")

    handles, labels_seen = axes[0, 0].get_legend_handles_labels()
    if handles and legend_inside:
        axes[0, -1].legend(handles, labels_seen, title="System", loc="best", frameon=True, fontsize=8, title_fontsize=9)
    elif handles:
        fig.legend(handles, labels_seen, title="System", loc="upper center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 1.04), frameon=True, fontsize=9, title_fontsize=10)
    if title and not title_as_subplot:
        fig.suptitle(title, y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if not legend_inside else 1.0))
    save_figure(fig, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: explainable f0 profiles.")
    add_common_graph_args(parser)
    parser.add_argument("--max-features", type=int, default=18, help=argparse.SUPPRESS)
    args = parser.parse_args()

    systems = all_systems(args.models)
    data = load_f0_profiles(args.results_root, systems)
    if data.empty:
        print("[WARN] No normalized f0 profile values found.")
        return 0

    subsets = [subset for subset in ["improvised", "naturalistic"] if subset in set(data["subset"])]
    labels = ["human"] + [system for system in systems if system != ORIGINAL]
    labels = [label for label in labels if label in set(data["label"])]
    palette = system_palette([ORIGINAL] + [system for system in systems if system != ORIGINAL])
    colors = {"human": palette[ORIGINAL]}
    colors.update({system: palette[system] for system in systems if system != ORIGINAL and system in palette})

    ignored_features = set(args.ignore_features or [])
    if ignored_features:
        data = data[~data["feature"].isin(ignored_features)]
        if data.empty:
            print("[WARN] No normalized f0 profile values left after applying ignored features.")
            return 0

    save_boxplot_figure(data, subsets, labels, colors, args.output_path, title="Normalized f0 Profiles Across Systems")
    if "combined" in set(data["subset"]):
        save_boxplot_figure(
            data,
            ["combined"],
            labels,
            colors,
            companion_output_path(args.output_path, "combined"),
            title="Normalized f0 Profiles Across Systems (Combined)",
            legend_inside=True,
            title_as_subplot=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
