#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import add_common_graph_args, all_systems, numeric, result_file, safe_read_csv, save_figure, system_palette


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


def profile_rows(frame: pd.DataFrame, label: str, subset: str) -> list[dict[str, float | str]]:
    rows = []
    for idx, feature in enumerate(F0_PROFILE_FEATURES):
        if feature not in frame.columns:
            continue
        values = numeric(frame[feature]).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "label": label,
                "subset": subset,
                "feature": feature,
                "x": idx,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return rows


def load_f0_profiles(results_root: Path, systems: list[str]) -> pd.DataFrame:
    rows = []
    model_systems = [system for system in systems if system != ORIGINAL]
    for subset in ["improvised", "naturalistic"]:
        original_answers = read_normalized_or_raw_f0(results_root, ORIGINAL, subset, raw_fallback=True)
        if original_answers is not None:
            rows.extend(profile_rows(original_answers, "original answers", subset))

        original_questions = read_normalized_or_raw_f0(results_root, ORIGINAL, subset, questions=True)
        if original_questions is not None:
            rows.extend(profile_rows(original_questions, "original questions", subset))

        for system in model_systems:
            frame = read_normalized_or_raw_f0(results_root, system, subset)
            if frame is not None:
                rows.extend(profile_rows(frame, system, subset))
    return pd.DataFrame(rows)


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
    labels = ["original questions", "original answers"] + [system for system in systems if system != ORIGINAL]
    labels = [label for label in labels if label in set(data["label"])]
    palette = system_palette([ORIGINAL] + [system for system in systems if system != ORIGINAL])
    colors = {"original answers": palette[ORIGINAL], "original questions": "#6B6B6B"}
    colors.update({system: palette[system] for system in systems if system != ORIGINAL and system in palette})

    fig, axes = plt.subplots(1, len(subsets), figsize=(9.5 * len(subsets), 12.0), sharey=True, squeeze=False)
    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset]
        for label in labels:
            series = sub[sub["label"] == label].sort_values("x")
            if series.empty:
                continue
            x = series["x"].to_numpy(dtype=float)
            mean = series["mean"].to_numpy(dtype=float)
            std = series["std"].to_numpy(dtype=float)
            color = colors.get(label)
            ax.plot(x, mean, marker="o", linewidth=2.0, label=label, color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        ax.set_title(subset.title())
        ax.set_xticks(range(len(F0_PROFILE_FEATURES)))
        ax.set_xticklabels(F0_PROFILE_FEATURES, rotation=30, ha="right")
        ax.set_xlabel("f0 feature")
        ax.grid(True, axis="y", alpha=0.35)
    axes[0, 0].set_ylabel("Normalized f0 value (mean +/- std)")

    handles, labels_seen = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_seen, title="System", loc="upper center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 1.03), frameon=True)
    fig.suptitle("Normalized f0 Profiles Across Systems", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
