#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from utils import add_common_graph_args, all_systems, load_stance_values, save_figure, system_palette


def companion_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def stance_question_labels(data: pd.DataFrame, questions: list[int]) -> list[str]:
    label_frame = data.copy()
    label_frame["question_index"] = pd.to_numeric(label_frame["question_index"], errors="coerce")
    labels = []
    for qidx in questions:
        values = (
            label_frame.loc[label_frame["question_index"] == qidx, "stance_label"]
            .dropna()
            .astype(str)
            .str.strip()
            if "stance_label" in label_frame.columns
            else pd.Series(dtype=str)
        )
        values = values[values != ""]
        label = values.iloc[0] if not values.empty else f"Q{qidx}"
        label = label.replace("|", " ").replace("/", " ").split()[0] if label.strip() else f"Q{qidx}"
        labels.append(label)
    return labels


def draw_stance_radar(
    ax_radar,
    data: pd.DataFrame,
    systems_with_data: list[str],
    palette: dict[str, str],
    questions: list[int],
    question_labels: list[str],
    *,
    legend_anchor=(-0.22, 1.05),
    label_size=10,
    legend_size: int | None = None,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(question_labels), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    for system in systems_with_data:
        sub = data[data["system"] == system].copy()
        sub["question_index"] = pd.to_numeric(sub["question_index"], errors="coerce").astype("Int64")
        sub["positive"] = (sub["system_polarity"] == "positive").astype(float)
        rates = sub.groupby("question_index")["positive"].mean().mul(100.0).reindex(questions).fillna(0.0).to_numpy(dtype=float)
        closed_rates = np.concatenate([rates, rates[:1]])
        ax_radar.plot(closed_angles, closed_rates, color=palette[system], linewidth=2.0, label=system)
        ax_radar.fill(closed_angles, closed_rates, color=palette[system], alpha=0.08)
    ax_radar.set_title("Positive STANCE Rate by Question", pad=24)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(question_labels, fontsize=label_size)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([0, 25, 50, 75, 100])
    ax_radar.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=max(8, label_size - 2))
    ax_radar.grid(True, alpha=0.45)
    legend_fontsize = legend_size if legend_size is not None else max(8, label_size - 2)
    ax_radar.legend(
        title="Model",
        loc="upper left",
        ncol=1,
        bbox_to_anchor=legend_anchor,
        frameon=True,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 1,
        borderpad=0.55,
        labelspacing=0.55,
        handlelength=1.7,
    )


def save_spider_only_figure(
    data: pd.DataFrame,
    systems_with_data: list[str],
    palette: dict[str, str],
    questions: list[int],
    question_labels: list[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 13))
    ax_radar = fig.add_subplot(1, 1, 1, projection="polar")
    draw_stance_radar(ax_radar, data, systems_with_data, palette, questions, question_labels, legend_anchor=(-0.18, 1.08), label_size=12, legend_size=12)
    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: stances.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    data = load_stance_values(args.results_root, systems)
    if data.empty:
        print("[WARN] No merged stance rows found.")
        return 0

    systems_with_data = [system for system in systems if system in set(data["system"])]
    palette = system_palette(systems)
    questions = sorted(int(qidx) for qidx in pd.to_numeric(data["question_index"], errors="coerce").dropna().unique())
    if not questions:
        print("[WARN] No stance question indices found.")
        return 0
    question_labels = stance_question_labels(data, questions)

    n_heatmaps = max(1, len(systems_with_data))
    heatmap_cols = max(1, int(np.ceil(n_heatmaps / 2)))
    fig = plt.figure(figsize=(max(10, 4.4 * heatmap_cols), 18))
    gs = fig.add_gridspec(3, heatmap_cols, height_ratios=[3.0, 1.0, 1.0])
    ax_radar = fig.add_subplot(gs[0, :], projection="polar")
    heatmap_axes = [fig.add_subplot(gs[1 + idx // heatmap_cols, idx % heatmap_cols]) for idx in range(n_heatmaps)]
    for idx in range(n_heatmaps, 2 * heatmap_cols):
        ax = fig.add_subplot(gs[1 + idx // heatmap_cols, idx % heatmap_cols])
        ax.set_axis_off()

    draw_stance_radar(ax_radar, data, systems_with_data, palette, questions, question_labels)

    polarity = ["negative", "positive"]
    for ax, system in zip(heatmap_axes, systems_with_data):
        sub = data[data["system"] == system]
        if sub.empty:
            ax.set_axis_off()
            continue
        counts = pd.crosstab(sub["original_polarity"], sub["system_polarity"]).reindex(index=polarity, columns=polarity, fill_value=0)
        total = counts.to_numpy().sum()
        values = counts / total * 100.0 if total else counts
        cmap = LinearSegmentedColormap.from_list(f"{system}_stance_confusion", ["#ffffff", palette[system]])
        sns.heatmap(values, annot=True, fmt=".1f", cmap=cmap, cbar=False, linewidths=0.5, linecolor="white", square=True, ax=ax)
        ax.set_title(system)
        ax.set_xlabel("Answer stance")
        ax.set_ylabel("Question stance")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("STANCE Polarity Across Systems", y=1.02)
    fig.tight_layout()
    save_figure(fig, args.output_path)
    save_spider_only_figure(data, systems_with_data, palette, questions, question_labels, companion_output_path(args.output_path, "spider"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
