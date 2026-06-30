#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from utils import ORIGINAL, add_common_graph_args, all_systems, load_stance_values, save_figure, system_palette


STANCE_LABEL_OVERRIDES = {
    "aggression": "Calmness",
    "inhibition": "Disinhibition",
    "callousness":"Empathy",
    "disorganization":"Organization"
}
INVERTED_SCORE_LABELS = {"organization", "calmness", "disinhibition", 'politeness', 'empathy'}


def companion_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def display_stance_label(label: str) -> str:
    label = label.strip()
    if not label:
        return label
    return STANCE_LABEL_OVERRIDES.get(label.lower(), label)


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
        label = display_stance_label(label)
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
        invert_mask = np.asarray([label.lower() in INVERTED_SCORE_LABELS for label in question_labels], dtype=bool)
        rates[invert_mask] = 100.0 - rates[invert_mask]
        closed_rates = np.concatenate([rates, rates[:1]])
        line_style = "--" if system == ORIGINAL else "-"
        ax_radar.plot(closed_angles, closed_rates, color=palette[system], linestyle=line_style, linewidth=4.0, label=system)
        fill_alpha = 0.08 if system == ORIGINAL else 0.0
        ax_radar.fill(closed_angles, closed_rates, color=palette[system], alpha=fill_alpha)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(question_labels, fontsize=3 * label_size)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([0, 25, 50, 75, 100])
    ax_radar.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=16)
    ax_radar.grid(True, alpha=0.45, linewidth=2.0)
    ax_radar.tick_params(axis="both", width=2.0, length=8)
    ax_radar.legend(
        loc="upper center",
        ncol=len(systems_with_data),
        bbox_to_anchor=legend_anchor,
        frameon=True,
        fontsize=16,
        borderpad=0.2,
        labelspacing=0.4,
        columnspacing=1.0,
        handlelength=1.6,
    )


def save_spider_only_figure(
    data: pd.DataFrame,
    systems_with_data: list[str],
    palette: dict[str, str],
    questions: list[int],
    question_labels: list[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(20,20))
    ax_radar = fig.add_subplot(1, 1, 1, projection="polar")
    draw_stance_radar(ax_radar, data, systems_with_data, palette, questions, question_labels, legend_anchor=(0.5, 1.1), label_size=12, legend_size=12)
    fig.tight_layout()
    save_figure(fig, output_path)


def stance_rate_frame(
    data: pd.DataFrame,
    systems_with_data: list[str],
    questions: list[int],
    question_labels: list[str],
) -> pd.DataFrame:
    rows = []
    label_by_question = dict(zip(questions, question_labels))
    for system in systems_with_data:
        sub = data[data["system"] == system].copy()
        sub["question_index"] = pd.to_numeric(sub["question_index"], errors="coerce").astype("Int64")
        sub = sub[sub["question_index"].isin(questions)]
        if sub.empty:
            continue
        sub["stance"] = sub["question_index"].astype(int).map(label_by_question)
        sub["score"] = (sub["system_polarity"] == "positive").astype(float).mul(100.0)
        invert_mask = sub["stance"].astype(str).str.lower().isin(INVERTED_SCORE_LABELS)
        sub.loc[invert_mask, "score"] = 100.0 - sub.loc[invert_mask, "score"]
        rows.append(sub[["system", "stance", "score"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["system", "stance", "score"])


def save_pointplot_figure(
    data: pd.DataFrame,
    systems_with_data: list[str],
    palette: dict[str, str],
    questions: list[int],
    question_labels: list[str],
    output_path: Path,
) -> None:
    plot_data = stance_rate_frame(data, systems_with_data, questions, question_labels)
    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(20, 9))
    sns.pointplot(
        data=plot_data,
        x="stance",
        y="score",
        hue="system",
        order=question_labels,
        hue_order=systems_with_data,
        palette=palette,
        dodge=0.35,
        errorbar=("ci", 95),
        capsize=0.08,
        err_kws={"linewidth": 1.2},
        markers="o",
        linestyles=["--" if system == ORIGINAL else "-" for system in systems_with_data],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Positive stance (%)", fontsize=30)
    ax.set_ylim(40, 100)
    ax.set_yticks([50, 75, 100])
    ax.set_yticklabels(["50%", "75%", "100%"])
    ax.tick_params(axis="x", rotation=25, labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(True, axis="y", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=len(handles)//2,
            frameon=True,
            fontsize=16,
            borderpad=0.25,
            columnspacing=1.0,
            handlelength=1.6,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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

    draw_stance_radar(ax_radar, data, systems_with_data, palette, questions, question_labels, legend_anchor=(0.5, 1.1))

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
        ax.tick_params(axis="x", rotation=0, labelsize=18, width=2.0, length=8)
        ax.tick_params(axis="y", rotation=0, labelsize=18, width=2.0, length=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, args.output_path)
    save_spider_only_figure(data, systems_with_data, palette, questions, question_labels, companion_output_path(args.output_path, "spider"))
    save_pointplot_figure(data, systems_with_data, palette, questions, question_labels, companion_output_path(args.output_path, "point"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
