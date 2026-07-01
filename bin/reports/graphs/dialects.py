#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from utils import (
    DIALECT_PROFILE_LABELS,
    ORIGINAL,
    add_common_graph_args,
    all_systems,
    load_dialect_change_values,
    load_dialect_score_values,
    numeric,
    save_figure,
    system_palette,
)


DIALECT_PROFILE_PLOT_LABELS = DIALECT_PROFILE_LABELS


def companion_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def draw_dialect_profile(ax_radar, scores, systems: list[str], palette: dict[str, str]) -> list[Line2D]:
    legend_handles = []
    if scores.empty:
        ax_radar.set_axis_off()
        return legend_handles

    scores = scores.copy()
    scores["score"] = numeric(scores["score"])
    profile = scores.groupby(["system", "dialect"], dropna=False)["score"].median().unstack("dialect").reindex(index=systems, columns=DIALECT_PROFILE_PLOT_LABELS)
    angles = np.linspace(0, 2 * np.pi, len(DIALECT_PROFILE_PLOT_LABELS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    score_floor = 1e-6
    max_score = float(np.nanmax(profile.to_numpy())) if profile.notna().any().any() else 1.0
    max_score = max(max_score, score_floor * 10.0)
    for system in systems:
        if system not in profile.index or profile.loc[system].isna().all():
            continue
        values = profile.loc[system].fillna(score_floor).clip(lower=score_floor).to_numpy(dtype=float)
        closed_values = np.concatenate([values, values[:1]])
        linestyle = "--" if system == ORIGINAL else "-"
        ax_radar.plot(closed_angles, closed_values, color=palette[system], linewidth=2.0, linestyle=linestyle, label=system)
        if system == ORIGINAL:
            ax_radar.fill(closed_angles, closed_values, color=palette[system], alpha=0.08)
        legend_handles.append(Line2D([0], [0], color=palette[system], linewidth=2.0, linestyle=linestyle, label=system))
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(DIALECT_PROFILE_PLOT_LABELS, fontsize=22)
    ax_radar.set_yscale("log")
    ax_radar.set_ylim(score_floor, max_score * 1.08)
    ax_radar.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax_radar.set_yticklabels(["1e-6", "", "1e-4", "", "1e-2", "", "1"], fontsize=20)
    ax_radar.grid(True, alpha=0.45)
    return legend_handles


def add_dialect_legend(
    ax_radar,
    legend_handles: list[Line2D],
    *,
    bbox_to_anchor=(0.5, 1.32),
    fontsize: int = 9,
    title_fontsize: int = 10,
) -> None:
    if not legend_handles:
        return
    ax_radar.legend(
        handles=legend_handles,
        title="System",
        loc="upper center",
        ncol=len(legend_handles),
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=1.6,
    )


def save_nochange_figure(scores, systems: list[str], palette: dict[str, str], output_path: Path) -> None:
    if scores.empty:
        return
    fig = plt.figure(figsize=(21, 18))
    ax_radar = fig.add_subplot(1, 1, 1, projection="polar")
    add_dialect_legend(
        ax_radar,
        draw_dialect_profile(ax_radar, scores, systems, palette),
        bbox_to_anchor=(0.5, 1.28),
        fontsize=12,
        title_fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, output_path)


def save_pointplot_figure(scores, systems: list[str], palette: dict[str, str], output_path: Path) -> None:
    if scores.empty:
        return

    plot_data = scores.copy()
    plot_data["score"] = numeric(plot_data["score"]).clip(lower=1e-6)
    plot_data = plot_data[plot_data["system"].isin(systems) & plot_data["dialect"].isin(DIALECT_PROFILE_PLOT_LABELS)]
    plot_data = plot_data.dropna(subset=["score"])
    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(24, 10))
    sns.pointplot(
        data=plot_data,
        x="dialect",
        y="score",
        hue="system",
        order=DIALECT_PROFILE_PLOT_LABELS,
        hue_order=systems,
        palette=palette,
        dodge=0.35,
        errorbar=("ci", 95),
        capsize=0.08,
        err_kws={"linewidth": 1.2},
        markers="o",
        linestyles=["--" if system == ORIGINAL else "-" for system in systems],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Dialect probability", fontsize=30)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticklabels(["1e-6", "", "1e-4", "", "1e-2", "", "1"])
    ax.tick_params(axis="x", rotation=25, labelsize=23)
    ax.tick_params(axis="y", labelsize=23)
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
    parser = argparse.ArgumentParser(description="Article graph: dialects.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    palette = system_palette(systems)
    scores = load_dialect_score_values(args.results_root, systems)
    changes = load_dialect_change_values(args.results_root, systems)
    if scores.empty and changes.empty:
        print("[WARN] No dialect rows found.")
        return 0

    fig = plt.figure(figsize=(21, 24))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_bar = fig.add_subplot(gs[1, 0])

    legend_handles = draw_dialect_profile(ax_radar, scores, systems, palette)

    if not changes.empty:
        summary = (
            changes.groupby("system", dropna=False)["changed"]
            .mean()
            .mul(100.0)
            .reset_index()
            .rename(columns={"changed": "changed_percent"})
        )
        sns.barplot(data=summary, x="system", y="changed_percent", hue="system", order=systems, hue_order=systems, palette=palette, legend=False, ax=ax_bar)
        ax_bar.set_xlabel("")
        ax_bar.set_ylabel("Changed (%)", fontsize=22)
        ax_bar.tick_params(axis="x", rotation=25, labelsize=22)
        ax_bar.tick_params(axis="y", labelsize=18)
    else:
        ax_bar.set_axis_off()

    add_dialect_legend(ax_radar, legend_handles)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, args.output_path)
    save_nochange_figure(scores, systems, palette, args.output_path.with_name("stage3_dialects_nochange.png"))
    save_pointplot_figure(scores, systems, palette, companion_output_path(args.output_path, "point"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
