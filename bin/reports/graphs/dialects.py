#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from utils import (
    DIALECT_PROFILE_LABELS,
    add_common_graph_args,
    all_systems,
    load_dialect_change_values,
    load_dialect_score_values,
    numeric,
    save_figure,
    system_palette,
)


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

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_bar = fig.add_subplot(gs[1, 0])

    legend_handles = []
    if not scores.empty:
        scores = scores.copy()
        scores["score"] = numeric(scores["score"])
        profile = scores.groupby(["system", "dialect"], dropna=False)["score"].median().unstack("dialect").reindex(index=systems, columns=DIALECT_PROFILE_LABELS)
        angles = np.linspace(0, 2 * np.pi, len(DIALECT_PROFILE_LABELS), endpoint=False)
        closed_angles = np.concatenate([angles, angles[:1]])
        score_floor = 1e-5
        max_score = float(np.nanmax(profile.to_numpy())) if profile.notna().any().any() else 1.0
        max_score = max(max_score, score_floor * 10.0)
        for system in systems:
            if system not in profile.index or profile.loc[system].isna().all():
                continue
            values = profile.loc[system].fillna(score_floor).clip(lower=score_floor).to_numpy(dtype=float)
            closed_values = np.concatenate([values, values[:1]])
            ax_radar.plot(closed_angles, closed_values, color=palette[system], linewidth=2.0, label=system)
            ax_radar.fill(closed_angles, closed_values, color=palette[system], alpha=0.08)
            legend_handles.append(Line2D([0], [0], color=palette[system], linewidth=2.0, label=system))
        ax_radar.set_title("Median Answer Dialect Score Profile", pad=24)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(DIALECT_PROFILE_LABELS, fontsize=11)
        ax_radar.set_yscale("log")
        ax_radar.set_ylim(score_floor, max_score * 1.08)
        ax_radar.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        ax_radar.set_yticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1"], fontsize=10)
        ax_radar.grid(True, alpha=0.45)
    else:
        ax_radar.set_axis_off()

    if not changes.empty:
        summary = (
            changes.groupby("system", dropna=False)["changed"]
            .mean()
            .mul(100.0)
            .reset_index()
            .rename(columns={"changed": "changed_percent"})
        )
        sns.barplot(data=summary, x="system", y="changed_percent", hue="system", order=systems, hue_order=systems, palette=palette, legend=False, ax=ax_bar)
        ax_bar.set_title("Question-to-Answer Dialect Change Rate")
        ax_bar.set_xlabel("")
        ax_bar.set_ylabel("Changed (%)")
        ax_bar.tick_params(axis="x", rotation=25, labelsize=11)
        ax_bar.tick_params(axis="y", labelsize=9)
    else:
        ax_bar.set_axis_off()

    if legend_handles:
        ax_radar.legend(
            handles=legend_handles,
            title="System",
            loc="upper left",
            ncol=1,
            bbox_to_anchor=(-0.32, 1.12),
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            borderpad=0.4,
            labelspacing=0.4,
            handlelength=1.4,
        )
    fig.suptitle("Dialects Across Systems", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
