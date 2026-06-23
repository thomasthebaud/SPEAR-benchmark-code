#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from utils import add_common_graph_args, all_systems, correlation_label, load_avd_values, remove_axis_legend, save_figure, system_palette


def draw_colored_rho_labels(ax, sub, systems: list[str], palette: dict[str, str]) -> None:
    y = 0.96
    for system in systems:
        rho = correlation_label(sub[sub["system"] == system])
        rho_text = "nan" if rho != rho else f"{rho:.2f}"
        ax.text(0.03, y, f"rho={rho_text}", transform=ax.transAxes, color=palette[system], ha="left", va="center", fontsize=9)
        y -= 0.065


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: AVD consistency.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    data = load_avd_values(args.results_root, systems)
    if data.empty:
        print("[WARN] No SER_AVD rows found.")
        return 0
    emotions = ["arousal", "dominance", "valence"]
    emotion_titles = {"arousal": "Arousal", "dominance": "Dominance", "valence": "Valence"}
    palette = system_palette(systems)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True, squeeze=False)
    plot_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    legend_ax = axes[1, 1]
    for ax, emotion in zip(plot_axes, emotions):
        sub = data[data["emotion"] == emotion]
        sns.scatterplot(data=sub, x="question", y="answer", hue="system", hue_order=systems, palette=palette, alpha=0.28, s=14, edgecolor="none", ax=ax)
        remove_axis_legend(ax)
        for system in systems:
            system_data = sub[sub["system"] == system].dropna(subset=["question", "answer"])
            if len(system_data) < 2:
                continue
            sns.regplot(data=system_data, x="question", y="answer", scatter=False, ci=None, color=palette[system], line_kws={"linewidth": 1.8}, ax=ax)
        draw_colored_rho_labels(ax, sub, systems, palette)
        ax.plot([-1, 1], [-1, 1], color="#555555", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "+1"])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["-1", "0", "+1"])
        ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(emotion_titles[emotion])
        ax.set_xlabel(f"Question {emotion_titles[emotion]}")
        ax.set_ylabel(f"Answer {emotion_titles[emotion]}")

    legend_ax.set_axis_off()
    handles = [Line2D([0], [0], color=palette[system], marker="o", linestyle="-", linewidth=2.0, markersize=6, label=system) for system in systems]
    legend_ax.legend(handles=handles, title="System", loc="center", frameon=True)
    fig.tight_layout()
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
