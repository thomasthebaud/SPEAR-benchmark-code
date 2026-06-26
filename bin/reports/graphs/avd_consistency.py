#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from utils import ORIGINAL, add_common_graph_args, all_systems, load_avd_values, remove_axis_legend, save_figure, system_palette


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

    fig, plot_axes = plt.subplots(
        1,
        3,
        figsize=(22, 6.6),
        sharex=True,
        sharey=False,
        squeeze=False,
        gridspec_kw={"wspace": 0.08},
    )
    plot_axes = plot_axes[0]
    for idx, (ax, emotion) in enumerate(zip(plot_axes, emotions)):
        sub = data[data["emotion"] == emotion]
        sns.scatterplot(data=sub, x="question", y="answer", hue="system", hue_order=systems, palette=palette, alpha=0.28, s=14, edgecolor="none", ax=ax)
        remove_axis_legend(ax)
        for system in systems:
            system_data = sub[sub["system"] == system].dropna(subset=["question", "answer"])
            if len(system_data) < 2:
                continue
            line_style = "--" if system == ORIGINAL else "-"
            sns.regplot(
                data=system_data,
                x="question",
                y="answer",
                scatter=False,
                ci=None,
                color=palette[system],
                line_kws={"linewidth": 1.8, "linestyle": line_style},
                ax=ax,
            )
        ax.plot([-1, 1], [-1, 1], color="#555555", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "+1"])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["-1", "0", "+1"])
        ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=(idx == 0))
        ax.yaxis.set_tick_params(labelleft=(idx == 0))
        for label in ax.get_yticklabels():
            label.set_visible(idx == 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(emotion_titles[emotion])
        ax.set_xlabel("Question")
        ax.set_ylabel("Answer" if idx == 0 else "")

    handles = [
        Line2D(
            [0],
            [0],
            color=palette[system],
            marker="o",
            linestyle="--" if system == ORIGINAL else "-",
            linewidth=2.0,
            markersize=6,
            label=system,
        )
        for system in systems
    ]
    fig.legend(
        handles=handles,
        title="System",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=max(1, (len(systems) + 1) // 2),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=0.4)
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
