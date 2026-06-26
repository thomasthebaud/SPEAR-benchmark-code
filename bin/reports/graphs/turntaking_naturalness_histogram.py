#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import ORIGINAL, SUBSETS, add_common_graph_args, all_systems, numeric, result_file, safe_read_csv, save_figure, system_palette


TURNTAKING_EXPERIMENT = "group4-dualturn-full-all6-fvad256"


def score_column(columns) -> str | None:
    if "naturalness_score" in columns:
        return "naturalness_score"
    if "nat_score" in columns:
        return "nat_score"
    return None


def load_turntaking_scores(results_root, systems: list[str]) -> pd.DataFrame:
    frames = []
    for subset in SUBSETS:
        for system in systems:
            path = result_file(results_root, system, "test", subset, f"turntaking.{TURNTAKING_EXPERIMENT}.csv")
            frame = safe_read_csv(path)
            if frame is None:
                continue
            column = score_column(frame.columns)
            if column is None:
                print(f"[WARN] No turn-taking naturalness score column found in {path}.")
                continue

            values = frame.copy()
            if "status" in values.columns:
                values = values[values["status"].astype(str).str.lower() == "ok"]
            values["turntaking_naturalness_score"] = (-1.0 * numeric(values[column])).clip(lower=0, upper=10)
            values = values.dropna(subset=["turntaking_naturalness_score"])
            if values.empty:
                continue

            frames.append(
                values.assign(
                    system=system,
                    subset=subset,
                )[["turntaking_naturalness_score", "system", "subset"]]
            )
    if not frames:
        return pd.DataFrame(columns=["turntaking_naturalness_score", "system", "subset"])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: turn-taking naturalness score histograms.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    data = load_turntaking_scores(args.results_root, systems)
    if data.empty:
        print("[WARN] No turn-taking naturalness scores found.")
        return 0

    palette = system_palette(systems)
    plot_systems = [system for system in systems if system in set(data["system"])]

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.48))
    legend_handles = []
    legend_labels = []
    sns.histplot(
        data=data,
        x="turntaking_naturalness_score",
        hue="system",
        hue_order=plot_systems,
        palette={system: palette[system] for system in plot_systems},
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.22,
        linewidth=1.4,
        ax=ax,
    )
    for system in plot_systems:
        sub = data.loc[data["system"] == system, "turntaking_naturalness_score"].dropna()
        if sub.nunique() < 2:
            continue
        sns.kdeplot(
            x=sub,
            color=palette[system],
            linestyle="--" if system == ORIGINAL else "-",
            linewidth=1.8,
            ax=ax,
        )
    legend = ax.get_legend()
    if legend is not None:
        legend_handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
        legend_labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
    ax.set_xlabel("Turn-taking naturalness score")
    ax.set_ylabel("Density")
    ax.tick_params(axis="both", labelsize=9)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="System",
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 1.14),
            fontsize=9,
            title_fontsize=10,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
