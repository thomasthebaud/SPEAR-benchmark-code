#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    add_common_graph_args,
    all_systems,
    available_base_metric_columns,
    load_base_metric_values,
    remove_iqr_outliers,
    save_figure,
    system_palette,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: intelligibility and speech quality.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    available = available_base_metric_columns(args.results_root, systems)
    metrics = [metric for metric in available if metric.startswith(("WER", "CER")) or metric == "UTMOS"]
    data = load_base_metric_values(args.results_root, systems, metrics)
    if data.empty:
        print("[WARN] No intelligibility or speech quality metrics found.")
        return 0

    data = data.copy()
    data["metric_group"] = data["metric"].astype(str).map(
        lambda metric: "CER" if metric.startswith("CER") else ("WER" if metric.startswith("WER") else metric)
    )
    data = data[data["metric_group"].isin(["CER", "WER", "UTMOS"])]
    data = remove_iqr_outliers(data, ["subset", "metric_group", "system"])
    if data.empty:
        print("[WARN] No intelligibility or speech quality rows remain after filtering.")
        return 0

    metric_order = [metric for metric in ["CER", "WER", "UTMOS"] if metric in set(data["metric_group"])]
    subset_order = [subset for subset in ["improvised", "naturalistic"] if subset in set(data["subset"])]
    palette = system_palette(systems)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), squeeze=False, sharey=False)
    for row_idx, metric in enumerate(["CER", "WER", "UTMOS"]):
        for col_idx, subset in enumerate(["improvised", "naturalistic"]):
            ax = axes[row_idx, col_idx]
            if metric not in metric_order or subset not in subset_order:
                ax.set_axis_off()
                continue
            sub = data[(data["metric_group"] == metric) & (data["subset"] == subset)]
            if sub.empty:
                ax.set_axis_off()
                continue
            plot_systems = [system for system in systems if not (metric in {"CER", "WER"} and system == "mini-omni")]
            sub = sub[sub["system"].isin(plot_systems)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sns.boxplot(
                data=sub,
                x="system",
                y="value",
                hue="system",
                order=plot_systems,
                hue_order=plot_systems,
                palette=palette,
                legend=False,
                showfliers=False,
                linewidth=0.9,
                ax=ax,
            )
            ax.set_title(subset.title() if row_idx == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(f"{metric} (%)" if metric in {"CER", "WER"} else metric)
            ax.tick_params(axis="x", rotation=25, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("Intelligibility and Speech Quality Across Systems", y=1.02)
    fig.tight_layout()
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
