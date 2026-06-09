#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils import add_common_graph_args, all_systems, load_base_metric_values, remove_iqr_outliers, save_figure, system_palette


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: interruptions and latency.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    streaming_systems = [system for system in ["original", "gpt-realtime-2"] if system in systems]
    if len(streaming_systems) < 2:
        print("[WARN] Expected original and gpt-realtime-2 for streaming latency/interrupted plots.")

    continuous_data = load_base_metric_values(args.results_root, streaming_systems, ["latency", "interrupted"])
    interruptions_data = load_base_metric_values(args.results_root, systems, ["interruptions"])
    if continuous_data.empty and interruptions_data.empty:
        print("[WARN] No latency or interruption metrics found.")
        return 0

    if not continuous_data.empty:
        continuous_data = remove_iqr_outliers(continuous_data, ["metric", "system"])
    metric_order = [metric for metric in ["latency", "interrupted time (s)"] if not continuous_data.empty and metric in set(continuous_data["metric"])]
    include_interruptions = not interruptions_data.empty
    n_rows = len(metric_order) + (1 if include_interruptions else 0)
    palette = system_palette(systems)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(4.2, 4.0 * n_rows)), squeeze=False)
    row_idx = 0
    for metric in metric_order:
        ax = axes[row_idx, 0]
        row_idx += 1
        sub = continuous_data[continuous_data["metric"] == metric]
        if metric == "interrupted time (s)" and not sub.empty:
            upper = sub["value"].quantile(0.95)
            sub = sub[sub["value"] <= upper]
        metric_systems = [system for system in streaming_systems if system in set(sub["system"])]
        if sub.empty or not metric_systems:
            ax.set_axis_off()
            continue
        sns.histplot(
            data=sub,
            x="value",
            hue="system",
            hue_order=metric_systems,
            bins=35,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            palette={system: palette[system] for system in metric_systems},
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("System")
        title = f"{metric} (top 5% removed)" if metric == "interrupted time (s)" else metric
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.tick_params(axis="both", labelsize=9)

    if include_interruptions:
        ax = axes[row_idx, 0]
        interruption_rates = (
            interruptions_data.groupby("system", dropna=False)["value"]
            .mean()
            .mul(100.0)
            .reset_index()
            .rename(columns={"value": "interruption_rate"})
        )
        plot_systems = [system for system in systems if system in set(interruption_rates["system"])]
        sns.barplot(
            data=interruption_rates,
            x="system",
            y="interruption_rate",
            hue="system",
            order=plot_systems,
            hue_order=plot_systems,
            palette={system: palette[system] for system in plot_systems},
            legend=False,
            ax=ax,
        )
        for patch in ax.patches:
            height = patch.get_height()
            if height == height:
                ax.text(patch.get_x() + patch.get_width() / 2.0, height, f"{height:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title("Interruption Rate")
        ax.set_xlabel("System")
        ax.set_ylabel("Interrupted samples (%)")
        ax.set_ylim(0, max(5.0, float(interruption_rates["interruption_rate"].max()) * 1.18))
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Interruptions and Latency Across Systems", y=1.02)
    fig.tight_layout()
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
