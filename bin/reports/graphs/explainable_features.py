#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils import add_common_graph_args, all_systems, load_feature_values, remove_axis_legend, save_figure, system_palette


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: explainable features.")
    add_common_graph_args(parser)
    parser.add_argument("--max-features", type=int, default=18)
    args = parser.parse_args()

    systems = all_systems(args.models)
    data = load_feature_values(args.results_root, systems, args.ignore_features, max_features=args.max_features)
    if data.empty:
        print("[WARN] No explainable feature values found.")
        return 0
    subsets = [subset for subset in ["improvised", "naturalistic"] if subset in set(data["subset"])]
    palette = system_palette(systems)
    max_features = max(data[data["subset"] == subset]["feature"].nunique() for subset in subsets)
    fig, axes = plt.subplots(1, len(subsets), figsize=(9.5 * len(subsets), max(14, 0.90 * max_features + 5.2)), sharex=True, squeeze=False)
    legend_handles = legend_labels = None
    for ax, subset in zip(axes[0], subsets):
        sub = data[data["subset"] == subset].copy()
        feature_order = sub.groupby("feature")["value"].median().sort_values(ascending=False).index.tolist()
        sns.violinplot(data=sub, x="value", y="feature", hue="system", hue_order=systems, order=feature_order, palette=palette, orient="h", inner="quartile", cut=0, linewidth=0.8, density_norm="width", ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles and legend_handles is None:
            legend_handles, legend_labels = handles, labels
        remove_axis_legend(ax)
        ax.set_title(subset.title())
        ax.set_xlabel("Feature value normalized to [0, 1]")
        ax.set_ylabel("Feature" if ax is axes[0, 0] else "")
        if ax is not axes[0, 0]:
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlim(-0.02, 1.02)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title="System", loc="upper center", ncol=min(4, len(systems)), bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Explainable Feature Distributions Across Systems", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
