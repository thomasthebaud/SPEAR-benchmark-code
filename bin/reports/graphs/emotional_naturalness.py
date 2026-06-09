#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    RELATIONSHIP_ORDER,
    add_common_graph_args,
    all_systems,
    load_naturalness_relationship_values,
    load_naturalness_values,
    save_figure,
    system_palette,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: emotional naturalness.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    scores = load_naturalness_values(args.results_root, systems)
    relationships = load_naturalness_relationship_values(args.results_root, systems)
    if scores.empty and relationships.empty:
        print("[WARN] No emotional naturalness rows found.")
        return 0

    system_colors = system_palette(systems)
    systems_with_relationships = [system for system in systems if not relationships.empty and system in set(relationships["system"])]
    relationship_order = [relationship for relationship in RELATIONSHIP_ORDER if not relationships.empty and relationship in set(relationships["relationship"])]
    n_relationship_panels = len(relationship_order) if not relationships.empty else 0
    n_panels = 1 + n_relationship_panels

    fig, axes = plt.subplots(n_panels, 1, figsize=(15, max(10, 2.2 * n_panels)), squeeze=False)
    axes = axes[:, 0]
    ax_summary = axes[0]
    relationship_axes = list(axes[1:])

    if not scores.empty:
        sns.boxplot(
            data=scores,
            x="naturalness_logit",
            y="system",
            hue="system",
            order=systems,
            hue_order=systems,
            palette=system_colors,
            legend=False,
            showfliers=False,
            linewidth=0.9,
            ax=ax_summary,
        )
        ax_summary.set_title("Overall")
        ax_summary.set_xlabel("Naturalness logit" if not relationship_axes else "")
        ax_summary.set_ylabel("")
        ax_summary.tick_params(axis="x", labelsize=9)
        ax_summary.tick_params(axis="y", labelsize=9)
    else:
        ax_summary.set_axis_off()

    if not relationships.empty and relationship_order and systems_with_relationships:
        for idx, (ax, relationship) in enumerate(zip(relationship_axes, relationship_order)):
            sub = relationships[relationships["relationship"] == relationship]
            sns.boxplot(
                data=sub,
                x="naturalness_logit",
                y="system",
                hue="system",
                order=systems_with_relationships,
                hue_order=systems_with_relationships,
                palette={system: system_colors[system] for system in systems_with_relationships},
                legend=False,
                showfliers=False,
                linewidth=0.9,
                orient="h",
                ax=ax,
            )
            ax.set_title(relationship)
            ax.set_xlabel("Naturalness logit" if idx == len(relationship_order) - 1 else "")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
    else:
        for ax in relationship_axes:
            ax.set_axis_off()

    fig.suptitle("Emotional Naturalness Across Systems and by relationship", y=1.02)
    fig.tight_layout()
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
