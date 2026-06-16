#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    ORIGINAL,
    RELATIONSHIP_ORDER,
    add_common_graph_args,
    all_systems,
    load_naturalness_relationship_values,
    load_naturalness_values,
    save_figure,
    system_palette,
)


def title_with_original_n(title: str, data) -> str:
    n_original = int((data["system"] == ORIGINAL).sum()) if not data.empty and "system" in data.columns else 0
    return f"{title} (n={n_original:,})"


def companion_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def draw_naturalness_panel(ax, data, title: str, plot_systems: list[str], system_colors: dict[str, str], *, kind: str) -> None:
    if data.empty or not plot_systems:
        ax.set_axis_off()
        return
    common_kwargs = dict(
        data=data,
        x="naturalness_logit",
        y="system",
        hue="system",
        order=plot_systems,
        hue_order=plot_systems,
        palette={system: system_colors[system] for system in plot_systems},
        legend=False,
        linewidth=0.9,
        ax=ax,
    )
    if kind == "violin":
        sns.violinplot(**common_kwargs, inner="quartile", cut=0)
    else:
        sns.boxplot(**common_kwargs, showfliers=False)
    ax.set_title(title_with_original_n(title, data))
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)


def draw_summary_panels(axes, scores, subset_order: list[str], systems: list[str], system_colors: dict[str, str], *, kind: str) -> None:
    summary_axes = list(axes)
    if not summary_axes:
        return
    draw_naturalness_panel(summary_axes[0], scores, "Overall", systems, system_colors, kind=kind)
    for ax, subset in zip(summary_axes[1:], subset_order):
        sub = scores[scores["subset"] == subset]
        plot_systems = [system for system in systems if system in set(sub["system"])]
        draw_naturalness_panel(ax, sub, subset.title(), plot_systems, system_colors, kind=kind)
    for ax in summary_axes[:-1]:
        ax.set_xlabel("")
    summary_axes[-1].set_xlabel("Naturalness logit")


def save_short_figure(scores, subset_order: list[str], systems: list[str], system_colors: dict[str, str], output_path: Path, *, kind: str) -> None:
    n_panels = 1 + len(subset_order)
    if scores.empty or n_panels == 0:
        return
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, max(7.5, 2.8 * n_panels)), squeeze=False)
    axes = axes[:, 0]
    draw_summary_panels(axes, scores, subset_order, systems, system_colors, kind=kind)
    title_suffix = " Violin Plots" if kind == "violin" else ""
    fig.suptitle(f"Emotional Naturalness Across Systems{title_suffix}", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


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
    subset_order = [subset for subset in ["naturalistic", "improvised"] if not scores.empty and subset in set(scores["subset"])]
    systems_with_relationships = [system for system in systems if not relationships.empty and system in set(relationships["system"])]
    relationship_order = [relationship for relationship in RELATIONSHIP_ORDER if not relationships.empty and relationship in set(relationships["relationship"])]
    n_relationship_panels = len(relationship_order) if not relationships.empty else 0
    n_summary_panels = 1 + len(subset_order)
    n_panels = n_summary_panels + n_relationship_panels

    fig, axes = plt.subplots(n_panels, 1, figsize=(15, max(10, 2.2 * n_panels)), squeeze=False)
    axes = axes[:, 0]
    summary_axes = list(axes[:n_summary_panels])
    relationship_axes = list(axes[n_summary_panels:])

    if not scores.empty:
        draw_summary_panels(summary_axes, scores, subset_order, systems, system_colors, kind="box")
        if relationship_axes:
            summary_axes[-1].set_xlabel("")
    else:
        for ax in summary_axes:
            ax.set_axis_off()

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
            ax.set_title(title_with_original_n(relationship, sub))
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

    save_short_figure(
        scores,
        subset_order,
        systems,
        system_colors,
        companion_output_path(args.output_path, "short"),
        kind="box",
    )
    save_short_figure(
        scores,
        subset_order,
        systems,
        system_colors,
        companion_output_path(args.output_path, "violin"),
        kind="violin",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
