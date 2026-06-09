#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import SUBSETS, add_common_graph_args, all_systems, numeric, result_file, safe_read_csv, save_figure, system_palette


def metadata_file(results_root: Path, system: str, subset: str) -> Path:
    protocol = results_root.name
    return Path("data") / protocol / "outputs" / system / "test" / subset / "metadata.csv"


def wer_columns(frame: pd.DataFrame) -> list[str]:
    specific = sorted(column for column in frame.columns if column.startswith("WER_"))
    if specific:
        return specific
    return ["WER"] if "WER" in frame.columns else []


def load_wer_answer_lengths(results_root: Path, systems: list[str]) -> pd.DataFrame:
    frames = []
    for subset in SUBSETS:
        for system in systems:
            metrics = safe_read_csv(result_file(results_root, system, "test", subset, "base_metrics.csv"))
            metadata = safe_read_csv(metadata_file(results_root, system, subset))
            if metrics is None or metadata is None:
                continue
            if "answer_audio_path" not in metrics.columns or "answer_audio_path" not in metadata.columns:
                print(f"[WARN] Missing answer_audio_path for {system}/{subset}.")
                continue
            if "answer_duration" not in metadata.columns:
                print(f"[WARN] Missing answer_duration for {system}/{subset}.")
                continue

            columns = wer_columns(metrics)
            if not columns:
                print(f"[WARN] No WER columns found for {system}/{subset}.")
                continue

            metric_values = metrics[["answer_audio_path", *columns]].copy()
            metric_values["answer_audio_path"] = metric_values["answer_audio_path"].astype(str)
            metric_values["wer"] = metric_values[columns].apply(numeric).mean(axis=1) * 100.0

            durations = metadata[["answer_audio_path", "answer_duration"]].copy()
            durations["answer_audio_path"] = durations["answer_audio_path"].astype(str)
            durations["answer_duration"] = numeric(durations["answer_duration"])

            joined = metric_values[["answer_audio_path", "wer"]].merge(durations, on="answer_audio_path", how="inner")
            joined = joined.dropna(subset=["wer", "answer_duration"])
            joined = joined[(joined["wer"] >= 0) & (joined["answer_duration"] >= 0)]
            if joined.empty:
                continue
            joined["system"] = system
            joined["subset"] = subset
            frames.append(joined)
    if not frames:
        return pd.DataFrame(columns=["answer_audio_path", "wer", "answer_duration", "system", "subset"])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Article graph: WER by answer audio length.")
    add_common_graph_args(parser)
    args = parser.parse_args()

    systems = all_systems(args.models)
    data = load_wer_answer_lengths(args.results_root, systems)
    if data.empty:
        print("[WARN] No WER and answer length rows found.")
        return 0

    subset_order = [subset for subset in SUBSETS if subset in set(data["subset"])]
    palette = system_palette(systems)

    fig, axes = plt.subplots(1, len(subset_order), figsize=(7.5 * len(subset_order), 6.2), squeeze=False, sharex=True, sharey=True)
    legend_handles = []
    legend_labels = []
    for col_idx, subset in enumerate(subset_order):
        ax = axes[0, col_idx]
        sub = data[data["subset"] == subset]
        plot_systems = [system for system in systems if system in set(sub["system"])]
        sns.scatterplot(
            data=sub,
            x="answer_duration",
            y="wer",
            hue="system",
            hue_order=plot_systems,
            palette={system: palette[system] for system in plot_systems},
            s=28,
            alpha=0.48,
            linewidth=0,
            ax=ax,
        )
        if not legend_handles:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(subset.title())
        ax.set_xlabel("Answer audio length (s)")
        ax.set_ylabel("WER (%)" if col_idx == 0 else "")
        ax.tick_params(axis="both", labelsize=9)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="System",
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
            title_fontsize=10,
        )
    fig.suptitle("WER by Answer Audio Length", y=1.08)
    fig.tight_layout()
    save_figure(fig, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
