#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel


SUBSETS = ["improvised", "naturalistic"]
SPLITS = ["dev", "test"]
ORIGINAL = "original"


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def pvalue_text(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.3g}"


def metric_row(
    metric: str,
    mean_diff: Any = np.nan,
    std_diff: Any = np.nan,
    p_value: Any = np.nan,
    *,
    n: Any = np.nan,
    auroc: Any = np.nan,
    accuracy: Any = np.nan,
    section: str = "",
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "metric": metric,
        "section": section,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "p_value": p_value,
        "n": n,
        "auroc": auroc,
        "accuracy": accuracy,
        "detail": detail,
    }


def audio_stem(frame: pd.DataFrame) -> pd.Series:
    return frame["audio_path"].astype(str).map(lambda value: Path(value).stem)


def paired_values(
    original: pd.DataFrame,
    model: pd.DataFrame,
    column: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    left = pd.DataFrame({"utterance_id": audio_stem(original), "original": numeric(original[column])})
    right = pd.DataFrame({"utterance_id": audio_stem(model), "model": numeric(model[column])})
    merged = left.merge(right, on="utterance_id", how="inner").dropna(subset=["original", "model"])
    return merged["original"], merged["model"], merged["model"] - merged["original"]


def paired_ttest(original_values: pd.Series, model_values: pd.Series) -> float:
    if len(original_values) < 2 or len(model_values) < 2:
        return np.nan
    try:
        return float(ttest_rel(model_values, original_values, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def welch_pvalue(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def feature_csv(args, model: str, split: str, subset: str) -> Path:
    return args.results_root / model / split / subset / "distrib_baselines_features.csv"


def append_metric_table_line(text_lines: List[str], name: str, row: Dict[str, Any]) -> None:
    text_lines.append(
        "\t".join(
            [
                name,
                fmt(row.get("n")),
                fmt(row.get("mean_diff")),
                fmt(row.get("std_diff")),
                pvalue_text(row.get("p_value", np.nan)),
                fmt(row.get("auroc")),
                fmt(row.get("accuracy")),
            ]
        )
    )


def setup_section(args, text_lines: List[str]) -> None:
    text_lines.append("Setup")
    text_lines.append("-----")
    text_lines.append("parameter\tvalue")
    setup_items = {
        "protocol": args.protocol,
        "selection_method": args.selection_method,
        "min_turns": args.min_turns,
        "min_speakers": args.min_speakers,
        "splits": "dev,test",
        "subsets": ",".join(SUBSETS),
        "asr_model": args.asr_model,
        "evaluated_llm_model": args.model,
        "stance_llm_model": args.stance_model,
        "sbert_model_for_naturalness_context": args.sbert_model,
        "data_root": str(args.data_root),
        "results_root": str(args.results_root),
    }
    for key, value in setup_items.items():
        text_lines.append(f"{key}\t{value}")
    text_lines.append("")


def data_statistics_section(args, text_lines: List[str]) -> None:
    text_lines.append("Data Statistics")
    text_lines.append("---------------")
    text_lines.append("model\tsplit\tsubset\tutterances\ttotal_audio_hours\tanswer_audio_hours")
    for model in [ORIGINAL, args.model]:
        for split in SPLITS:
            for subset in SUBSETS:
                path = args.data_root / "outputs" / model / split / subset / f"{split}_{subset}_metadata.csv"
                frame = safe_read_csv(path)
                if frame is None:
                    text_lines.append(f"{model}\t{split}\t{subset}\tmissing\tmissing\tmissing")
                    continue

                n_rows = len(frame)
                duration = numeric(frame["total_duration"]).sum() if "total_duration" in frame else np.nan
                answer_duration = np.nan
                if {"total_duration", "question_end_time"}.issubset(frame.columns):
                    answer_duration = (numeric(frame["total_duration"]) - numeric(frame["question_end_time"])).clip(lower=0).sum()

                text_lines.append(
                    f"{model}\t{split}\t{subset}\t{n_rows}\t{duration / 3600.0:.2f}\t{answer_duration / 3600.0:.2f}"
                )
    text_lines.append("")



def base_metrics_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "base_metrics.csv"


def base_metrics_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Basic Metrics")
    text_lines.append("-------------")
    text_lines.append("metric\tn\tmean_diff\tstd_diff\tp_value\tauroc\taccuracy")
    metric_cols = ["CER", "WER", "UTMOS", "latency"]
    for subset in SUBSETS:
        model_frame = safe_read_csv(base_metrics_csv(args, args.model, subset))
        original_frame = safe_read_csv(base_metrics_csv(args, ORIGINAL, subset))
        if model_frame is None:
            text_lines.append(f"Basic metrics ({subset})\tmissing\tmissing\tmissing\tmissing\tmissing\tmissing")
            continue
        for metric in metric_cols:
            if metric not in model_frame.columns:
                continue
            model_values = numeric(model_frame[metric]).dropna()
            if model_values.empty:
                continue
            detail = "model mean/std; original base_metrics.csv unavailable"
            mean_value = model_values.mean()
            std_value = model_values.std()
            pvalue = np.nan
            n_value = len(model_values)
            if original_frame is not None and metric in original_frame.columns:
                original_values = numeric(original_frame[metric]).dropna()
                if not original_values.empty:
                    mean_value = model_values.mean() - original_values.mean()
                    std_value = np.sqrt(model_values.var(ddof=1) + original_values.var(ddof=1))
                    pvalue = welch_pvalue(original_values, model_values)
                    n_value = min(len(original_values), len(model_values))
                    detail = "model-original mean difference; p-value is Welch t-test"
            row = metric_row(
                metric,
                mean_value,
                std_value,
                pvalue,
                n=n_value,
                section="Basic Metrics",
                detail=detail,
            )
            metrics_by_subset[subset].append(row)
            append_metric_table_line(text_lines, f"{metric} ({subset})", row)
    text_lines.append("")

def emotional_naturalness_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Emotional Naturalness")
    text_lines.append("---------------------")
    text_lines.append("metric\tn\tmean_diff\tstd_diff\tp_value\tauroc\taccuracy")
    for subset in SUBSETS:
        original_path = args.results_root / ORIGINAL / "test" / subset / "naturalness_scores.csv"
        model_path = args.results_root / args.model / "test" / subset / "naturalness_scores.csv"
        original = safe_read_csv(original_path)
        model = safe_read_csv(model_path)
        if original is None or model is None or "naturalness_logit" not in original or "naturalness_logit" not in model:
            text_lines.append(f"Emotional naturalness ({subset})\tmissing\tmissing\tmissing\tmissing\tmissing\tmissing")
            continue

        original_values, model_values, diff = paired_values(original, model, "naturalness_logit")
        pvalue = paired_ttest(original_values, model_values)
        row = metric_row(
            "Emotional naturalness",
            diff.mean(),
            diff.std(),
            pvalue,
            n=len(diff),
            section="Emotional Naturalness",
            detail="naturalness_logit model-original",
        )
        metrics_by_subset[subset].append(row)
        append_metric_table_line(text_lines, f"Emotional naturalness ({subset})", row)
    text_lines.append("")


def stances_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("STANCEs")
    text_lines.append("-------")
    text_lines.append("metric\tn\tmean_diff\tstd_diff\tp_value\tauroc\taccuracy")

    subset = "improvised"
    path = args.results_root / args.model / "test" / subset / "merged_stances.csv"
    frame = safe_read_csv(path)
    if frame is None:
        text_lines.append("STANCE improvised\tmissing\tmissing\tmissing\tmissing\tmissing\tmissing")
        text_lines.append("")
        return
    if not {"question_index", "score_original", "score_llm"}.issubset(frame.columns):
        text_lines.append("STANCE improvised\tmissing_columns\tmissing\tmissing\tmissing\tmissing\tmissing")
        text_lines.append("")
        return

    for qidx, group in frame.groupby("question_index", sort=True):
        original = numeric(group["score_original"])
        model = numeric(group["score_llm"])
        valid = original.notna() & model.notna()
        original = original.loc[valid]
        model = model.loc[valid]
        diff = model - original
        pvalue = paired_ttest(original, model)
        question = str(group["stance_question"].dropna().iloc[0]) if "stance_question" in group and group["stance_question"].notna().any() else ""
        row = metric_row(
            f"Q{qidx}",
            diff.mean(),
            diff.std(),
            pvalue,
            n=len(diff),
            section="STANCEs",
            detail=question,
        )
        metrics_by_subset[subset].append(row)
        append_metric_table_line(text_lines, f"Q{qidx}", row)
    text_lines.append("")


def feature_stats(args, subset: str, feature: str) -> Tuple[float, float, float, int]:
    original = safe_read_csv(feature_csv(args, ORIGINAL, "test", subset))
    model = safe_read_csv(feature_csv(args, args.model, "test", subset))
    if original is None or model is None or feature not in original or feature not in model:
        return np.nan, np.nan, np.nan, 0

    original_values = numeric(original[feature]).dropna()
    model_values = numeric(model[feature]).dropna()
    pvalue = welch_pvalue(original_values, model_values)
    diff_mean = model_values.mean() - original_values.mean()
    diff_std = np.sqrt(model_values.var(ddof=1) + original_values.var(ddof=1))
    return diff_mean, diff_std, pvalue, int(min(len(original_values), len(model_values)))


def add_feature_metric(
    args,
    metrics_by_subset: Dict[str, List[Dict[str, Any]]],
    subset: str,
    feature: str,
    auroc: float,
    accuracy: float,
) -> Dict[str, Any]:
    mean_diff, std_diff, pvalue, n = feature_stats(args, subset, feature)
    row = metric_row(
        feature,
        mean_diff,
        std_diff,
        pvalue,
        n=n,
        auroc=auroc,
        accuracy=accuracy,
        section="Explainable Features",
        detail="feature value model-original; p-value is Welch t-test",
    )
    metrics_by_subset[subset].append(row)
    return row


def explainable_features_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Explainable Features")
    text_lines.append("--------------------")
    path = args.results_root / args.model / "distrib_baselines_summary.csv"
    frame = safe_read_csv(path)
    if frame is None or "feature" not in frame or "auroc" not in frame:
        text_lines.append("Baseline summary\tmissing")
        text_lines.append("")
        return

    frame = frame.copy()
    frame["auroc"] = numeric(frame["auroc"])
    frame["accuracy"] = numeric(frame["accuracy"]) if "accuracy" in frame else np.nan
    frame = frame.dropna(subset=["auroc"])

    for subset in sorted(frame["subset"].dropna().unique()) if "subset" in frame else ["all"]:
        sub = frame[frame["subset"] == subset].copy() if "subset" in frame else frame.copy()
        text_lines.append(f"{subset}")
        text_lines.append("rank_group\tmetric\tn\tmean_diff\tstd_diff\tp_value\tauroc\taccuracy")
        for rank_group, block in [
            ("highest_auroc", sub.sort_values("auroc", ascending=False).head(10)),
            ("lowest_auroc", sub.sort_values("auroc", ascending=True).head(10)),
        ]:
            for _, source_row in block.iterrows():
                feature = str(source_row.get("feature", ""))
                auroc = pd.to_numeric(source_row.get("auroc"), errors="coerce")
                accuracy = pd.to_numeric(source_row.get("accuracy"), errors="coerce")
                row = add_feature_metric(args, metrics_by_subset, subset, feature, auroc, accuracy)
                text_lines.append(
                    "\t".join(
                        [
                            rank_group,
                            feature,
                            fmt(row["n"]),
                            fmt(row["mean_diff"]),
                            fmt(row["std_diff"]),
                            pvalue_text(row["p_value"]),
                            fmt(row["auroc"]),
                            fmt(row["accuracy"]),
                        ]
                    )
                )
    text_lines.append("")


def write_outputs(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.txt"

    for subset, rows in metrics_by_subset.items():
        metrics_path = args.output_dir / f"metrics-{subset}.csv"
        columns = ["metric", "section", "mean_diff", "std_diff", "p_value", "n", "auroc", "accuracy", "detail"]
        pd.DataFrame(rows, columns=columns).to_csv(metrics_path, index=False)
        print(f"Wrote metrics: {metrics_path}")

    report_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--stance-model", required=True)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--selection-method", default="end_with_question")
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics_by_subset: Dict[str, List[Dict[str, Any]]] = {subset: [] for subset in SUBSETS}
    text_lines = [
        "SPEARBench Short Report",
        "=======================",
        "",
    ]

    setup_section(args, text_lines)
    data_statistics_section(args, text_lines)
    base_metrics_section(args, metrics_by_subset, text_lines)
    emotional_naturalness_section(args, metrics_by_subset, text_lines)
    stances_section(args, metrics_by_subset, text_lines)
    explainable_features_section(args, metrics_by_subset, text_lines)

    text_lines.extend(
        [
            "Files",
            "-----",
            f"metrics-improvised.csv\t{args.output_dir / 'metrics-improvised.csv'}",
            f"metrics-naturalistic.csv\t{args.output_dir / 'metrics-naturalistic.csv'}",
            f"report.txt\t{args.output_dir / 'report.txt'}",
        ]
    )
    write_outputs(args, metrics_by_subset, text_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
