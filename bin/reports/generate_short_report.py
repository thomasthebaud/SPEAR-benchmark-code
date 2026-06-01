#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon


SUBSETS = ["improvised", "naturalistic"]
SPLITS = ["dev", "test"]
ORIGINAL = "original"
EXCLUDED_REPORT_METRICS = {"question_end_time"}
STATISTICAL_TESTS = {"Welch t-test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"}
DIALECT_LABELS = [
    "East Asia",
    "English",
    "Germanic",
    "Irish",
    "North America",
    "Northern Irish",
    "Oceania",
    "Other",
    "Romance",
    "Scottish",
    "Semitic",
    "Slavic",
    "South African",
    "Southeast Asia",
    "South Asia",
    "Welsh",
]


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


def statistical_pvalue(original_values: pd.Series, model_values: pd.Series, test_name: str, paired: bool = False) -> float:
    original_values = pd.Series(original_values).dropna()
    model_values = pd.Series(model_values).dropna()
    if len(original_values) < 2 or len(model_values) < 2:
        return np.nan
    try:
        if test_name == "Welch t-test":
            return float(ttest_ind(original_values, model_values, equal_var=False, nan_policy="omit").pvalue)
        if test_name == "Mann-Whitney U Test":
            return float(mannwhitneyu(original_values, model_values, alternative="two-sided").pvalue)
        if test_name == "Wilcoxon Signed-Rank Test":
            if len(original_values) != len(model_values):
                return np.nan
            return float(wilcoxon(model_values, original_values, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return np.nan
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


def append_percentage_metric(
    metrics_by_subset: Dict[str, List[Dict[str, Any]]],
    text_lines: List[str],
    subset: str,
    metric: str,
    numerator: int,
    denominator: int,
    detail: str,
) -> None:
    percentage = (100.0 * numerator / denominator) if denominator else np.nan
    row = metric_row(
        metric,
        percentage,
        np.nan,
        np.nan,
        n=denominator,
        section="STANCEs",
        detail=f"{detail}; numerator={numerator}; denominator={denominator}",
    )
    metrics_by_subset[subset].append(row)
    append_metric_table_line(text_lines, metric, row)


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
        "statistical_test": args.statistical_test,
        "data_root": str(args.data_root),
        "results_root": str(args.results_root),
    }
    for key, value in setup_items.items():
        text_lines.append(f"{key}\t{value}")
    text_lines.append("")


def data_statistics_section(args, text_lines: List[str]) -> None:
    text_lines.append("Data Statistics")
    text_lines.append("---------------")
    text_lines.append("model	split	subset	utterances	total_audio_hours")
    for model in [ORIGINAL, args.model]:
        for split in SPLITS:
            for subset in SUBSETS:
                path = args.data_root / "outputs" / model / split / subset / f"{split}_{subset}_metadata.csv"
                frame = safe_read_csv(path)
                if frame is None:
                    text_lines.append(f"{model}	{split}	{subset}	missing	missing")
                    continue

                n_rows = len(frame)
                duration = numeric(frame["total_duration"]).sum() if "total_duration" in frame else np.nan
                text_lines.append(f"{model}	{split}	{subset}	{n_rows}	{duration / 3600.0:.2f}")
    text_lines.append("")



def base_metrics_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "base_metrics.csv"


def language_id_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "language_id.csv"


def dialect_id_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "dialect_id.csv"


def english_language_count(frame: pd.DataFrame) -> Tuple[int, int]:
    if "language" not in frame.columns:
        return 0, 0
    languages = frame["language"].astype(str).str.strip().str.lower()
    valid = languages.notna() & (languages != "") & (languages != "nan")
    return int((languages.loc[valid] == "eng").sum()), int(valid.sum())


def language_id_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Language ID")
    text_lines.append("-----------")
    text_lines.append("subset	% Eng in models' answers	% Eng in original answers")
    for subset in SUBSETS:
        percentages: Dict[str, Any] = {"model": np.nan, "original": np.nan}
        for model, label in [(ORIGINAL, "original"), (args.model, "model")]:
            frame = safe_read_csv(language_id_csv(args, model, subset))
            metric = f"English language detected in {label} answers (%)"
            if frame is None or "language" not in frame.columns:
                continue
            numerator, denominator = english_language_count(frame)
            percentage = (100.0 * numerator / denominator) if denominator else np.nan
            percentages[label] = percentage
            row = metric_row(
                metric,
                percentage,
                np.nan,
                np.nan,
                n=denominator,
                section="Language ID",
                detail=f"percentage of language_id.csv rows with language=eng; numerator={numerator}; denominator={denominator}",
            )
            metrics_by_subset[subset].append(row)
        text_lines.append(f"{subset}	{fmt(percentages['model'])}	{fmt(percentages['original'])}")
    text_lines.append("")


def append_dialect_percentage_rows(
    metrics_by_subset: Dict[str, List[Dict[str, Any]]],
    text_lines: List[str],
    subset: str,
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    values = frame[column].astype(str).str.strip()
    valid = values[values.notna() & (values != "") & (values.str.lower() != "nan")]
    denominator = int(len(valid))
    counts = valid.value_counts()
    for dialect in DIALECT_LABELS:
        numerator = int(counts.get(dialect, 0))
        percentage = (100.0 * numerator / denominator) if denominator else np.nan
        metric = f"{label}: {dialect} (%)"
        row = metric_row(
            metric,
            percentage,
            np.nan,
            np.nan,
            n=denominator,
            section="Dialect ID",
            detail=f"percentage of dialect_id.csv rows where {column}={dialect}; numerator={numerator}; denominator={denominator}",
        )
        metrics_by_subset[subset].append(row)
        append_metric_table_line(text_lines, f"{metric} ({subset})", row)


def dialect_id_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Dialect ID")
    text_lines.append("----------")
    text_lines.append("	".join(["Subset", "Question/Answer", *DIALECT_LABELS]))
    for subset in SUBSETS:
        frame = safe_read_csv(dialect_id_csv(args, args.model, subset))
        if frame is None:
            text_lines.append("	".join([subset, "missing", *(["missing"] * len(DIALECT_LABELS))]))
            continue
        for column, label in [("dialect", "Question"), ("answer_dialect", "Answer")]:
            if column not in frame.columns:
                text_lines.append("	".join([subset, label, *(["missing"] * len(DIALECT_LABELS))]))
                continue
            before_count = len(metrics_by_subset[subset])
            append_dialect_percentage_rows(metrics_by_subset, [], subset, frame, column, f"{label} dialect")
            rows = metrics_by_subset[subset][before_count:]
            percentages = {row["metric"].split(": ", 1)[1].removesuffix(" (%)"): row["mean_diff"] for row in rows}
            text_lines.append("	".join([subset, label, *(fmt(percentages.get(dialect)) for dialect in DIALECT_LABELS)]))
    text_lines.append("")


def asr_error_metric_columns(*frames: Optional[pd.DataFrame]) -> List[str]:
    columns = set()
    for frame in frames:
        if frame is None:
            continue
        columns.update(
            column
            for column in frame.columns
            if column in {"CER", "WER"} or column.startswith("CER_") or column.startswith("WER_")
        )

    ordered: List[str] = []
    for prefix in ["CER", "WER"]:
        specific = sorted(column for column in columns if column.startswith(f"{prefix}_"))
        if specific:
            ordered.extend(specific)
        elif prefix in columns:
            ordered.append(prefix)
    return ordered


def base_metric_columns(model_frame: pd.DataFrame, original_frame: Optional[pd.DataFrame]) -> List[str]:
    columns = asr_error_metric_columns(model_frame, original_frame)
    for metric in ["UTMOS", "latency", "interrupted"]:
        if metric in model_frame.columns or (original_frame is not None and metric in original_frame.columns):
            columns.append(metric)
    return columns


def add_basic_metric_row(
    args,
    metrics_by_subset: Dict[str, List[Dict[str, Any]]],
    text_lines: List[str],
    subset: str,
    metric: str,
    model_frame: pd.DataFrame,
    original_frame: Optional[pd.DataFrame],
) -> None:
    if metric not in model_frame.columns:
        return
    model_values = numeric(model_frame[metric]).dropna()
    if model_values.empty:
        return
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
            pvalue = statistical_pvalue(original_values, model_values, args.statistical_test)
            n_value = min(len(original_values), len(model_values))
            detail = f"model-original mean difference; p-value is {args.statistical_test}"
    if metric == "interrupted":
        detail += "; interrupted overlap duration in ms, with 0 for non-interruptions"
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


def add_interruption_count_row(
    args,
    metrics_by_subset: Dict[str, List[Dict[str, Any]]],
    text_lines: List[str],
    subset: str,
    model_frame: pd.DataFrame,
    original_frame: Optional[pd.DataFrame],
) -> None:
    if "interruptions" not in model_frame.columns:
        return
    model_values = numeric(model_frame["interruptions"]).fillna(0)
    model_count = int((model_values > 0).sum())
    mean_value = model_count
    detail = f"model interruption count={model_count}"
    if original_frame is not None and "interruptions" in original_frame.columns:
        original_values = numeric(original_frame["interruptions"]).fillna(0)
        original_count = int((original_values > 0).sum())
        mean_value = model_count - original_count
        detail = f"model-original interruption count difference; model_count={model_count}; original_count={original_count}"
    row = metric_row(
        "interruptions count",
        mean_value,
        np.nan,
        np.nan,
        n=len(model_values),
        section="Basic Metrics",
        detail=detail,
    )
    metrics_by_subset[subset].append(row)
    append_metric_table_line(text_lines, f"interruptions count ({subset})", row)


def base_metrics_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Basic Metrics")
    text_lines.append("-------------")
    text_lines.append("metric	n	mean_diff	std_diff	p_value	auroc	accuracy")
    for subset in SUBSETS:
        model_frame = safe_read_csv(base_metrics_csv(args, args.model, subset))
        original_frame = safe_read_csv(base_metrics_csv(args, ORIGINAL, subset))
        if model_frame is None:
            text_lines.append(f"Basic metrics ({subset})	missing	missing	missing	missing	missing	missing")
            continue
        for metric in base_metric_columns(model_frame, original_frame):
            add_basic_metric_row(args, metrics_by_subset, text_lines, subset, metric, model_frame, original_frame)
        add_interruption_count_row(args, metrics_by_subset, text_lines, subset, model_frame, original_frame)
    text_lines.append("")

def naturalness_scores_with_relationship(args, model: str, subset: str) -> Optional[pd.DataFrame]:
    scores_path = args.results_root / model / "test" / subset / "naturalness_scores.csv"
    rel_path = args.results_root / model / "test" / subset / "naturalness_inference_input.csv"
    scores = safe_read_csv(scores_path)
    rels = safe_read_csv(rel_path)
    if scores is None or rels is None or "naturalness_logit" not in scores or "rel_detail" not in rels:
        return None

    scores = scores.copy().reset_index(drop=True)
    rels = rels.copy().reset_index(drop=True)
    if "audio_path" in scores:
        scores["utterance_id"] = audio_stem(scores)
    else:
        scores["utterance_id"] = scores.index.astype(str)

    if "utterance_id" in rels.columns:
        rel_map = rels[["utterance_id", "rel_detail"]].drop_duplicates("utterance_id")
        merged = scores.merge(rel_map, on="utterance_id", how="left")
    elif len(scores) == len(rels):
        merged = scores
        merged["rel_detail"] = rels["rel_detail"]
    else:
        return None

    merged["relationship_type"] = merged["rel_detail"].astype(str).replace({"": "unknown", "nan": "unknown"})
    merged["naturalness_logit"] = numeric(merged["naturalness_logit"])
    return merged.dropna(subset=["naturalness_logit"])


def emotional_naturalness_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Emotional Naturalness")
    text_lines.append("---------------------")
    text_lines.append("metric	n	mean_diff	std_diff	p_value	auroc	accuracy")
    for subset in SUBSETS:
        text_lines.append(f"{subset}")
        original_path = args.results_root / ORIGINAL / "test" / subset / "naturalness_scores.csv"
        model_path = args.results_root / args.model / "test" / subset / "naturalness_scores.csv"
        original = safe_read_csv(original_path)
        model = safe_read_csv(model_path)
        if original is None or model is None or "naturalness_logit" not in original or "naturalness_logit" not in model:
            text_lines.append(f"Emotional naturalness ({subset})	missing	missing	missing	missing	missing	missing")
            continue

        original_values, model_values, diff = paired_values(original, model, "naturalness_logit")
        pvalue = statistical_pvalue(original_values, model_values, args.statistical_test, paired=True)
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

        original_rel = naturalness_scores_with_relationship(args, ORIGINAL, subset)
        model_rel = naturalness_scores_with_relationship(args, args.model, subset)
        if original_rel is None or model_rel is None:
            text_lines.append(f"Emotional naturalness by relationship ({subset})	missing	missing	missing	missing	missing	missing")
            continue

        original_grouped = original_rel.groupby("relationship_type")["naturalness_logit"].agg(["mean", "count"]).rename(columns={"mean": "original_mean", "count": "original_n"})
        model_grouped = model_rel.groupby("relationship_type")["naturalness_logit"].agg(["mean", "count"]).rename(columns={"mean": "model_mean", "count": "model_n"})
        grouped = original_grouped.join(model_grouped, how="outer")
        for relationship, group_row in grouped.sort_index().iterrows():
            original_mean = group_row.get("original_mean", np.nan)
            model_mean = group_row.get("model_mean", np.nan)
            original_n = group_row.get("original_n", np.nan)
            model_n = group_row.get("model_n", np.nan)
            mean_diff = model_mean - original_mean if pd.notna(model_mean) and pd.notna(original_mean) else np.nan
            n_value = int(np.nanmin([original_n, model_n])) if pd.notna(original_n) and pd.notna(model_n) else np.nan
            rel_row = metric_row(
                f"Emotional naturalness by relationship: {relationship}",
                mean_diff,
                np.nan,
                np.nan,
                n=n_value,
                section="Emotional Naturalness",
                detail=f"relationship_type={relationship}; original_mean={fmt(original_mean)}; model_mean={fmt(model_mean)}; original_n={fmt(original_n)}; model_n={fmt(model_n)}",
            )
            metrics_by_subset[subset].append(rel_row)
            append_metric_table_line(text_lines, f"Emotional naturalness {relationship} ({subset})", rel_row)
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

    all_original = numeric(frame["score_original"])
    all_model = numeric(frame["score_llm"])
    all_valid = all_original.notna() & all_model.notna()
    all_original = all_original.loc[all_valid]
    all_model = all_model.loc[all_valid]
    original_negative = all_original < 0
    original_positive = all_original > 0

    append_percentage_metric(
        metrics_by_subset,
        text_lines,
        subset,
        "STANCE same sign (%)",
        int((np.sign(all_original) == np.sign(all_model)).sum()),
        len(all_original),
        "percentage of valid paired rows where score_original and score_llm have the same sign, with zero treated as neutral",
    )
    append_percentage_metric(
        metrics_by_subset,
        text_lines,
        subset,
        "STANCE model positive when original negative (%)",
        int(((all_model > 0) & original_negative).sum()),
        int(original_negative.sum()),
        "percentage among valid paired rows with score_original < 0",
    )
    append_percentage_metric(
        metrics_by_subset,
        text_lines,
        subset,
        "STANCE model negative when original positive (%)",
        int(((all_model < 0) & original_positive).sum()),
        int(original_positive.sum()),
        "percentage among valid paired rows with score_original > 0",
    )

    for qidx, group in frame.groupby("question_index", sort=True):
        original = numeric(group["score_original"])
        model = numeric(group["score_llm"])
        valid = original.notna() & model.notna()
        original = original.loc[valid]
        model = model.loc[valid]
        diff = model - original
        pvalue = statistical_pvalue(original, model, args.statistical_test, paired=True)
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
    pvalue = statistical_pvalue(original_values, model_values, args.statistical_test)
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
        detail=f"feature value model-original; p-value is {args.statistical_test}",
    )
    metrics_by_subset[subset].append(row)
    return row




def cluster_values_path(args, model: str, subset: str, threshold: str) -> Path:
    return args.results_root / model / "test" / subset / f"correlation_cluster_features_rho{threshold}.csv"


def cluster_groups_path(args, model: str, subset: str, threshold: str) -> Path:
    return args.results_root / model / "test" / subset / f"correlation_feature_groups_rho{threshold}.csv"


def cluster_threshold_suffixes(args, subset: str) -> List[str]:
    folder = args.results_root / args.model / "test" / subset
    if not folder.exists():
        return []
    suffixes = []
    for path in sorted(folder.glob("correlation_cluster_features_rho*.csv")):
        suffixes.append(path.stem.replace("correlation_cluster_features_rho", ""))
    return suffixes


def add_cluster_feature_metrics(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str], subset: str) -> None:
    for threshold in cluster_threshold_suffixes(args, subset):
        model_values = safe_read_csv(cluster_values_path(args, args.model, subset, threshold))
        groups = safe_read_csv(cluster_groups_path(args, args.model, subset, threshold))
        if model_values is None or groups is None or "cluster_feature" not in groups.columns:
            continue
        original_values = safe_read_csv(cluster_values_path(args, ORIGINAL, subset, threshold))
        text_lines.append(f"cluster_features_rho{threshold}")
        group_by_feature = groups.drop_duplicates("cluster_feature").set_index("cluster_feature")
        for feature in [str(value) for value in groups["cluster_feature"].dropna().unique() if str(value) not in EXCLUDED_REPORT_METRICS]:
            if feature not in model_values.columns:
                continue
            model_series = numeric(model_values[feature]).dropna()
            if model_series.empty:
                continue
            mean_value = model_series.mean()
            std_value = model_series.std()
            pvalue = np.nan
            n_value = len(model_series)
            detail = "script 41 PCA cluster feature; model mean/std; original cluster values unavailable"
            if original_values is not None and feature in original_values.columns:
                original_series = numeric(original_values[feature]).dropna()
                if not original_series.empty:
                    mean_value = model_series.mean() - original_series.mean()
                    std_value = np.sqrt(model_series.var(ddof=1) + original_series.var(ddof=1))
                    pvalue = statistical_pvalue(original_series, model_series, args.statistical_test)
                    n_value = min(len(original_series), len(model_series))
                    detail = f"script 41 PCA cluster feature model-original; p-value is {args.statistical_test}"
            if feature in group_by_feature.index:
                group_row = group_by_feature.loc[feature]
                detail += (
                    f"; source_features={group_row.get('features', '')}"
                    f"; validates_in_test={group_row.get('validates_in_test', '')}"
                    f"; test_mean_abs_spearman={fmt(pd.to_numeric(group_row.get('test_mean_abs_spearman', np.nan), errors='coerce'))}"
                    f"; pca_explained_variance_ratio={fmt(pd.to_numeric(group_row.get('pca_explained_variance_ratio', np.nan), errors='coerce'))}"
                )
            row = metric_row(
                feature,
                mean_value,
                std_value,
                pvalue,
                n=n_value,
                section="Explainable Features",
                detail=detail,
            )
            metrics_by_subset[subset].append(row)
            append_metric_table_line(text_lines, feature, row)

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
    frame = frame[~frame["feature"].astype(str).isin(EXCLUDED_REPORT_METRICS)]
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

    for subset in SUBSETS:
        add_cluster_feature_metrics(args, metrics_by_subset, text_lines, subset)
    text_lines.append("")


def align_tabular_blocks(text_lines: List[str], padding: int = 2) -> List[str]:
    aligned: List[str] = []
    block: List[List[str]] = []

    def flush_block() -> None:
        nonlocal block
        if not block:
            return
        widths = [max(len(row[i]) if i < len(row) else 0 for row in block) for i in range(max(len(row) for row in block))]
        sep = " " * padding
        for row in block:
            cells = [row[i] if i < len(row) else "" for i in range(len(widths))]
            aligned.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip())
        block = []

    for line in text_lines:
        if "\t" in line:
            block.append(line.split("\t"))
        else:
            flush_block()
            aligned.append(line)
    flush_block()
    return aligned


def write_outputs(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.txt"

    for subset, rows in metrics_by_subset.items():
        metrics_path = args.output_dir / f"metrics-{subset}.csv"
        columns = ["metric", "section", "mean_diff", "std_diff", "p_value", "n", "auroc", "accuracy", "detail"]
        pd.DataFrame(rows, columns=columns).to_csv(metrics_path, index=False)
        print(f"Wrote metrics: {metrics_path}")

    report_path.write_text("\n".join(align_tabular_blocks(text_lines)) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--stance-model", required=True)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--statistical-test", default="Welch t-test", choices=sorted(STATISTICAL_TESTS))
    parser.add_argument("--selection-method", default="end_with_question")
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-features", nargs="*", default=[])
    args = parser.parse_args()
    EXCLUDED_REPORT_METRICS.update(args.ignore_features)

    metrics_by_subset: Dict[str, List[Dict[str, Any]]] = {subset: [] for subset in SUBSETS}
    text_lines = [
        "SPEARBench Short Report",
        "=======================",
        "",
    ]

    setup_section(args, text_lines)
    data_statistics_section(args, text_lines)
    base_metrics_section(args, metrics_by_subset, text_lines)
    language_id_section(args, metrics_by_subset, text_lines)
    dialect_id_section(args, metrics_by_subset, text_lines)
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
