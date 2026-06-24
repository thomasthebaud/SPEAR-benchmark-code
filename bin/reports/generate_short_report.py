#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon


SUBSETS = ["improvised", "naturalistic"]
SPLITS = ["dev", "test"]
ORIGINAL = "original"
EXCLUDED_REPORT_METRICS = {"question_end_time"}
REPORTED_EXPLAINABLE_FEATURES = {"total_duration_s", "voiced_duration_s", "voiced_ratio"}
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


def prepared_dialogue_ids(metadata: pd.DataFrame) -> set[str]:
    if "conversation_id" in metadata.columns:
        return set(metadata["conversation_id"].dropna().astype(str))
    return set()


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
    model_value: Any = np.nan,
    original_value: Any = np.nan,
) -> Dict[str, Any]:
    return {
        "metric": metric,
        "section": section,
        "model_value": model_value,
        "original_value": original_value,
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
    return args.results_root / model / split / subset / "distrib_baselines_features_normalized.csv"


def is_reported_explainable_feature(feature: str) -> bool:
    return feature in REPORTED_EXPLAINABLE_FEATURES or feature.startswith("f0_p")


def available_explainable_features(args, subset: str) -> List[str]:
    features = set()
    for model in [ORIGINAL, args.model]:
        frame = safe_read_csv(feature_csv(args, model, "test", subset))
        if frame is None:
            continue
        features.update(
            str(column)
            for column in frame.columns
            if is_reported_explainable_feature(str(column)) and str(column) not in EXCLUDED_REPORT_METRICS
        )
    return sorted(features)


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
    text_lines.append("Models")
    text_lines.append("------")
    text_lines.append("parameter	value")
    setup_items = {
        "LLM model used for inference": args.model,
        "ASR models used": ", ".join(args.asr_models),
        "Language ID model": args.language_id_model,
        "Dialect ID model": args.dialect_id_model,
        "LLM used for stance": args.stance_model,
        "Statistical tests used for p-values": args.statistical_test,
        "UTMOS model": args.utmos_model,
        "VAD model": args.vad_model,
    }
    for key, value in setup_items.items():
        text_lines.append(f"{key}	{value}")
    text_lines.append("")


def data_statistics_section(args, text_lines: List[str]) -> None:
    text_lines.append("Data")
    text_lines.append("----")
    columns = [("improvised", "dev"), ("improvised", "test"), ("naturalistic", "dev"), ("naturalistic", "test")]
    summaries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for subset, split in columns:
        original_path = args.data_root / "outputs" / ORIGINAL / split / subset / "metadata.csv"
        model_path = args.data_root / "outputs" / args.model / split / subset / "metadata.csv"
        original = safe_read_csv(original_path)
        model = safe_read_csv(model_path)
        summaries[(subset, split)] = {
            "unique_original_dialogues": len(prepared_dialogue_ids(original)) if original is not None else np.nan,
            "selected_dialogues": len(original) if original is not None else np.nan,
            "question_hours": numeric(original["question_end_time"]).sum() / 3600.0 if original is not None and "question_end_time" in original else np.nan,
            "original_answer_hours": numeric(original["answer_duration"]).sum() / 3600.0 if original is not None and "answer_duration" in original else np.nan,
            "original_total_hours": (
                (numeric(original["question_end_time"]).sum() + numeric(original["answer_duration"]).sum()) / 3600.0
                if original is not None and {"question_end_time", "answer_duration"}.issubset(original.columns)
                else np.nan
            ),
            "generated_answer_hours": numeric(model["answer_duration"]).sum() / 3600.0 if model is not None and "answer_duration" in model else np.nan,
            "original_answer_mean_seconds": numeric(original["answer_duration"]).mean() if original is not None and "answer_duration" in original else np.nan,
            "generated_answer_mean_seconds": numeric(model["answer_duration"]).mean() if model is not None and "answer_duration" in model else np.nan,
        }

    def total_for(values: List[Any]) -> float:
        valid = [float(value) for value in values if pd.notna(value)]
        return float(np.sum(valid)) if valid else np.nan

    def row_values(key: str, digits: int = 2, suffix: str = "") -> List[str]:
        vals = [summaries[item][key] for item in columns]
        output = [fmt(value, digits) + suffix if pd.notna(value) else "missing" for value in vals]
        total = total_for(vals)
        output.append(fmt(total, digits) + suffix if pd.notna(total) else "missing")
        return output

    text_lines.append("metric	Improvised dev	Improvised test	Naturalistic dev	Naturalistic test	Total")
    text_lines.append("	".join(["Unique original dialogues", *row_values("unique_original_dialogues", 0)]))
    text_lines.append("	".join(["Selected dialogues", *row_values("selected_dialogues", 0)]))
    text_lines.append("	".join(["Hours of questions", *row_values("question_hours", 2, "h")]))
    text_lines.append("	".join(["Hours of original answers", *row_values("original_answer_hours", 2, "h")]))
    text_lines.append("	".join(["Original total hours", *row_values("original_total_hours", 2, "h")]))
    text_lines.append("	".join(["Hours of generated answers", *row_values("generated_answer_hours", 2, "h")]))
    original_mean_values = [summaries[item]["original_answer_mean_seconds"] for item in columns]
    text_lines.append("	".join(["Mean original answer length", *(fmt(value, 2) + "s" if pd.notna(value) else "missing" for value in original_mean_values), ""]))
    mean_values = [summaries[item]["generated_answer_mean_seconds"] for item in columns]
    text_lines.append("	".join(["Mean generated answer length", *(fmt(value, 2) + "s" if pd.notna(value) else "missing" for value in mean_values), ""]))
    text_lines.append("")


def setup_details_section(args, text_lines: List[str]) -> None:
    text_lines.append("Setup")
    text_lines.append("-----")
    text_lines.append("parameter	value")
    setup_items = {
        "protocol": args.protocol,
        "selection_method": args.selection_method,
        "min_turns": args.min_turns,
        "min_speakers": args.min_speakers,
        "splits": "dev,test",
        "subsets": ",".join(SUBSETS),
        "data_root": str(args.data_root),
        "results_root": str(args.results_root),
    }
    for key, value in setup_items.items():
        text_lines.append(f"{key}	{value}")
    text_lines.append("")

def base_metrics_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "base_metrics.csv"


def language_id_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "language_id.csv"


def dialect_id_csv(args, model: str, subset: str) -> Path:
    return args.results_root / model / "test" / subset / "dialect_id.csv"


def dialect_logits_csv(args, model: str) -> Path:
    return args.results_root / model / "test" / "dialect_logits.csv"


def parse_vector(value) -> Optional[np.ndarray]:
    if pd.isna(value):
        return None
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)) or not parsed:
        return None
    try:
        vector = np.asarray(parsed, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or not np.isfinite(vector).all():
        return None
    return vector


def least_squares_slope_flattened(question_vectors: List[np.ndarray], answer_vectors: List[np.ndarray]) -> float:
    if not question_vectors or not answer_vectors:
        return np.nan
    question = np.concatenate([vector.ravel() for vector in question_vectors])
    answer = np.concatenate([vector.ravel() for vector in answer_vectors])
    finite = np.isfinite(question) & np.isfinite(answer)
    question = question[finite]
    answer = answer[finite]
    if len(question) < 2 or np.unique(question).size < 2 or np.unique(answer).size < 2:
        return np.nan
    question = (question - question.min()) / (question.max() - question.min())
    answer = (answer - answer.min()) / (answer.max() - answer.min())
    slope, _ = np.polyfit(question, answer, deg=1)
    return float(slope)


def answer_total_variance(answer_vectors: List[np.ndarray]) -> float:
    if not answer_vectors:
        return np.nan
    matrix = np.vstack(answer_vectors)
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.nanvar(matrix, axis=0, ddof=1).sum())


def dialect_logit_metrics(args, subset: str) -> Tuple[float, float, int]:
    frame = safe_read_csv(dialect_logits_csv(args, args.model))
    if frame is None or not {"question_log_logits", "answer_log_logits"}.issubset(frame.columns):
        return np.nan, np.nan, 0
    if "subset" in frame.columns:
        frame = frame[frame["subset"].astype(str) == subset]
    question_vectors: List[np.ndarray] = []
    answer_vectors: List[np.ndarray] = []
    for _, row in frame.iterrows():
        question = parse_vector(row["question_log_logits"])
        answer = parse_vector(row["answer_log_logits"])
        if question is None or answer is None or question.shape != answer.shape:
            continue
        question_vectors.append(question)
        answer_vectors.append(answer)
    return least_squares_slope_flattened(question_vectors, answer_vectors), answer_total_variance(answer_vectors), len(answer_vectors)


LANGUAGE_NAME_OVERRIDES = {
    "eng": "English",
    "fra": "French",
    "fre": "French",
    "spa": "Spanish",
    "deu": "German",
    "ger": "German",
    "ita": "Italian",
    "por": "Portuguese",
    "nld": "Dutch",
    "dut": "Dutch",
    "rus": "Russian",
    "zho": "Chinese",
    "chi": "Chinese",
    "cmn": "Mandarin Chinese",
    "jpn": "Japanese",
    "kor": "Korean",
    "ara": "Arabic",
    "hin": "Hindi",
    "urd": "Urdu",
    "ben": "Bengali",
    "tur": "Turkish",
    "vie": "Vietnamese",
    "tha": "Thai",
    "ind": "Indonesian",
    "msa": "Malay",
    "tgl": "Tagalog",
    "fil": "Filipino",
}


def language_display_name(code: str) -> str:
    code = str(code).strip().lower()
    if not code:
        return ""
    if code in LANGUAGE_NAME_OVERRIDES:
        return LANGUAGE_NAME_OVERRIDES[code]
    try:
        import pycountry

        language = pycountry.languages.get(alpha_3=code) or pycountry.languages.get(alpha_2=code)
        if language is not None:
            return getattr(language, "name", code)
    except Exception:
        pass
    return code


def language_counts(frame: pd.DataFrame) -> Tuple[pd.Series, int]:
    if "language" not in frame.columns:
        return pd.Series(dtype=int), 0
    languages = frame["language"].astype(str).str.strip().str.lower()
    valid = languages.notna() & (languages != "") & (languages != "nan")
    valid &= ~languages.str.startswith(("unknown", "unkown"), na=False)
    counts = languages.loc[valid].value_counts()
    return counts, int(valid.sum())


def english_language_count(frame: pd.DataFrame) -> Tuple[int, int]:
    counts, denominator = language_counts(frame)
    return int(counts.get("eng", 0)), denominator


def second_language_stats(frame: pd.DataFrame) -> Tuple[str, float, int, int]:
    counts, denominator = language_counts(frame)
    if denominator == 0 or len(counts) < 2:
        return "", np.nan, 0, denominator
    language = str(counts.index[1])
    numerator = int(counts.iloc[1])
    return language_display_name(language), 100.0 * numerator / denominator, numerator, denominator


def language_id_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Language and Dialect ID")
    text_lines.append("-----------------------")
    rows: List[Dict[str, Any]] = []

    for subset in SUBSETS:
        row: Dict[str, Any] = {
            "subset": subset,
            "model_eng": np.nan,
            "original_eng": np.nan,
            "model_second_language": "",
            "model_second_percentage": np.nan,
            "original_second_language": "",
            "original_second_percentage": np.nan,
        }
        for model, label in [(ORIGINAL, "original"), (args.model, "model")]:
            frame = safe_read_csv(language_id_csv(args, model, subset))
            metric = f"English language detected in {label} answers (%)"
            if frame is None or "language" not in frame.columns:
                continue
            numerator, denominator = english_language_count(frame)
            percentage = (100.0 * numerator / denominator) if denominator else np.nan
            row[f"{label}_eng"] = percentage
            metrics_by_subset[subset].append(
                metric_row(
                    metric,
                    percentage,
                    np.nan,
                    np.nan,
                    n=denominator,
                    section="Language and Dialect ID",
                    detail=f"percentage of language_id.csv rows with language=eng; numerator={numerator}; denominator={denominator}",
                )
            )

            language, second_percentage, second_numerator, second_denominator = second_language_stats(frame)
            row[f"{label}_second_language"] = language
            row[f"{label}_second_percentage"] = second_percentage
            metrics_by_subset[subset].append(
                metric_row(
                    f"Second most spoken language in {label} answers",
                    second_percentage,
                    np.nan,
                    np.nan,
                    n=second_denominator,
                    section="Language and Dialect ID",
                    detail=f"language={language}; numerator={second_numerator}; denominator={second_denominator}",
                )
            )
        rows.append(row)

    include_model_second = any(pd.notna(row["model_eng"]) and float(row["model_eng"]) < 100.0 for row in rows)
    include_original_second = any(pd.notna(row["original_eng"]) and float(row["original_eng"]) < 100.0 for row in rows)
    headers = ["subset", "% Eng in models' answers", "% Eng in original answers"]
    if include_model_second:
        headers.extend(["second most spoken language in models' answers", "percentage 2nd language in models' answers"])
    if include_original_second:
        headers.extend(["second most spoken language in original answers", "percentage 2nd language in original answers"])
    text_lines.append("	".join(headers))

    for row in rows:
        cells = [row["subset"], fmt(row["model_eng"]), fmt(row["original_eng"])]
        if include_model_second:
            cells.extend([str(row["model_second_language"]), fmt(row["model_second_percentage"])])
        if include_original_second:
            cells.extend([str(row["original_second_language"]), fmt(row["original_second_percentage"])])
        text_lines.append("	".join(cells))
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
            section="Language and Dialect ID",
            detail=f"percentage of dialect_id.csv rows where {column}={dialect}; numerator={numerator}; denominator={denominator}",
        )
        metrics_by_subset[subset].append(row)
        append_metric_table_line(text_lines, f"{metric} ({subset})", row)


def dialect_id_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Dialect distribution")
    text_lines.append("~~~~~~~~~~~~~~~~~~~~")
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

    text_lines.append("Dialectal logit metrics")
    text_lines.append("subset	dialectal_entrainment_spearman	dialectal_variance	n")
    for subset in SUBSETS:
        entrainment, variance, n_value = dialect_logit_metrics(args, subset)
        for metric, value in [("Dialectal entrainment", entrainment), ("Dialectal variance", variance)]:
            metrics_by_subset[subset].append(
                metric_row(
                    metric,
                    value,
                    np.nan,
                    np.nan,
                    n=n_value,
                    section="Language and Dialect ID",
                    detail="computed from dialect_logits.csv question_log_logits and answer_log_logits",
                )
            )
        text_lines.append("	".join([subset, fmt(entrainment), fmt(variance), fmt(n_value, 0)]))
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
    for metric in ["UTMOS", "latency", "interruption_segments", "interrupted"]:
        if metric in model_frame.columns or (original_frame is not None and metric in original_frame.columns):
            columns.append(metric)
    return columns


def basic_metric_display_name(metric: str) -> str:
    if metric == "interruption_segments":
        return "average number of interruptions per dialogues"
    if metric == "interrupted":
        return "interrupted time (s)"
    if metric == "interruptions count":
        return "number of dialogues with interruption"
    return metric


def basic_metric_values(metric: str, series: pd.Series) -> pd.Series:
    values = numeric(series).dropna()
    if metric.startswith(("CER", "WER")):
        values = values * 100.0
    elif metric == "interrupted":
        values = values / 1000.0
    return values


def basic_metric_fmt(metric: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    if metric == "interruption_segments":
        return f"{float(value):.2f}"
    if metric == "interrupted":
        return f"{float(value):.3f}"
    if metric.startswith(("CER", "WER")):
        return f"{float(value):.2f}"
    return fmt(value)


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
    model_values = basic_metric_values(metric, model_frame[metric])
    if model_values.empty:
        return
    model_mean = model_values.mean()
    original_mean = np.nan
    if original_frame is not None and metric in original_frame.columns:
        original_values = basic_metric_values(metric, original_frame[metric])
        if not original_values.empty:
            original_mean = original_values.mean()
    mean_diff = model_mean - original_mean if pd.notna(original_mean) else np.nan
    display_metric = basic_metric_display_name(metric)
    row = metric_row(
        display_metric,
        mean_diff,
        np.nan,
        np.nan,
        n=len(model_values),
        section="Intelligibility and Interruption Metrics",
        detail=metric,
        model_value=model_mean,
        original_value=original_mean,
    )
    metrics_by_subset[subset].append(row)
    text_lines.append("	".join([subset, display_metric, basic_metric_fmt(metric, model_mean), basic_metric_fmt(metric, original_mean), basic_metric_fmt(metric, mean_diff)]))


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
    model_value = int((model_values > 0).sum())
    original_value = np.nan
    if original_frame is not None and "interruptions" in original_frame.columns:
        original_values = numeric(original_frame["interruptions"]).fillna(0)
        original_value = int((original_values > 0).sum())
    mean_diff = model_value - original_value if pd.notna(original_value) else np.nan
    original_total = len(original_values) if original_frame is not None and "interruptions" in original_frame.columns else np.nan
    row = metric_row(
        "number of dialogues with interruption",
        mean_diff,
        np.nan,
        np.nan,
        n=len(model_values),
        section="Intelligibility and Interruption Metrics",
        detail="interruptions count",
        model_value=f"{model_value}/{len(model_values)}",
        original_value=f"{original_value}/{original_total}" if pd.notna(original_total) else np.nan,
    )
    metrics_by_subset[subset].append(row)
    original_text = f"{original_value}/{original_total}" if pd.notna(original_total) else "nan"
    text_lines.append("	".join([subset, "number of dialogues with interruption", f"{model_value}/{len(model_values)}", original_text, fmt(mean_diff, 0)]))


def base_metrics_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("Intelligibility and Interruption Metrics")
    text_lines.append("----------------------------------------")
    text_lines.append("subset	metric	model_mean	original_mean	mean_diff")
    for subset in SUBSETS:
        model_frame = safe_read_csv(base_metrics_csv(args, args.model, subset))
        original_frame = safe_read_csv(base_metrics_csv(args, ORIGINAL, subset))
        if model_frame is None:
            text_lines.append(f"{subset}	missing	missing	missing	missing")
            continue
        for metric in base_metric_columns(model_frame, original_frame):
            add_basic_metric_row(args, metrics_by_subset, text_lines, subset, metric, model_frame, original_frame)
        add_interruption_count_row(args, metrics_by_subset, text_lines, subset, model_frame, original_frame)
    text_lines.append("")

def naturalness_scores_with_relationship(args, model: str, subset: str) -> Optional[pd.DataFrame]:
    scores_path = args.results_root / model / "test" / subset / "naturalness_scores_normalized.csv"
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
    text_lines.append("See detailed report graphs.")
    for subset in SUBSETS:
        original_path = args.results_root / ORIGINAL / "test" / subset / "naturalness_scores_normalized.csv"
        model_path = args.results_root / args.model / "test" / subset / "naturalness_scores_normalized.csv"
        original = safe_read_csv(original_path)
        model = safe_read_csv(model_path)
        if original is None or model is None or "naturalness_logit" not in original or "naturalness_logit" not in model:
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
            detail="normalized naturalness_logit model-original",
        )
        metrics_by_subset[subset].append(row)

        original_rel = naturalness_scores_with_relationship(args, ORIGINAL, subset)
        model_rel = naturalness_scores_with_relationship(args, args.model, subset)
        if original_rel is None or model_rel is None:
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
    text_lines.append("")


def stances_section(args, metrics_by_subset: Dict[str, List[Dict[str, Any]]], text_lines: List[str]) -> None:
    text_lines.append("STANCE")
    text_lines.append("------")

    subset = "improvised"
    path = args.results_root / args.model / "test" / subset / "merged_stances.csv"
    frame = safe_read_csv(path)
    if frame is None or not {"question_index", "score_original", "score_llm"}.issubset(frame.columns):
        text_lines.append("STANCE improvised	missing")
        text_lines.append("")
        return

    frame = frame.copy()
    frame["score_original"] = numeric(frame["score_original"])
    frame["score_llm"] = numeric(frame["score_llm"])

    text_lines.append("Stance descriptions")
    text_lines.append("question	positive_original_%	negative_original_%	definition")
    for qidx, group in frame.groupby("question_index", sort=True):
        original = numeric(group["score_original"]).dropna()
        denominator = len(original)
        pos = 100.0 * float((original > 0).sum()) / denominator if denominator else np.nan
        neg = 100.0 * float((original < 0).sum()) / denominator if denominator else np.nan
        question = str(group["stance_question"].dropna().iloc[0]) if "stance_question" in group and group["stance_question"].notna().any() else ""
        metrics_by_subset[subset].append(
            metric_row(
                f"Q{qidx} stance description",
                np.nan,
                np.nan,
                np.nan,
                n=denominator,
                section="STANCE Descriptions",
                detail=question,
                model_value=pos,
                original_value=neg,
            )
        )
        text_lines.append("	".join([f"Q{qidx}", fmt(pos), fmt(neg), question]))

    all_original = numeric(frame["score_original"])
    all_model = numeric(frame["score_llm"])
    valid = all_original.notna() & all_model.notna()
    all_original = all_original.loc[valid]
    all_model = all_model.loc[valid]
    denominator = len(all_original)
    model_same = 100.0 * float((np.sign(all_original) == np.sign(all_model)).sum()) / denominator if denominator else np.nan
    model_more_positive = 100.0 * float((all_model > all_original).sum()) / denominator if denominator else np.nan
    model_more_negative = 100.0 * float((all_model < all_original).sum()) / denominator if denominator else np.nan

    result_rows = [
        ("original", 100.0 if denominator else np.nan, 0.0 if denominator else np.nan, 0.0 if denominator else np.nan),
        (args.model, model_same, model_more_positive, model_more_negative),
    ]
    text_lines.append("")
    text_lines.append("STANCE results")
    text_lines.append("dataset	STANCE same sign (%)	More positive (%)	More negative (%)")
    for dataset, same, more_positive, more_negative in result_rows:
        metrics_by_subset[subset].append(
            metric_row(
                f"STANCE results: {dataset}",
                same,
                np.nan,
                np.nan,
                n=denominator,
                section="STANCE Results",
                detail="",
                model_value=more_positive,
                original_value=more_negative,
            )
        )
        text_lines.append("	".join([dataset, fmt(same), fmt(more_positive), fmt(more_negative)]))
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
        frame = pd.DataFrame(columns=["feature", "subset", "auroc", "accuracy"])
    else:
        frame = frame.copy()
        frame["feature"] = frame["feature"].astype(str)
        frame = frame[frame["feature"].map(is_reported_explainable_feature)]
        frame = frame[~frame["feature"].isin(EXCLUDED_REPORT_METRICS)]
        frame["auroc"] = numeric(frame["auroc"])
        frame["accuracy"] = numeric(frame["accuracy"]) if "accuracy" in frame else np.nan

    for subset in SUBSETS:
        features = available_explainable_features(args, subset)
        if not features:
            continue
        sub = frame[frame["subset"] == subset].copy() if "subset" in frame else frame.copy()
        summary_by_feature = sub.drop_duplicates("feature").set_index("feature") if not sub.empty else pd.DataFrame()
        text_lines.append(f"{subset}")
        text_lines.append("rank_group\tmetric\tn\tmean_diff\tstd_diff\tp_value\tauroc\taccuracy")
        for feature in features:
            if not summary_by_feature.empty and feature in summary_by_feature.index:
                source_row = summary_by_feature.loc[feature]
                auroc = pd.to_numeric(source_row.get("auroc"), errors="coerce")
                accuracy = pd.to_numeric(source_row.get("accuracy"), errors="coerce")
            else:
                auroc = np.nan
                accuracy = np.nan
            row = add_feature_metric(args, metrics_by_subset, subset, feature, auroc, accuracy)
            text_lines.append(
                "\t".join(
                    [
                        "reported",
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
        columns = ["metric", "section", "model_value", "original_value", "mean_diff", "std_diff", "p_value", "n", "auroc", "accuracy", "detail"]
        pd.DataFrame(rows, columns=columns).to_csv(metrics_path, index=False)
        print(f"Wrote metrics: {metrics_path}")

    report_path.write_text("\n".join(align_tabular_blocks(text_lines)) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--asr-models", nargs="+", required=True)
    parser.add_argument("--language-id-model", default="facebook/mms-lid-126")
    parser.add_argument("--dialect-id-model", default="tiantiaf/voxlect-english-dialect-whisper-large-v3")
    parser.add_argument("--utmos-model", default="tarepan/SpeechMOS:v1.2.0 utmos22_strong")
    parser.add_argument("--vad-model", default="silero-vad")
    parser.add_argument("--stance-model", required=True)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--statistical-test", default="Welch t-test", choices=sorted(STATISTICAL_TESTS))
    parser.add_argument("--selection-method", default="end_with_question")
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--min-speakers", type=int, default=2)
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

    data_statistics_section(args, text_lines)
    setup_section(args, text_lines)
    setup_details_section(args, text_lines)
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
