#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


SUBSETS = ["improvised", "naturalistic"]
F0_PROFILE_FEATURES = [
    "f0_min_raw",
    "f0_p10",
    "f0_p52",
    "f0_median_raw",
    "f0_p75",
    "f0_p90",
    "f0_max_raw",
]
F0_FEATURE_FALLBACKS = {"f0_p52": "f0_p25"}
TURNTAKING_EXPERIMENT = "group4-dualturn-full-all6-fvad256"
TURNTAKING_FILE = f"turntaking.{TURNTAKING_EXPERIMENT}.csv"
MODELS_WITHOUT_INTERRUPTION_METRICS = {"gpt-audio-1.5", "mini-omni", "qwen2.5-omni-7b"}
STANCE_LABEL_OVERRIDES = {
    "aggression": "Calmness",
    "inhibition": "Disinhibition",
    "callousness": "Empathy",
    "disorganization": "Organization",
}


def display_model_name(model: str) -> str:
    return "human" if str(model) == "original" else str(model)


def safe_read_csv(path: Path, *, warn_missing: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if warn_missing:
            print(f"[WARN] Missing CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)



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


def valid_text(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    normalized = values.str.lower()
    valid = values.notna() & (values != "") & (normalized != "nan")
    valid &= ~normalized.str.startswith(("unknown", "unkown"), na=False)
    return values[valid]


def percentage(mask: pd.Series | np.ndarray, denominator: int) -> float:
    if denominator <= 0:
        return np.nan
    return float(100.0 * np.asarray(mask, dtype=bool).sum() / denominator)


def base_metrics_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "base_metrics.csv"


def language_id_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "language_id.csv"


def dialect_id_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "dialect_id.csv"


def dialect_logits_path(results_root: Path, model: str) -> Path:
    return results_root / model / "test" / "dialect_logits.csv"


def naturalness_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "naturalness_scores_normalized.csv"


def turntaking_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / TURNTAKING_FILE


def stances_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "merged_stances.csv"


def ser_avd_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "SER_AVD.csv"


def explainable_features_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "distrib_baselines_features_normalized.csv"


def cluster_feature_paths(results_root: Path, model: str, subset: str) -> list[Path]:
    folder = results_root / model / "test" / subset
    if not folder.exists():
        return []
    return sorted(folder.glob("correlation_cluster_features_rho*.csv"))


def mean_or_nan(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    return float(series.mean()) if not series.empty else np.nan


def base_metric_frames(results_root: Path, model: str, subsets: list[str]) -> list[pd.DataFrame]:
    frames = []
    for subset in subsets:
        frame = safe_read_csv(base_metrics_path(results_root, model, subset))
        if frame is not None:
            frames.append(frame)
    return frames


def average_column(frames: list[pd.DataFrame], column: str, *, scale: float = 1.0) -> float:
    values = [numeric(frame[column]) * scale for frame in frames if column in frame.columns]
    if not values:
        return np.nan
    return float(pd.concat(values, ignore_index=True).dropna().mean())


def concat_columns(frames: list[pd.DataFrame], columns: list[str], *, scale: float = 1.0) -> pd.Series:
    values = []
    for frame in frames:
        for column in columns:
            if column in frame.columns:
                values.append(numeric(frame[column]) * scale)
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True).dropna()


def mean_std(values: pd.Series) -> tuple[float, float]:
    values = pd.Series(values, dtype="float64").dropna()
    if values.empty:
        return np.nan, np.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def add_mean_std(row: dict[str, object], name: str, values: pd.Series, *, use_std: bool) -> None:
    mean, std = mean_std(values)
    row[name] = mean
    if use_std:
        row[f"{name}_std"] = std


def metric_columns(frames: list[pd.DataFrame], metric: str) -> list[str]:
    columns = set()
    for frame in frames:
        columns.update(column for column in frame.columns if column == metric or column.startswith(f"{metric}_"))
    specific = sorted(column for column in columns if column.startswith(f"{metric}_"))
    return specific if specific else ([metric] if metric in columns else [])


def dialogue_interruption_fraction(frames: list[pd.DataFrame]) -> str:
    interrupted = 0
    total = 0
    for frame in frames:
        total += len(frame)
        if "interruptions" in frame.columns:
            interrupted += int((numeric(frame["interruptions"]).fillna(0) > 0).sum())
        elif "interrupted" in frame.columns:
            interrupted += int((numeric(frame["interrupted"]).fillna(0) > 0).sum())
    return f"{interrupted}/{total}" if total else ""


def metric_mean_std(frames: list[pd.DataFrame], metric: str, *, scale: float = 1.0) -> tuple[float, float]:
    return mean_std(concat_columns(frames, metric_columns(frames, metric), scale=scale))


def interruption_rate_values(frames: list[pd.DataFrame]) -> pd.Series:
    values = []
    for frame in frames:
        if "interruptions" in frame.columns:
            values.append((numeric(frame["interruptions"]).fillna(0) > 0).astype(float) * 100.0)
        elif "interrupted" in frame.columns:
            values.append((numeric(frame["interrupted"]).fillna(0) > 0).astype(float) * 100.0)
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True).dropna()


def interruption_metrics(frames: list[pd.DataFrame]) -> tuple[float, str, float]:
    avg_interruptions = average_column(frames, "interruption_segments")
    fraction = dialogue_interruption_fraction(frames)
    interrupted_time_s = average_column(frames, "interrupted", scale=1.0 / 1000.0)
    return avg_interruptions, fraction, interrupted_time_s


def language_counts(results_root: Path, model: str, subsets: list[str]) -> tuple[pd.Series, int]:
    counts = []
    denominator = 0
    for subset in subsets:
        frame = safe_read_csv(language_id_path(results_root, model, subset))
        if frame is None or "language" not in frame.columns:
            continue
        languages = valid_text(frame["language"]).str.lower()
        denominator += int(len(languages))
        counts.append(languages.value_counts())
    if not counts:
        return pd.Series(dtype=int), 0
    merged = pd.concat(counts, axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
    return merged.astype(int), denominator


def second_language(results_root: Path, model: str, subsets: list[str]) -> tuple[str, float]:
    counts, denominator = language_counts(results_root, model, subsets)
    if denominator == 0 or len(counts) < 2:
        return "", np.nan
    language = str(counts.index[1])
    return language_display_name(language), float(100.0 * int(counts.iloc[1]) / denominator)

def english_percentage(results_root: Path, model: str, subsets: list[str]) -> float:
    counts, denominator = language_counts(results_root, model, subsets)
    return 100.0 * int(counts.get("eng", 0)) / denominator if denominator else np.nan


def dialect_percentages(results_root: Path, model: str, subsets: list[str]) -> tuple[float, float]:
    same_numerator = 0
    same_denominator = 0
    north_america_numerator = 0
    north_america_denominator = 0

    for subset in subsets:
        frame = safe_read_csv(dialect_id_path(results_root, model, subset))
        if frame is None or "answer_dialect" not in frame.columns:
            continue
        answer = frame["answer_dialect"].astype(str).str.strip()
        answer_valid = answer.notna() & (answer != "") & (answer.str.lower() != "nan")
        north_america_numerator += int((answer.loc[answer_valid] == "North America").sum())
        north_america_denominator += int(answer_valid.sum())

        if "dialect" not in frame.columns:
            continue
        question = frame["dialect"].astype(str).str.strip()
        pair_valid = answer_valid & question.notna() & (question != "") & (question.str.lower() != "nan")
        same_numerator += int((answer.loc[pair_valid] == question.loc[pair_valid]).sum())
        same_denominator += int(pair_valid.sum())

    same_pct = 100.0 * same_numerator / same_denominator if same_denominator else np.nan
    north_america_pct = 100.0 * north_america_numerator / north_america_denominator if north_america_denominator else np.nan
    return same_pct, north_america_pct


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


def least_squares_slope_flattened(question_vectors: list[np.ndarray], answer_vectors: list[np.ndarray]) -> float:
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


def answer_total_variance(answer_vectors: list[np.ndarray]) -> float:
    if not answer_vectors:
        return np.nan
    matrix = np.vstack(answer_vectors)
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.nanvar(matrix, axis=0, ddof=1).sum())


def dialect_logit_metrics(results_root: Path, model: str, subsets: list[str]) -> tuple[float, float]:
    frame = safe_read_csv(dialect_logits_path(results_root, model))
    if frame is None or not {"question_log_logits", "answer_log_logits"}.issubset(frame.columns):
        return np.nan, np.nan
    if "subset" in frame.columns:
        frame = frame[frame["subset"].isin(subsets)]
    question_vectors = []
    answer_vectors = []
    for _, row in frame.iterrows():
        question = parse_vector(row["question_log_logits"])
        answer = parse_vector(row["answer_log_logits"])
        if question is None or answer is None or question.shape != answer.shape:
            continue
        question_vectors.append(question)
        answer_vectors.append(answer)
    return least_squares_slope_flattened(question_vectors, answer_vectors), answer_total_variance(answer_vectors)


def naturalness_values(results_root: Path, model: str, subsets: list[str]) -> pd.Series:
    values = []
    for subset in subsets:
        frame = safe_read_csv(naturalness_path(results_root, model, subset))
        if frame is not None and "naturalness_logit" in frame.columns:
            values.append(numeric(frame["naturalness_logit"]))
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True).dropna()


def turntaking_naturalness_values(results_root: Path, model: str, subsets: list[str]) -> pd.Series:
    values = []
    for subset in subsets:
        frame = safe_read_csv(turntaking_path(results_root, model, subset))
        if frame is not None and "naturalness_score" in frame.columns:
            values.append(numeric(frame["naturalness_score"]))
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True).dropna()


def avd_correlation(results_root: Path, model: str, subsets: list[str], dimension: str) -> float:
    question_values = []
    answer_values = []
    q_col = f"question_{dimension}"
    a_col = f"answer_{dimension}"
    for subset in subsets:
        frame = safe_read_csv(ser_avd_path(results_root, model, subset))
        if frame is None or not {q_col, a_col}.issubset(frame.columns):
            continue
        valid = pd.DataFrame({"question": numeric(frame[q_col]), "answer": numeric(frame[a_col])}).dropna()
        if valid.empty:
            continue
        question_values.append(valid["question"])
        answer_values.append(valid["answer"])
    if not question_values:
        return np.nan
    question = pd.concat(question_values, ignore_index=True)
    answer = pd.concat(answer_values, ignore_index=True)
    if len(question) < 2 or question.nunique(dropna=True) < 2 or answer.nunique(dropna=True) < 2:
        return np.nan
    return float(question.corr(answer))


def explainable_values(results_root: Path, model: str, subsets: list[str], column: str) -> pd.Series:
    values = []
    for subset in subsets:
        frame = safe_read_csv(explainable_features_path(results_root, model, subset))
        if frame is not None and column in frame.columns:
            values.append(numeric(frame[column]))
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True).dropna()


def normalized_f0_std_average(results_root: Path, model: str, subsets: list[str]) -> float:
    stds = []
    for feature in F0_PROFILE_FEATURES:
        values = explainable_values(results_root, model, subsets, feature)
        if values.empty and feature in F0_FEATURE_FALLBACKS:
            values = explainable_values(results_root, model, subsets, F0_FEATURE_FALLBACKS[feature])
        if len(values) > 1:
            stds.append(float(values.std(ddof=1)))
        elif len(values) == 1:
            stds.append(0.0)
    return float(pd.Series(stds, dtype="float64").mean()) if stds else np.nan


def known_stance_reference(frame: pd.DataFrame) -> pd.Series:
    if "score_reference" in frame.columns:
        reference = numeric(frame["score_reference"])
        if reference.notna().any():
            return reference

    if not {"target_category", "stance_related_categories"}.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    def row_reference(row: pd.Series) -> float:
        categories = [str(category).strip().lower() for category in str(row.get("stance_related_categories", "")).split("|")]
        target_category = str(row.get("target_category", "")).strip().lower()
        if len(categories) >= 1 and target_category == categories[0]:
            return 2.0
        if len(categories) >= 2 and target_category == categories[1]:
            return -2.0
        return np.nan

    return frame.apply(row_reference, axis=1).astype("float64")


def stance_row_label(row: pd.Series) -> str:
    for column in ["target_category", "stance_related_categories", "stance_question"]:
        value = str(row.get(column, "")).strip()
        if not value or value.lower() in {"nan", "none"}:
            continue
        return value.replace("|", " ").replace("/", " ").split()[0].lower()
    return ""


def orient_stance_scores(frame: pd.DataFrame, scores: pd.Series) -> pd.Series:
    inverted = frame.apply(lambda row: stance_row_label(row) in STANCE_LABEL_OVERRIDES, axis=1)
    multipliers = pd.Series(np.where(inverted, -1.0, 1.0), index=frame.index)
    return scores * multipliers


def stance_metrics(results_root: Path, model: str, subsets: list[str]) -> tuple[float, float, float]:
    same_numerator = 0
    same_denominator = 0
    more_negative = 0
    more_positive = 0
    comparison_denominator = 0

    for subset in subsets:
        frame = safe_read_csv(stances_path(results_root, model, subset), warn_missing=(subset != "naturalistic"))
        if frame is None or "score_llm" not in frame.columns:
            continue
        reference = orient_stance_scores(frame, known_stance_reference(frame))
        model_scores = orient_stance_scores(frame, numeric(frame["score_llm"]))
        valid = reference.notna() & model_scores.notna()
        reference = reference.loc[valid]
        model_scores = model_scores.loc[valid]
        same_numerator += int((np.sign(reference) == np.sign(model_scores)).sum())
        same_denominator += int(len(reference))
        more_negative += int((model_scores < reference).sum())
        more_positive += int((model_scores > reference).sum())
        comparison_denominator += int(len(reference))

    same_pct = 100.0 * same_numerator / same_denominator if same_denominator else np.nan
    more_negative_pct = 100.0 * more_negative / comparison_denominator if comparison_denominator else np.nan
    more_positive_pct = 100.0 * more_positive / comparison_denominator if comparison_denominator else np.nan
    return same_pct, more_negative_pct, more_positive_pct


def general_explainable_average(results_root: Path, model: str, subsets: list[str]) -> float:
    values = []
    for subset in subsets:
        for path in cluster_feature_paths(results_root, model, subset):
            frame = safe_read_csv(path)
            if frame is None:
                continue
            for column in frame.columns:
                if column.startswith("general_explainable_feature"):
                    values.append(numeric(frame[column]))
    if not values:
        return np.nan
    return float(pd.concat(values, ignore_index=True).dropna().mean())


def resolve_output_path(args: argparse.Namespace) -> Path:
    output = args.output_csv or args.output_dir
    if output is None:
        return Path("reports") / args.protocol / "benchmark" / f"{args.model}.csv"
    output = Path(output)
    if output.suffix.lower() == ".csv":
        return output
    return output / f"{args.model}.csv"


def compute_benchmark_line(args: argparse.Namespace) -> dict[str, object]:
    frames = base_metric_frames(args.results_root, args.model, args.subsets)
    suppress_interruption_metrics = args.model.lower() in MODELS_WITHOUT_INTERRUPTION_METRICS
    same_dialect_pct, north_america_dialect_pct = dialect_percentages(args.results_root, args.model, args.subsets)
    dialect_entrainment, dialect_variance = dialect_logit_metrics(args.results_root, args.model, args.subsets)
    stance_same_pct, stance_more_negative_pct, stance_more_positive_pct = stance_metrics(args.results_root, args.model, args.subsets)

    row: dict[str, object] = {
        "protocol": args.protocol,
        "model": display_model_name(args.model),
    }
    add_mean_std(row, "UTMOS", concat_columns(frames, ["UTMOS"]), use_std=args.use_std)
    add_mean_std(row, "WER_%", concat_columns(frames, metric_columns(frames, "WER"), scale=100.0), use_std=args.use_std)
    add_mean_std(row, "CER_%", concat_columns(frames, metric_columns(frames, "CER"), scale=100.0), use_std=args.use_std)
    latency_values = concat_columns(frames, ["latency"])
    latency_values = latency_values[latency_values >= 0.0]
    add_mean_std(row, "latency_ms", latency_values, use_std=args.use_std)

    if suppress_interruption_metrics:
        row["interrupted_time_ms"] = np.nan
        if args.use_std:
            row["interrupted_time_ms_std"] = np.nan
        interruptions_pct = np.nan
    else:
        add_mean_std(row, "interrupted_time_ms", concat_columns(frames, ["interrupted"]), use_std=args.use_std)
        interruptions_pct = mean_std(interruption_rate_values(frames))[0]

    row.update(
        {
            "interruptions_%": interruptions_pct,
            "english_answers_%": english_percentage(args.results_root, args.model, args.subsets),
            "same_dialect_as_question_%": same_dialect_pct,
            "north_american_dialect_%": north_america_dialect_pct,
            "dialectal_entrainment_spearman": dialect_entrainment,
            "dialectal_variance": dialect_variance,
        }
    )
    row["turn_taking_naturalness"] = mean_std(turntaking_naturalness_values(args.results_root, args.model, args.subsets))[0]
    add_mean_std(row, "emotional_naturalness_logit", naturalness_values(args.results_root, args.model, args.subsets), use_std=args.use_std)
    row.update(
        {
            "arousal_question_answer_corr": avd_correlation(args.results_root, args.model, args.subsets, "arousal"),
            "valence_question_answer_corr": avd_correlation(args.results_root, args.model, args.subsets, "valence"),
            "dominance_question_answer_corr": avd_correlation(args.results_root, args.model, args.subsets, "dominance"),
            "stance_same_as_question_%": stance_same_pct,
            "stance_more_negative_%": stance_more_negative_pct,
            "stance_more_positive_%": stance_more_positive_pct,
        }
    )
    add_mean_std(row, "explainable_duration_s", explainable_values(args.results_root, args.model, args.subsets, "total_duration_s"), use_std=args.use_std)
    add_mean_std(row, "explainable_voiced_ratio", explainable_values(args.results_root, args.model, args.subsets, "voiced_ratio"), use_std=args.use_std)
    row["explainable_normalized_f0_std_avg"] = normalized_f0_std_average(args.results_root, args.model, args.subsets)
    return row

def main() -> int:
    parser = argparse.ArgumentParser(description="Write one CSV line of aggregate benchmark metrics for a model.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Backward-compatible alias; may be a directory or a .csv path.")
    parser.add_argument("--subsets", nargs="+", default=SUBSETS)
    parser.add_argument("--n-digits", type=int, help="Number of digits to keep after the decimal point for numeric CSV values.")
    parser.add_argument("--use-std", action="store_true", help="Include standard deviation columns for mean-valued metrics.")
    parser.add_argument("--ignore-features", nargs="*", default=[], help="Accepted for compatibility with 52_benchmark.sh.")
    args = parser.parse_args()
    if args.n_digits is not None and args.n_digits < 0:
        parser.error("--n-digits must be non-negative")

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = compute_benchmark_line(args)
    float_format = f"%.{args.n_digits}f" if args.n_digits is not None else None
    pd.DataFrame([row]).to_csv(output_path, index=False, float_format=float_format)
    print(f"Wrote benchmark line: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
