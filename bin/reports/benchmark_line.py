#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


SUBSETS = ["improvised", "naturalistic"]


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


def naturalness_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "naturalness_scores_normalized.csv"


def stances_path(results_root: Path, model: str, subset: str) -> Path:
    return results_root / model / "test" / subset / "merged_stances.csv"


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


def wer_columns(frames: list[pd.DataFrame]) -> list[str]:
    columns = set()
    for frame in frames:
        columns.update(column for column in frame.columns if column == "WER" or column.startswith("WER_"))
    specific = sorted(column for column in columns if column.startswith("WER_"))
    return specific if specific else (["WER"] if "WER" in columns else [])


def wer_metrics(frames: list[pd.DataFrame]) -> tuple[float, float]:
    columns = wer_columns(frames)
    if not columns:
        return np.nan, np.nan

    all_values = []
    asr_means = []
    for column in columns:
        column_values = [numeric(frame[column]) for frame in frames if column in frame.columns]
        if not column_values:
            continue
        values = pd.concat(column_values, ignore_index=True).dropna()
        if values.empty:
            continue
        values = values * 100.0
        all_values.append(values)
        asr_means.append(float(values.mean()))

    average_wer = float(pd.concat(all_values, ignore_index=True).mean()) if all_values else np.nan
    wer_std_between_asr = float(pd.Series(asr_means, dtype="float64").std(ddof=0)) if len(asr_means) > 1 else 0.0 if asr_means else np.nan
    return average_wer, wer_std_between_asr


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


def naturalness_average(results_root: Path, model: str, subsets: list[str]) -> float:
    values = []
    for subset in subsets:
        frame = safe_read_csv(naturalness_path(results_root, model, subset))
        if frame is not None and "naturalness_logit" in frame.columns:
            values.append(numeric(frame["naturalness_logit"]))
    if not values:
        return np.nan
    return float(pd.concat(values, ignore_index=True).dropna().mean())


def stance_metrics(results_root: Path, model: str, subsets: list[str]) -> tuple[float, float, float]:
    same_numerator = 0
    same_denominator = 0
    more_negative = 0
    more_positive = 0
    comparison_denominator = 0

    for subset in subsets:
        frame = safe_read_csv(stances_path(results_root, model, subset), warn_missing=(subset != "naturalistic"))
        if frame is None or not {"score_original", "score_llm"}.issubset(frame.columns):
            continue
        original = numeric(frame["score_original"])
        model_scores = numeric(frame["score_llm"])
        valid = original.notna() & model_scores.notna()
        original = original.loc[valid]
        model_scores = model_scores.loc[valid]
        same_numerator += int((np.sign(original) == np.sign(model_scores)).sum())
        same_denominator += int(len(original))
        more_negative += int((model_scores < original).sum())
        more_positive += int((model_scores > original).sum())
        comparison_denominator += int(len(original))

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
    average_wer, wer_std_between_asr = wer_metrics(frames)
    avg_interruptions, interruption_fraction, interrupted_time_s = interruption_metrics(frames)
    same_dialect_pct, north_america_dialect_pct = dialect_percentages(args.results_root, args.model, args.subsets)
    second_lang, second_lang_pct = second_language(args.results_root, args.model, args.subsets)
    stance_same_pct, stance_more_negative_pct, stance_more_positive_pct = stance_metrics(args.results_root, args.model, args.subsets)

    return {
        "protocol": args.protocol,
        "model": args.model,
        "avg_latency": average_column(frames, "latency"),
        "avg_UTMOS": average_column(frames, "UTMOS"),
        "avg_WER_%": average_wer,
        "WER_%_std_between_asr_models": wer_std_between_asr,
        "average_number_of_interruptions_per_dialogue": avg_interruptions,
        "number_of_dialogues_with_interruption": interruption_fraction,
        "interrupted_time_s": interrupted_time_s,
        "EN_lang_%": english_percentage(args.results_root, args.model, args.subsets),
        "second_most_spoken_language": second_lang,
        "percentage_2nd_language": second_lang_pct,
        "same_dialect_%": same_dialect_pct,
        "NA_dialect_%": north_america_dialect_pct,
        "avg_emo_naturalness_logit": naturalness_average(args.results_root, args.model, args.subsets),
        "same_stance_as_question_%": stance_same_pct,
        "more_negative_stance_%": stance_more_negative_pct,
        "more_positive_stance_%": stance_more_positive_pct,
        "avg_general_expl_feat": general_explainable_average(args.results_root, args.model, args.subsets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write one CSV line of aggregate benchmark metrics for a model.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Backward-compatible alias; may be a directory or a .csv path.")
    parser.add_argument("--subsets", nargs="+", default=SUBSETS)
    parser.add_argument("--n-digits", type=int, help="Number of digits to keep after the decimal point for numeric CSV values.")
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
