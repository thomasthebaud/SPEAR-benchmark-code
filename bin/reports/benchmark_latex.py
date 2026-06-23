#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


GROUPS = {
    "UTMOS": "Speech quality",
    "UTMOS_std": "Speech quality",
    "WER_%": "Speech quality",
    "WER_%_std": "Speech quality",
    "CER_%": "Speech quality",
    "CER_%_std": "Speech quality",
    "latency_ms": "Interruptions",
    "latency_ms_std": "Interruptions",
    "interrupted_time_ms": "Interruptions",
    "interrupted_time_ms_std": "Interruptions",
    "interruptions_%": "Interruptions",
    "english_answers_%": "Language and Dialect",
    "same_dialect_as_question_%": "Language and Dialect",
    "north_american_dialect_%": "Language and Dialect",
    "dialectal_entrainment_spearman": "Language and Dialect",
    "dialectal_variance": "Language and Dialect",
    "emotional_naturalness_logit": "Emotions",
    "emotional_naturalness_logit_std": "Emotions",
    "arousal_question_answer_corr": "Emotions",
    "valence_question_answer_corr": "Emotions",
    "dominance_question_answer_corr": "Emotions",
    "stance_same_as_question_%": "Stances",
    "stance_more_negative_%": "Stances",
    "stance_more_positive_%": "Stances",
    "explainable_duration_s": "Explainable Features",
    "explainable_duration_s_std": "Explainable Features",
    "explainable_voiced_ratio": "Explainable Features",
    "explainable_voiced_ratio_std": "Explainable Features",
    "explainable_normalized_f0_std_avg": "Explainable Features",
}
GROUP_ORDER = ["Speech quality", "Interruptions", "Language and Dialect", "Emotions", "Stances", "Explainable Features"]
ROW_END = " " + chr(92) * 2

METRIC_ORDER = [
    "UTMOS",
    "WER_%",
    "CER_%",
    "latency_ms",
    "interrupted_time_ms",
    "interruptions_%",
    "english_answers_%",
    "dialectal_entrainment_spearman",
    "dialectal_variance",
    "emotional_naturalness_logit",
    "arousal_question_answer_corr",
    "valence_question_answer_corr",
    "dominance_question_answer_corr",
    "stance_same_as_question_%",
    "stance_more_negative_%",
    "stance_more_positive_%",
    "explainable_duration_s",
    "explainable_voiced_ratio",
    "explainable_normalized_f0_std_avg",
]

HEADER_ROWS = {
    "UTMOS": ("UTMOS", "", "1-5"),
    "WER_%": ("WER", "", r"\%"),
    "CER_%": ("CER", "", r"\%"),
    "latency_ms": ("Latency", "", "(ms)"),
    "interrupted_time_ms": ("Interr.", "time", "(ms)"),
    "interruptions_%": ("Interr.", "", r"\%"),
    "english_answers_%": ("English", "answers", r"\%"),
    "dialectal_entrainment_spearman": ("Dialectal", "entrain.", r"$\beta$"),
    "dialectal_variance": ("Dialectal", "variance", r"$tr(\Sigma)$"),
    "emotional_naturalness_logit": ("Emotional", "Naturalness", "logit"),
    "arousal_question_answer_corr": ("Arousal", "corr.", r"$\rho$"),
    "valence_question_answer_corr": ("Valence", "corr.", r"$\rho$"),
    "dominance_question_answer_corr": ("Dominance", "corr.", r"$\rho$"),
    "stance_same_as_question_%": ("Same", "stance", r"\%"),
    "stance_more_negative_%": ("More", "negative", r"\%"),
    "stance_more_positive_%": ("More", "positive", r"\%"),
    "explainable_duration_s": ("Answer", "Duration", "s"),
    "explainable_voiced_ratio": ("Voiced", "ratio", ""),
    "explainable_normalized_f0_std_avg": ("Pitch", "Variation", "(std)"),
}

LABELS = {
    "UTMOS": "UTMOS",
    "UTMOS_std": "UTMOS std",
    "WER_%": "WER",
    "WER_%_std": "WER std",
    "CER_%": "CER",
    "CER_%_std": "CER std",
    "latency_ms": "Latency",
    "latency_ms_std": "Latency std",
    "interrupted_time_ms": "Interrupted time",
    "interrupted_time_ms_std": "Interrupted time std",
    "interruptions_%": "Interruptions",
    "english_answers_%": "English answers",
    "same_dialect_as_question_%": "Same dialect",
    "north_american_dialect_%": "North American",
    "dialectal_entrainment_spearman": "Dialectal entrainment",
    "dialectal_variance": "Dialectal variance",
    "emotional_naturalness_logit": "Naturalness",
    "emotional_naturalness_logit_std": "Naturalness std",
    "arousal_question_answer_corr": "Arousal corr.",
    "valence_question_answer_corr": "Valence corr.",
    "dominance_question_answer_corr": "Dominance corr.",
    "stance_same_as_question_%": "Same stance",
    "stance_more_negative_%": "More negative",
    "stance_more_positive_%": "More positive",
    "explainable_duration_s": "Duration",
    "explainable_duration_s_std": "Duration std",
    "explainable_voiced_ratio": "Voiced ratio",
    "explainable_voiced_ratio_std": "Voiced ratio std",
    "explainable_normalized_f0_std_avg": "Norm. f0 std avg.",
}

UNITS = {
    "UTMOS": "",
    "UTMOS_std": "",
    "WER_%": r"\%",
    "WER_%_std": r"\%",
    "CER_%": r"\%",
    "CER_%_std": r"\%",
    "latency_ms": "ms",
    "latency_ms_std": "ms",
    "interrupted_time_ms": "ms",
    "interrupted_time_ms_std": "ms",
    "interruptions_%": r"\%",
    "english_answers_%": r"\%",
    "same_dialect_as_question_%": r"\%",
    "north_american_dialect_%": r"\%",
    "dialectal_entrainment_spearman": r"$\beta$",
    "dialectal_variance": r"$tr(\Sigma)$",
    "emotional_naturalness_logit": "logit",
    "emotional_naturalness_logit_std": "logit",
    "arousal_question_answer_corr": "r",
    "valence_question_answer_corr": "r",
    "dominance_question_answer_corr": "r",
    "stance_same_as_question_%": r"\%",
    "stance_more_negative_%": r"\%",
    "stance_more_positive_%": r"\%",
    "explainable_duration_s": "s",
    "explainable_duration_s_std": "s",
    "explainable_voiced_ratio": "ratio",
    "explainable_voiced_ratio_std": "ratio",
    "explainable_normalized_f0_std_avg": "norm.",
}


def latex_escape(value) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_value(value, column: str) -> str:
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "nan":
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return latex_escape(value)
    unit = UNITS.get(column, "")
    if unit == r"\%":
        return f"{number:.1f}"
    if unit == "ms":
        return str(int(round(number)))
    if column == "dialectal_variance":
        return f"{number:.1f}"
    if column == "emotional_naturalness_logit":
        return f"{number:.2f}"
    if unit == "s":
        return f"{number:.1f}"
    if unit in {"r", r"$\rho$", r"$\beta$"}:
        return f"{number:.2f}"
    return latex_escape(value)


def metric_columns(fieldnames: list[str]) -> list[str]:
    available = set(fieldnames)
    return [column for column in METRIC_ORDER if column in available]


def column_groups(columns: list[str]) -> list[tuple[str, int]]:
    groups = []
    idx = 0
    while idx < len(columns):
        group = GROUPS[columns[idx]]
        span = 1
        while idx + span < len(columns) and GROUPS[columns[idx + span]] == group:
            span += 1
        groups.append((group, span))
        idx += span
    return groups


def tabular_spec(columns: list[str]) -> str:
    parts = ["l"]
    parts.extend("c" * span for _, span in column_groups(columns))
    return "|".join(parts)


def grouped_header(columns: list[str]) -> list[str]:
    cells = [""]
    groups = column_groups(columns)
    for idx, (group, span) in enumerate(groups):
        align = "c" if idx == len(groups) - 1 else "c|"
        cells.append(rf"\multicolumn{{{span}}}{{{align}}}{{{latex_escape(group)}}}")
    return cells


def split_header_rows(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    top = [r"\multirow{2}{*}{Model}"]
    middle = [""]
    units = [""]
    for column in columns:
        first, second, unit = HEADER_ROWS.get(column, (LABELS.get(column, column), "", UNITS.get(column, "")))
        top.append(first)
        middle.append(second)
        units.append(unit)
    return top, middle, units


def row_for(frame_row: dict[str, str], columns: list[str]) -> str:
    cells = [latex_escape(frame_row.get("model", ""))]
    cells.extend(format_value(frame_row.get(column), column) for column in columns)
    return " & ".join(cells) + ROW_END

def make_table(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    columns = metric_columns(fieldnames)
    if not columns:
        raise ValueError("No recognized benchmark metric columns found.")

    original = [row for row in rows if row.get("model", "") == "original"]
    models = [row for row in rows if row.get("model", "") != "original"]

    header_top, header_middle, header_units = split_header_rows(columns)
    lines = [
        rf"\begin{{tabular}}{{{tabular_spec(columns)}}}",
        r"\toprule",
        " & ".join(grouped_header(columns)) + ROW_END,
        " & ".join(header_top) + ROW_END,
        " & ".join(header_middle) + ROW_END,
        " & ".join(header_units) + ROW_END,
        r"\midrule",
    ]
    if original:
        for row in original:
            lines.append(row_for(row, columns))
        lines.append(r"\midrule")
    for row in models:
        lines.append(row_for(row, columns))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"

def main() -> int:
    parser = argparse.ArgumentParser(description="Convert benchmark.csv to a clean LaTeX table.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    args = parser.parse_args()

    with args.input_csv.open(newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    table = make_table(rows, fieldnames)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(table)
    print(f"Wrote benchmark LaTeX table: {args.output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
