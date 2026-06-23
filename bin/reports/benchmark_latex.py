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
    "english_answers_%": "Language",
    "same_dialect_as_question_%": "Language",
    "north_american_dialect_%": "Language",
    "emotional_naturalness_logit": "Emotions",
    "emotional_naturalness_logit_std": "Emotions",
    "arousal_question_answer_corr": "Emotions",
    "valence_question_answer_corr": "Emotions",
    "dominance_question_answer_corr": "Emotions",
    "stance_same_as_question_%": "Stances",
    "stance_more_negative_%": "Stances",
    "stance_more_positive_%": "Stances",
    "explainable_duration_s": "Explainable",
    "explainable_duration_s_std": "Explainable",
    "explainable_voiced_ratio": "Explainable",
    "explainable_voiced_ratio_std": "Explainable",
    "explainable_normalized_f0_std_avg": "Explainable",
}
GROUP_ORDER = ["Speech quality", "Interruptions", "Language", "Emotions", "Stances", "Explainable"]
ROW_END = " " + chr(92) * 2

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
    if unit == "s":
        return f"{number:.1f}"
    if unit == "r":
        return f"{number:.2f}"
    return latex_escape(value)


def metric_columns(fieldnames: list[str]) -> list[str]:
    return [column for column in fieldnames if column in GROUPS]


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

    lines = [
        rf"\begin{{tabular}}{{{tabular_spec(columns)}}}",
        r"\toprule",
        " & ".join(grouped_header(columns)) + ROW_END,
        " & ".join(["Model"] + [latex_escape(LABELS[column]) for column in columns]) + ROW_END,
        " & ".join([""] + [UNITS[column] for column in columns]) + ROW_END,
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
