#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


STANCE_FILE_RE = re.compile(r"stance_metrics_Q(?P<idx>\d+)\.csv$")

IDENTITY_COLUMNS = [
    "question_index",
    "row_idx",
    "initial_conversation",
    "speakers",
    "target_speaker",
    "target_side",
    "target_role",
    "target_category",
    "stance_question",
    "stance_related_categories",
    "relationship_detail",
    "asset_prompt_id_unique",
    "asset_a_id",
    "asset_b_id",
]

OUTPUT_COLUMNS = IDENTITY_COLUMNS + [
    "audio_path_original",
    "audio_path_llm",
    "score_original",
    "score_llm",
    "score_change",
    "score_abs_change",
    "score_relative_change",
    "score_abs_relative_change",
    "confidence_original",
    "confidence_llm",
    "confidence_change",
    "confidence_abs_change",
    "confidence_relative_change",
    "confidence_abs_relative_change",
    "choice_original",
    "choice_llm",
    "error_original",
    "error_llm",
]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text.strip()


def safe_float(value: Any) -> float:
    text = safe_str(value)
    if not text:
        return math.nan
    try:
        number = float(text)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def relative_change(new: float, old: float) -> float:
    if not math.isfinite(new) or not math.isfinite(old) or old == 0.0:
        return math.nan
    return (new - old) / abs(old)


def mean(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return math.nan
    return sum(valid) / len(valid)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stance_files(directory: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(directory.glob("stance_metrics_Q*.csv")):
        match = STANCE_FILE_RE.match(path.name)
        if match:
            files[int(match.group("idx"))] = path
    return files


MergeKey = tuple[str, int, str]


def question_number(row: dict[str, str], question_index: int) -> int:
    q_text = safe_str(row.get("question_index"))
    return int(q_text) if q_text.isdigit() else int(question_index)


def normalized_audio_id(row: dict[str, str]) -> str:
    for column in ("audio_path", "answer_audio_path"):
        audio_path = safe_str(row.get(column))
        if not audio_path:
            continue
        name = Path(audio_path).name
        if name.startswith("answer_"):
            name = name[len("answer_") :]
        return name
    return ""


def merge_key(row: dict[str, str], question_index: int) -> MergeKey:
    question = question_number(row, question_index)
    audio_id = normalized_audio_id(row)
    if audio_id:
        return "audio", question, audio_id
    return "row_idx", question, safe_str(row.get("row_idx"))


def keyed_rows(path: Path, question_index: int) -> dict[MergeKey, dict[str, str]]:
    rows: dict[MergeKey, dict[str, str]] = {}
    duplicate_keys = 0
    for row in read_csv_rows(path):
        key = merge_key(row, question_index)
        if key[2]:
            if key in rows:
                duplicate_keys += 1
                continue
            rows[key] = row
    if duplicate_keys:
        print(f"[WARN] {path}: skipped {duplicate_keys} duplicate merge key(s)")
    return rows


def sort_key(key: MergeKey) -> tuple[int, str, int | str]:
    key_type, question, identifier = key
    return question, key_type, int(identifier) if identifier.isdigit() else identifier


def merge_rows(original: dict[str, str], llm: dict[str, str]) -> dict[str, str]:
    score_original = safe_float(original.get("stance_score"))
    score_llm = safe_float(llm.get("stance_score"))
    confidence_original = safe_float(original.get("stance_probability"))
    confidence_llm = safe_float(llm.get("stance_probability"))

    score_change = score_llm - score_original
    confidence_change = confidence_llm - confidence_original
    score_relative_change = relative_change(score_llm, score_original)
    confidence_relative_change = relative_change(confidence_llm, confidence_original)

    merged: dict[str, str] = {}
    for column in IDENTITY_COLUMNS:
        merged[column] = safe_str(original.get(column)) or safe_str(llm.get(column))

    merged.update(
        {
            "audio_path_original": safe_str(original.get("audio_path")),
            "audio_path_llm": safe_str(llm.get("audio_path")),
            "score_original": fmt_float(score_original),
            "score_llm": fmt_float(score_llm),
            "score_change": fmt_float(score_change),
            "score_abs_change": fmt_float(abs(score_change)),
            "score_relative_change": fmt_float(score_relative_change),
            "score_abs_relative_change": fmt_float(abs(score_relative_change)),
            "confidence_original": fmt_float(confidence_original),
            "confidence_llm": fmt_float(confidence_llm),
            "confidence_change": fmt_float(confidence_change),
            "confidence_abs_change": fmt_float(abs(confidence_change)),
            "confidence_relative_change": fmt_float(confidence_relative_change),
            "confidence_abs_relative_change": fmt_float(abs(confidence_relative_change)),
            "choice_original": safe_str(original.get("stance_choice")),
            "choice_llm": safe_str(llm.get("stance_choice")),
            "error_original": safe_str(original.get("stance_error")),
            "error_llm": safe_str(llm.get("stance_error")),
        }
    )
    return merged


def print_summary(
    merged_rows: list[dict[str, str]],
    question_indices: list[int],
    unmatched_counts: dict[int, tuple[int, int]],
) -> None:
    rows_by_question: dict[int, list[dict[str, str]]] = {idx: [] for idx in question_indices}
    for row in merged_rows:
        question = safe_str(row.get("question_index"))
        if question.isdigit():
            rows_by_question.setdefault(int(question), []).append(row)

    print("")
    print("STANCE summary")
    print(
        "stance,evaluated,original_only,llm_only,"
        "score_mean_change,score_mean_abs_change,score_mean_relative_change,score_mean_abs_relative_change,"
        "confidence_mean_change,confidence_mean_abs_change,confidence_mean_relative_change,confidence_mean_abs_relative_change"
    )
    for idx in question_indices:
        rows = rows_by_question.get(idx, [])
        original_only, llm_only = unmatched_counts.get(idx, (0, 0))
        score_change = [safe_float(row.get("score_change")) for row in rows]
        score_abs_change = [safe_float(row.get("score_abs_change")) for row in rows]
        score_relative_change = [safe_float(row.get("score_relative_change")) for row in rows]
        score_abs_relative_change = [safe_float(row.get("score_abs_relative_change")) for row in rows]
        confidence_change = [safe_float(row.get("confidence_change")) for row in rows]
        confidence_abs_change = [safe_float(row.get("confidence_abs_change")) for row in rows]
        confidence_relative_change = [safe_float(row.get("confidence_relative_change")) for row in rows]
        confidence_abs_relative_change = [safe_float(row.get("confidence_abs_relative_change")) for row in rows]

        print(
            f"Q{idx},"
            f"{len(rows)},"
            f"{original_only},"
            f"{llm_only},"
            f"{fmt_float(mean(score_change))},"
            f"{fmt_float(mean(score_abs_change))},"
            f"{fmt_float(mean(score_relative_change))},"
            f"{fmt_float(mean(score_abs_relative_change))},"
            f"{fmt_float(mean(confidence_change))},"
            f"{fmt_float(mean(confidence_abs_change))},"
            f"{fmt_float(mean(confidence_relative_change))},"
            f"{fmt_float(mean(confidence_abs_relative_change))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=None, help="Accepted for compatibility; not used.")
    parser.add_argument("--stances-original", type=Path, required=True)
    parser.add_argument("--stances-llm", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-stances", type=int, default=10)
    args = parser.parse_args()

    original_files = stance_files(args.stances_original)
    llm_files = stance_files(args.stances_llm)
    question_indices = list(range(args.num_stances))

    merged: list[dict[str, str]] = []
    unmatched_counts: dict[int, tuple[int, int]] = {}
    for idx in question_indices:
        original_path = original_files.get(idx)
        llm_path = llm_files.get(idx)
        if original_path is None or llm_path is None:
            print(
                f"[WARN] Q{idx}: missing file(s): "
                f"original={original_path if original_path else 'missing'} "
                f"llm={llm_path if llm_path else 'missing'}"
            )
            unmatched_counts[idx] = (0, 0)
            continue

        original_rows = keyed_rows(original_path, idx)
        llm_rows = keyed_rows(llm_path, idx)
        original_keys = set(original_rows)
        llm_keys = set(llm_rows)
        common_keys = sorted(original_keys & llm_keys, key=sort_key)
        unmatched_counts[idx] = (len(original_keys - llm_keys), len(llm_keys - original_keys))

        for key in common_keys:
            merged.append(merge_rows(original_rows[key], llm_rows[key]))

    write_csv_rows(args.output_csv, merged)
    print(f"Wrote {len(merged)} merged STANCE rows to {args.output_csv}")
    print_summary(merged, question_indices, unmatched_counts)


if __name__ == "__main__":
    main()
