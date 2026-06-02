#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SCORE_COLUMNS = ("arousal", "dominance", "valence")
TURN_SCORE_COLUMNS = {column: f"turn_{column}" for column in SCORE_COLUMNS}
OUTPUT_COLUMNS = [
    "question_arousal",
    "question_dominance",
    "question_valence",
    "answer_arousal",
    "answer_dominance",
    "answer_valence",
    "question_num_chunks",
    "answer_num_chunks",
]
NAN_VALUE = "nan"


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def load_feature_scores(features_csv: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    features = pd.read_csv(features_csv)
    missing_columns = [column for column in TURN_SCORE_COLUMNS.values() if column not in features.columns]
    if missing_columns:
        raise RuntimeError(
            f"{features_csv} does not contain full-turn emotion columns: {', '.join(missing_columns)}. "
            "Rerun 20_naturalness_feats.sh --extract before aggregation."
        )

    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}

    for (pair_stem, vad_source), group in features.groupby(["pair_stem", "vad_source"], dropna=False):
        source = str(vad_source).strip().lower()
        if source.startswith("existing_"):
            source = source.removeprefix("existing_")
        if source not in {"question", "answer"}:
            continue

        values: Dict[str, str] = {}
        for column in SCORE_COLUMNS:
            turn_column = TURN_SCORE_COLUMNS[column]
            scores = group[turn_column].map(safe_float).dropna()
            if scores.empty:
                values[column] = NAN_VALUE
            else:
                values[column] = format_float(float(scores.iloc[0]))
        values["num_chunks"] = str(len(group))

        grouped.setdefault(str(pair_stem).strip().lower(), {})[source] = values

    return grouped


def build_output_rows(metadata_csv: Path, scores: Dict[str, Dict[str, Dict[str, str]]]) -> tuple[List[dict], List[str]]:
    metadata = pd.read_csv(metadata_csv)
    metadata_fields = list(metadata.columns)
    fieldnames = metadata_fields + [field for field in OUTPUT_COLUMNS if field not in metadata_fields]

    rows: List[dict] = []
    for row in metadata.to_dict(orient="records"):
        out = dict(row)
        pair_stem = Path(str(row.get("audio_path", ""))).stem.lower()
        by_source = scores.get(pair_stem, {})
        question = by_source.get("question")
        answer = by_source.get("answer")
        if question is None:
            question = {column: NAN_VALUE for column in SCORE_COLUMNS}
            question["num_chunks"] = "0"
        if answer is None:
            answer = {column: NAN_VALUE for column in SCORE_COLUMNS}
            answer["num_chunks"] = "0"

        for column in SCORE_COLUMNS:
            out[f"question_{column}"] = question[column]
            out[f"answer_{column}"] = answer[column]
        out["question_num_chunks"] = question["num_chunks"]
        out["answer_num_chunks"] = answer["num_chunks"]
        rows.append(out)

    return rows, fieldnames


def write_rows(output_csv: Path, rows: List[dict], fieldnames: List[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full-turn VoxProfile A/V/D scores by question and answer.")
    parser.add_argument("--metadata", type=Path, required=True, help="Benchmark metadata.csv.")
    parser.add_argument("--features", type=Path, default=None, help="VoxProfile feature metadata.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Output SER_AVD.csv.")
    args = parser.parse_args()

    features_csv = args.features or (args.metadata.parent / "naturalness" / "voxprofile_features" / "metadata.csv")
    if not features_csv.exists():
        raise FileNotFoundError(f"VoxProfile feature metadata not found: {features_csv}")

    scores = load_feature_scores(features_csv)
    rows, fieldnames = build_output_rows(args.metadata, scores)
    write_rows(args.output, rows, fieldnames)
    print(f"Wrote SER AVD scores to {args.output}")


if __name__ == "__main__":
    main()
