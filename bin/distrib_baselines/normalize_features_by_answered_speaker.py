#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


F0_FEATURES = [
    "f0_mean_raw",
    "f0_median_raw",
    "f0_std_raw",
    "f0_min_raw",
    "f0_max_raw",
    "f0_range_raw",
    "f0_p10",
    "f0_p90",
    "f0_range_p10_p90",
    "f0_mean_p10_p90",
    "f0_std_p10_p90",
    "f0_p25",
    "f0_p75",
    "f0_range_p25_p75",
    "f0_mean_p25_p75",
    "f0_std_p25_p75",
]
RENAME_COLUMNS = {
    "f0_total_duration_s": "total_duration_s",
    "f0_voiced_duration_s": "voiced_duration_s",
    "f0_voiced_ratio": "voiced_ratio",
    "f0_n_voiced_frames": "n_voiced_frames",
}


def audio_stem(value) -> str:
    if pd.isna(value):
        return ""
    return Path(str(value)).stem


def answered_speaker_from_question(value) -> str:
    if pd.isna(value):
        return ""
    matches = re.findall(r"([A-Za-z0-9]+):", str(value))
    return matches[-1] if matches else ""


def load_answered_speakers(metadata_path: Path, *, questions: bool = False) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    key_column = "audio_path" if questions else ("answer_audio_path" if "answer_audio_path" in metadata.columns else "audio_path")
    metadata["_audio_stem"] = metadata[key_column].map(audio_stem)
    if "transcript_question" in metadata.columns:
        metadata["answered_speaker"] = metadata["transcript_question"].map(answered_speaker_from_question)
    else:
        metadata["answered_speaker"] = ""
    return metadata[["_audio_stem", "answered_speaker"]].drop_duplicates("_audio_stem")


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["_answered_speaker_mean_f0"] = (
        pd.to_numeric(out["f0_mean_raw"], errors="coerce")
        .groupby(out["answered_speaker"], dropna=False)
        .transform("mean")
    )
    valid_denominator = out["_answered_speaker_mean_f0"].replace(0, np.nan)
    for column in F0_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") / valid_denominator
    out = out.drop(columns=["_answered_speaker_mean_f0"])
    return out.rename(columns={old: new for old, new in RENAME_COLUMNS.items() if old in out.columns})


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize distribution baseline f0 features by answered speaker.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", action="store_true", help="Match feature rows to metadata by question audio_path instead of answer_audio_path.")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    if "orig_id" not in features.columns:
        raise ValueError(f"Missing required column 'orig_id' in {args.features}")
    if "f0_mean_raw" not in features.columns:
        raise ValueError(f"Missing required column 'f0_mean_raw' in {args.features}")

    answered_speakers = load_answered_speakers(args.metadata, questions=args.questions)
    features = features.copy()
    features["_audio_stem"] = features["orig_id"].astype(str)
    merged = features.merge(answered_speakers, on="_audio_stem", how="left")
    merged["answered_speaker"] = merged["answered_speaker"].fillna("")
    normalized = normalize_features(merged).drop(columns=["_audio_stem"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)
    print(f"Wrote {len(normalized)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
