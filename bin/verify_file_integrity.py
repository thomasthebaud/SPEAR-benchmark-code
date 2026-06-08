#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return path
    return base_dir / path


def path_exists(path_value, base_dir: Path) -> bool:
    if pd.isna(path_value) or str(path_value).strip() == "":
        return False
    return resolve_path(str(path_value), base_dir).exists()


def transcript_key(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def verify_lengths(metadata: pd.DataFrame, transcript_files: list[Path]) -> None:
    for transcript_file in transcript_files:
        transcripts = pd.read_csv(transcript_file)
        if len(transcripts) != len(metadata):
            raise AssertionError(
                f"Length mismatch: metadata has {len(metadata)} rows but "
                f"{transcript_file} has {len(transcripts)} rows."
            )


def missing_audio_mask(metadata: pd.DataFrame, base_dir: Path) -> pd.Series:
    mask = pd.Series(False, index=metadata.index)
    for column in ["audio_path", "answer_audio_path"]:
        if column not in metadata.columns:
            raise AssertionError(f"Missing required metadata column: {column}")
        exists = metadata[column].map(lambda value: path_exists(value, base_dir))
        mask = mask | ~exists
    return mask


def build_input_audio_lookup(inputs_metadata: pd.DataFrame) -> dict[str, str]:
    if "transcript_question" not in inputs_metadata.columns:
        raise AssertionError("inputs metadata is missing transcript_question")
    if "audio_path" not in inputs_metadata.columns:
        raise AssertionError("inputs metadata is missing audio_path")

    lookup = {}
    duplicates = set()
    for _, row in inputs_metadata.iterrows():
        key = transcript_key(row["transcript_question"])
        if not key:
            continue
        if key in lookup and lookup[key] != row["audio_path"]:
            duplicates.add(key)
        lookup[key] = row["audio_path"]

    for key in duplicates:
        lookup.pop(key, None)
    if duplicates:
        print(f"Warning: ignored {len(duplicates)} duplicate transcript_question keys in inputs metadata.", flush=True)
    return lookup


def correct_metadata(metadata: pd.DataFrame, inputs_metadata: pd.DataFrame, base_dir: Path) -> tuple[pd.DataFrame, int, int]:
    lookup = build_input_audio_lookup(inputs_metadata)
    corrected = metadata.copy()
    missing_before = missing_audio_mask(corrected, base_dir)
    corrected_count = 0
    dropped_count = 0
    keep_rows = []

    for idx, row in corrected.iterrows():
        if missing_before.loc[idx]:
            key = transcript_key(row.get("transcript_question", ""))
            replacement_audio_path = lookup.get(key)
            if replacement_audio_path is not None:
                if row.get("audio_path") != replacement_audio_path:
                    corrected.at[idx, "audio_path"] = replacement_audio_path
                    corrected_count += 1

        row_ok = True
        for column in ["audio_path", "answer_audio_path"]:
            if not path_exists(corrected.at[idx, column], base_dir):
                row_ok = False
                break
        if row_ok:
            keep_rows.append(idx)
        else:
            dropped_count += 1

    return corrected.loc[keep_rows].reset_index(drop=True), corrected_count, dropped_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True, help="LLM metadata.csv to verify.")
    parser.add_argument("--inputs-metadata", type=Path, required=True, help="Prepared input metadata.csv for the same split/subset.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory for relative audio paths.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Defaults to metadata_verified.csv next to metadata.")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    transcript_files = sorted(args.metadata.parent.glob("*_transcripts.csv"))
    verify_lengths(metadata, transcript_files)
    print(f"Verified transcript lengths for {len(transcript_files)} ASR file(s): {args.metadata.parent}", flush=True)

    missing = missing_audio_mask(metadata, args.base_dir)
    missing_count = int(missing.sum())
    if missing_count == 0:
        print(f"All audio files exist for {args.metadata}", flush=True)
        return 0

    print(f"Found {missing_count}/{len(metadata)} metadata row(s) with at least one missing audio file in {args.metadata}", flush=True)
    inputs_metadata = pd.read_csv(args.inputs_metadata)
    verified, corrected_count, dropped_count = correct_metadata(metadata, inputs_metadata, args.base_dir)

    output_path = args.output or (args.metadata.parent / "metadata_verified.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verified.to_csv(output_path, index=False)
    print(
        f"Wrote {output_path}: kept={len(verified)}/{len(metadata)}, "
        f"audio_path_corrected={corrected_count}, dropped={dropped_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
