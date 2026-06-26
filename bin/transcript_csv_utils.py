from pathlib import Path

import pandas as pd

COMPLETE_EXIT_CODE = 10


def asr_output_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].replace(" ", "_")


def has_transcript(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def audio_paths_match(current, previous) -> bool:
    if pd.isna(current) or pd.isna(previous):
        return False
    current = str(current)
    previous = str(previous)
    return current == previous or Path(current).name == Path(previous).name


def strip_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]


def build_transcripts_by_audio_path(transcripts: pd.DataFrame) -> dict[str, str]:
    by_audio_path = {}
    if "answer_audio_path" not in transcripts.columns:
        return by_audio_path

    for _, row in transcripts.iterrows():
        transcript = row.get("ASR_transcript_answer")
        audio_path = row.get("answer_audio_path")
        if not has_transcript(transcript) or pd.isna(audio_path):
            continue
        by_audio_path[str(audio_path)] = str(transcript)
        by_audio_path[Path(str(audio_path)).name] = str(transcript)
    return by_audio_path


def reusable_transcript_for_row(idx, row, transcripts: pd.DataFrame, transcripts_by_audio_path: dict[str, str]):
    if idx in transcripts.index:
        candidate = transcripts.at[idx, "ASR_transcript_answer"]
        if has_transcript(candidate):
            if "answer_audio_path" not in transcripts.columns:
                return str(candidate)
            current_audio = row.get("answer_audio_path")
            previous_audio = transcripts.at[idx, "answer_audio_path"]
            if audio_paths_match(current_audio, previous_audio):
                return str(candidate)

    if "answer_audio_path" not in row.index:
        return None

    audio_path = row.get("answer_audio_path")
    if pd.isna(audio_path):
        return None

    transcript = transcripts_by_audio_path.get(str(audio_path))
    if transcript is None:
        transcript = transcripts_by_audio_path.get(Path(str(audio_path)).name)
    return transcript


def count_missing_transcripts(metadata: pd.DataFrame, transcripts: pd.DataFrame) -> int:
    if "ASR_transcript_answer" not in transcripts.columns:
        return len(metadata)

    transcripts_by_audio_path = build_transcripts_by_audio_path(transcripts)
    missing = 0
    for idx, row in metadata.iterrows():
        transcript = reusable_transcript_for_row(idx, row, transcripts, transcripts_by_audio_path)
        if not has_transcript(transcript):
            missing += 1
    return missing
