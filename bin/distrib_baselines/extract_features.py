#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import parselmouth
import spacy
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
F0_SCRIPT = SCRIPT_DIR / "run_f0_extraction.py"
LEXICAL_SCRIPT = SCRIPT_DIR / "run_lexical_extraction.py"
TEMPORAL_SCRIPT = SCRIPT_DIR / "run_temporal_extraction.py"

F0_FLOOR = 75
F0_CEILING = 500
MIN_VOICED_RATIO = 0.05
LOW_CONF_THRESH = 0.7
MATTR_SMALL = 50
MATTR_LARGE = 500
MTLD_TTR_THRESHOLD = 0.72
MIN_WORDS_LD = 50
PAUSE_THRESHOLD_S = 0.2
MERGE_GAP_THRESHOLD_S = 1.0
MIN_STRETCH_DURATION_S = 12.1


def load_functions(script_path: Path, names: List[str]) -> Dict[str, Any]:
    """Load only selected function definitions from runner scripts with top-level side effects."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))

    def is_uppercase_assign(node: ast.AST) -> bool:
        if not isinstance(node, ast.Assign):
            return False
        return all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)

    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or is_uppercase_assign(node)
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: Dict[str, Any] = {"__file__": str(script_path)}
    exec(compile(module, str(script_path), "exec"), namespace)
    return {name: namespace[name] for name in names if name in namespace}


helpers: Dict[str, Any] = {}
helpers.update(load_functions(F0_SCRIPT, ["robust_stats"]))
helpers.update(
    load_functions(
        LEXICAL_SCRIPT,
        ["normalize_word", "compute_entropy", "compute_mattr", "compute_mtld"],
    )
)
helpers.update(
    load_functions(
        TEMPORAL_SCRIPT,
        [
            "merge_vad_segments",
            "filter_min_duration",
            "words_in_stretches",
            "assign_stretch_index_for_words",
            "compute_pause_stats_within_stretches",
            "compute_metrics_for_file",
        ],
    )
)


def resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def load_relationships(path: Optional[Path]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    relationship_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    if path is None or not path.exists():
        return relationship_map
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            vendor_id = row.get("vendor_id", "")
            session_id = row.get("session_id", "")
            if vendor_id and session_id:
                relationship_map[(vendor_id, session_id)] = (
                    row.get("relationship", "UNKNOWN"),
                    row.get("relationship_detail", "UNKNOWN"),
                )
    return relationship_map


def parse_ids(orig_id: str) -> Dict[str, str]:
    parts = orig_id.split("_")
    vendor_id = parts[0] if len(parts) > 0 else ""
    session_id = parts[1][1:] if len(parts) > 1 and parts[1].startswith("S") else (parts[1] if len(parts) > 1 else "")
    conversation_id = "_".join(parts[:3]) if len(parts) >= 3 else orig_id
    return {
        "orig_id": orig_id,
        "vendor_id": vendor_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
    }


def compute_f0_features(audio_path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"f0_status": "OK"}
    snd = parselmouth.Sound(str(audio_path))
    total_dur = snd.get_total_duration()
    pitch = snd.to_pitch_ac(pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEILING)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    total_frames = len(f0)
    voiced_frames = len(voiced)
    voiced_ratio = voiced_frames / total_frames if total_frames else 0.0

    row.update(
        {
            "f0_total_duration_s": total_dur,
            "f0_voiced_duration_s": voiced_ratio * total_dur,
            "f0_voiced_ratio": voiced_ratio,
            "f0_n_voiced_frames": voiced_frames,
        }
    )

    if voiced_frames == 0:
        row["f0_status"] = "NO_VOICED_FRAMES"
        return row

    row.update(
        {
            "f0_mean_raw": float(np.mean(voiced)),
            "f0_median_raw": float(np.median(voiced)),
            "f0_std_raw": float(np.std(voiced)),
            "f0_min_raw": float(np.min(voiced)),
            "f0_max_raw": float(np.max(voiced)),
            "f0_range_raw": float(np.max(voiced) - np.min(voiced)),
        }
    )

    p10, p90, r1090, m1090, s1090 = helpers["robust_stats"](voiced, 10, 90)
    row.update(
        {
            "f0_p10": p10,
            "f0_p90": p90,
            "f0_range_p10_p90": r1090,
            "f0_mean_p10_p90": m1090,
            "f0_std_p10_p90": s1090,
        }
    )

    p25, p75, r2575, m2575, s2575 = helpers["robust_stats"](voiced, 25, 75)
    row.update(
        {
            "f0_p25": p25,
            "f0_p75": p75,
            "f0_range_p25_p75": r2575,
            "f0_mean_p25_p75": m2575,
            "f0_std_p25_p75": s2575,
        }
    )

    if voiced_ratio < MIN_VOICED_RATIO:
        row["f0_status"] = "LOW_VOICED_RATIO"
    return row


def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        return spacy.blank("en")


def iter_text_tokens(text: str) -> List[str]:
    return [
        helpers["normalize_word"](token)
        for token in re.findall(r"\b[\w']+\b", text)
        if helpers["normalize_word"](token)
    ]


def compute_lexical_features(text: str, nlp) -> Dict[str, Any]:
    row: Dict[str, Any] = {"lexical_status": "OK", "lexical_status_reason": ""}
    tokens = iter_text_tokens(text)
    total_words = len(tokens)
    if total_words == 0:
        row["lexical_status"] = "EMPTY"
        return row

    row["lexical_total_words"] = total_words
    row["lexical_unique_words"] = len(set(tokens))
    row["lexical_mean_asr_confidence"] = np.nan
    row["lexical_low_conf_flag"] = ""

    doc = nlp(" ".join(tokens))
    pos_tags = [token.pos_ for token in doc]
    content_count = sum(pos in {"NOUN", "VERB", "ADJ", "ADV"} for pos in pos_tags)
    if not any(pos_tags):
        content_count = 0

    row["lexical_content_word_count"] = content_count
    row["lexical_function_word_count"] = total_words - content_count
    row["lexical_density"] = content_count / total_words
    row["lexical_ttr"] = row["lexical_unique_words"] / total_words

    if total_words >= MIN_WORDS_LD:
        mattr_small = helpers["compute_mattr"](tokens, MATTR_SMALL)
        mattr_large = helpers["compute_mattr"](tokens, min(MATTR_LARGE, total_words))
        row["lexical_mattr_small"] = mattr_small
        row["lexical_mattr_large"] = mattr_large
        row["lexical_mattr_ratio"] = mattr_small / mattr_large if mattr_small and mattr_large else np.nan
        row["lexical_mtld"] = helpers["compute_mtld"](tokens, MTLD_TTR_THRESHOLD)
    else:
        row["lexical_mattr_small"] = np.nan
        row["lexical_mattr_large"] = np.nan
        row["lexical_mattr_ratio"] = np.nan
        row["lexical_mtld"] = np.nan
        row["lexical_status_reason"] = "SHORT_TEXT"

    freq = Counter(tokens)
    row["lexical_hapax_ratio"] = sum(1 for count in freq.values() if count == 1) / total_words
    row["lexical_entropy"] = helpers["compute_entropy"](freq)

    sentences = [sent.strip().lower() for sent in re.split(r"[.!?]+", text) if sent.strip()]
    backchannels = 0
    discourse = 0
    for sent in sentences:
        words = sent.split()
        if len(words) <= 2:
            if any(word in {"yeah", "okay", "ok", "uh", "um"} for word in words):
                backchannels += 1
        elif any(phrase in sent for phrase in ["you know", "i mean", "kind of", "sort of"]):
            discourse += 1

    row["lexical_backchannel_ratio"] = backchannels / len(sentences) if sentences else 0
    row["lexical_discourse_marker_ratio"] = discourse / len(sentences) if sentences else 0
    return row


def transcript_text(row: pd.Series, *, questions: bool = False) -> str:
    if questions:
        text = row.get("transcript_question", "")
        return text if isinstance(text, str) else ""
    answer_text = row.get("transcript_answer", "")
    if isinstance(answer_text, str) and answer_text.strip():
        return answer_text.strip()
    asr_text = row.get("ASR_transcript", "")
    if isinstance(asr_text, str) and asr_text.strip():
        return asr_text.strip()
    text = row.get("transcript", "")
    return text if isinstance(text, str) else ""



def transcript_text_from_json(json_path: Path) -> str:
    with json_path.open(encoding="utf-8") as infile:
        data = json.load(infile)

    texts = []
    for segment in data.get("metadata:transcript", []):
        transcript = segment.get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            texts.append(transcript.strip())
            continue

        words = [
            str(word.get("word", "")).strip()
            for word in segment.get("words", [])
            if str(word.get("word", "")).strip()
        ]
        if words:
            texts.append(" ".join(words))

    return " ".join(texts).strip()


def feature_text(row: pd.Series, *, questions: bool = False) -> str:
    json_path = preprocessed_json_path(row, Path.cwd(), questions=questions)
    if json_path is not None:
        text = transcript_text_from_json(json_path)
        if text:
            return text
    return transcript_text(row, questions=questions)



def extract_words_from_json(json_path: Path) -> List[Dict[str, float]]:
    with json_path.open(encoding="utf-8") as infile:
        data = json.load(infile)

    words: List[Dict[str, float]] = []
    for segment in data.get("metadata:transcript", []):
        for word in segment.get("words", []):
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                continue
            if end_f <= start_f:
                continue
            words.append({"start": start_f, "end": end_f})

    words.sort(key=lambda item: item["start"])
    return words


def extract_vad_from_json(json_path: Path) -> List[Tuple[float, float]]:
    with json_path.open(encoding="utf-8") as infile:
        data = json.load(infile)

    segments: List[Tuple[float, float]] = []
    for segment in data.get("metadata:vad", []):
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        segments.append((start_f, end_f))

    segments.sort(key=lambda item: item[0])
    return segments


def preprocessed_json_path(row: pd.Series, base_dir: Path, *, questions: bool = False) -> Optional[Path]:
    audio_column = "audio_path" if questions else "answer_audio_path"
    audio_value = row.get(audio_column)
    if not isinstance(audio_value, str) or not audio_value.strip():
        return None

    audio_path = resolve_path(audio_value, base_dir)
    search_dirs = []
    if audio_path.parent:
        search_dirs.append(audio_path.parent / "baseline_prepreprocess")
    metadata_output_dir = audio_path.parent.parent if audio_path.parent.name in {"audio", "audios"} else audio_path.parent
    search_dirs.append(metadata_output_dir / "baseline_prepreprocess")

    for directory in search_dirs:
        candidate = directory / f"{audio_path.stem}.json"
        if candidate.exists():
            return candidate
    return None


def estimate_timed_words(text: str, start: float, end: float) -> List[Dict[str, float]]:
    tokens = iter_text_tokens(text)
    if not tokens or end <= start:
        return []
    duration = end - start
    step = duration / len(tokens)
    word_duration = step * 0.8
    return [
        {"start": start + idx * step, "end": min(end, start + idx * step + word_duration)}
        for idx in range(len(tokens))
    ]


def build_words_and_vad(row: pd.Series, *, questions: bool = False) -> Tuple[List[Dict[str, float]], List[Tuple[float, float]], str]:
    json_path = preprocessed_json_path(row, Path.cwd(), questions=questions)
    if json_path is not None:
        words = extract_words_from_json(json_path)
        vad = extract_vad_from_json(json_path)
        return words, vad, "baseline_prepreprocess_json"

    duration_column = "question_end_time" if questions else "answer_duration"
    duration = row.get(duration_column, row.get("total_duration", 0.0))
    try:
        duration_f = float(duration)
    except (TypeError, ValueError):
        duration_f = 0.0

    text = transcript_text(row, questions=questions)
    start = 0.0
    end = max(duration_f, 0.0)
    words = estimate_timed_words(text, start, end)
    vad = [(start, end)] if words and end > start else []
    return words, vad, "estimated_from_metadata"


def compute_temporal_features(row: pd.Series, *, questions: bool = False) -> Dict[str, Any]:
    words, vad, source = build_words_and_vad(row, questions=questions)
    out: Dict[str, Any] = {"temporal_alignment_source": source}
    if len(words) < 2:
        out["temporal_status"] = "TOO_FEW_WORDS"
        return out
    if not vad:
        out["temporal_status"] = "NO_VAD"
        return out

    merged = helpers["merge_vad_segments"](vad, MERGE_GAP_THRESHOLD_S)
    valid_stretches = helpers["filter_min_duration"](merged, MIN_STRETCH_DURATION_S)
    if not valid_stretches:
        valid_stretches = merged

    metrics = helpers["compute_metrics_for_file"](words, valid_stretches, PAUSE_THRESHOLD_S)
    if metrics is None:
        out["temporal_status"] = "INSUFFICIENT_TIMED_WORDS_IN_STRETCHES"
        return out

    out.update({f"temporal_{key}": value for key, value in metrics.items() if key != "status"})
    out["temporal_status"] = metrics.get("status", "OK")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--relationships-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", action="store_true", help="Extract features from question audio/text instead of answer audio/text.")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    relationships = load_relationships(args.relationships_csv)
    nlp = load_spacy_model()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    base_dir = Path.cwd()

    for _, meta_row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"extracting explainable feats from {args.metadata.parent}"):
        audio_column = "audio_path" if args.questions else "answer_audio_path"
        audio_path = resolve_path(meta_row[audio_column], base_dir)
        orig_id = audio_path.stem
        ids = parse_ids(orig_id)
        relationship, relationship_detail = relationships.get(
            (ids["vendor_id"], ids["session_id"]), ("UNKNOWN", "UNKNOWN")
        )
        out: Dict[str, Any] = {
            **ids,
            "audio_path": str(audio_path),
            "speakers": meta_row.get("speakers", ""),
            "relationship": relationship,
            "relationship_detail": relationship_detail,
            "total_duration": meta_row.get("total_duration", ""),
            "question_end_time": meta_row.get("question_end_time", ""),
            "extraction_status": "OK",
        }

        try:
            out.update(compute_f0_features(audio_path))
        except Exception as exc:
            out["f0_status"] = f"ERROR: {exc}"
            traceback.print_exc()

        try:
            out.update(compute_lexical_features(feature_text(meta_row, questions=args.questions), nlp))
        except Exception as exc:
            out["lexical_status"] = "ERROR"
            out["lexical_status_reason"] = str(exc)
            traceback.print_exc()

        try:
            out.update(compute_temporal_features(meta_row, questions=args.questions))
        except Exception as exc:
            out["temporal_status"] = f"ERROR: {exc}"
            traceback.print_exc()

        rows.append(out)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
