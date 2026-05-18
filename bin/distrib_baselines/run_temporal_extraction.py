#!/usr/bin/env python3
"""
Temporal feature extraction (sharded)

Scientifically-backed speech-rate computation:
- Use speaker VAD ("metadata:vad") to define voiced stretches.
- Merge adjacent VAD segments if gap <= MERGE_GAP_THRESHOLD_S (default 1.0s).
- Keep only *continuous* voiced stretches with duration >= MIN_STRETCH_DURATION_S (default 12.1s).
- Compute BOTH speech rate and articulation rate on the SAME selected material (same stretches):
    * speech_rate: words / total_stretch_time   (includes pauses inside stretches)
    * articulation_rate: words / (total_stretch_time - pauses>=PAUSE_THRESHOLD_S inside stretches)
- Pause statistics are computed from word timing gaps >= PAUSE_THRESHOLD_S, but only within the selected stretches.
- No 15s segmentation output (interaction/file-level only).

Output CSV per shard: temporal_interaction_shard_XXXX.csv
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# -----------------------
# Defaults / constants
# -----------------------
PAUSE_THRESHOLD_S = 0.2          # keep as requested
MERGE_GAP_THRESHOLD_S = 1.0      # requested change (VAD merge gap)
MIN_STRETCH_DURATION_S = 12.1    # Arantes-like stability criterion
OUT_DIR = Path("/home/ahallur1/spear/Seamless_Experiments/Temporal/shard_csvs")
ALL_JSON = Path("/home/ahallur1/spear/Seamless_Experiments/Lexical/all_jsons.txt")

BASE_FIELDS = [
    "orig_id",
    "wav_path",
    "total_duration_s",
    "speech_active_time_s",
    "pause_count",
    "pause_total_duration_s",
    "pause_mean_duration_s",
    "pause_ratio",
    "speech_rate_wps",
    "speech_rate_wpm",
    "articulation_rate_wps",
    "articulation_rate_wpm",
    "status",
]


# -----------------------
# Helpers
# -----------------------
def load_all_jsons(list_path: Path) -> List[str]:
    with open(list_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def extract_words(json_path: Path) -> List[Dict[str, float]]:
    """
    Extract timed words from metadata:transcript.
    Drops words with missing/invalid timestamps or non-positive durations.
    Returns sorted list of dicts: {"start": float, "end": float}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    words: List[Dict[str, float]] = []
    for seg in data.get("metadata:transcript", []):
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")

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

    words.sort(key=lambda x: x["start"])
    return words


def extract_vad_segments(json_path: Path) -> List[Tuple[float, float]]:
    """
    Extract VAD segments from metadata:vad.
    Returns sorted list of (start, end) floats with end > start.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    vad = data.get("metadata:vad", [])
    segments: List[Tuple[float, float]] = []
    for seg in vad:
        s = seg.get("start")
        e = seg.get("end")
        if s is None or e is None:
            continue
        try:
            s_f = float(s)
            e_f = float(e)
        except (TypeError, ValueError):
            continue
        if e_f <= s_f:
            continue
        segments.append((s_f, e_f))

    segments.sort(key=lambda t: t[0])
    return segments


def merge_vad_segments(vad_segments: List[Tuple[float, float]], merge_gap_s: float) -> List[Tuple[float, float]]:
    """
    Merge adjacent VAD segments if the gap between them is <= merge_gap_s.
    """
    if not vad_segments:
        return []

    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = vad_segments[0]

    for s, e in vad_segments[1:]:
        gap = s - cur_e
        if gap <= merge_gap_s:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged


def filter_min_duration(stretches: List[Tuple[float, float]], min_dur_s: float) -> List[Tuple[float, float]]:
    return [(s, e) for (s, e) in stretches if (e - s) >= min_dur_s]


def words_in_stretches(words: List[Dict[str, float]], stretches: List[Tuple[float, float]]) -> List[Dict[str, float]]:
    """
    Keep only words fully contained within any stretch.
    Assumes words sorted by start and stretches sorted by start.
    """
    if not words or not stretches:
        return []

    out: List[Dict[str, float]] = []
    j = 0
    for w in words:
        ws, we = w["start"], w["end"]

        # advance stretch pointer while word starts after stretch ends
        while j < len(stretches) and ws > stretches[j][1]:
            j += 1
        if j >= len(stretches):
            break

        ss, se = stretches[j]
        if ws >= ss and we <= se:
            out.append(w)

    return out


def assign_stretch_index_for_words(
    words: List[Dict[str, float]],
    stretches: List[Tuple[float, float]]
) -> List[int]:
    """
    For each word (assumed already filtered to be inside stretches), return the index
    of the containing stretch. Words must be sorted; stretches must be sorted.
    """
    idxs: List[int] = []
    j = 0
    for w in words:
        ws, we = w["start"], w["end"]
        while j < len(stretches) and ws > stretches[j][1]:
            j += 1
        if j >= len(stretches):
            # should not happen if words are filtered
            idxs.append(-1)
            continue
        ss, se = stretches[j]
        if ws >= ss and we <= se:
            idxs.append(j)
        else:
            # should not happen if words are filtered
            idxs.append(-1)
    return idxs


def compute_pause_stats_within_stretches(
    words: List[Dict[str, float]],
    stretch_ids: List[int],
    pause_threshold_s: float
) -> Tuple[int, float, float]:
    """
    Compute pauses between consecutive words, but only when both words are in the same stretch.
    Pause = next_start - prev_end, counted if >= pause_threshold_s.
    """
    pauses: List[float] = []
    for i in range(1, len(words)):
        if stretch_ids[i] != stretch_ids[i - 1]:
            continue  # boundary between stretches; do not count as a pause
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= pause_threshold_s:
            pauses.append(gap)

    pause_count = len(pauses)
    pause_total = float(np.sum(pauses)) if pauses else 0.0
    pause_mean = float(np.mean(pauses)) if pauses else 0.0
    return pause_count, pause_total, pause_mean


def compute_metrics_for_file(
    words_all: List[Dict[str, float]],
    stretches_valid: List[Tuple[float, float]],
    pause_threshold_s: float
) -> Optional[Dict[str, float]]:
    """
    Compute temporal metrics using only words within valid stretches (>=30s),
    ensuring speech rate and articulation rate are computed on the same material.
    """
    if not words_all or not stretches_valid:
        return None

    words_sel = words_in_stretches(words_all, stretches_valid)
    if len(words_sel) < 2:
        return None

    stretch_ids = assign_stretch_index_for_words(words_sel, stretches_valid)

    total_stretch_time = float(sum(e - s for s, e in stretches_valid))

    pause_count, pause_total, pause_mean = compute_pause_stats_within_stretches(
        words_sel, stretch_ids, pause_threshold_s
    )

    # "speech-active time" for articulation denominator = stretch time excluding pauses (>= threshold) within stretches
    speech_active_time = max(total_stretch_time - pause_total, 0.0)

    total_words = len(words_sel)

    # Speech rate: includes pauses (within stretches) because denominator is total stretch time
    speech_rate_wps = total_words / total_stretch_time if total_stretch_time > 0 else np.nan
    speech_rate_wpm = speech_rate_wps * 60.0

    # Articulation rate: excludes pauses (>= threshold) within the same stretches
    articulation_rate_wps = total_words / speech_active_time if speech_active_time > 0 else np.nan
    articulation_rate_wpm = articulation_rate_wps * 60.0

    return {
        "total_duration_s": total_stretch_time,
        "speech_active_time_s": speech_active_time,
        "pause_count": pause_count,
        "pause_total_duration_s": pause_total,
        "pause_mean_duration_s": pause_mean,
        "pause_ratio": (pause_total / total_stretch_time) if total_stretch_time > 0 else np.nan,
        "speech_rate_wps": speech_rate_wps,
        "speech_rate_wpm": speech_rate_wpm,
        "articulation_rate_wps": articulation_rate_wps,
        "articulation_rate_wpm": articulation_rate_wpm,
        "status": "OK",
    }


# -----------------------
# Main
# -----------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_jsons_txt", type=str, default = str(ALL_JSON))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)

    ap.add_argument("--pause_threshold_s", type=float, default=PAUSE_THRESHOLD_S)
    ap.add_argument("--merge_gap_threshold_s", type=float, default=MERGE_GAP_THRESHOLD_S)
    ap.add_argument("--min_stretch_duration_s", type=float, default=MIN_STRETCH_DURATION_S)

    args = ap.parse_args()

    all_jsons_txt = Path(args.all_jsons_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jsons = load_all_jsons(all_jsons_txt)
    shard_jsons = all_jsons[args.shard_idx :: args.num_shards]
    print(f"Shard {args.shard_idx}/{args.num_shards}: processing {len(shard_jsons)} JSON files")

    out_csv = out_dir / f"temporal_interaction_shard_{args.shard_idx:04d}.csv"

    with open(out_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=BASE_FIELDS)
        writer.writeheader()

        for p in tqdm(shard_jsons):
            json_path = Path(p)
            orig_id = json_path.stem

            try:
                # Extract
                words = extract_words(json_path)
                if len(words) < 2:
                    writer.writerow({
                        "orig_id": orig_id,
                        "wav_path": orig_id,
                        "status": "TOO_FEW_WORDS",
                    })
                    continue

                vad = extract_vad_segments(json_path)
                if not vad:
                    writer.writerow({
                        "orig_id": orig_id,
                        "wav_path": orig_id,
                        "status": "NO_VAD",
                    })
                    continue

                # Build valid stretches
                merged = merge_vad_segments(vad, args.merge_gap_threshold_s)
                valid_stretches = filter_min_duration(merged, args.min_stretch_duration_s)

                if not valid_stretches:
                    writer.writerow({
                        "orig_id": orig_id,
                        "wav_path": orig_id,
                        "status": "NO_VALID_STRETCH",
                    })
                    continue

                metrics = compute_metrics_for_file(
                    words_all=words,
                    stretches_valid=valid_stretches,
                    pause_threshold_s=args.pause_threshold_s,
                )

                if metrics is None:
                    writer.writerow({
                        "orig_id": orig_id,
                        "wav_path": orig_id,
                        "status": "INSUFFICIENT_TIMED_WORDS_IN_STRETCHES",
                    })
                    continue

                metrics.update({
                    "orig_id": orig_id,
                    "wav_path": orig_id,  # consistent with your earlier convention
                })
                writer.writerow(metrics)

            except Exception as e:
                traceback.print_exc()
                writer.writerow({
                    "orig_id": orig_id,
                    "wav_path": str(json_path),
                    "status": f"ERROR: {e}",
                })

    print(f"Shard {args.shard_idx} finished. Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())