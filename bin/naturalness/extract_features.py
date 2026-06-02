import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import os
import shutil
import pandas as pd

from whisper_emotion import WhisperWrapper

TURN_SCORE_COLUMNS = ["turn_valence", "turn_arousal", "turn_dominance"]
try:
    from .extract import (
        CSV_FIELDS,
        TARGET_SR,
        chunk_sliding_pad,
        embedding_paths,
        embeddings_exist,
        forward_fast,
        npy_shape_header_only,
        pad_to_len,
        to_16k,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from extract import (
        CSV_FIELDS,
        TARGET_SR,
        chunk_sliding_pad,
        embedding_paths,
        embeddings_exist,
        forward_fast,
        npy_shape_header_only,
        pad_to_len,
        to_16k,
    )


FEATURE_CSV_FIELDS = list(CSV_FIELDS) + [column for column in TURN_SCORE_COLUMNS if column not in CSV_FIELDS]


def waveform_to_mono_np(waveform):
    if torch.is_tensor(waveform):
        y = waveform.detach().cpu().numpy()
    else:
        y = np.asarray(waveform)
    if y.ndim == 2:
        # soundfile returns (frames, channels); torchaudio-style tensors are
        # often (channels, frames). Support both layouts.
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            y = y.mean(axis=0)
        else:
            y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)


def expected_num_chunks(audio_info, args):
    try:
        info = sf.info(audio_info["audio1"])
        sr = int(info.samplerate)
        n_samples = int(float(audio_info["end"]) * sr) - int(float(audio_info["start"]) * sr)
        n_samples = max(0, (n_samples * TARGET_SR + sr - 1) // sr)
    except Exception:
        return 0

    win = int(round(args.win_sec * TARGET_SR))
    hop = int(round(args.hop_sec * TARGET_SR))
    if win <= 0 or hop <= 0 or n_samples <= 0:
        return 0
    if args.pad_last:
        return (n_samples + hop - 1) // hop
    if n_samples < win:
        return 0
    return 1 + (n_samples - win) // hop



def load_audio(audio_path, start_time=0.0, end_time=None):
    waveform, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    y = waveform_to_mono_np(waveform)
    start_sample = max(0, int(round(float(start_time) * sr)))
    end_sample = len(y)
    if end_time is not None:
        end_sample = min(end_sample, max(start_sample, int(round(float(end_time) * sr))))
    y = y[start_sample:end_sample]
    return to_16k(y, sr, TARGET_SR)


def chunk_audio(y, sr, args, turn_name, start_offset=0.0):
    win_n = int(round(args.win_sec * sr))
    chunks = []

    spans = chunk_sliding_pad(y, sr, args.win_sec, args.hop_sec, pad_last=args.pad_last)
    for ci, (i0, i1_true, t0, t1_true, padded) in enumerate(spans):
        ch = y[i0:i1_true]
        if padded:
            ch = pad_to_len(ch, win_n)
            t1 = t0 + args.win_sec
        else:
            t1 = t1_true

        chunks.append(
            {
                "audio": ch.astype(np.float32, copy=False),
                "turn_name": turn_name,
                "turn_chunk_index": ci,
                "start": float(start_offset + t0),
                "end": float(start_offset + t1),
                "true_end": float(start_offset + t1_true),
                "padded": bool(padded),
            }
        )

    return chunks, len(y) / sr


def compute_chunks(chunks, model, device, args):
    computed = []
    pending_audio = []
    pending_chunks = []

    def flush_compute():
        if not pending_audio:
            return
        val, aro, dom, pooled_vec, last_btd = forward_fast(
            model=model,
            device=device,
            batch_audio_np=pending_audio,
            pool=args.pool,
            amp=args.amp,
        )
        for i, chunk in enumerate(pending_chunks):
            computed.append(
                {
                    **chunk,
                    "valence": float(val[i]),
                    "arousal": float(aro[i]),
                    "dominance": float(dom[i]),
                    "pooled_vec": pooled_vec[i],
                    "last_btd": last_btd[i],
                }
            )
        pending_audio.clear()
        pending_chunks.clear()

    for chunk in chunks:
        pending_audio.append(chunk["audio"])
        pending_chunks.append(chunk)
        if len(pending_audio) >= args.batch_size:
            flush_compute()

    flush_compute()
    return computed


def compute_turn_scores(audio, model, device, args):
    if len(audio) == 0:
        return {"turn_valence": "", "turn_arousal": "", "turn_dominance": ""}
    val, aro, dom, _pooled_vec, _last_btd = forward_fast(
        model=model,
        device=device,
        batch_audio_np=[audio.astype(np.float32, copy=False)],
        pool=args.pool,
        amp=args.amp,
    )
    return {
        "turn_valence": f"{float(val[0]):.6f}",
        "turn_arousal": f"{float(aro[0]):.6f}",
        "turn_dominance": f"{float(dom[0]):.6f}",
    }


def feature_metadata_is_old_version(output_dir):
    csv_path = output_dir / "metadata.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    try:
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, [])
    except Exception:
        return False
    return not set(TURN_SCORE_COLUMNS).issubset(header)


def warn_old_feature_metadata(output_dir):
    csv_path = output_dir / "metadata.csv"
    raise RuntimeError(
        f"Old VoxProfile feature metadata detected at {csv_path}; it does not contain "
        f"the full-turn columns {', '.join(TURN_SCORE_COLUMNS)}. Remove the existing "
        "naturalness/voxprofile_features directory and rerun 20_naturalness_feats.sh --extract "
        "to recompute a clean new-version feature CSV."
    )


def feature_metadata_has_turn_scores(row, output_dir):
    csv_path = output_dir / "metadata.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    key = Path(row["audio_path"]).stem.lower()
    required_sources = {"question", "answer"}
    seen_sources = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or not set(TURN_SCORE_COLUMNS).issubset(reader.fieldnames):
                return False
            for metadata_row in reader:
                if str(metadata_row.get("pair_stem", "")).strip().lower() != key:
                    continue
                source = str(metadata_row.get("vad_source", "")).strip().lower()
                if source.startswith("existing_"):
                    source = source.removeprefix("existing_")
                if source not in required_sources:
                    continue
                if all(str(metadata_row.get(column, "")).strip() for column in TURN_SCORE_COLUMNS):
                    seen_sources.add(source)
                if seen_sources == required_sources:
                    return True
    except Exception:
        return False
    return False


def num_chunks_for_duration(duration, args):
    n_samples = int(round(max(0.0, float(duration)) * TARGET_SR))
    win = int(round(args.win_sec * TARGET_SR))
    hop = int(round(args.hop_sec * TARGET_SR))
    if win <= 0 or hop <= 0 or n_samples <= 0:
        return 0
    if args.pad_last:
        return (n_samples + hop - 1) // hop
    if n_samples < win:
        return 0
    return 1 + (n_samples - win) // hop


def audio_duration(path):
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def combined_embeddings_exist(row, args, output_dir):
    question_path = Path(row["audio_path"])
    answer_path = Path(row["answer_audio_path"])
    key = question_path.stem.lower()

    question_duration = max(0.0, float(row["question_end_time"]) - float(row["context_end_time"]))
    answer_duration = audio_duration(answer_path)
    expected_chunks = (
        num_chunks_for_duration(question_duration, args)
        + num_chunks_for_duration(answer_duration, args)
    )
    if expected_chunks <= 0:
        return False

    emb_vec_root = output_dir / "embeds_vec"
    emb_seq_root = output_dir / "embeds_seq"
    for ci in range(expected_chunks):
        seg_stem = f"{key}__ch{ci:05d}"
        vec_path, seq_path = embedding_paths(
            emb_vec_root, emb_seq_root, key, seg_stem, args.save_full_seq
        )
        if not embeddings_exist(vec_path, seq_path):
            return False
    return True


def write_embedding_row(writer, chunk, key, ci, vec_path, seq_path, segment_path, args, turn_scores):
    writer.writerow(
        {
            "dyad_id": key.lower(),
            "pair_stem": key.lower(),
            "speaker_id": key,
            "seg_stem": f"{key.lower()}__ch{ci:05d}",
            "chunk_index": ci,
            "start": f"{chunk['start']:.3f}",
            "end": f"{chunk['end']:.3f}",
            "true_end": f"{chunk['true_end']:.3f}",
            "duration": f"{(chunk['end'] - chunk['start']):.3f}",
            "true_duration": f"{(chunk['true_end'] - chunk['start']):.3f}",
            "padded": "1" if chunk["padded"] else "0",
            "valence": f"{chunk['valence']:.6f}",
            "arousal": f"{chunk['arousal']:.6f}",
            "dominance": f"{chunk['dominance']:.6f}",
            "turn_valence": turn_scores[chunk["turn_name"]].get("turn_valence", ""),
            "turn_arousal": turn_scores[chunk["turn_name"]].get("turn_arousal", ""),
            "turn_dominance": turn_scores[chunk["turn_name"]].get("turn_dominance", ""),
            "vad_source": chunk["turn_name"],
            "emb_vec_path": str(vec_path),
            "emb_vec_shape": str(tuple(chunk["pooled_vec"].shape)),
            "emb_seq_path": str(seq_path) if seq_path is not None else "",
            "emb_seq_shape": str(tuple(chunk["last_btd"].shape)) if seq_path is not None else "",
            "segment_path": segment_path,
            "win_sec": str(args.win_sec),
            "hop_sec": str(args.hop_sec),
            "pool": args.pool,
        }
    )


def save_computed_chunks(computed_chunks, sr, args, key, output_dir, turn_scores):
    dyad_id = key.lower()
    seg_base = key.lower()

    seg_root = output_dir / "segments"
    emb_vec_root = output_dir / "embeds_vec"
    emb_seq_root = output_dir / "embeds_seq"
    emb_vec_root.mkdir(parents=True, exist_ok=True)
    if args.save_full_seq:
        emb_seq_root.mkdir(parents=True, exist_ok=True)

    seg_dir = (seg_root / dyad_id) if args.save_segments else None
    if seg_dir is not None:
        seg_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metadata.csv"
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FEATURE_CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()

        for ci, chunk in enumerate(computed_chunks):
            seg_stem = f"{seg_base}__ch{ci:05d}"
            vec_path, seq_path = embedding_paths(
                emb_vec_root, emb_seq_root, dyad_id, seg_stem, args.save_full_seq
            )
            vec_path.parent.mkdir(parents=True, exist_ok=True)
            if seq_path is not None:
                seq_path.parent.mkdir(parents=True, exist_ok=True)

            segment_path = ""
            if seg_dir is not None:
                seg_file = seg_dir / f"{seg_stem}.{args.seg_format}"
                segment_path = str(seg_file)
                if not seg_file.exists():
                    sf.write(segment_path, chunk["audio"], sr)

            if embeddings_exist(vec_path, seq_path):
                try:
                    vshape = str(npy_shape_header_only(vec_path))
                except Exception:
                    vshape = ""
                try:
                    sshape = str(npy_shape_header_only(seq_path)) if seq_path is not None else ""
                except Exception:
                    sshape = ""
                writer.writerow(
                    {
                        "dyad_id": dyad_id,
                        "pair_stem": seg_base,
                        "speaker_id": key,
                        "seg_stem": seg_stem,
                        "chunk_index": ci,
                        "start": f"{chunk['start']:.3f}",
                        "end": f"{chunk['end']:.3f}",
                        "true_end": f"{chunk['true_end']:.3f}",
                        "duration": f"{(chunk['end'] - chunk['start']):.3f}",
                        "true_duration": f"{(chunk['true_end'] - chunk['start']):.3f}",
                        "padded": "1" if chunk["padded"] else "0",
                        "valence": "",
                        "arousal": "",
                        "dominance": "",
                        "turn_valence": turn_scores[chunk["turn_name"]].get("turn_valence", ""),
                        "turn_arousal": turn_scores[chunk["turn_name"]].get("turn_arousal", ""),
                        "turn_dominance": turn_scores[chunk["turn_name"]].get("turn_dominance", ""),
                        "vad_source": f"existing_{chunk['turn_name']}",
                        "emb_vec_path": str(vec_path),
                        "emb_vec_shape": vshape,
                        "emb_seq_path": str(seq_path) if seq_path is not None else "",
                        "emb_seq_shape": sshape,
                        "segment_path": segment_path,
                        "win_sec": str(args.win_sec),
                        "hop_sec": str(args.hop_sec),
                        "pool": args.pool,
                    }
                )
                continue

            np.save(vec_path, chunk["pooled_vec"])
            if seq_path is not None:
                np.save(seq_path, chunk["last_btd"])
            write_embedding_row(writer, chunk, key, ci, vec_path, seq_path, segment_path, args, turn_scores)


def process_audio(audio_path, model, device, args, key, output_dir, start_time=0.0):
    y, sr = load_audio(audio_path, start_time=start_time)
    chunks, _duration = chunk_audio(y, sr, args, "audio", start_offset=0.0)
    computed = compute_chunks(chunks, model, device, args)
    turn_scores = {"audio": compute_turn_scores(y, model, device, args)}
    save_computed_chunks(computed, sr, args, key, output_dir, turn_scores)


def process_question_answer(row, model, device, args, output_dir):
    question_path = Path(row["audio_path"])
    answer_path = Path(row["answer_audio_path"])
    base_key = question_path.stem

    question_audio, question_sr = load_audio(
        question_path,
        start_time=row["context_end_time"],
        end_time=row["question_end_time"],
    )
    answer_audio, answer_sr = load_audio(answer_path, start_time=0.0)
    if question_sr != answer_sr:
        raise ValueError(f"Question and answer sample rates differ after resampling: {question_sr} != {answer_sr}")

    question_chunks, question_duration = chunk_audio(
        question_audio, question_sr, args, "question", start_offset=0.0
    )
    answer_chunks, _answer_duration = chunk_audio(
        answer_audio, answer_sr, args, "answer", start_offset=question_duration
    )

    turn_scores = {
        "question": compute_turn_scores(question_audio, model, device, args),
        "answer": compute_turn_scores(answer_audio, model, device, args),
    }
    question_embeddings = compute_chunks(question_chunks, model, device, args)
    answer_embeddings = compute_chunks(answer_chunks, model, device, args)
    save_computed_chunks(question_embeddings + answer_embeddings, question_sr, args, base_key, output_dir, turn_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, help="metadata file")

    parser.add_argument("--win-sec", type=float, default=3.0, help="Window size seconds.")
    parser.add_argument("--min-len-question", type=float, default=0.5, help="Minimum question duration in seconds to process (default 0.5s).")
    parser.add_argument("--hop-sec", type=float, default=1.0, help="Hop size seconds (hop < win => sliding window).")
    parser.add_argument("--pad-last", action="store_true", help="Pad last short window to full length (default ON).")
    parser.add_argument("--no-pad-last", dest="pad_last", action="store_false", help="Drop trailing short window.")
    parser.set_defaults(pad_last=True)

    parser.add_argument("--save-segments", action="store_true", help="Save 16k mono segments to out-dir/segments/<dyad>/")
    parser.add_argument("--seg-format", type=str, default="wav", choices=["wav", "flac"])

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument("--pool", choices=["mean", "max"], default="mean", help="Pool encoder [T,D] -> [D].")
    parser.add_argument("--save-full-seq", action="store_true", help="Also save per-window full encoder [T,D].")

    parser.add_argument("--amp", action="store_true", help="Use autocast fp16 on CUDA for faster encoder.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--num-gpus", type=int, default=0, help="0 = all visible CUDA devices.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Remove existing naturalness/voxprofile_features for this metadata directory before extraction.",
    )

    args = parser.parse_args()

    output_dir = args.metadata.parent / 'naturalness/voxprofile_features'
    if args.force_recompute and output_dir.exists():
        print(f"Force recompute requested; removing existing feature directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if feature_metadata_is_old_version(output_dir):
        warn_old_feature_metadata(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion-dim").to(device)
    model.eval()
    print("Model Loaded. Starting feature extraction...")

    metadata = pd.read_csv(args.metadata)
    too_short = 0
    already_computed = 0
    for idx, row in tqdm(metadata.iterrows(), desc=f"Extracting emotions", total=len(metadata)):
        if float(row['question_end_time'] - row["context_end_time"]) < args.min_len_question:
            too_short += 1
            continue
        if combined_embeddings_exist(row, args, output_dir) and feature_metadata_has_turn_scores(row, output_dir):
            already_computed += 1
            continue
        process_question_answer(row, model, device, args, output_dir)
    print(
        f"Finished processing. {already_computed}/{len(metadata)} pairs skipped because embeddings already exist; "
        f"{too_short}/{len(metadata)} pairs skipped due to short question length (< {args.min_len_question} seconds)."
    )
