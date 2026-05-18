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
import pandas as pd

from whisper_emotion import WhisperWrapper
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



def waveform_to_mono_np(waveform):
    if torch.is_tensor(waveform):
        y = waveform.detach().cpu().numpy()
    else:
        y = np.asarray(waveform)
    if y.ndim == 2:
        y = y.mean(axis=0)
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



def process_audio(audio_path, model, device, args, key, output_dir):
    waveform, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    y = waveform_to_mono_np(waveform)
    
    y, sr = to_16k(y, sr, TARGET_SR)

    dyad_id = key.lower()
    seg_base = key.lower()
    win_n = int(round(args.win_sec * sr))

    seg_root = output_dir / "segments"
    emb_vec_root = output_dir / "embeds_vec"
    emb_seq_root = output_dir / "embeds_seq"
    emb_vec_root.mkdir(parents=True, exist_ok=True)
    if args.save_full_seq:
        emb_seq_root.mkdir(parents=True, exist_ok=True)

    seg_dir = (seg_root / dyad_id) if args.save_segments else None
    if seg_dir is not None:
        seg_dir.mkdir(parents=True, exist_ok=True)

    # csv_path = output_dir / f"{dyad_id}.csv"
    csv_path = output_dir / "metadata.csv"
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()

    pending_audio = []
    pending_meta = []

    def write_row(meta, v, a, d, vad_source, vec_path, vec_shape, seq_path, seq_shape):
        writer.writerow(
            {
                "dyad_id": meta["dyad_id"],
                "pair_stem": meta["pair_stem"],
                "speaker_id": meta["speaker_id"],
                "seg_stem": meta["seg_stem"],
                "chunk_index": meta["chunk_index"],
                "start": f"{meta['start']:.3f}",
                "end": f"{meta['end']:.3f}",
                "true_end": f"{meta['true_end']:.3f}",
                "duration": f"{(meta['end'] - meta['start']):.3f}",
                "true_duration": f"{(meta['true_end'] - meta['start']):.3f}",
                "padded": "1" if meta["padded"] else "0",
                "valence": v,
                "arousal": a,
                "dominance": d,
                "vad_source": vad_source,
                "emb_vec_path": vec_path,
                "emb_vec_shape": vec_shape,
                "emb_seq_path": seq_path,
                "emb_seq_shape": seq_shape,
                "segment_path": meta["segment_path"],
                "win_sec": str(args.win_sec),
                "hop_sec": str(args.hop_sec),
                "pool": args.pool,
            }
        )

    def save_and_write(meta, v, a, d, pooled_vec_i, last_btd_i):
        vec_path, seq_path = embedding_paths(
            emb_vec_root, emb_seq_root, meta["dyad_id"], meta["seg_stem"], args.save_full_seq
        )
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        if seq_path is not None:
            seq_path.parent.mkdir(parents=True, exist_ok=True)

        if not vec_path.exists():
            np.save(vec_path, pooled_vec_i)
        if seq_path is not None and not seq_path.exists():
            np.save(seq_path, last_btd_i)

        write_row(
            meta=meta,
            v=f"{v:.6f}",
            a=f"{a:.6f}",
            d=f"{d:.6f}",
            vad_source="computed",
            vec_path=str(vec_path),
            vec_shape=str(tuple(pooled_vec_i.shape)),
            seq_path=str(seq_path) if seq_path is not None else "",
            seq_shape=str(tuple(last_btd_i.shape)) if seq_path is not None else "",
        )

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
        for i, meta in enumerate(pending_meta):
            save_and_write(
                meta,
                float(val[i]),
                float(aro[i]),
                float(dom[i]),
                pooled_vec[i],
                last_btd[i],
            )
        pending_audio.clear()
        pending_meta.clear()

    spans = chunk_sliding_pad(y, sr, args.win_sec, args.hop_sec, pad_last=args.pad_last)
    # print(
    #     f"Audio chunked into {len(spans)} segments with window size "
    #     f"{args.win_sec} sec and hop size {args.hop_sec} sec."
    # )

    for ci, (i0, i1_true, t0, t1_true, padded) in enumerate(spans):
        ch = y[i0:i1_true]
        if padded:
            ch = pad_to_len(ch, win_n)
            t1 = t0 + args.win_sec
        else:
            t1 = t1_true

        seg_stem = f"{seg_base}__ch{ci:05d}"
        vec_path, seq_path = embedding_paths(
            emb_vec_root, emb_seq_root, dyad_id, seg_stem, args.save_full_seq
        )

        segment_path = ""
        if seg_dir is not None:
            seg_file = seg_dir / f"{seg_stem}.{args.seg_format}"
            segment_path = str(seg_file)
            if not seg_file.exists():
                sf.write(segment_path, ch, sr)

        meta = {
            "dyad_id": dyad_id,
            "pair_stem": seg_base,
            "speaker_id": key,
            "seg_stem": seg_stem,
            "chunk_index": ci,
            "start": float(t0),
            "end": float(t1),
            "true_end": float(t1_true),
            "padded": bool(padded),
            "segment_path": segment_path,
        }

        if embeddings_exist(vec_path, seq_path):
            try:
                vshape = str(npy_shape_header_only(vec_path))
            except Exception:
                vshape = ""
            try:
                sshape = str(npy_shape_header_only(seq_path)) if seq_path is not None else ""
            except Exception:
                sshape = ""
            write_row(meta, "", "", "", "missing", str(vec_path), vshape, str(seq_path) if seq_path else "", sshape)
            continue

        pending_audio.append(ch.astype(np.float32, copy=False))
        pending_meta.append(meta)
        if len(pending_audio) >= args.batch_size:
            flush_compute()

    flush_compute()
    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, help="metadata file")

    parser.add_argument("--win-sec", type=float, default=3.0, help="Window size seconds.")
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

    args = parser.parse_args()

    output_dir = args.metadata.parent / 'naturalness/voxprofile_features'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion-dim").to(device)
    model.eval()
    print("Model Loaded. Starting feature extraction...")

    metadata = pd.read_csv(args.metadata)

    for idx, row in tqdm(metadata.iterrows(), desc=f"Processing file {args.metadata.stem}", total=len(metadata)):
        process_audio(row['audio_path'], model, device, args, Path(row['audio_path']).stem, output_dir)

