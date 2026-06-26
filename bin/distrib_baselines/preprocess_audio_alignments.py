#!/usr/bin/env python3
"""Create Seamless-style word/VAD JSON files for SPEARBench answer audio."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import site
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


TARGET_SAMPLE_RATE = 16000


def disable_user_site_packages() -> None:
    """Avoid importing incompatible packages from ~/.local into the ASR env."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        user_site = Path(site.getusersitepackages()).resolve()
    except Exception:
        user_site = None
    try:
        user_base = Path(site.getuserbase()).resolve()
    except Exception:
        user_base = None

    cleaned = []
    for raw_path in sys.path:
        if not raw_path:
            continue
        try:
            path = Path(raw_path).resolve()
        except Exception:
            cleaned.append(raw_path)
            continue
        if user_site and path == user_site:
            continue
        if user_base and (path == user_base or user_base in path.parents):
            continue
        cleaned.append(raw_path)
    sys.path[:] = cleaned


disable_user_site_packages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe and word-align audio paths from a SPEARBench metadata CSV, "
            "writing Seamless-style JSONs into baseline_prepreprocess/."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-column", default="answer_audio_path")
    parser.add_argument("--model", default="large-v2", help="WhisperX/Whisper model name.")
    parser.add_argument("--language", default="en", help="Language code, or empty string to auto-detect.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="WhisperX compute type. Use auto, int8, int8_float16, float16, or float32.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--align-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for WhisperX word alignment. CPU is usually safer on mixed GPU nodes.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-shard smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--failures-jsonl", type=Path)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(compute_type_arg: str, device: str) -> str:
    if compute_type_arg != "auto":
        return compute_type_arg
    return "int8"


def torch_cuda_is_usable() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        _ = torch.zeros(1, device="cuda") + 1
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        logging.warning("PyTorch CUDA is not usable; using CPU for torch stages: %s", exc)
        return False


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def load_metadata_audio_paths(metadata: Path, audio_column: str, base_dir: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    with metadata.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if audio_column not in (reader.fieldnames or []):
            raise ValueError(f"{metadata} does not contain column {audio_column!r}")
        for idx, row in enumerate(reader):
            value = (row.get(audio_column) or "").strip()
            if not value:
                continue
            rows.append((idx, resolve_path(value, base_dir)))
    return rows


def select_shard(items: list[tuple[int, Path]], shard_index: int, num_shards: int) -> list[tuple[int, Path]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def output_json_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}.json"


def load_silero_vad() -> tuple[Any, Any, Any]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

        model = load_silero_vad()
        return model, get_speech_timestamps, read_audio
    except Exception as package_exc:
        logging.info("Falling back to torch.hub Silero VAD load: %s", package_exc)

    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps, _, read_audio, _, _ = utils
    return model.cpu().eval(), get_speech_timestamps, read_audio


def read_wav_mono_16k(wav_path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> Any:
    """Read PCM wav without torchaudio/sox and return a float32 torch tensor."""
    import numpy as np
    import torch

    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported wav sample width {sample_width} bytes: {wav_path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_sample_rate and audio.size:
        try:
            from scipy.signal import resample_poly

            gcd = math.gcd(sample_rate, target_sample_rate)
            audio = resample_poly(audio, target_sample_rate // gcd, sample_rate // gcd).astype(np.float32)
        except Exception:
            duration = audio.size / float(sample_rate)
            old_t = np.linspace(0.0, duration, num=audio.size, endpoint=False)
            new_size = int(round(duration * target_sample_rate))
            new_t = np.linspace(0.0, duration, num=new_size, endpoint=False)
            audio = np.interp(new_t, old_t, audio).astype(np.float32)

    return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))


def vad_segments_for_file(
    wav_path: Path,
    vad_model: Any,
    get_speech_timestamps: Any,
    read_audio: Any,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    del read_audio
    audio = read_wav_mono_16k(wav_path, TARGET_SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        return_seconds=True,
    )
    return [
        {"start": float(seg["start"]), "end": float(seg["end"])}
        for seg in timestamps
        if float(seg["end"]) > float(seg["start"])
    ]


def load_whisperx(model_name: str, device: str, compute_type: str, language: str | None) -> Any:
    import whisperx

    kwargs: dict[str, Any] = {
        "compute_type": compute_type,
        "vad_method": "silero",
    }
    if language:
        kwargs["language"] = language
    return whisperx.load_model(model_name, device=device, **kwargs)


def transcript_segments_for_file(
    wav_path: Path,
    asr_model: Any,
    align_cache: dict[str, tuple[Any, Any]],
    device: str,
    align_device: str,
    batch_size: int,
    language: str | None,
) -> list[dict[str, Any]]:
    import whisperx

    audio = whisperx.load_audio(str(wav_path))
    result = asr_model.transcribe(audio, batch_size=batch_size, language=language)
    language_code = language or result.get("language") or "en"

    if language_code not in align_cache:
        align_cache[language_code] = whisperx.load_align_model(
            language_code=language_code,
            device=align_device,
        )
    align_model, align_metadata = align_cache[language_code]
    aligned = whisperx.align(
        result.get("segments", []),
        align_model,
        align_metadata,
        audio,
        align_device,
        return_char_alignments=False,
    )

    output: list[dict[str, Any]] = []
    for segment in aligned.get("segments", []):
        words = []
        for word in segment.get("words", []):
            word_text = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")
            if not word_text or start is None or end is None:
                continue
            item: dict[str, Any] = {
                "word": word_text,
                "start": float(start),
                "end": float(end),
            }
            if word.get("score") is not None:
                item["score"] = float(word["score"])
            words.append(item)

        transcript = str(segment.get("text", "")).strip()
        if not transcript:
            transcript = " ".join(word["word"] for word in words).strip()
        if not transcript and not words:
            continue

        if words:
            start = words[0]["start"]
            end = words[-1]["end"]
        else:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))

        output.append(
            {
                "words": words,
                "start": float(start),
                "end": float(end),
                "transcript": transcript,
            }
        )
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(json_ready(data), handle, indent=4, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, path)


def append_failure(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")


def transcribe_one(
    wav_path: Path,
    args: argparse.Namespace,
    vad_model: Any,
    get_speech_timestamps: Any,
    read_audio: Any,
    asr_model: Any,
    align_cache: dict[str, tuple[Any, Any]],
    device: str,
) -> dict[str, Any]:
    vad = vad_segments_for_file(
        wav_path,
        vad_model,
        get_speech_timestamps,
        read_audio,
        args,
    )
    transcript = []
    if vad:
        transcript = transcript_segments_for_file(
            wav_path,
            asr_model,
            align_cache,
            device,
            args.align_device,
            args.batch_size,
            args.language or None,
        )

    return {
        "id": wav_path.stem,
        "audio_path": str(wav_path),
        "metadata:transcript": transcript,
        "metadata:vad": vad,
    }


def write_manifest(output_dir: Path, records: list[dict[str, Any]]) -> None:
    path = output_dir / "manifest.csv"
    fields = ["row_index", "audio_path", "json_path", "status"]
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    configure_logging()
    args = parse_args()
    base_dir = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures_jsonl = args.failures_jsonl or args.output_dir / "transcription_failures.jsonl"

    all_items = load_metadata_audio_paths(args.metadata, args.audio_column, base_dir)
    selected = select_shard(all_items, args.shard_index, args.num_shards)
    if args.limit > 0:
        selected = selected[: args.limit]

    pending = [
        (row_idx, wav_path)
        for row_idx, wav_path in selected
        if args.overwrite or not output_json_path(wav_path, args.output_dir).exists()
    ]

    logging.info("Metadata: %s", args.metadata)
    logging.info("Output dir: %s", args.output_dir)
    logging.info(
        "Found %d audio rows; shard %d/%d selected %d; pending %d",
        len(all_items),
        args.shard_index,
        args.num_shards,
        len(selected),
        len(pending),
    )
    if args.dry_run:
        for _, wav_path in pending:
            logging.info("DRY-RUN %s -> %s", wav_path, output_json_path(wav_path, args.output_dir))
        return 0
    if not pending:
        write_manifest(
            args.output_dir,
            [
                {
                    "row_index": row_idx,
                    "audio_path": str(wav_path),
                    "json_path": str(output_json_path(wav_path, args.output_dir)),
                    "status": "exists",
                }
                for row_idx, wav_path in selected
            ],
        )
        logging.info("Nothing to do.")
        return 0

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
    if args.align_device == "cuda" and not torch_cuda_is_usable():
        args.align_device = "cpu"

    logging.info("Loading Silero VAD")
    vad_model, get_speech_timestamps, read_audio = load_silero_vad()
    logging.info(
        "Loading WhisperX model=%s device=%s compute_type=%s align_device=%s",
        args.model,
        device,
        compute_type,
        args.align_device,
    )
    try:
        asr_model = load_whisperx(args.model, device, compute_type, args.language or None)
    except Exception as exc:
        if device == "cuda" and not args.no_cpu_fallback:
            logging.warning("CUDA WhisperX load failed; retrying ASR on CPU with int8: %s", exc)
            device = "cpu"
            compute_type = "int8"
            asr_model = load_whisperx(args.model, device, compute_type, args.language or None)
        else:
            raise

    align_cache: dict[str, tuple[Any, Any]] = {}
    manifest_records: list[dict[str, Any]] = []
    failures = 0
    started = time.time()

    progress = tqdm(pending, total=len(pending), unit="file", desc="Preprocessing audio alignments")
    for row_idx, wav_path in progress:
        output_path = output_json_path(wav_path, args.output_dir)
        progress.set_postfix_str(wav_path.name, refresh=False)
        try:
            if not wav_path.exists():
                raise FileNotFoundError(wav_path)
            data = transcribe_one(
                wav_path,
                args,
                vad_model,
                get_speech_timestamps,
                read_audio,
                asr_model,
                align_cache,
                device,
            )
            atomic_write_json(output_path, data)
            manifest_records.append(
                {
                    "row_index": row_idx,
                    "audio_path": str(wav_path),
                    "json_path": str(output_path),
                    "status": "ok",
                }
            )
        except Exception as exc:
            failures += 1
            logging.exception("Failed %s", wav_path)
            manifest_records.append(
                {
                    "row_index": row_idx,
                    "audio_path": str(wav_path),
                    "json_path": str(output_path),
                    "status": f"error: {exc}",
                }
            )
            append_failure(
                failures_jsonl,
                {
                    "row_index": row_idx,
                    "wav": str(wav_path),
                    "json": str(output_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    write_manifest(args.output_dir, manifest_records)
    elapsed = time.time() - started
    logging.info("Finished pending=%d failures=%d elapsed_sec=%.1f", len(pending), failures, elapsed)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
