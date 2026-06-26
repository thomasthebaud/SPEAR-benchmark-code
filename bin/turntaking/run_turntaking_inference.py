#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    class _NoOpTqdm:
        def __init__(self, iterable=None, **_: Any) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, _: int = 1) -> None:
            pass

        def set_postfix(self, **_: Any) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(iterable=None, **kwargs):
        return _NoOpTqdm(iterable, **kwargs)

SCRIPT_DIR = Path(__file__).resolve().parent
DUALTURN_SCRIPTS = SCRIPT_DIR / "dualturn" / "scripts"
for path in (SCRIPT_DIR, DUALTURN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score_fvad_checkpoint import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    build_model,
    load_training_checkpoint,
    resolve_experiment,
    saved,
)
from train_fvad_head import NaturalnessFiveTypeEvaluator, vap_bin_times_to_frames  # noqa: E402


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_completed_output_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    completed: dict[int, dict[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "row_index" not in reader.fieldnames:
                return {}
            for row in reader:
                if str(row.get("status", "")).strip().lower() != "ok":
                    continue
                try:
                    row_index = int(row.get("row_index", ""))
                except (TypeError, ValueError):
                    continue
                completed[row_index] = row
    except Exception as exc:
        tqdm.write(f"Warning: could not read existing turn-taking output {path}: {exc}")
        return {}
    return completed


def resolve_audio_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def finite_or_empty(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_stereo_for_merge(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.float()


def merged_question_answer_audio(
    question_audio_path: Path,
    answer_audio_path: Path,
    answer_start_time: float,
    *,
    sample_rate: int,
) -> torch.Tensor:
    question = load_stereo_for_merge(question_audio_path, sample_rate)
    answer = load_stereo_for_merge(answer_audio_path, sample_rate)
    answer_start_sample = max(0, question.shape[-1] + int(round(answer_start_time * sample_rate)))
    total_samples = max(question.shape[-1], answer_start_sample + answer.shape[-1])
    merged = torch.zeros((2, total_samples), dtype=torch.float32)
    merged[:, : question.shape[-1]] += question
    merged[:, answer_start_sample : answer_start_sample + answer.shape[-1]] += answer
    peak = float(merged.abs().max()) if merged.numel() else 0.0
    if peak > 1.0:
        merged = merged / peak
    return merged


def prepare_merged_segment(
    evaluator: NaturalnessFiveTypeEvaluator,
    item: dict[str, Any],
    question_audio_path: Path,
    answer_audio_path: Path,
    answer_start_time: float,
) -> dict[str, Any]:
    model_audio = merged_question_answer_audio(
        question_audio_path,
        answer_audio_path,
        answer_start_time,
        sample_rate=evaluator.sample_rate,
    )
    vad_audio = merged_question_answer_audio(
        question_audio_path,
        answer_audio_path,
        answer_start_time,
        sample_rate=16_000,
    )
    duration_s = float(model_audio.shape[-1] / evaluator.sample_rate)
    vad_pack = evaluator.base.observed_vad_50hz(
        vad_audio,
        sr=16_000,
        vad_source=evaluator.vad_source,
        silero_models=evaluator.silero_models,
        rms_threshold=evaluator.rms_threshold,
        silero_threshold=evaluator.silero_threshold,
        silero_min_speech_ms=evaluator.silero_min_speech_ms,
        silero_min_silence_ms=evaluator.silero_min_silence_ms,
        clean_min_speech_ms=evaluator.clean_min_speech_ms,
        clean_min_silence_ms=evaluator.clean_min_silence_ms,
    )
    units = evaluator.base.extract_utterance_units(
        vad_pack["clean"],
        frame_hz=50.0,
        duration_s=duration_s,
        unit_pre_s=evaluator.unit_pre_s,
        unit_post_s=evaluator.unit_post_s,
        min_utterance_s=evaluator.min_utterance_s,
        unit_mode=evaluator.unit_mode,
        utterance_merge_gap_s=evaluator.utterance_merge_gap_s,
        utterance_merge_other_max_ratio=evaluator.utterance_merge_other_max_ratio,
    )
    return {**item, "audio": model_audio, "duration_s": duration_s, "vad_clean": vad_pack["clean"], "units": units}


def make_segment_item(row: dict[str, str], audio_path: Path, index: int) -> dict[str, Any]:
    stem = audio_path.stem
    conversation_id = row.get("conversation_id", "")
    return {
        "segment_id": stem or f"row_{index}",
        "condition": "answer",
        "version": "answer",
        "pair_id": conversation_id or stem or f"row_{index}",
        "edit_type": "spearbench_answer",
        "audio_path": audio_path,
    }


def output_row(
    source_row: dict[str, str],
    *,
    row_index: int,
    answer_audio_path: Path | None,
    scored: dict[str, Any] | None,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    audio_path = source_row.get("audio_path", "")
    answer_audio = str(answer_audio_path) if answer_audio_path is not None else source_row.get("answer_audio_path", "")
    base = {
        "row_index": row_index,
        "segment_id": Path(answer_audio).stem if answer_audio else "",
        "conversation_id": source_row.get("conversation_id", ""),
        "question_audio_path": audio_path,
        "answer_audio_path": answer_audio,
        "speakers": source_row.get("speakers", ""),
        "context_end_time": source_row.get("context_end_time", ""),
        "question_end_time": source_row.get("question_end_time", ""),
        "answer_start_time": source_row.get("answer_start_time", ""),
        "answer_duration": source_row.get("answer_duration", ""),
        "status": status,
        "error": error,
    }
    if scored is None:
        base.update({
            "duration_s": "",
            "mean_nll": "",
            "tail_nll": "",
            "dialog_nll": "",
            "nat_score": "",
            "naturalness_score": "",
            "num_units": 0,
            "tail_k": 0,
            "num_nll_frames": 0,
            "unit_nll": "[]",
            "unit_type_counts": "{}",
        })
        return base
    unit_counts: dict[str, int] = {}
    for unit in scored.get("units", []) or []:
        key = str(unit.get("unit_type", "unknown"))
        unit_counts[key] = unit_counts.get(key, 0) + 1
    base.update({
        "duration_s": finite_or_empty(scored.get("duration_s")),
        "mean_nll": finite_or_empty(scored.get("mean_nll")),
        "tail_nll": finite_or_empty(scored.get("tail_nll")),
        "dialog_nll": finite_or_empty(scored.get("dialog_nll")),
        "nat_score": finite_or_empty(scored.get("nat_score")),
        "naturalness_score": finite_or_empty(scored.get("naturalness_score")),
        "num_units": int(scored.get("num_units", 0) or 0),
        "tail_k": int(scored.get("tail_k", 0) or 0),
        "num_nll_frames": int(scored.get("num_nll_frames", 0) or 0),
        "unit_nll": json.dumps(scored.get("unit_nll", []), ensure_ascii=False),
        "unit_type_counts": json.dumps(dict(sorted(unit_counts.items())), ensure_ascii=False),
    })
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DualTurn FVAD turn-taking inference on SPEARBench metadata rows.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to turntaking.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vad-source", choices=["silero", "rms"], default="silero")
    parser.add_argument("--rms-threshold", type=float, default=0.015)
    parser.add_argument("--context-s", type=float, default=3.0)
    parser.add_argument("--unit-pre-s", type=float, default=2.0)
    parser.add_argument("--unit-post-s", type=float, default=0.0)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory for relative audio paths.")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute all rows even when output already has completed rows.")
    args = parser.parse_args()

    if not args.metadata.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {args.metadata}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    rows = read_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]

    payload = load_training_checkpoint(args.checkpoint)
    _experiment, spec = resolve_experiment(payload, args.experiment)
    device = torch.device(args.device)
    model = build_model(payload, spec, local_files_only=args.local_files_only).to(device).eval()

    frame_hz = 50.0 if spec.backbone == "vap" else 12.5
    sample_rate = 16_000 if spec.backbone == "vap" else 24_000
    manifest_stub = args.output.with_suffix(".manifest_stub.csv")
    write_csv(manifest_stub, [], ["audio_path", "natural_audio_path", "session_id", "pair_id", "edit_type"])
    evaluator = NaturalnessFiveTypeEvaluator(
        manifest=manifest_stub,
        output_dir=args.output.parent,
        sample_rate=sample_rate,
        samples_per_frame=int(sample_rate / frame_hz),
        model_frame_hz=frame_hz,
        bin_frames_50hz=vap_bin_times_to_frames([0.2, 0.4, 0.6, 0.8], 50.0),
        threshold_ratio=float(saved(payload, "threshold_ratio", 0.5)),
        bernoulli_head_reduction="sum",
        batch_size=args.batch_size,
        limit=None,
        vad_source=args.vad_source,
        rms_threshold=args.rms_threshold,
        silero_threshold=0.5,
        silero_min_speech_ms=100,
        silero_min_silence_ms=50,
        clean_min_speech_ms=150,
        clean_min_silence_ms=150,
        context_s=args.context_s,
        tail_gamma=0.25,
        lambda_mean=0.5,
        unit_pre_s=args.unit_pre_s,
        unit_post_s=args.unit_post_s,
        min_unit_frames=5,
        min_utterance_s=0.5,
        utterance_merge_gap_s=1.0,
        utterance_merge_other_max_ratio=0.2,
        unit_mode="boundaries",
    )

    completed_rows = {} if args.force_recompute else read_completed_output_rows(args.output)
    out_rows: list[dict[str, Any] | None] = []
    for idx, source_row in enumerate(rows):
        completed = completed_rows.get(idx)
        expected_answer_stem = Path(source_row.get("answer_audio_path", "")).stem
        completed_answer_stem = Path(str(completed.get("answer_audio_path", ""))).stem if completed else ""
        if completed is not None and expected_answer_stem and completed_answer_stem == expected_answer_stem:
            out_rows.append(completed)
        else:
            out_rows.append(None)
    skipped_count = sum(1 for row in out_rows if row is not None)
    if skipped_count:
        tqdm.write(f"Loaded {skipped_count} completed row(s) from existing output: {args.output}")

    prepared_batch: list[dict[str, Any]] = []
    batch_indices: list[int] = []
    batch_audio_paths: list[Path] = []
    progress = tqdm(
        total=len(rows),
        desc=f"Turn-taking {args.output.parent.name}",
        unit="file",
        dynamic_ncols=True,
        mininterval=2.0,
        miniters=max(1, min(args.batch_size, 16)),
        leave=True,
    )
    ok_count = 0
    error_count = 0
    last_postfix_update = 0

    def mark_progress(status: str, count: int = 1) -> None:
        nonlocal ok_count, error_count, last_postfix_update
        if status == "ok":
            ok_count += count
        else:
            error_count += count
        progress.update(count)
        processed = ok_count + error_count
        if processed == len(rows) or processed - last_postfix_update >= max(args.batch_size, 16):
            progress.set_postfix(ok=ok_count, error=error_count, refresh=False)
            last_postfix_update = processed

    def flush_batch() -> None:
        nonlocal prepared_batch, batch_indices, batch_audio_paths
        if not prepared_batch:
            return
        try:
            scored_batch = evaluator._score_prepared_batch(
                model,
                prepared_batch,
                device=device,
                use_autocast=device.type == "cuda",
                autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
            )
            for idx, audio_path, scored in zip(batch_indices, batch_audio_paths, scored_batch):
                out_rows[idx] = output_row(rows[idx], row_index=idx, answer_audio_path=audio_path, scored=scored, status="ok")
            mark_progress("ok", len(scored_batch))
        except Exception as exc:
            if len(prepared_batch) == 1:
                idx = batch_indices[0]
                out_rows[idx] = output_row(rows[idx], row_index=idx, answer_audio_path=batch_audio_paths[0], scored=None, status="error", error=str(exc))
                mark_progress("error")
            else:
                saved_batch = list(zip(batch_indices, batch_audio_paths, prepared_batch))
                prepared_batch = []
                batch_indices = []
                batch_audio_paths = []
                for idx, audio_path, item in saved_batch:
                    prepared_batch = [item]
                    batch_indices = [idx]
                    batch_audio_paths = [audio_path]
                    flush_batch()
                return
        prepared_batch = []
        batch_indices = []
        batch_audio_paths = []

    for idx, row in enumerate(rows):
        if out_rows[idx] is not None:
            mark_progress("ok")
            continue
        raw_question_path = row.get("audio_path", "")
        raw_answer_path = row.get("answer_audio_path", "")
        if not raw_question_path:
            out_rows[idx] = output_row(row, row_index=idx, answer_audio_path=None, scored=None, status="error", error="metadata row has no audio_path")
            mark_progress("error")
            continue
        if not raw_answer_path:
            out_rows[idx] = output_row(row, row_index=idx, answer_audio_path=None, scored=None, status="error", error="metadata row has no answer_audio_path")
            mark_progress("error")
            continue
        question_audio_path = resolve_audio_path(raw_question_path, args.base_dir)
        answer_audio_path = resolve_audio_path(raw_answer_path, args.base_dir)
        if not question_audio_path.is_file():
            out_rows[idx] = output_row(row, row_index=idx, answer_audio_path=answer_audio_path, scored=None, status="error", error=f"audio_path not found: {question_audio_path}")
            mark_progress("error")
            continue
        if not answer_audio_path.is_file():
            out_rows[idx] = output_row(row, row_index=idx, answer_audio_path=answer_audio_path, scored=None, status="error", error=f"answer_audio_path not found: {answer_audio_path}")
            mark_progress("error")
            continue
        try:
            item = make_segment_item(row, answer_audio_path, idx)
            prepared = prepare_merged_segment(
                evaluator,
                item,
                question_audio_path,
                answer_audio_path,
                parse_float(row.get("answer_start_time", 0.0)),
            )
        except Exception as exc:
            out_rows[idx] = output_row(row, row_index=idx, answer_audio_path=answer_audio_path, scored=None, status="error", error=str(exc))
            mark_progress("error")
            continue
        prepared_batch.append(prepared)
        batch_indices.append(idx)
        batch_audio_paths.append(answer_audio_path)
        if len(prepared_batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    progress.close()

    final_rows = [row if row is not None else output_row(rows[idx], row_index=idx, answer_audio_path=None, scored=None, status="error", error="not processed") for idx, row in enumerate(out_rows)]
    fields = [
        "row_index",
        "segment_id",
        "conversation_id",
        "question_audio_path",
        "answer_audio_path",
        "speakers",
        "context_end_time",
        "question_end_time",
        "answer_start_time",
        "answer_duration",
        "duration_s",
        "mean_nll",
        "tail_nll",
        "dialog_nll",
        "nat_score",
        "naturalness_score",
        "num_units",
        "tail_k",
        "num_nll_frames",
        "unit_nll",
        "unit_type_counts",
        "status",
        "error",
    ]
    write_csv(args.output, final_rows, fields)
    ok = sum(1 for row in final_rows if row.get("status") == "ok")
    tqdm.write(f"Wrote {args.output} ({ok}/{len(final_rows)} rows scored; {skipped_count} reused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
