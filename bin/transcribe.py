import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import logging
from contextlib import contextmanager

@contextmanager
def suppress_pad_token_warning():
    message = "Setting `pad_token_id` to `eos_token_id`"

    class PadTokenWarningFilter(logging.Filter):
        def filter(self, record):
            return message not in record.getMessage()

    hf_logger = logging.getLogger("transformers.generation.utils")
    filt = PadTokenWarningFilter()
    hf_logger.addFilter(filt)
    try:
        yield
    finally:
        hf_logger.removeFilter(filt)

def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def resolve_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    if model_name.startswith("whisper-"):
        return f"openai/{model_name}"
    if model_name.startswith("Qwen3-ASR-"):
        return f"Qwen/{model_name}"
    return model_name


def asr_output_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].replace(" ", "_")


def is_qwen_asr_model(model_name: str) -> bool:
    return "Qwen3-ASR" in model_name


def load_audio_for_asr(audio_path: Path, start_time: float = 0.0) -> tuple:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    start_sample = min(len(audio), int(start_time * sr))
    audio = audio[start_sample:]
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio, sr


def build_qwen_asr_model(resolved_model_name: str, device: str) -> dict:
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ImportError(
            "Qwen3-ASR models require the qwen-asr package. "
            "Install it with: pip install -U qwen-asr"
        ) from exc

    model = Qwen3ASRModel.from_pretrained(
        resolved_model_name,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=device,
        max_new_tokens=256,
    )
    return {
        "backend": "qwen",
        "model": model,
        "device": device,
        "torch_dtype": get_torch_dtype(device),
    }


def build_whisper_asr_model(resolved_model_name: str, device: str) -> dict:
    torch_dtype = get_torch_dtype(device)
    processor = AutoProcessor.from_pretrained(resolved_model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        resolved_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    return {
        "backend": "whisper",
        "processor": processor,
        "model": model,
        "device": device,
        "torch_dtype": torch_dtype,
    }


def build_asr_model(model_name: str) -> dict:
    device = get_device()
    resolved_model_name = resolve_model_name(model_name)
    print(f"Loading ASR model {resolved_model_name} on {device}")

    if is_qwen_asr_model(resolved_model_name):
        return build_qwen_asr_model(resolved_model_name, device)
    return build_whisper_asr_model(resolved_model_name, device)


def transcribe_audio(asr_model: dict, audio_path: Path, start_time: float = 0.0) -> str:
    audio, sr = load_audio_for_asr(audio_path, start_time=start_time)
    if len(audio) ==0:
        return ""

    model = asr_model["model"]
    if asr_model["backend"] == "qwen":
        try:
            with suppress_pad_token_warning():
                results = model.transcribe(audio=(audio, sr), language=None)
            if not results:
                return ""
            return getattr(results[0], "text", str(results[0])).strip()
        except Exception as exc:
            print(f"Error during Qwen ASR transcription of {audio_path}: {exc}, returning empty string")
            return ""

    processor = asr_model["processor"]
    input_features = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features.to(device=asr_model["device"], dtype=asr_model["torch_dtype"])
    try: 
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                task="transcribe",
            )
    except Exception as exc:
        print(f"Error during Whisper ASR transcription of {audio_path}: {exc}, returning empty string")
        return ""

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def resolve_audio_path(audio_path: str, data_dir: Path) -> Path:
    path = Path(audio_path)
    if path.exists():
        return path

    data_dir_relative = data_dir / path.name
    if data_dir_relative.exists():
        return data_dir_relative

    return path


def has_transcript(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def answer_path_matches(current, previous) -> bool:
    if pd.isna(current) or pd.isna(previous):
        return False
    current = str(current)
    previous = str(previous)
    return current == previous or Path(current).name == Path(previous).name


def load_reusable_transcripts(output_path: Path, metadata: pd.DataFrame, *, force_recompute: bool) -> pd.Series:
    reusable = pd.Series("", index=metadata.index, dtype="object")
    if force_recompute or not output_path.exists():
        return reusable

    try:
        previous = pd.read_csv(output_path)
    except Exception as exc:
        print(f"Warning: could not read existing transcripts {output_path}: {exc}; recomputing all rows")
        return reusable

    if "ASR_transcript_answer" not in previous.columns:
        return reusable

    previous_by_audio_path = {}
    if "answer_audio_path" in previous.columns:
        for _, row in previous.iterrows():
            transcript = row.get("ASR_transcript_answer")
            audio_path = row.get("answer_audio_path")
            if has_transcript(transcript) and not pd.isna(audio_path):
                previous_by_audio_path[str(audio_path)] = str(transcript)
                previous_by_audio_path[Path(str(audio_path)).name] = str(transcript)

    for idx in metadata.index:
        if idx in previous.index:
            transcript = previous.at[idx, "ASR_transcript_answer"]
            if has_transcript(transcript):
                if "answer_audio_path" not in previous.columns or answer_path_matches(metadata.at[idx, "answer_audio_path"], previous.at[idx, "answer_audio_path"]):
                    reusable.at[idx] = str(transcript)
                    continue

        audio_path = metadata.at[idx, "answer_audio_path"]
        if not pd.isna(audio_path):
            reusable_transcript = previous_by_audio_path.get(str(audio_path))
            if reusable_transcript is None:
                reusable_transcript = previous_by_audio_path.get(Path(str(audio_path)).name)
            if reusable_transcript is not None:
                reusable.at[idx] = reusable_transcript
    return reusable


def validate_shard_args(shard_index: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards - 1}], got {shard_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory containing audio files")
    parser.add_argument("--model", help="Model to use for ASR")
    parser.add_argument("--output_dir", required=True, help="Directory where the model-specific transcripts CSV will be written")
    parser.add_argument("--split",default='test', help="Data split to run inference on (e.g., 'test', 'dev')")
    parser.add_argument("--subset",default='improvised', help="Data subset to run inference on (e.g., 'improvised', 'naturalistic')")
    parser.add_argument("--force-recompute", action='store_true')
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to process.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of metadata shards.")
    parser.add_argument("--output-csv-name", default=None, help="Output transcript CSV filename inside output_dir.")

    args = parser.parse_args()
    validate_shard_args(args.shard_index, args.num_shards)
    print(f"Starting transcription for {args.split} {args.subset}")
    input_dir = Path(args.data_dir) / args.split / args.subset
    metadata_path = input_dir / f"metadata.csv"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_output_name = f"{asr_output_name(args.model)}_transcripts.csv"
    output_path = output_dir / (args.output_csv_name or default_output_name)
    reusable_output_path = output_dir / default_output_name

    if output_path.exists() and args.force_recompute:
        print(f"{output_path} already computed, recomputing")

    metadata = pd.read_csv(metadata_path)
    shard_indices = [idx for position, idx in enumerate(metadata.index) if position % args.num_shards == args.shard_index]
    metadata = metadata.loc[shard_indices].copy()

    if args.num_shards > 1:
        print(f"Processing shard {args.shard_index + 1}/{args.num_shards} with {len(metadata)} row(s)")

    # metadata['ASR_transcript_question'] = ''
    reusable_path = reusable_output_path if reusable_output_path.exists() else output_path
    metadata['ASR_transcript_answer'] = load_reusable_transcripts(
        reusable_path,
        metadata,
        force_recompute=args.force_recompute,
    )
    missing_indices = [idx for idx in metadata.index if not has_transcript(metadata.at[idx, 'ASR_transcript_answer'])]

    if not missing_indices:
        metadata.to_csv(output_path, index=False)
        print(f"##### {output_path} already complete ({len(metadata)} rows) #####")
        raise SystemExit(0)

    reused = len(metadata) - len(missing_indices)
    if reused:
        print(f"Reusing {reused} existing transcript row(s); computing {len(missing_indices)} missing row(s)")
    else:
        print(f"Computing {len(missing_indices)} transcript row(s)")

    asr_model = build_asr_model(args.model)

    for idx in tqdm(missing_indices, total=len(missing_indices), desc=f'transcribing answers with {args.model}', mininterval=64):
        row = metadata.loc[idx]
        audio_path = resolve_audio_path(row['answer_audio_path'], input_dir)
        metadata.at[idx, 'ASR_transcript_answer'] = transcribe_audio(
            asr_model,
            audio_path,
        )

    metadata.to_csv(output_path, index=False)
    print(f"Saved transcripts to {output_path} ({reused} reused, {len(missing_indices)} computed)")
