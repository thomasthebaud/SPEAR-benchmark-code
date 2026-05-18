import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def resolve_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    if model_name.startswith("whisper-"):
        return f"openai/{model_name}"
    return model_name


def load_audio_for_asr(audio_path: Path, start_time: float = 0.0) -> tuple:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    start_sample = min(len(audio), int(start_time * sr))
    audio = audio[start_sample:]
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio, sr


def build_asr_model(model_name: str) -> dict:
    device = get_device()
    torch_dtype = get_torch_dtype(device)
    resolved_model_name = resolve_model_name(model_name)
    print(f"Loading ASR model {resolved_model_name} on {device}")

    processor = AutoProcessor.from_pretrained(resolved_model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        resolved_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "torch_dtype": torch_dtype,
    }


def transcribe_audio(asr_model: dict, audio_path: Path, start_time: float = 0.0) -> str:
    audio, sr = load_audio_for_asr(audio_path, start_time=start_time)
    if len(audio) == 0:
        return ""

    processor = asr_model["processor"]
    model = asr_model["model"]
    input_features = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features.to(device=asr_model["device"], dtype=asr_model["torch_dtype"])

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def resolve_audio_path(audio_path: str, data_dir: Path) -> Path:
    path = Path(audio_path)
    if path.exists():
        return path

    data_dir_relative = data_dir / path.name
    if data_dir_relative.exists():
        return data_dir_relative

    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory containing audio files")
    parser.add_argument("--model", help="Model to use for ASR")
    parser.add_argument("--split",default='test', help="Data split to run inference on (e.g., 'test', 'dev')")
    parser.add_argument("--subset",default='improvised', help="Data subset to run inference on (e.g., 'improvised', 'naturalistic')")
    parser.add_argument("--force-recompute", action='store_true')

    args = parser.parse_args()
    print(f"Starting transcription for {args.split} {args.subset}")
    input_dir = Path(args.data_dir) / args.split / args.subset
    metadata_path = input_dir / f"{args.split}_{args.subset}_metadata.csv"

    metadata = pd.read_csv(metadata_path)
    if 'ASR_transcript' not in metadata.columns: metadata['ASR_transcript'] = ''
    elif not args.force_recompute: exit(f"##### {args.split} {args.subset} already computed #####")
    else: print(f"{args.split} {args.subset} already computed, recomputing")

    asr_model = build_asr_model(args.model)

    no_answer=0
    for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc='transcribing'):
        if row['question_end_time']==row['total_duration']:#file not computed
            no_answer+=1
            continue
        audio_path = resolve_audio_path(row['audio_path'], input_dir)
        metadata.at[idx, 'ASR_transcript'] = transcribe_audio(
            asr_model,
            audio_path,
            start_time=float(row['question_end_time']),
        )

    metadata.to_csv(metadata_path, index=False)
    print(f"Saved transcribed metadata to {metadata_path}")
    print(f"{no_answer}/{len(metadata)} had no answers from the LLM")
