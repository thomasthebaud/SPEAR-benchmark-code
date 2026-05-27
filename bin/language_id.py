import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as F
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

try:
    import pycountry
except ImportError:
    pycountry = None


ISO_639_3_TO_639_2_OVERRIDES = {
    "cmn": "zho",
    "yue": "zho",
    "nan": "zho",
    "arb": "ara",
    "acm": "ara",
    "apc": "ara",
    "arz": "ara",
    "sou": "tha",
    "nod": "tha",
    "khb": "tha",
    "tts": "lao",
}


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def build_language_id_model(model_name: str) -> dict:
    device = get_device()
    torch_dtype = get_torch_dtype(device)
    print(f"Loading language ID model {model_name} on {device}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    return {
        "feature_extractor": feature_extractor,
        "model": model,
        "device": device,
        "torch_dtype": torch_dtype,
    }


def iso_639_3_to_639_2(language_code: str) -> str:
    code = str(language_code or "").strip().lower()
    if len(code) != 3:
        return code

    if code in ISO_639_3_TO_639_2_OVERRIDES:
        return ISO_639_3_TO_639_2_OVERRIDES[code]

    if pycountry is None:
        return code

    language = pycountry.languages.get(alpha_3=code)
    if language is None:
        return code

    if hasattr(language, "terminology"):
        return language.terminology
    if hasattr(language, "bibliographic"):
        return language.alpha_3

    return language.alpha_3


def resolve_audio_path(audio_path: str, metadata_path: Path) -> Path:
    path = Path(audio_path)
    if path.exists():
        return path

    metadata_relative = metadata_path.parent / path.name
    if metadata_relative.exists():
        return metadata_relative

    return path


def load_answer_audio(row: pd.Series, metadata_path: Path, target_sr: int) -> torch.Tensor:
    audio_path = resolve_audio_path(row["answer_audio_path"], metadata_path)
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    waveform = torch.from_numpy(audio)
    if sr != target_sr and waveform.numel() > 0:
        waveform = F.resample(waveform, sr, target_sr)

    return waveform


def predict_language(language_id_model: dict, waveform: torch.Tensor) -> tuple:
    if waveform.numel() == 0:
        return "", float("nan")

    feature_extractor = language_id_model["feature_extractor"]
    model = language_id_model["model"]
    target_sr = int(feature_extractor.sampling_rate)

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    converted_inputs = {}
    for key, value in inputs.items():
        if value.is_floating_point():
            value = value.to(
                device=language_id_model["device"],
                dtype=language_id_model["torch_dtype"],
            )
        else:
            value = value.to(device=language_id_model["device"])
        converted_inputs[key] = value
    inputs = converted_inputs

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1)
        score, predicted_id = torch.max(probs, dim=-1)

    predicted_id = int(predicted_id.item())
    language = model.config.id2label.get(predicted_id, str(predicted_id))
    language = iso_639_3_to_639_2(language)
    return language, float(score.item())


def load_or_initialize_outputs(metadata: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    base = pd.DataFrame({"audio_path": metadata["audio_path"]})
    if not output_path.exists():
        return base

    outputs = pd.read_csv(output_path)
    outputs = outputs.loc[:, ~outputs.columns.str.startswith("Unnamed:")]
    if "audio_path" not in outputs.columns:
        return base

    outputs = base.merge(outputs, on="audio_path", how="left")
    if "language" in outputs.columns:
        outputs["language"] = outputs["language"].map(iso_639_3_to_639_2)
    return outputs


def rows_missing_language(metadata: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    if "language" not in outputs.columns:
        return metadata

    language_by_audio = outputs.set_index("audio_path")["language"]
    values = metadata["audio_path"].map(language_by_audio)
    return metadata[values.isna() | (values == "")]


def save_outputs(outputs: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadatafile", "--metadata-file", required=True, help="Metadata CSV")
    parser.add_argument("--output", required=True, help="CSV to store language predictions")
    parser.add_argument("--model", required=True, help="Hugging Face audio language ID model")
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()

    metadata_path = Path(args.metadatafile)
    output_path = Path(args.output)
    metadata = pd.read_csv(metadata_path)
    outputs = load_or_initialize_outputs(metadata, output_path)

    if args.force_recompute:
        missing = metadata
    else:
        missing = rows_missing_language(metadata, outputs)

    if missing.empty:
        save_outputs(outputs, output_path)
        print(f"Skipping {output_path}: already computed.")
        exit()

    language_id_model = build_language_id_model(args.model)
    target_sr = int(language_id_model["feature_extractor"].sampling_rate)

    if "language" not in outputs.columns:
        outputs["language"] = ""
    if "language_score" not in outputs.columns:
        outputs["language_score"] = float("nan")

    for _, row in tqdm(missing.iterrows(), total=missing.shape[0], desc="predicting language"):
        waveform = load_answer_audio(row, metadata_path, target_sr)
        if len(waveform)/target_sr < 1.0: language, score = "unknown - too short", float("nan")
        else:
            try:
                language, score = predict_language(language_id_model, waveform)
                if score<0.5: language = "unknown - low confidence"
                if str(score)=="nan": language = "unknown"
            except Exception as exc:
                print(f"Warning: language ID failed for {row['audio_path']}: {exc}")
                language, score = "unknown", float("nan")
        row_mask = outputs["audio_path"] == row["audio_path"]
        outputs.loc[row_mask, "language"] = language
        outputs.loc[row_mask, "language_score"] = score

    save_outputs(outputs, output_path)
