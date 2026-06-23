import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as torch_f
import torchaudio.functional as audio_f
from tqdm import tqdm

VOXLECT_DIR = Path(__file__).resolve().parent / "voxlect"
sys.path.insert(0, str(VOXLECT_DIR))
from voxlect_classifier import Voxlect_classifier


VOXLECT_MODELS = {
    "spanish": "tiantiaf/voxlect-spanish-dialect-whisper-large-v3",
    "english": "tiantiaf/voxlect-english-dialect-whisper-large-v3",
    "mandarin-cantonese": "tiantiaf/voxlect-mandarin-cantonese-dialect-whisper-large-v3",
    "thai": "tiantiaf/voxlect-thai-dialect-whisper-large-v3",
    "french": "tiantiaf/voxlect-french-dialect-whisper-large-v3",
    "german": "tiantiaf/voxlect-german-dialect-whisper-large-v3",
    "arabic": "tiantiaf/voxlect-arabic-dialect-whisper-large-v3",
    "tibetan": "tiantiaf/voxlect-tibetan-dialect-whisper-large-v3",
}


LANGUAGE_ALIASES = {
    "ar": "arabic",
    "ara": "arabic",
    "arabic": "arabic",
    "arb": "arabic",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "german": "german",
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "french": "french",
    "bo": "tibetan",
    "bod": "tibetan",
    "tib": "tibetan",
    "tibetan": "tibetan",
    "th": "thai",
    "tha": "thai",
    "thai": "thai",
    "cmn": "mandarin-cantonese",
    "mandarin": "mandarin-cantonese",
    "mandarin chinese": "mandarin-cantonese",
    "yue": "mandarin-cantonese",
    "cantonese": "mandarin-cantonese",
    "zh": "mandarin-cantonese",
    "zho": "mandarin-cantonese",
    "chi": "mandarin-cantonese",
    "chinese": "mandarin-cantonese",
}


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def normalize_language(language: str) -> str:
    return str(language or "").strip().lower().replace("_", " ").replace("-", " ")


def get_voxlect_model_name(language: str) -> str:
    normalized = normalize_language(language)
    model_key = LANGUAGE_ALIASES.get(normalized, normalized.replace(" ", "-"))
    return VOXLECT_MODELS.get(model_key, "")


def is_english(language: str) -> bool:
    return LANGUAGE_ALIASES.get(normalize_language(language), normalize_language(language)) == "english"


def build_dialect_id_model(model_name: str) -> dict:
    device = get_device()
    if torch.cuda.is_available():
        print("GPU available, use GPU")
    else:
        print("Warning: GPU unavailable. VoxLect Whisper models expect CUDA and may fail on CPU.")

    print(f"Loading dialect ID model {model_name} on {device}")
    classifier = Voxlect_classifier(model_name, device)

    return {
        "classifier": classifier,
        "device": device,
        "model_name": model_name,
    }



def load_audio(audio_path: Path, target_sr: int = 16000) -> torch.Tensor:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    waveform = torch.from_numpy(audio)
    if sr != target_sr and waveform.numel() > 0:
        waveform = audio_f.resample(waveform, sr, target_sr)

    return waveform.unsqueeze(0)


def infer_dialect_labels(classifier: Voxlect_classifier, num_classes: int) -> list:
    config = getattr(classifier.model, "config", None)
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) == num_classes:
        return [id2label.get(idx, id2label.get(str(idx), f"class_{idx}")) for idx in range(num_classes)]

    dialect_list = getattr(classifier, "dialect_list", [])
    if len(dialect_list) == num_classes:
        return dialect_list

    return [f"class_{idx}" for idx in range(num_classes)]


def predict_dialect(dialect_id_model: dict, waveform: torch.Tensor, volume: float = 1.0):
    if waveform.numel() == 0:
        return "unknown", "nan"

    classifier = dialect_id_model["classifier"]
    device = dialect_id_model["device"]

    with torch.no_grad():
        probs, label = classifier.process(waveform, device, sr=16000)

    probs_list = probs[0].tolist()
    return label, str(probs_list)


def load_or_initialize_outputs(metadata: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    base = pd.DataFrame({"audio_path": metadata["audio_path"]})
    if not output_path.exists():
        return base

    outputs = pd.read_csv(output_path)
    outputs = outputs.loc[:, ~outputs.columns.str.startswith("Unnamed:")]
    if "audio_path" not in outputs.columns:
        return base

    return base.merge(outputs, on="audio_path", how="left")


def rows_missing_dialect(metadata: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    if "dialect" not in outputs.columns:
        return metadata

    dialect_by_audio = outputs.set_index("audio_path")["dialect"]
    values = metadata["audio_path"].map(dialect_by_audio)
    return metadata[values.isna() | (values == "")]


def load_language_predictions(metadata: pd.DataFrame, output_path: Path, language_file: str) -> pd.Series:
    if "language" in metadata.columns:
        return metadata["language"]

    if language_file:
        language_path = Path(language_file)
    else:
        language_path = output_path.parent / "language_id.csv"

    if not language_path.exists():
        raise FileNotFoundError(
            f"Could not find per-utterance language predictions at {language_path}"
        )

    language_outputs = pd.read_csv(language_path)
    if "audio_path" not in language_outputs.columns or "language" not in language_outputs.columns:
        raise ValueError(f"{language_path} must contain audio_path and language columns")

    language_by_audio = language_outputs.set_index("audio_path")["language"]
    return metadata["audio_path"].map(language_by_audio).fillna("unknown")


def save_outputs(outputs: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadatafile", "--metadata-file", required=True, help="Metadata CSV")
    parser.add_argument("--output", required=True, help="CSV to store dialect predictions")
    parser.add_argument(
        "--language",
        required=True,
        help="Language to select the VoxLect model",
    )
    parser.add_argument(
        "--languagefile",
        "--language-file",
        default="",
        help="CSV with audio_path and language columns. Defaults to language_id.csv next to --output.",
    )
    parser.add_argument("--volume", type=float, default=1.0, help="Audio volume multiplier")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--eval-question",
        action="store_true",
        help="Only predict dialect for the question audio_path; skip answer_audio_path.",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadatafile)
    output_path = Path(args.output)
    metadata = pd.read_csv(metadata_path)
    outputs = load_or_initialize_outputs(metadata, output_path)

    language_path = Path(args.languagefile) if args.languagefile else output_path.parent / "language_id.csv"
    if language_path.exists():
        metadata["detected_language"] = load_language_predictions(
            metadata,
            output_path,
            str(language_path),
        )
    else:
        metadata["detected_language"] = args.language

    if args.force_recompute:
        missing = metadata
    else:
        missing = rows_missing_dialect(metadata, outputs)

    if missing.empty:
        print(f"Skipping {output_path}: already computed.")
        exit()

    if "language" not in outputs.columns:
        outputs["language"] = ""
    if "dialect" not in outputs.columns:
        outputs["dialect"] = ""
    if "dialect_score" not in outputs.columns:
        outputs["dialect_score"] = "nan"
    else:
        outputs["dialect_score"] = outputs["dialect_score"].astype("object")
    if "dialect_model" not in outputs.columns:
        outputs["dialect_model"] = ""

    requested_language = args.language
    model_name = get_voxlect_model_name(requested_language)
    model = build_dialect_id_model(model_name)

    for _, row in tqdm(missing.iterrows(), total=missing.shape[0], desc="predicting dialect"):
        detected_language = row["detected_language"]
        
        row_mask = outputs["audio_path"] == row["audio_path"]
        outputs.loc[row_mask, "language"] = detected_language

        if not is_english(detected_language):
            outputs.loc[row_mask, "dialect"] = "unpredicted"
            outputs.loc[row_mask, "dialect_score"] = "nan"
            outputs.loc[row_mask, "dialect_model"] = ""
            continue

        if not model_name or not is_english(requested_language):
            outputs.loc[row_mask, "dialect"] = "unpredicted"
            outputs.loc[row_mask, "dialect_score"] = "nan"
            outputs.loc[row_mask, "dialect_model"] = ""
            continue

        waveform = load_audio(Path(row["audio_path"]))
        dialect, score = predict_dialect(model, waveform, volume=args.volume)
        outputs.loc[row_mask, "dialect"] = dialect
        outputs.loc[row_mask, "dialect_score"] = score
        outputs.loc[row_mask, "dialect_model"] = model_name

        if args.eval_question:
            continue

        waveform = load_audio(Path(row["answer_audio_path"]))
        dialect, score = predict_dialect(model, waveform, volume=args.volume)
        outputs.loc[row_mask, "answer_dialect"] = dialect
        outputs.loc[row_mask, "answer_dialect_score"] = score
        outputs.loc[row_mask, "answer_dialect_model"] = model_name

    save_outputs(outputs, output_path)
