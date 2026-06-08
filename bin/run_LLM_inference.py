import argparse
import io
import importlib.util
import sys
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import base64
import numpy as np
import soundfile as sf
import os

def resample_audio(audio, source_sr, target_sr):
    if source_sr == target_sr:
        return audio
    output_length = max(1, round(audio.shape[0] * target_sr / source_sr))
    source_positions = np.linspace(0, audio.shape[0] - 1, num=output_length)
    resampled_channels = [
        np.interp(source_positions, np.arange(audio.shape[0]), audio[:, channel])
        for channel in range(audio.shape[1])
    ]
    return np.stack(resampled_channels, axis=1).astype(np.float32)


def load_proxy_module(model_name: str):
    model_name_lower = model_name.lower()
    proxy_dir = Path(__file__).resolve().parent / "llm_proxies"

    proxy_path = proxy_dir / f"{model_name_lower}.py"

    if not proxy_path.exists():
        raise FileNotFoundError(f"Proxy implementation not found at {proxy_path}")

    spec = importlib.util.spec_from_file_location(f"llm_proxies.{proxy_path.stem}", proxy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import proxy module from {proxy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "get_reply_with_audio"):
        raise AttributeError(
            f"Proxy module '{proxy_path}' must define get_reply_with_audio(...)"
        )
    return module


def ensure_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ["answer_audio_path", "transcript_answer", "answer_start_time", "answer_duration", "finish_reason"]:
        if col not in frame.columns:
            frame[col] = pd.Series(index=frame.index, dtype="object")
        else:
            frame[col] = frame[col].astype("object")
    return frame


def save_output_metadata(frame: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    frame.to_csv(tmp_csv, index=False)
    tmp_csv.replace(output_csv)


def load_existing_output(output_csv: Path) -> pd.DataFrame | None:
    if not output_csv.exists():
        return None
    try:
        frame = pd.read_csv(output_csv)
    except Exception as exc:
        print(f"Warning: could not read existing output metadata {output_csv}: {exc}", flush=True)
        return None
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--output_dir", help="Directory to save inference outputs")
    parser.add_argument("--model", help="Model to use for inference")
    parser.add_argument("--prompt", help="Prompt for inference")
    parser.add_argument("--split",default='test', help="Data split to run inference on (e.g., 'test', 'dev')")
    parser.add_argument("--subset",default='improvised', help="Data subset to run inference on (e.g., 'improvised', 'naturalistic')")
    parser.add_argument("--openai-api-key",type=str, help="")
    parser.add_argument("--org",type=str, help="")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop at the first failed inference instead of skipping failed rows.")

    args = parser.parse_args()
    model_proxy = load_proxy_module(args.model)
    sr = 16_000 

    input_dir = Path(args.audio_dir) / args.split / args.subset
    output_dir = Path(args.output_dir) / args.split / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "metadata.csv"

    if args.prompt=='None':print("Warning: No prompt provided, only feeding the audios.")

    metadata = pd.read_csv(input_dir / "metadata.csv")
    print(f"found {len(metadata)} rows in {input_dir}/metadata.csv")

    existing_output = load_existing_output(output_csv)
    if existing_output is not None and len(existing_output) == len(metadata):
        print(f"### Subset {args.split}/{args.subset} already processed, moving on. ###")
        exit()

    if existing_output is not None:
        output_metadata = ensure_output_columns(existing_output)
        processed_audio_paths = set(output_metadata["audio_path"].dropna().astype(str)) if "audio_path" in output_metadata.columns else set()
        print(f"Resuming from {len(processed_audio_paths)}/{len(metadata)} completed rows in {output_csv}")
    else:
        output_metadata = ensure_output_columns(metadata.iloc[0:0].copy())
        processed_audio_paths = set()
        save_output_metadata(output_metadata, output_csv)

    failed_indices = {}

    remaining = metadata[~metadata["audio_path"].astype(str).isin(processed_audio_paths)]

    for idx, row in tqdm(remaining.iterrows(), total=remaining.shape[0], desc=f"Processing {args.split}/{args.subset}"):
        input_audio_path = Path(row['audio_path'])
        output_path = output_dir / f"audio/{input_audio_path.stem}.wav"
        try:
            # get answer
            audio_answer_bytes, transcript_answer, finish_reason, success, answer_start_time = model_proxy.get_reply_with_audio(
                audio_path=input_audio_path,
                instruction=args.prompt,
                model_name=args.model,
                org=args.org,
                api_key=args.openai_api_key,
            )
            if not success:
                if finish_reason not in failed_indices:
                    failed_indices[finish_reason] = []
                failed_indices[finish_reason].append(idx)
                print(f"Warning: failed to process {input_audio_path} because {finish_reason}" , flush=True)
                if args.stop_on_fail:
                    raise RuntimeError(f"Failed to process {input_audio_path} because {finish_reason}")
                continue
        except Exception as exc:
            print(f"Warning: failed to process {input_audio_path}: {exc}", flush=True)
            finish_reason = str(exc)
            if finish_reason not in failed_indices:
                failed_indices[finish_reason] = []
            failed_indices[finish_reason].append(idx)
            if args.stop_on_fail:
                raise
            continue
        # print("finish reason", finish_reason, "transcript:", transcript_answer)
        # save the answer
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_answer, answer_sr = sf.read(io.BytesIO(audio_answer_bytes), dtype="float32", always_2d=True)
        audio_answer = resample_audio(audio_answer, answer_sr, sr)
        audio_output = audio_answer #keeping the answer separated
        # audio_output = np.concatenate([audio, audio_answer], axis=0) 
        sf.write(output_path, audio_output, sr)

        if answer_start_time is None: answer_start_time = row['question_end_time']
        # 0 if no delay or non streaming model, negative if interruption, positive if delayed
        output_row = row.copy()
        output_row['transcript_answer'] = transcript_answer
        output_row['answer_start_time'] = f"{answer_start_time - row['question_end_time']:.3f}"
        output_row['answer_audio_path'] = str(output_path)
        output_row['answer_duration'] = len(audio_output) / sr
        output_row['finish_reason'] = finish_reason

        output_metadata = pd.concat([output_metadata, pd.DataFrame([output_row])], ignore_index=True)
        output_metadata = ensure_output_columns(output_metadata)
        save_output_metadata(output_metadata, output_csv)
        processed_audio_paths.add(str(row['audio_path']))
        # except Exception as exc:
        #     print(f"Warning: failed to process {input_audio_path}: {exc}")
        #     failed_indices.append(idx)

        # exit("Exiting after first iteration for testing purposes") # --- IGNORE ---

    total_failed = 0
    for reason in failed_indices:
        print(f"Failed due to {reason}: {len(failed_indices[reason])}")
        total_failed += len(failed_indices[reason])

    print(f"Completed audios = {len(output_metadata)}/{len(metadata)}")
    print(f"Failed audios = {total_failed}/{remaining.shape[0]}")
