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

    args = parser.parse_args()
    model_proxy = load_proxy_module(args.model)
    sr = 16_000 

    input_dir = Path(args.audio_dir) / args.split / args.subset
    output_dir = Path(args.output_dir) / args.split / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(output_dir / f"metadata.csv"):
        print(f"### Subset {args.split}/{args.subset} already processed, moving on. ###")
        exit()

    if args.prompt=='None':print("Warning: No prompt provided, only feeding the audios.")

    metadata = pd.read_csv(input_dir / f"metadata.csv")
    print(f"found {len(metadata)} rows in {input_dir}/metadata.csv")
    output_metadata = metadata.copy()
    for col in ["answer_audio_path", "transcript_answer", "answer_start_time", "finish_reason"]:
        if col not in output_metadata.columns:
            output_metadata[col] = pd.Series(index=output_metadata.index, dtype="object")
        else:
            output_metadata[col] = output_metadata[col].astype("object")

    failed_indices = {}

    for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
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
                # print(f"Warning: failed to process {input_audio_path} because {finish_reason}")
                continue
        except Exception as exc:
            print(f"Warning: failed to process {input_audio_path}: {exc}")
            finish_reason = str(exc)
            if finish_reason not in failed_indices:
                failed_indices[finish_reason] = []
            failed_indices[finish_reason].append(idx)
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
        output_metadata.at[idx, 'transcript_answer'] = transcript_answer
        output_metadata.at[idx, 'answer_start_time'] = f"{answer_start_time - row['question_end_time']:.3f}"
        output_metadata.at[idx, 'answer_audio_path'] = str(output_path)
        output_metadata.at[idx, 'answer_duration'] = len(audio_output) / sr
        output_metadata.at[idx, 'finish_reason'] = finish_reason
        # except Exception as exc:
        #     print(f"Warning: failed to process {input_audio_path}: {exc}")
        #     failed_indices.append(idx)

        # exit("Exiting after first iteration for testing purposes") # --- IGNORE ---

    total_failed = 0
    for reason in failed_indices:
        print(f"Failed due to {reason}: {failed_indices[reason]}")
        output_metadata = output_metadata.drop(index=failed_indices[reason])
        total_failed += len(failed_indices[reason])

    output_metadata.to_csv(output_dir / f"metadata.csv", index=False)
    print(f"Failed audios = {len(total_failed)}/{len(output_metadata)+len(total_failed)}")
