import argparse
import io
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from openai import OpenAI
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


def get_reply_with_audio(audio_path, instruction, model_name, org, api_key, temp=0.7):
    try:
        client = OpenAI(api_key=api_key, organization=org)

        with open(audio_path, "rb") as infile:
            audio_b64 = base64.b64encode(infile.read()).decode("utf-8")

        # Build the user message with audio input
        user_content = []
        if instruction != 'None':
            user_content.append({"type": "text", "text": instruction})
        user_content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": audio_path.suffix.lstrip(".")
                }
            })
        # Send to the model
        response = client.chat.completions.create(
            model=model_name,
            temperature=temp,
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can understand and respond to speech."},
                {"role": "user", "content": user_content}
            ]
        )

        message = response.choices[0].message
        if message.audio is None or message.audio.data is None:
            print(f"Warning: model failed to return audio for {audio_path}")
            return None, None, None, False

        audio_bytes = base64.b64decode(message.audio.data)
        transcript = message.audio.transcript

        return audio_bytes, transcript, response.choices[0].finish_reason, True
    except Exception as exc:
        print(f"Warning: model failed for {audio_path}: {exc}")
        return None, None, None, False



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

    
    input_dir = Path(args.audio_dir) / args.split / args.subset
    output_dir = Path(args.output_dir) / args.split / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(output_dir / f"{args.split}_{args.subset}_metadata.csv"):
        print(f"### Subset {args.split}/{args.subset} already processed, moving on. ###")
        exit()

    if args.prompt=='None':print("Warning: No prompt provided, only feeding the audios.")

    metadata = pd.read_csv(input_dir / f"{args.split}_{args.subset}_metadata.csv")
    print(f"found {len(metadata)} rows in {input_dir}/{args.split}_{args.subset}_metadata.csv")
    output_metadata = metadata.copy()
    output_metadata['question_end_time'] = output_metadata['total_duration']
    failed_indices = []

    for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        input_audio_path = Path(row['audio_path'])
        output_path = output_dir / f"audio/{input_audio_path.stem}.wav"
        try:
            #Load audio
            audio, sr = sf.read(input_audio_path, dtype="float32", always_2d=True)
            if audio.shape[1] > 2: audio = np.sum(audio, axis=1, keepdims=True) # Convert to mono if more than 2 channels
            # print(f"audio shape: {audio.shape}")
            # get answer
            audio_answer_bytes, transcript_answer, finish_reason, success = get_reply_with_audio(audio_path=input_audio_path, instruction=args.prompt, model_name=args.model, org=args.org, api_key=args.openai_api_key)
            if not success:
                failed_indices.append(idx)
                print(f"Warning: failed to process {input_audio_path}")
                continue
            # print("finish reason", finish_reason, "transcript:", transcript_answer)
            # save the answer
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio_answer, answer_sr = sf.read(io.BytesIO(audio_answer_bytes), dtype="float32", always_2d=True)
            audio_answer = resample_audio(audio_answer, answer_sr, sr)
            if audio_answer.shape[1] != audio.shape[1]:
                audio_answer = np.tile(audio_answer[:, :1], (1, audio.shape[1]))
            audio_output = np.concatenate([audio, audio_answer], axis=0) 
            sf.write(output_path, audio_output, sr)

            trancript_output = row['transcript'] + '<...>' + transcript_answer
            output_metadata.at[idx, 'transcript'] = trancript_output
            output_metadata.at[idx, 'audio_path'] = output_path
            output_metadata.at[idx, 'total_duration'] = len(audio_output) / sr
        except Exception as exc:
            print(f"Warning: failed to process {input_audio_path}: {exc}")
            failed_indices.append(idx)

    if failed_indices:
        output_metadata = output_metadata.drop(index=failed_indices)

    output_metadata.to_csv(output_dir / f"{args.split}_{args.subset}_metadata.csv", index=False)
    print(f"Failed audios = {len(failed_indices)}/{len(output_metadata)+len(failed_indices)}")
