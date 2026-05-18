import os
import argparse
import pandas as pd
import csv
import json
import struct
import wave
from pathlib import Path
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000

def iter_dyads(split_name: str, data_dir: Path, SET: str) -> list[dict]:
    dyads: list[dict] = []
    lookup_path = data_dir / "assets/dyad_lookup.csv"
    reader = pd.read_csv(lookup_path)
    for _, row in reader.iterrows():
        relpath1 = Path(row["participant1_relpath"])
        relpath2 = Path(row["participant2_relpath"])
        if relpath1.parts[1:3] != (SET, split_name):
            continue
        if relpath2.parts[1:3] != (SET, split_name):
            continue
        dyads.append(row)
    return dyads

def wav_relpath_to_json_path(relpath: str, data_dir: Path) -> Path:
    return (data_dir / "assets" / relpath).resolve().with_suffix(".json")

def extract_turns(record: dict, speaker_id: str) -> list[dict]:
    transcript_data = record.get("transcript")
    if transcript_data is None:
        transcript_data = record.get("metadata:transcript", [])

    turns: list[dict] = []
    if isinstance(transcript_data, str):
        text = transcript_data.strip()
        if text:
            turns.append(
                {
                    "speaker_id": speaker_id,
                    "start": float("inf"),
                    "end": float("inf"),
                    "text": text,
                }
            )
        return turns

    if not isinstance(transcript_data, list):
        return turns

    for index, item in enumerate(transcript_data):
        if isinstance(item, str):
            text = item.strip()
            start = float(index)
            end = float(index)
        elif isinstance(item, dict):
            text = str(item.get("transcript", "")).strip()
            start = float(item.get("start", index))
            end = float(item.get("end", start))
        else:
            continue

        if text:
            turns.append(
                {
                    "speaker_id": speaker_id,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    return turns

def load_record(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)
    
def conversation_stem(path: Path) -> str:
    stem = path.stem
    participant_suffix = stem.rsplit("_P", 1)
    if len(participant_suffix) == 2:
        return participant_suffix[0]
    return stem

def load_transcripts(split_name: str, data_dir: Path, SET: str) -> int:
    transcripts = {}
    audios = {}
    for row in tqdm(iter_dyads(split_name, data_dir, SET), desc=f"Loading transcripts for {split_name} {SET}"):
        speaker1_id = row["participant1_id"]
        speaker2_id = row["participant2_id"]
        json_path1 = wav_relpath_to_json_path(row["participant1_relpath"], data_dir=data_dir)
        json_path2 = wav_relpath_to_json_path(row["participant2_relpath"], data_dir=data_dir)

        if not json_path1.exists() or not json_path2.exists():
            continue

        turns = extract_turns(load_record(json_path1), speaker1_id)
        turns.extend(extract_turns(load_record(json_path2), speaker2_id))
        turns.sort(key=lambda turn: (turn["start"], turn["end"], turn["speaker_id"], turn["text"]))

        stem = conversation_stem(Path(row["participant1_relpath"]))
        transcripts[f"{stem}_{speaker1_id}_{speaker2_id}"] = turns
        audios[f"{stem}_{speaker1_id}_{speaker2_id}"] = {
            'conversation_id': stem,
            'spk1': speaker1_id,
            'spk2': speaker2_id,
            'audio1': str(data_dir / row["participant1_relpath"][3:]),
            'audio2': str(data_dir / row["participant2_relpath"][3:]),
        }

    return transcripts, audios

def keep_questions(transcripts: dict[str, list[dict]], audios: dict[str, dict], N_max=5) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    filtered_transcripts = {}
    filtered_audios = {}
    for key, turns in transcripts.items():
        idx=0
        for row, turn in enumerate(turns[:-1]):
            if turn["text"].strip().endswith("?") and turn["speaker_id"] != turns[row+1]["speaker_id"]:
                start_row = max(0, row - N_max + 1)
                selected_turns = turns[start_row:row + 1]
                filtered_transcripts[key+f'_{idx}'] = selected_turns

                selected_audio = audios[key].copy()
                selected_audio['start'] = selected_turns[0]["start"]
                selected_audio['end'] = selected_turns[-1]["end"]
                selected_audio['question_end_time'] = ""
                filtered_audios[key+f'_{idx}'] = selected_audio
                idx += 1

    print(f"After selecting for sub-dialogues ending with a question of maximum {N_max} turns, {len(filtered_transcripts)} were extracted from {len(transcripts)} dialogues.")
    return filtered_transcripts, filtered_audios


def keep_answers(transcripts: dict[str, list[dict]], audios: dict[str, dict], N_max=5) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    filtered_transcripts = {}
    filtered_audios = {}
    for key, turns in transcripts.items():
        idx=0
        for row, turn in enumerate(turns[:-1]):
            if turn["text"].strip().endswith("?") and turn["speaker_id"] != turns[row+1]["speaker_id"]:
                K=1
                while row+K < len(turns) and turns[row+K]["speaker_id"] != turn["speaker_id"]:
                    K+=1
                start_row = max(0, row - N_max + 1)
                selected_turns = turns[start_row:row + K]
                filtered_transcripts[key+f'_{idx}'] = selected_turns

                selected_audio = audios[key].copy()
                selected_audio['start'] = selected_turns[0]["start"]
                selected_audio['end'] = selected_turns[-1]["end"]
                selected_audio['question_end_time'] = turn["end"] - selected_turns[0]["start"]
                filtered_audios[key+f'_{idx}'] = selected_audio
                idx += 1

    print(f"Keeping the answers as well: {len(filtered_transcripts)} answers kept.")
    return filtered_transcripts, filtered_audios


def read_wav_segment_mono(audio_path: Path, start_time: float, end_time: float) -> tuple[list[float], int]:
    """Read one source channel from PCM or IEEE-float WAV without optional audio deps."""
    with audio_path.open("rb") as infile:
        if infile.read(4) != b"RIFF":
            raise ValueError(f"{audio_path} is not a RIFF WAV file.")
        infile.seek(8)
        if infile.read(4) != b"WAVE":
            raise ValueError(f"{audio_path} is not a WAVE file.")

        audio_format = channels = sample_rate = bits_per_sample = None
        data_offset = data_size = None
        while True:
            header = infile.read(8)
            if len(header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", header)
            chunk_start = infile.tell()
            if chunk_id == b"fmt ":
                fmt_data = infile.read(chunk_size)
                audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack("<HHIIHH", fmt_data[:16])
            elif chunk_id == b"data":
                data_offset = chunk_start
                data_size = chunk_size
                infile.seek(chunk_size, 1)
            else:
                infile.seek(chunk_size, 1)
            if chunk_size % 2:
                infile.seek(1, 1)

        if None in (audio_format, channels, sample_rate, bits_per_sample, data_offset, data_size):
            raise ValueError(f"{audio_path} is missing WAV fmt or data chunks.")

        bytes_per_sample = bits_per_sample // 8
        frame_size = channels * bytes_per_sample
        start_frame = max(0, int(start_time * sample_rate))
        end_frame = max(start_frame, int(end_time * sample_rate + 0.999999))
        available_frames = data_size // frame_size
        frame_count = max(0, min(end_frame, available_frames) - start_frame)

        infile.seek(data_offset + start_frame * frame_size)
        raw = infile.read(frame_count * frame_size)

    samples = []
    for offset in range(0, len(raw), frame_size):
        sample = raw[offset:offset + bytes_per_sample]
        if audio_format == 3 and bits_per_sample == 32:
            value = struct.unpack("<f", sample)[0]
        elif audio_format == 1 and bits_per_sample == 16:
            value = struct.unpack("<h", sample)[0] / 32768.0
        elif audio_format == 1 and bits_per_sample == 32:
            value = struct.unpack("<i", sample)[0] / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV format in {audio_path}: format={audio_format}, bits={bits_per_sample}")
        samples.append(value)

    return samples, sample_rate


def resample_linear(samples: list[float], source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> list[float]:
    if source_rate == target_rate:
        return samples
    if not samples:
        return []

    output_length = max(1, round(len(samples) * target_rate / source_rate))
    if output_length == 1:
        return [samples[0]]

    ratio = source_rate / target_rate
    resampled = []
    last_index = len(samples) - 1
    for output_index in range(output_length):
        source_position = output_index * ratio
        left = min(int(source_position), last_index)
        right = min(left + 1, last_index)
        weight = source_position - left
        resampled.append(samples[left] * (1.0 - weight) + samples[right] * weight)
    return resampled


def fit_length(samples: list[float], target_frames: int) -> list[float]:
    if len(samples) >= target_frames:
        return samples[:target_frames]
    return samples + [0.0] * (target_frames - len(samples))


def write_stereo_wav(audio_path: Path, channel_a: list[float], channel_b: list[float], sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as outfile:
        outfile.setnchannels(2)
        outfile.setsampwidth(2)
        outfile.setframerate(sample_rate)
        frames = bytearray()
        for sample_a, sample_b in zip(channel_a, channel_b):
            for sample in (sample_a, sample_b):
                clipped = max(-1.0, min(1.0, sample))
                frames.extend(struct.pack("<h", int(clipped * 32767.0)))
        outfile.writeframes(frames)


def export_segment_audio(audio_path: Path, audio_info: dict) -> tuple[Path, float]:
    start_time = float(audio_info["start"])
    end_time = float(audio_info["end"])
    target_frames = max(1, round((end_time - start_time) * TARGET_SAMPLE_RATE))

    if not os.path.exists(audio_path):
        speaker_a, rate_a = read_wav_segment_mono(Path(audio_info["audio1"]), start_time, end_time)
        speaker_b, rate_b = read_wav_segment_mono(Path(audio_info["audio2"]), start_time, end_time)
        speaker_a = fit_length(resample_linear(speaker_a, rate_a), target_frames)
        speaker_b = fit_length(resample_linear(speaker_b, rate_b), target_frames)

        write_stereo_wav(audio_path, speaker_a, speaker_b)
    return target_frames / TARGET_SAMPLE_RATE


def format_transcript(turns: list[dict]) -> str:
    return " ".join(f"{turn['speaker_id']}: {turn['text']}" for turn in turns)


def save_processed_dataset(
    output_path: Path,
    split: str,
    subset: str,
    transcripts: dict[str, list[dict]],
    audios: dict[str, dict],
    label: str
) -> tuple[Path, Path, Path]:
    output_path.mkdir(parents=True, exist_ok=True)

    transcript_output_path = output_path / f"{split}_{subset}_transcripts_processed.json"
    with transcript_output_path.open("w", encoding="utf-8") as outfile:
        json.dump(transcripts, outfile, indent=2)

    audios_output_path = output_path / f"{split}_{subset}_audios_processed.json"
    with audios_output_path.open("w", encoding="utf-8") as outfile:
        json.dump(audios, outfile, indent=2)

    metadata_output_path = output_path / f"{split}_{subset}_metadata.csv"
    fieldnames = [
        "audio_path",
        "total_duration",
        "speakers",
        "initial_conversation",
        "transcript",
        "question_end_time",
    ]
    with metadata_output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for audio_id, audio_info in tqdm(audios.items(), desc=f"Saving audio {label} for {split} {subset}"):
            audio_path = output_path / "audios" / f"{audio_id}.wav"
            duration = export_segment_audio(audio_path, audio_info)
            writer.writerow(
                {
                    "audio_path": str(audio_path),
                    "total_duration": f"{duration:.3f}",
                    "speakers": f"{audio_info['spk1']}|{audio_info['spk2']}",
                    "initial_conversation": audio_info["conversation_id"],
                    "transcript": format_transcript(transcripts[audio_id]),
                    "question_end_time": audio_info.get("question_end_time", ""),
                }
            )

    return transcript_output_path, audios_output_path, metadata_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the seamless data.")
    parser.add_argument("--questions_output_dir", type=str, required=True, help="Path to the directory where the processed questions data will be saved.")
    parser.add_argument("--answers_output_dir", type=str, required=True, help="Path to the directory where the processed answers data will be saved.")
    parser.add_argument("--min_turns", type=int, default=1, help="Minimum number of turns in a dialogue.")  
    parser.add_argument("--min_speakers", type=int, default=1, help="Minimum number of speakers in a dialogue.")
    parser.add_argument("--method", type=str, default='end_with_question', help="Method to select the last turn of the dialogue. Options: 'end_with_question'.")
    parser.add_argument("--split",default='test', help="Data split to run inference on (e.g., 'test', 'dev')")
    parser.add_argument("--subset",default='improvised', help="Data subset to run inference on (e.g., 'improvised', 'naturalistic')")

    args = parser.parse_args()

    print(f"Processing data from {args.data_dir} and saving to {args.questions_output_dir} and {args.answers_output_dir} with min_turns={args.min_turns} and min_speakers={args.min_speakers}.")
    split, subset = args.split, args.subset
    print(f"Now processing {split}, {subset}")
    # Load the seamless data

    # Create output directory if it doesn't exist
    questions_output_path = Path(args.questions_output_dir) / f"{split}/{subset}"
    answers_output_path = Path(args.answers_output_dir) / f"{split}/{subset}"
    os.makedirs(questions_output_path, exist_ok=True)
    os.makedirs(answers_output_path, exist_ok=True)
    
    #Loading sentences
    transcripts, audios = load_transcripts(split_name=split, data_dir=Path(args.data_dir), SET=subset)

    #filtering dialogues based on min_turns and min_speakers
    L = len(transcripts)
    if args.min_turns > 0:
        audios = {k: v for k, v in audios.items() if len(transcripts[k]) >= args.min_turns}
        transcripts = {k: v for k, v in transcripts.items() if len(v) >= args.min_turns}
        print(f"After filtering for minimum turns ({args.min_turns}), {len(transcripts)}/{L} dialogues remain in {split} {subset}.")
        L = len(transcripts)
    if args.min_speakers > 0:
        audios = {k: v for k, v in audios.items() if len(transcripts[k]) >= args.min_speakers}
        transcripts = {k: v for k, v in transcripts.items() if len(v) >= args.min_speakers}
        print(f"After filtering for minimum speakers ({args.min_speakers}), {len(transcripts)}/{L} dialogues remain in {split} {subset}.")
        L = len(transcripts)

    assert L>0, f"No dialogues remain in {split} {subset} after filtering. Please adjust the min_turns and min_speakers parameters."

    if args.method == 'end_with_question':
        questions_transcripts, questions_audios = keep_questions(transcripts, audios)
        answers_transcripts, answers_audios = keep_answers(transcripts, audios)
    else:
        raise ValueError(f"Unsupported method: {args.method}. Supported methods: 'end_with_question'.")
    
    # Save the processed data
    transcript_output_path, audios_output_path, metadata_output_path = save_processed_dataset(
        questions_output_path,
        split,
        subset,
        questions_transcripts,
        questions_audios,
        'questions'
    )
    answers_transcript_output_path, answers_audios_output_path, answers_metadata_output_path = save_processed_dataset(
        answers_output_path,
        split,
        subset,
        answers_transcripts,
        answers_audios,
        'answers'
    )
    print(f"Saved processed data for {split} {subset} to {transcript_output_path}, {audios_output_path}, {metadata_output_path}, {answers_transcript_output_path}, {answers_audios_output_path}, and {answers_metadata_output_path}.")
