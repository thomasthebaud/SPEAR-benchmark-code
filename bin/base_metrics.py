import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import numpy as np
from jiwer import wer, cer
import tempfile
import os
import re
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

utmos_model = None
utmos_device = None
vad_model = None
interruption_cache = {}

def get_utmos_device():
    global utmos_device
    if utmos_device is None:
        utmos_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {utmos_device} for UTMOS inference")
    return utmos_device


def get_utmos_model():
    global utmos_model
    if utmos_model is None:
        device = get_utmos_device()
        utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        )
        utmos_model = utmos_model.to(device).eval()
    return utmos_model


def get_vad_model():
    global vad_model
    if vad_model is None:
        vad_model = load_silero_vad()
    return vad_model

def load_audio_mono(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    return waveform.squeeze(0), sample_rate


def load_answer(row):
    waveform, sample_rate = load_audio_mono(row['answer_audio_path'])
    return waveform.unsqueeze(0), sample_rate

def clean_reference(text: str) -> str:
    # if '<...>' in text: return text.split('<...>')[-1]
    speaker_pattern = r'(?:P\d{4}A?:\s*)'
    heading_pattern = r"^\s*\*\*[^*]+\*\*\s*[:\-–—]?\s*"
    text = re.sub(speaker_pattern, '', str(text))
    text = re.sub(heading_pattern, '', text)
    if '   ' in text:text=text.split('   ')[-1]
    return re.sub(speaker_pattern, '', text)

def get_metric(metadata, fn, name):
    metrics = []
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc=f'measuring {name}'):
        metrics.append(fn(row))
    return metrics

def compute_wer(row):
    reference, hypothesis = row['transcript_answer'], row['ASR_transcript']
    if str(reference)=='nan': return np.nan
    reference = clean_reference(reference).strip('.? ')
    hypothesis = hypothesis.strip('.? ')
    w = wer(reference, hypothesis)
    if w>1:print(f"Warning: WER for file {row['answer_audio_path']} is greater than 1 (WER={w}).\nReference: '{reference}',\nHypothesis: '{hypothesis}'")
    return wer(reference, hypothesis)

def compute_cer(row):
    reference, hypothesis = row['transcript_answer'], row['ASR_transcript']
    if str(reference)=='nan': return np.nan
    reference = clean_reference(reference).strip('.? ')
    hypothesis = hypothesis.strip('.? ')
    return cer(reference, hypothesis)

def compute_latency(row):
    question_end_time = float(row['question_end_time'])
    answer_start_time = float(row.get('answer_start_time', 0.0))
    answer_vad_start_time = max(0.0, -answer_start_time)

    question_segments = vad_segments(row['audio_path'])
    answer_segments = vad_segments(row['answer_audio_path'], start_time=answer_vad_start_time)

    question_voice_end = question_segments[-1][1] if question_segments else question_end_time
    answer_voice_start = answer_segments[0][0] if answer_segments else 0.0

    question_end_delay = max(0.0, question_end_time - question_voice_end)
    answer_start_delay = max(0.0, answer_voice_start)
    latency_seconds = question_end_delay + max(0.0, answer_start_time) + answer_start_delay
    return 1000.0 * latency_seconds


def vad_segments(audio_path, start_time=0.0):
    waveform, sr = load_audio_mono(audio_path)
    if start_time > 0.0:
        waveform = waveform[int(round(start_time * sr)):]
    timestamps = get_speech_timestamps(
        waveform,
        get_vad_model(),
        sampling_rate=sr,
        return_seconds=True,
    )
    return [(float(item['start']), float(item['end'])) for item in timestamps]


def clip_segments(segments, start_time, end_time):
    clipped = []
    for start, end in segments:
        start = max(float(start), float(start_time))
        end = min(float(end), float(end_time))
        if end > start:
            clipped.append((start, end))
    return clipped


def overlap_duration(segment, segments):
    start, end = segment
    overlap = 0.0
    for other_start, other_end in segments:
        overlap += max(0.0, min(end, other_end) - max(start, other_start))
    return overlap


def compute_interruption_overlap(row):
    cache_key = row['answer_audio_path']
    if cache_key in interruption_cache:
        return interruption_cache[cache_key]

    question_end_time = float(row['question_end_time'])
    context_end_time = float(row.get('context_end_time', 0.0))
    answer_start_time = float(row.get('answer_start_time', 0.0))

    question_segments = clip_segments(vad_segments(row['audio_path']), context_end_time, question_end_time)
    question_segments = [(start - question_end_time, end - question_end_time) for start, end in question_segments]
    answer_segments = [
        (start + answer_start_time, end + answer_start_time)
        for start, end in vad_segments(row['answer_audio_path'])
    ]

    overlapping_answer_segments = 0
    total_overlap = 0.0
    for answer_segment in answer_segments:
        segment_overlap = overlap_duration(answer_segment, question_segments)
        if segment_overlap > 0.0:
            overlapping_answer_segments += 1
            total_overlap += segment_overlap

    result = {
        'interruption_segments': overlapping_answer_segments,
        'interrupted': 1000.0 * total_overlap,
        'interruptions': int(overlapping_answer_segments > 0),
    }
    interruption_cache[cache_key] = result
    return result


def compute_interruption_segments(row):
    return compute_interruption_overlap(row)['interruption_segments']


def compute_interruptions(row):
    return compute_interruption_overlap(row)['interruptions']


def compute_interrupted(row):
    return compute_interruption_overlap(row)['interrupted']

def compute_UTMOS(row):
    waveform_segment, sr = load_answer(row)
    try:
        device = get_utmos_device()
        waveform_segment = waveform_segment.to(device)
        with torch.inference_mode():
            score = get_utmos_model()(waveform_segment, sr)
        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().item()
    except Exception as exc:
        print(f"Warning: file {row['answer_audio_path']} failed UTMOS: {exc}")
        score = 'nan'
    return float(score)

def load_or_initialize_outputs(metadata, output_path):
    if output_path.exists():
        metrics = pd.read_csv(output_path)
        metrics = metrics.loc[:, ~metrics.columns.str.startswith("Unnamed:")]
        if "answer_audio_path" not in metrics.columns:
            metrics = pd.DataFrame({"answer_audio_path": metadata["answer_audio_path"]})
        else:
            metrics = pd.DataFrame({"answer_audio_path": metadata["answer_audio_path"]}).merge(metrics, on="answer_audio_path", how="left")
    else:
        metrics = pd.DataFrame({"answer_audio_path": metadata["answer_audio_path"]})
    return metrics


def save_outputs(metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def rows_missing_metric(metadata, metrics, metric_name):
    if metric_name not in metrics.columns:
        return metadata
    metric_by_audio = metrics.set_index("answer_audio_path")[metric_name]
    values = metadata["answer_audio_path"].map(metric_by_audio)
    return metadata[values.isna()]

def print_stats(metric_name, metrics):
    print(f"{metric_name}: {np.mean(metrics[metric_name]):.3f}(+-{np.std(metrics[metric_name]):.2f})")

def compute_and_save_metric(metadata, metrics, output_path, metric_name, fn, display_name=None, force_recompute=False):
    rows_to_compute = metadata if force_recompute else rows_missing_metric(metadata, metrics, metric_name)
    if rows_to_compute.empty:
        print(f"Skipping {metric_name}: already computed.")
        print_stats(metric_name, metrics)
        return metrics

    if force_recompute:
        print(f"Force recomputing {metric_name}.")

    values = get_metric(rows_to_compute, fn, display_name or metric_name)
    if metric_name not in metrics.columns:
        metrics[metric_name] = np.nan

    value_by_audio = pd.Series(values, index=rows_to_compute["answer_audio_path"]).groupby(level=0).last()
    rows_with_new_values = metrics["answer_audio_path"].isin(value_by_audio.index)
    metrics.loc[rows_with_new_values, metric_name] = metrics.loc[rows_with_new_values, "answer_audio_path"].map(value_by_audio)
    save_outputs(metrics, output_path)
    print_stats(metric_name, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", help="Directory containing audio files")
    parser.add_argument("--outputs", help="CSV to store the metrics")
    parser.add_argument("--asr-models", nargs="+", help="ASR models to compute WER/CER for (e.g., 'Qwen3-ASR-0.6B', 'whisper-large-v3')")
    parser.add_argument("--VAD-model", type=str, help="Model to use for VAD", default="silero_vad", choices=["silero_vad"])
    parser.add_argument("--UTMOS-model", type=str, help="Model to use for UTMOS", default="SpeechMOS/utmos22_strong", choices=["SpeechMOS/utmos22_strong"])
    parser.add_argument(
        "--force-recompute",
        nargs="*",
        default=[],
        help="Metric column names to recompute even if already present. Use all to recompute every metric.",
    )

    args = parser.parse_args()
    print(f"Evaluating metrics")
    force_recompute = set(args.force_recompute)
    force_all_metrics = "all" in force_recompute
    should_force_recompute = lambda metric_name: force_all_metrics or metric_name in force_recompute
    asr_models = {'Qwen3-ASR-0.6B':False, 'whisper-large-v3':False}
    asr_models = {model:False for model in args.asr_models}
    metadata = pd.read_csv(args.metadata)
    if 'answer_start_time' not in metadata.columns: metadata['answer_start_time'] = 0.0

    for asr_model in asr_models:
        transcript_file = Path(args.metadata).parent / f"{asr_model}_transcripts.csv"
        if os.path.exists(transcript_file):
            transcripts = pd.read_csv(transcript_file)
            transcripts.fillna('', inplace=True)
            if len(transcripts) == len(metadata): metadata[f'{asr_model}_transcript'] = transcripts['ASR_transcript_answer']
            else: metadata[f'{asr_model}_transcript'] = metadata['answer_audio_path'].map(transcripts.set_index('answer_audio_path')['ASR_transcript_answer']).fillna('')
            asr_models[asr_model] = True
    output_path = Path(args.outputs)
    metrics = load_or_initialize_outputs(metadata, output_path)
    save_outputs(metrics, output_path)

    for asr_model in asr_models:
        if asr_models[asr_model]:
            metadata_with_asr = metadata[metadata[f'{asr_model}_transcript']!='']
            L = len(metadata_with_asr)
            print(f"Found {L}/{len(metadata)} lines with transcripts from model {asr_model}.")
            metadata_with_asr['ASR_transcript']=metadata_with_asr[f'{asr_model}_transcript']
            metric_name = f'WER_{asr_model}'
            metrics = compute_and_save_metric(metadata_with_asr, metrics, output_path, metric_name, compute_wer, metric_name, should_force_recompute('WER'))
            metric_name = f'CER_{asr_model}'
            metrics = compute_and_save_metric(metadata_with_asr, metrics, output_path, metric_name, compute_cer, metric_name, should_force_recompute('CER'))
        else:print(f"Warning: no ASR outputs computed for model {asr_model}, WER/CER will not be measured.")

    metrics = compute_and_save_metric(metadata, metrics, output_path, 'latency', compute_latency, 'latency (ms)', should_force_recompute('latency'))
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'interruption_segments', compute_interruption_segments, 'answer segments overlapping question speech', should_force_recompute('interrupt'))
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'interrupted', compute_interrupted, 'interrupted time (ms)', should_force_recompute('interrupt'))
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'interruptions', compute_interruptions, 'dialogues with interruption', should_force_recompute('interrupt'))
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'UTMOS', compute_UTMOS, 'UTMOS', should_force_recompute('UTMOS'))
