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
vad_model = None

def get_utmos_model():
    global utmos_model
    if utmos_model is None:
        utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        )
    return utmos_model


def get_vad_model():
    global vad_model
    if vad_model is None:
        vad_model = load_silero_vad()
    return vad_model

def load_answer(row):
    waveform, sample_rate = torchaudio.load(row['audio_path'])
    start = min(int(row['question_end_time'] * sample_rate), int(row['total_duration'] * sample_rate))
    waveform_segment = waveform[:, start:]
    if waveform_segment.shape[0]>1: waveform_segment = waveform_segment.mean(dim=0).unsqueeze(0)
    return waveform_segment, sample_rate

def clean_reference(text: str) -> str:
    if '<...>' in text: return text.split('<...>')[-1]
    pattern = r'(P\d{4}A?:\s*)'
    utterances = re.split(pattern, text)
    spks, utts = [s for idx, s in enumerate(utterances) if idx%2==1], [s for idx, s in enumerate(utterances) if idx%2==0]
    tgt_spk = spks[-1]
    # print(utts, spks)
    utts_ = []
    for spk, utt in zip(spks[::-1], utts[::-1]):
        # print(spk, utt)
        if spk==tgt_spk: utts_.append(utt)
        else:break
    return ' '.join(utts_[::-1])

def get_metric(metadata, fn, name):
    metrics = []
    for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc=f'measuring {name}'):
        metrics.append(fn(row))
    print(f"{name}: {np.mean(metrics)}(+-{np.std(metrics)})")
    return metrics

def compute_wer(row):
    reference, hypothesis = row['transcript'], row['ASR_transcript']
    reference = clean_reference(reference).strip('.? ')
    hypothesis = hypothesis.strip('.? ')
    # print('REF:', reference)
    # print('HYP:', hypothesis)
    # print(wer(reference, hypothesis))
    return wer(reference, hypothesis)

def compute_cer(row):
    reference, hypothesis = row['transcript'], row['ASR_transcript']
    reference = clean_reference(reference).strip('.? ')
    hypothesis = hypothesis.strip('.? ')
    return cer(reference, hypothesis)

def compute_latency(row):
    waveform_segment, sr = load_answer(row)
    speech_timestamps = get_speech_timestamps(
    waveform_segment,
    get_vad_model(),
    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    if len(speech_timestamps)>0:return 1000*float(speech_timestamps[0]['start'])
    else: return 0

def compute_UTMOS(row):
    waveform_segment, sr = load_answer(row)
    try:
        with torch.no_grad():
            score = get_utmos_model()(waveform_segment, sr)
    except:
        print(f"Warning: file {row['audio_path']} failed UTMOS")
        score = 'nan'
    return float(score)

def load_or_initialize_outputs(metadata, output_path):
    if output_path.exists():
        metrics = pd.read_csv(output_path)
        metrics = metrics.loc[:, ~metrics.columns.str.startswith("Unnamed:")]
        if "audio_path" not in metrics.columns:
            metrics = pd.DataFrame({"audio_path": metadata["audio_path"]})
        else:
            metrics = pd.DataFrame({"audio_path": metadata["audio_path"]}).merge(metrics, on="audio_path", how="left")
    else:
        metrics = pd.DataFrame({"audio_path": metadata["audio_path"]})
    return metrics


def save_outputs(metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def rows_missing_metric(metadata, metrics, metric_name):
    if metric_name not in metrics.columns:
        return metadata
    metric_by_audio = metrics.set_index("audio_path")[metric_name]
    values = metadata["audio_path"].map(metric_by_audio)
    return metadata[values.isna()]


def compute_and_save_metric(metadata, metrics, output_path, metric_name, fn, display_name=None):
    missing = rows_missing_metric(metadata, metrics, metric_name)
    if missing.empty:
        print(f"Skipping {metric_name}: already computed.")
        return metrics

    values = get_metric(missing, fn, display_name or metric_name)
    if metric_name not in metrics.columns:
        metrics[metric_name] = np.nan

    value_by_audio = pd.Series(values, index=missing["audio_path"])
    metrics[metric_name] = metrics["audio_path"].map(value_by_audio).combine_first(metrics[metric_name])
    save_outputs(metrics, output_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", help="Directory containing audio files")
    parser.add_argument("--outputs", help="CSV to store the metrics")

    args = parser.parse_args()
    print(f"Evaluating metrics")

    metadata = pd.read_csv(args.metadata)
    output_path = Path(args.outputs)
    metrics = load_or_initialize_outputs(metadata, output_path)
    save_outputs(metrics, output_path)

    if 'ASR_transcript' not in metadata.columns:
        print(f"Warning: no ASR outputs computed, WER/CER will not be measured.")
        metadata_with_asr = metadata.iloc[0:0].copy()
    else:
        L = len(metadata)
        metadata_with_asr = metadata.dropna(subset='ASR_transcript')
        print(f"Found {L-len(metadata_with_asr)}/{L} lines without ASR transcripts.")

    if not metadata_with_asr.empty:
        metrics = compute_and_save_metric(metadata_with_asr, metrics, output_path, 'WER', compute_wer, 'WER')
        metrics = compute_and_save_metric(metadata_with_asr, metrics, output_path, 'CER', compute_cer, 'CER')

    metrics = compute_and_save_metric(metadata, metrics, output_path, 'latency', compute_latency, 'latency (ms)')
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'UTMOS', compute_UTMOS, 'UTMOS')
