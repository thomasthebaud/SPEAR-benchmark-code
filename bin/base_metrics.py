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
    waveform, sample_rate = torchaudio.load(row['answer_audio_path'])
    if waveform.shape[0]>1: waveform = waveform.mean(dim=0).unsqueeze(0)
    return waveform, sample_rate

def clean_reference(text: str) -> str:
    # if '<...>' in text: return text.split('<...>')[-1]
    pattern = r'(P\d{4}A?:\s*)'
    utterances = re.split(pattern, text)
    # spks, utts = [s for idx, s in enumerate(utterances) if idx%2==1], [s for idx, s in enumerate(utterances) if idx%2==0]
    # tgt_spk = spks[-1]
    # # print(utts, spks)
    # utts_ = []
    # for spk, utt in zip(spks[::-1], utts[::-1]):
    #     # print(spk, utt)
    #     if spk==tgt_spk: utts_.append(utt)
    #     else:break
    return ' '.join(utterances)

def get_metric(metadata, fn, name):
    metrics = []
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc=f'measuring {name}'):
        metrics.append(fn(row))
    return metrics

def compute_wer(row):
    reference, hypothesis = row['transcript_answer'], row['ASR_transcript']
    reference = clean_reference(reference).strip('.? ')
    hypothesis = hypothesis.strip('.? ')
    return wer(reference, hypothesis)

def compute_cer(row):
    reference, hypothesis = row['transcript_answer'], row['ASR_transcript']
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

def compute_interruptions(row):return int(row['answer_start_time']<0.0)

def compute_interrupted(row):return 1000*float(-row['answer_start_time']) if row['answer_start_time']<0.0 else 0.0

def compute_UTMOS(row):
    waveform_segment, sr = load_answer(row)
    try:
        with torch.no_grad():
            score = get_utmos_model()(waveform_segment, sr)
    except:
        print(f"Warning: file {row['answer_audio_path']} failed UTMOS")
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

def compute_and_save_metric(metadata, metrics, output_path, metric_name, fn, display_name=None):
    missing = rows_missing_metric(metadata, metrics, metric_name)
    if missing.empty:
        print(f"Skipping {metric_name}: already computed.")
        print_stats(metric_name, metrics)
        return metrics

    values = get_metric(missing, fn, display_name or metric_name)
    if metric_name not in metrics.columns:
        metrics[metric_name] = np.nan

    value_by_audio = pd.Series(values, index=missing["answer_audio_path"])
    metrics[metric_name] = metrics["answer_audio_path"].map(value_by_audio).combine_first(metrics[metric_name])
    save_outputs(metrics, output_path)
    print_stats(metric_name, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", help="Directory containing audio files")
    parser.add_argument("--outputs", help="CSV to store the metrics")

    args = parser.parse_args()
    print(f"Evaluating metrics")
    asr_models = {'Qwen3-ASR-0.6B':False, 'whisper-large-v3':False}
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
            print(f"Found {L}/{len(metadata)} lines without transcripts from model {asr_model}.")
            metadata_with_asr['ASR_transcript']=metadata_with_asr[f'{asr_model}_transcript']
            metrics = compute_and_save_metric(metadata_with_asr, metrics, output_path, 'WER', compute_wer, f'WER_{asr_model}')
            metrics = compute_and_save_metric(metadata_with_asr, metrics, output_path, 'CER', compute_cer, f'CER_{asr_model}')
        else:print(f"Warning: no ASR outputs computed for model {asr_model}, WER/CER will not be measured.")

    metrics = compute_and_save_metric(metadata, metrics, output_path, 'latency', compute_latency, 'latency (ms)')
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'interruptions', compute_interruptions, 'interruptions')
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'interrupted', compute_interrupted, 'interrupted time (ms)')
    metrics = compute_and_save_metric(metadata, metrics, output_path, 'UTMOS', compute_UTMOS, 'UTMOS')
