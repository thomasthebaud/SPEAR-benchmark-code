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

utmos_model = torch.hub.load(
    "tarepan/SpeechMOS:v1.2.0",
    "utmos22_strong",
    trust_repo=True
)

model = load_silero_vad()

def load_answer(row):
    waveform, sample_rate = torchaudio.load(row['audio_path'])
    waveform_segment = waveform[:, int(row['question_end_time'] * sample_rate):]
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
    model,
    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    if len(speech_timestamps)>0:return 1000*float(speech_timestamps[0]['start'])
    else: return 0

def compute_UTMOS(row):
    waveform_segment, sr = load_answer(row)

    with torch.no_grad():
        score = utmos_model(waveform_segment, sr)
    return float(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", help="Directory containing audio files")
    parser.add_argument("--outputs", help="CSV to store the metrics")

    args = parser.parse_args()
    print(f"Evaluating metrics")

    metadata = pd.read_csv(args.metadata)
    
    if 'ASR_transcript' not in metadata.columns: print(f"Warning: no ASR outputs computed, WER/CER will not be measured.")
    L = len(metadata)
    metadata = metadata.dropna(subset='ASR_transcript')
    print(f"Found {L-len(metadata)}/{L} lines without ASR transcripts.")


    metrics = {'id':[], 'CER':[], 'WER':[], 'UTMOS':[],'latency':[]}

    WER = get_metric(metadata, compute_wer, 'WER')
    CER = get_metric(metadata, compute_cer, 'CER')
    latency = get_metric(metadata, compute_latency, 'latency (ms)')
    UTMOS = get_metric(metadata, compute_UTMOS, 'UTMOS')

    metrics = pd.DataFrame({'audio_path':metadata['audio_path'], 'CER':CER, 'WER':WER, 'UTMOS':UTMOS,'latency':latency})
    metrics.to_csv(args.outputs)