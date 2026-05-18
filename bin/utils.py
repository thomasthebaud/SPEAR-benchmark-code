import soundfile as sf
import torch


def load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), sr

def load_audios(audio_info) -> dict:
    audio1_waveform, audio1_sr = load(audio_info['audio1'])
    audio2_waveform, audio2_sr = load(audio_info['audio2'])
    audio_waveform = audio1_waveform + audio2_waveform
    audio_waveform = audio_waveform[:, int(audio_info['start'] * audio1_sr):int(audio_info['end'] * audio1_sr)]
    assert audio1_sr == audio2_sr, "Sample rates of the two audio files must be the same."
    return audio_waveform, audio1_sr
