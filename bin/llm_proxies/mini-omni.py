import io
import os
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional
import torch

"""
Adapted from the github https://github.com/gpt-omni/mini-omni/blob/main/server.py 
So the inference is more straighforward and doesn't require running a separate server process. 
This is used as a direct in-process proxy for run_LLM_inference.py, 
and is not intended to be a general-purpose Mini-Omni client implementation.
"""
NON_STREAMING_ANSWER_START_S = 0.0
MIN_WAV_BYTES = 2000  # roughly >60 ms at 16 kHz mono 16-bit PCM
MINI_OMNI_ROOT = Path(__file__).resolve().parent / "mini-omni-utils"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[2] / "models" / "mini-omni-checkpoint"
DEFAULT_STREAM_STRIDE = 4
DEFAULT_MAX_TOKENS = 2048
OUTPUT_SAMPLE_RATE = 24000

_CLIENT = None
_CLIENT_CONFIG: tuple[str, str] | None = None
_CLIENT_LOCK = threading.Lock()


def convert_to_wav_strict(input_path: Path) -> Path:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(out.name)
    out.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(input_path),
        "-map_metadata", "-1",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def is_wav_long_enough(wav_path: Path, min_bytes: int = MIN_WAV_BYTES) -> bool:
    return wav_path.stat().st_size >= min_bytes


def get_default_device() -> str:
    return os.environ.get("MINI_OMNI_DEVICE", "cuda:0")


def get_checkpoint_dir() -> Path:
    return Path(os.environ.get("MINI_OMNI_CKPT_DIR", str(DEFAULT_CKPT_DIR)))


def import_omni_inference():
    inference_path = MINI_OMNI_ROOT / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(
            f"Mini-Omni inference.py not found at {inference_path}. "
            "The vendored Mini-Omni runtime files should live under bin/llm_proxies/mini-omni-utils."
        )
    root = str(MINI_OMNI_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Mini-Omni ships a patched litgpt package with audio generation helpers
    # that are not present in the pip litgpt release. Make sure the vendored
    # package is imported even if another litgpt was imported earlier.
    for name in list(sys.modules):
        if name == "litgpt" or name.startswith("litgpt."):
            del sys.modules[name]

    import importlib.util

    spec = importlib.util.spec_from_file_location("mini_omni_vendored_inference", inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Mini-Omni inference module from {inference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OmniInference


def load_mini_omni_model(ckpt_dir: Optional[Path] = None, device: Optional[str] = None, warm_up: Optional[bool] = None):
    """Load and cache a Mini-Omni OmniInference instance."""
    global _CLIENT, _CLIENT_CONFIG

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else get_checkpoint_dir()
    device = device or get_default_device()
    config = (str(ckpt_dir), str(device))
    if _CLIENT is not None and _CLIENT_CONFIG == config:
        return _CLIENT

    OmniInference = import_omni_inference()
    print(f"[mini-omni] Loading model from {ckpt_dir} on {device}", flush=True)
    client = OmniInference(str(ckpt_dir), device)

    if warm_up is None:
        warm_up = os.environ.get("MINI_OMNI_WARM_UP", "0") == "1"
    if warm_up:
        sample = MINI_OMNI_ROOT / "data" / "samples" / "output1.wav"
        if sample.exists():
            print(f"[mini-omni] Warming up with {sample}", flush=True)
            client.warm_up(str(sample))
        else:
            print(f"[mini-omni] Warm-up sample not found: {sample}; skipping warm-up.", flush=True)

    _CLIENT = client
    _CLIENT_CONFIG = config
    return _CLIENT


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def clear_mini_omni_runtime(client) -> None:
    try:
        client.model.clear_kv_cache()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def maybe_synchronize_cuda() -> None:
    if os.environ.get("MINI_OMNI_SYNC_CUDA", "0") != "1":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def infer_one_audio(
    audio_path: Path,
    *,
    stream_stride: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.9,
    top_k: int = 1,
    top_p: float = 1.0,
) -> tuple[bytes, str]:
    """Run Mini-Omni audio-to-audio inference for one audio file and return a complete WAV."""
    max_tokens = int(os.environ.get("MINI_OMNI_MAX_TOKENS", max_tokens or DEFAULT_MAX_TOKENS))
    client = load_mini_omni_model()

    wav_path = convert_to_wav_strict(audio_path)
    try:
        if not is_wav_long_enough(wav_path):
            raise ValueError(f"Input audio is too short after WAV conversion: {audio_path}")
        try:
            with _CLIENT_LOCK:
                maybe_synchronize_cuda()
                pcm_audio, transcript = client.run_AA_nonstream(
                    str(wav_path),
                    max_returned_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                maybe_synchronize_cuda()
        except Exception as exc:
            clear_mini_omni_runtime(client)
            raise RuntimeError(f"Mini-Omni inference failed for {audio_path}: {exc}") from exc
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not pcm_audio:
        raise RuntimeError(f"Mini-Omni produced no audio for {audio_path}")
    return pcm16_to_wav_bytes(pcm_audio), transcript


def get_reply_with_audio(
    audio_path: Path,
    instruction: str,
    model_name: str,
    org: str | None,
    api_key: str | None,
    temp: float = 0.7,
):
    """Direct in-process Mini-Omni proxy compatible with run_LLM_inference.py."""

    try:
        audio_bytes_out, transcript = infer_one_audio(audio_path, temperature=temp)
    except Exception as exc:
        if os.environ.get("MINI_OMNI_RAISE_ERRORS", "0") == "1":
            raise
        return None, None, str(exc), False, None

    finish_reason = "direct_nonstream_inference"
    return audio_bytes_out, transcript, finish_reason, True, NON_STREAMING_ANSWER_START_S
