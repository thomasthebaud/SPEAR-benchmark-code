from __future__ import annotations

import io
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


MODEL_REPO = "Qwen/Qwen2.5-Omni-7B"
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_SPEAKER = "Ethan"
USE_AUDIO_IN_VIDEO = True
MAX_NEW_TOKENS = 8192

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_MODEL_BUNDLE = None
_MODEL_LOCK = threading.Lock()


def _resolve_model_name(model_name: str) -> str:
    if model_name and "/" in model_name:
        return model_name
    return MODEL_REPO


def _get_audio_duration_s(audio_path: Path) -> float:
    try:
        info = sf.info(audio_path)
        if info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    return 0.0


def _load_model_bundle(model_name: str) -> dict:
    global _MODEL_BUNDLE

    resolved_model_name = _resolve_model_name(model_name)

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None and _MODEL_BUNDLE["model_name"] == resolved_model_name:
            return _MODEL_BUNDLE

        try:
            import torch
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-Omni requires a new-enough Transformers install plus torch. "
                "The model card recommends `pip install git+https://github.com/huggingface/transformers`, "
                "`pip install accelerate`, and `pip install qwen-omni-utils -U`."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2.5-Omni proxy requires CUDA for inference, but torch.cuda.is_available() is False.")

        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "enable_audio_output": True,
        }

        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                resolved_model_name,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception as exc:
            if "flash" not in str(exc).lower():
                raise
            print(
                "Warning: failed to load Qwen2.5-Omni with flash_attention_2; "
                "falling back to the default attention implementation.",
                flush=True,
            )
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                resolved_model_name,
                **model_kwargs,
            )

        model.eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(resolved_model_name)

        _MODEL_BUNDLE = {
            "model_name": resolved_model_name,
            "model": model,
            "processor": processor,
        }
        return _MODEL_BUNDLE


def _model_input_device(model):
    thinker = getattr(model, "thinker", None)
    if thinker is not None and hasattr(thinker, "device"):
        return thinker.device
    return model.device


def _prepare_inputs(processor, model, audio_path: Path, instruction: str):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": DEFAULT_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
            ],
        },
    ]

    if instruction and instruction != "None":
        messages[1]["content"].append({"type": "text", "text": instruction})

    inputs = processor.apply_chat_template(
        messages,
        load_audio_from_video=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    return inputs.to(_model_input_device(model))


def _decode_response_text(processor, text_ids, prompt_token_count: int) -> str:
    sequences = getattr(text_ids, "sequences", text_ids)
    if sequences.shape[1] > prompt_token_count:
        sequences = sequences[:, prompt_token_count:]

    decoded = processor.batch_decode(
        sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def _audio_tensor_to_wav_bytes(audio) -> Optional[bytes]:
    if audio is None:
        return None

    samples = audio.reshape(-1).detach().float().cpu().numpy()
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32)

    if samples.size == 0:
        return None

    buffer = io.BytesIO()
    sf.write(buffer, samples, OUTPUT_SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def _run_generate_with_streaming(
    model,
    processor,
    inputs,
    temp: float,
) -> Tuple[object, object, str, Optional[float]]:
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=1.0,
    )
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "speaker": DEFAULT_SPEAKER,
        "return_audio": True,
        "thinker_max_new_tokens": MAX_NEW_TOKENS,
        "thinker_do_sample": temp > 0,
        "thinker_temperature": max(float(temp), 1e-5),
        "thinker_top_p": 0.95,
        "thinker_top_k": 20,
        "talker_do_sample": False,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO,
    }

    def target() -> None:
        try:
            result_queue.put(model.generate(**generation_kwargs))
        except Exception as exc:
            if "speaker" not in str(exc):
                streamer.end()
                result_queue.put(exc)
                return

            fallback_kwargs = dict(generation_kwargs)
            fallback_kwargs.pop("speaker", None)
            try:
                result_queue.put(model.generate(**fallback_kwargs))
            except Exception as fallback_exc:
                streamer.end()
                result_queue.put(fallback_exc)

    start_time = time.monotonic()
    first_text_delta_s = None
    text_fragments = []
    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    while thread.is_alive():
        try:
            fragment = next(streamer)
        except queue.Empty:
            continue
        except StopIteration:
            break

        if fragment:
            if first_text_delta_s is None:
                first_text_delta_s = time.monotonic() - start_time
            text_fragments.append(fragment)

    while True:
        try:
            fragment = next(streamer)
        except (queue.Empty, StopIteration):
            break

        if fragment:
            if first_text_delta_s is None:
                first_text_delta_s = time.monotonic() - start_time
            text_fragments.append(fragment)

    thread.join()
    result = result_queue.get()

    if isinstance(result, Exception):
        raise result

    text_ids, audio = result
    return text_ids, audio, "".join(text_fragments).strip(), first_text_delta_s


def get_reply_with_audio(
    audio_path: Path,
    instruction: str,
    model_name: str,
    org: str,
    api_key: str,
    temp: float = 0.7,
) -> Tuple[Optional[bytes], str, Optional[str], bool, Optional[float]]:
    """
    SPEARBench-compatible proxy for Qwen2.5-Omni-7B.

    The model runs locally on CUDA. Text output is streamed with
    TextIteratorStreamer while the final generated speech is returned as WAV
    bytes, matching the benchmark runner's proxy contract.
    """
    del org, api_key

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return None, "", f"Audio file does not exist: {audio_path}", False, None

    try:
        bundle = _load_model_bundle(model_name)
        model = bundle["model"]
        processor = bundle["processor"]
        inputs = _prepare_inputs(processor, model, audio_path, instruction)
        prompt_token_count = inputs["input_ids"].shape[1]
        input_duration_s = _get_audio_duration_s(audio_path)

        text_ids, audio, streamed_text, first_text_delta_s = _run_generate_with_streaming(
            model,
            processor,
            inputs,
            temp,
        )

        transcript = _decode_response_text(processor, text_ids, prompt_token_count)
        if not transcript:
            transcript = streamed_text

        wav_bytes = _audio_tensor_to_wav_bytes(audio)
        if wav_bytes is None:
            return None, transcript, "no_audio_output", False, None

        answer_start_s = None
        if first_text_delta_s is not None:
            answer_start_s = input_duration_s + first_text_delta_s

        return wav_bytes, transcript, "completed", True, answer_start_s

    except Exception as exc:
        return None, "", f"error: {exc}", False, None
