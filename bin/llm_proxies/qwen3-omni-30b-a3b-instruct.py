from __future__ import annotations

import io
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


MODEL_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_SPEAKER = "Ethan"
USE_AUDIO_IN_VIDEO = True
MAX_NEW_TOKENS = 8192

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Respond naturally and concisely. "
    "Your output should be only the spoken content that the user should hear."
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
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3-Omni requires a source/new-enough Transformers install plus torch. "
                "The model card recommends `pip install git+https://github.com/huggingface/transformers`, "
                "`pip install accelerate`, and `pip install qwen-omni-utils -U`."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-Omni proxy requires CUDA for inference, but torch.cuda.is_available() is False.")

        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
        }

        try:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                resolved_model_name,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception as exc:
            if "flash" not in str(exc).lower():
                raise
            print(
                "Warning: failed to load Qwen3-Omni with flash_attention_2; "
                "falling back to the default attention implementation.",
                flush=True,
            )
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                resolved_model_name,
                **model_kwargs,
            )

        model.eval()
        processor = Qwen3OmniMoeProcessor.from_pretrained(resolved_model_name)

        _MODEL_BUNDLE = {
            "model_name": resolved_model_name,
            "model": model,
            "processor": processor,
        }
        return _MODEL_BUNDLE


def _prepare_inputs(processor, model, audio_path: Path, instruction: str):
    from qwen_omni_utils import process_mm_info

    messages = [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
            ],
        },
    ]

    if instruction and instruction != "None":
        messages[1]["content"].append({"type": "text", "text": instruction})

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    return inputs.to(model.device).to(model.dtype)


def _decode_response_text(processor, text_ids, prompt_token_count: int) -> str:
    sequences = getattr(text_ids, "sequences", text_ids)
    decoded = processor.batch_decode(
        sequences[:, prompt_token_count:],
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
        "thinker_return_dict_in_generate": True,
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
        except TypeError as exc:
            if "speaker" not in str(exc):
                streamer.end()
                result_queue.put(exc)
                return

            fallback_kwargs = dict(generation_kwargs)
            fallback_kwargs.pop("speaker", None)
            fallback_kwargs["spk"] = DEFAULT_SPEAKER
            try:
                result_queue.put(model.generate(**fallback_kwargs))
            except Exception as fallback_exc:
                streamer.end()
                result_queue.put(fallback_exc)
        except Exception as exc:
            streamer.end()
            result_queue.put(exc)

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
    SPEARBench-compatible proxy for Qwen3-Omni-30B-A3B-Instruct.

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
