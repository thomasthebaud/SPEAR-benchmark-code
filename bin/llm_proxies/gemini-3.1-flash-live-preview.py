import base64
import io
import json
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import quote

import numpy as np
import soundfile as sf

try:
    import websocket
except ImportError as exc:
    raise ImportError(
        "The Gemini native audio proxy requires websocket-client. "
        "Install it with `pip install websocket-client`."
    ) from exc


GEMINI_LIVE_URL_TEMPLATE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    "?key={api_key}"
)

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
MODEL_ALIASES = {
    "gemini-3.1-flash-live-preview": DEFAULT_MODEL,
}

INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_MS = 40
BYTES_PER_SAMPLE = 2
CHANNELS = 1

POST_STREAM_WAIT_S = 30.0
SETUP_TIMEOUT_S = 10.0
QUIET_AFTER_TURN_COMPLETE_S = 1.0

DEFAULT_INSTRUCTIONS = (
    "You are participating in a natural spoken conversation.\n"
    "The user audio is being streamed to you in real time from a file.\n"
    "Answer when it feels natural, including before the entire audio file has finished if appropriate.\n"
    "Keep responses conversational and concise."
)


def _resample_mono_audio(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int = INPUT_RATE,
) -> np.ndarray:
    """Convert input audio to mono float32 at target_rate."""
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    samples = samples.astype(np.float32, copy=False)

    if samples.size == 0 or source_rate == target_rate:
        return samples

    output_length = max(1, round(samples.shape[0] * target_rate / source_rate))
    target_positions = np.linspace(0, samples.shape[0] - 1, num=output_length)

    return np.interp(
        target_positions,
        np.arange(samples.shape[0]),
        samples,
    ).astype(np.float32)


def _read_audio_as_pcm16(audio_path: Path) -> bytes:
    """Load an audio file and return mono PCM16 little-endian bytes at 16 kHz."""
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = _resample_mono_audio(audio, sample_rate, INPUT_RATE)
    mono = np.clip(mono, -1.0, 1.0)
    return (mono * 32767.0).astype("<i2").tobytes()


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = OUTPUT_RATE) -> bytes:
    """Wrap raw PCM16 bytes into a WAV container."""
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return buffer.getvalue()


def _json_send(ws: websocket.WebSocket, payload: Dict) -> None:
    ws.send(json.dumps(payload))


def _parse_server_messages(raw) -> Iterable[Dict]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    raw = raw.strip()
    if not raw:
        return

    try:
        yield json.loads(raw)
        return
    except json.JSONDecodeError:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _connect(api_key: str) -> websocket.WebSocket:
    url = GEMINI_LIVE_URL_TEMPLATE.format(api_key=quote(api_key, safe=""))
    return websocket.create_connection(url)


def _resolve_model_name(model_name: str) -> str:
    if not model_name:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(model_name, model_name)


def _configure_session(
    ws: websocket.WebSocket,
    model_name: str,
    session_instructions: str,
    temp: float,
) -> None:
    _json_send(
        ws,
        {
            "setup": {
                "model": f"models/{_resolve_model_name(model_name)}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "temperature": temp,
                },
                "systemInstruction": {
                    "parts": [{"text": session_instructions}],
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": True,
                    },
                    "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                    "turnCoverage": "TURN_INCLUDES_ONLY_ACTIVITY",
                },
            },
        },
    )


def _receiver_loop(
    ws: websocket.WebSocket,
    audio_fragments: list,
    text_fragments: list,
    event_log: list,
    error_queue: queue.Queue,
    stop_event: threading.Event,
    setup_done_event: threading.Event,
    turn_done_event: threading.Event,
    timing_state: dict,
) -> None:
    ws.settimeout(1.0)

    while not stop_event.is_set():
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as exc:
            if not stop_event.is_set():
                error_queue.put(exc)
            return

        for event in _parse_server_messages(raw):
            if "setupComplete" in event:
                event_log.append("setupComplete")
                setup_done_event.set()
                continue

            if "goAway" in event:
                event_log.append("goAway")

            if "toolCall" in event:
                event_log.append("toolCall")

            server_content = event.get("serverContent")
            if not server_content:
                continue

            if server_content.get("generationComplete"):
                event_log.append("generationComplete")

            if server_content.get("turnComplete"):
                event_log.append("turnComplete")
                turn_done_event.set()

            if server_content.get("interrupted"):
                event_log.append("interrupted")

            output_transcription = server_content.get("outputTranscription")
            if output_transcription and output_transcription.get("text"):
                text_fragments.append(output_transcription["text"])

            model_turn = server_content.get("modelTurn") or {}
            for part in model_turn.get("parts", []):
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    if timing_state["answer_start_s"] is None:
                        input_start_s = timing_state.get("input_stream_start_s")
                        if input_start_s is not None:
                            timing_state["answer_start_s"] = max(
                                0.0,
                                time.monotonic() - input_start_s,
                            )
                    audio_fragments.append(inline_data["data"])

                text = part.get("text")
                if text:
                    text_fragments.append(text)


def _wait_for_setup(
    setup_done_event: threading.Event,
    error_queue: queue.Queue,
    timeout_s: float = SETUP_TIMEOUT_S,
) -> None:
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if not error_queue.empty():
            raise error_queue.get()

        if setup_done_event.is_set():
            return

        time.sleep(0.05)

    raise TimeoutError("Timed out waiting for Gemini setupComplete.")


def _stream_audio(ws: websocket.WebSocket, pcm_bytes: bytes, timing_state: dict) -> None:
    chunk_size = int(INPUT_RATE * BYTES_PER_SAMPLE * (CHUNK_MS / 1000.0))

    _json_send(ws, {"realtimeInput": {"activityStart": {}}})
    timing_state["input_stream_start_s"] = time.monotonic()

    for offset in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[offset : offset + chunk_size]
        _json_send(
            ws,
            {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(chunk).decode("ascii"),
                        "mimeType": f"audio/pcm;rate={INPUT_RATE}",
                    },
                },
            },
        )
        time.sleep(CHUNK_MS / 1000.0)

    _json_send(ws, {"realtimeInput": {"activityEnd": {}}})


def get_reply_with_audio(
    audio_path: Path,
    instruction: str,
    model_name: str,
    org: Optional[str],
    api_key: str,
    temp: float = 0.7,
) -> Tuple[Optional[bytes], str, Optional[str], bool, Optional[float]]:
    """
    SPEARBench-compatible Gemini Live native-audio proxy.

    The benchmark runner still passes the key through --openai-api-key; for this
    proxy that value should be the Gemini API key from openai_keys.sh.
    """
    del org  # Gemini API keys are project-scoped; no OpenAI organization header.

    if not api_key or api_key.strip().lower() == "will be added later":
        raise ValueError(
            "Gemini API key is required. Source openai_keys.sh and pass "
            "$gemini_api_key as --openai-api-key for this proxy."
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return None, "", f"Audio file does not exist: {audio_path}", False, None

    try:
        pcm_bytes = _read_audio_as_pcm16(audio_path)
    except Exception as exc:
        return None, "", f"failed_to_read_audio: {exc}", False, None

    if not pcm_bytes:
        return None, "", "empty_audio", False, None

    session_instructions = DEFAULT_INSTRUCTIONS
    if instruction and instruction != "None":
        session_instructions = f"{DEFAULT_INSTRUCTIONS}\n\n{instruction}"

    ws = None
    receiver = None

    stop_event = threading.Event()
    setup_done_event = threading.Event()
    turn_done_event = threading.Event()
    error_queue: queue.Queue = queue.Queue()

    audio_fragments = []
    text_fragments = []
    event_log = []
    timing_state = {
        "input_stream_start_s": None,
        "answer_start_s": None,
    }

    try:
        ws = _connect(api_key)
        receiver = threading.Thread(
            target=_receiver_loop,
            args=(
                ws,
                audio_fragments,
                text_fragments,
                event_log,
                error_queue,
                stop_event,
                setup_done_event,
                turn_done_event,
                timing_state,
            ),
            daemon=True,
        )
        receiver.start()

        _configure_session(ws, model_name or DEFAULT_MODEL, session_instructions, temp)
        _wait_for_setup(setup_done_event, error_queue)
        _stream_audio(ws, pcm_bytes, timing_state)

        deadline = time.time() + POST_STREAM_WAIT_S
        last_audio_count = len(audio_fragments)
        last_text_count = len(text_fragments)
        last_activity_time = time.time()

        while time.time() < deadline:
            if not error_queue.empty():
                raise error_queue.get()

            current_audio_count = len(audio_fragments)
            current_text_count = len(text_fragments)
            if current_audio_count != last_audio_count or current_text_count != last_text_count:
                last_audio_count = current_audio_count
                last_text_count = current_text_count
                last_activity_time = time.time()

            if audio_fragments and turn_done_event.is_set():
                if time.time() - last_activity_time > QUIET_AFTER_TURN_COMPLETE_S:
                    break

            time.sleep(0.05)

        transcript = "".join(text_fragments).strip()

        if not audio_fragments:
            debug_tail = ", ".join(str(evt) for evt in event_log[-30:])
            return (
                None,
                transcript,
                f"no_audio_fragments; recent_events=[{debug_tail}]",
                False,
                None,
            )

        try:
            raw_audio = b"".join(
                base64.b64decode(fragment)
                for fragment in audio_fragments
            )
        except Exception as exc:
            return None, transcript, f"failed_to_decode_audio: {exc}", False, None

        if not raw_audio:
            debug_tail = ", ".join(str(evt) for evt in event_log[-30:])
            return (
                None,
                transcript,
                f"empty_decoded_audio; recent_events=[{debug_tail}]",
                False,
                None,
            )

        wav_bytes = _pcm16_to_wav_bytes(raw_audio, OUTPUT_RATE)
        finish_reason = "completed" if turn_done_event.is_set() else "timeout"

        return wav_bytes, transcript, finish_reason, True, timing_state["answer_start_s"]

    except Exception as exc:
        transcript = "".join(text_fragments).strip()
        return None, transcript, f"error: {exc}", False, None

    finally:
        stop_event.set()

        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        if receiver is not None:
            try:
                receiver.join(timeout=2.0)
            except Exception:
                pass
