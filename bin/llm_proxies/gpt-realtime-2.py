import base64
import io
import json
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import websocket
except ImportError as exc:
    raise ImportError(
        "The gpt-realtime-2 proxy requires websocket-client. "
        "Install it in the SPEARBench conda env with `pip install websocket-client`."
    ) from exc

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"
TARGET_RATE = 24000
CHUNK_MS = 40
BYTES_PER_SAMPLE = 2

DEFAULT_INSTRUCTIONS = (
    "You are participating in a natural spoken conversation.\n"
    "Answer when it feels natural, not only at the very end.\n"
    "Keep responses conversational and concise.\n"
    "If the user interrupts, stop and respond to the latest user speech."
)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_RATE) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    output_length = max(1, round(samples.shape[0] * target_rate / source_rate))
    positions = np.linspace(0, samples.shape[0] - 1, num=output_length)
    return np.interp(positions, np.arange(samples.shape[0]), samples).astype(np.float32)


def read_audio_data(audio_path: Path) -> bytes:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio[:, 0]

    audio = resample_audio(audio, sample_rate, TARGET_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def create_wav_bytes(pcm_bytes: bytes, sample_rate: int = TARGET_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as outfile:
        outfile.setnchannels(1)
        outfile.setsampwidth(BYTES_PER_SAMPLE)
        outfile.setframerate(sample_rate)
        outfile.writeframes(pcm_bytes)
    return buffer.getvalue()


def json_send(ws: websocket.WebSocket, payload: dict):
    ws.send(json.dumps(payload))


def parse_server_messages(raw: str):
    for line in raw.splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def get_reply_with_audio(audio_path: Path, instruction: str, model_name: str, org: str, api_key: str, temp: float = 0.7):
    if not api_key:
        raise ValueError("API key is required for gpt-realtime-2 proxy")

    session_instructions = DEFAULT_INSTRUCTIONS
    if instruction and instruction != "None":
        session_instructions = f"{DEFAULT_INSTRUCTIONS}\n\n{instruction}"

    pcm_bytes = read_audio_data(audio_path)
    if len(pcm_bytes) == 0:
        return None, "", None, False

    headers = [
        f"Authorization: Bearer {api_key}",
    ]
    ws = websocket.create_connection(OPENAI_REALTIME_URL, header=headers)
    ws.settimeout(5.0)

    json_send(ws, {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": session_instructions,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": TARGET_RATE,
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "create_response": True,
                        "interrupt_response": True,
                        "eagerness": "auto",
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": TARGET_RATE,
                    },
                    "voice": "alloy",
                },
            },
        },
    })

    text_fragments = []
    audio_fragments = []
    response_finished = False
    start_time = time.time()
    timeout = 60.0

    def drain_initial_messages():
        try:
            while True:
                raw = ws.recv()
                for event in parse_server_messages(raw):
                    if event.get("type") in {"session.update", "session.started", "session.ready"}:
                        continue
        except websocket.WebSocketTimeoutException:
            return

    drain_initial_messages()

    chunk_size = int(TARGET_RATE * BYTES_PER_SAMPLE * (CHUNK_MS / 1000.0))
    for offset in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[offset : offset + chunk_size]
        json_send(ws, {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        })
        time.sleep(CHUNK_MS / 1000.0)

    ws.settimeout(1.0)
    while time.time() - start_time < timeout and not response_finished:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as exc:
            ws.close()
            return None, "", None, False

        for event in parse_server_messages(raw):
            event_type = event.get("type")
            if event_type == "response.audio.delta":
                audio_payload = event.get("audio")
                if audio_payload:
                    audio_fragments.append(audio_payload)
            elif event_type == "response.text.delta":
                text = event.get("text")
                if text:
                    text_fragments.append(text)
            elif event_type == "response.delta":
                delta = event.get("delta")
                if isinstance(delta, dict):
                    text = delta.get("content") or delta.get("text")
                    if text:
                        text_fragments.append(text)
            elif event_type == "response.completed":
                response_finished = True
            elif event_type in {"response.error", "session.error"}:
                ws.close()
                return None, "", None, False

    ws.close()

    if not audio_fragments:
        return None, "", None, False

    try:
        audio_bytes = b"".join(base64.b64decode(fragment) for fragment in audio_fragments)
    except Exception:
        return None, "", None, False

    wav_bytes = create_wav_bytes(audio_bytes, TARGET_RATE)
    transcript = "".join(text_fragments).strip()
    finish_reason = "completed" if response_finished else "timeout"
    return wav_bytes, transcript, finish_reason, True
