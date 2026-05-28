import base64
import subprocess
import tempfile
from pathlib import Path
from openai import OpenAI

MIN_WAV_BYTES = 2000  # roughly >60 ms at 16 kHz mono 16-bit PCM
NON_STREAMING_ANSWER_START_S = 0.0

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

def get_reply_with_audio(
    audio_path: Path,
    instruction: str,
    model_name: str,
    org: str | None,
    api_key: str,
    temp: float = 0.7,
):
    client = OpenAI(api_key=api_key, organization=org)

    wav_path = convert_to_wav_strict(audio_path)



    audio_bytes_in = wav_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes_in).decode("ascii")

    user_content = []

    if instruction and instruction != "None":
        user_content.append({"type": "text", "text": instruction})

    user_content.append(
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_b64,
                "format": "wav",
            },
        }
    )

    response = client.chat.completions.create(
        model=model_name,  # "gpt-audio-1.5"
        temperature=temp,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can understand and respond to speech.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    message = response.choices[0].message

    if message.audio is None or message.audio.data is None:
        print(f"Warning: model failed to return audio for {audio_path}")
        return None, None, response.choices[0].finish_reason, False, None

    audio_bytes_out = base64.b64decode(message.audio.data)
    transcript = message.audio.transcript
    return audio_bytes_out, transcript, response.choices[0].finish_reason, True, NON_STREAMING_ANSWER_START_S
