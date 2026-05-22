import base64
from pathlib import Path
from openai import OpenAI


def get_reply_with_audio(audio_path: Path, instruction: str, model_name: str, org: str, api_key: str, temp: float = 0.7):
    """Send audio and instruction to the gpt-4o-audio model and return the response audio and transcript."""
    client = OpenAI(api_key=api_key, organization=org)

    with open(audio_path, "rb") as infile:
        audio_b64 = base64.b64encode(infile.read()).decode("utf-8")

    user_content = []
    if instruction != 'None':
        user_content.append({"type": "text", "text": instruction})
    user_content.append({
        "type": "input_audio",
        "input_audio": {
            "data": audio_b64,
            "format": audio_path.suffix.lstrip(".")
        }
    })

    response = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can understand and respond to speech."},
            {"role": "user", "content": user_content}
        ]
    )

    message = response.choices[0].message
    if message.audio is None or message.audio.data is None:
        print(f"Warning: model failed to return audio for {audio_path}")
        return None, None, None, False

    audio_bytes = base64.b64decode(message.audio.data)
    transcript = message.audio.transcript
    return audio_bytes, transcript, response.choices[0].finish_reason, True
