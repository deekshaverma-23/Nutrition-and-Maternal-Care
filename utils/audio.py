from gtts import gTTS
from pathlib import Path

AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def text_to_speech_to_file(text: str, lang: str, key: str) -> str:

    path = AUDIO_DIR / f"{key}.mp3"

    if path.exists():
        return str(path)

    tts = gTTS(text=text, lang=lang)
    tts.save(str(path))

    return str(path)