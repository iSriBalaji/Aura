"""Text-to-Speech (TTS) Voice Service for Aura.
Zero-cost voice generation using open-source gTTS (Google Translate Speech Engine)
with natural markdown text cleaning for spoken conversation.
"""

import io
import re
from typing import Optional


def clean_markdown_for_speech(text: str, max_chars: int = 2500) -> str:
    """Transform markdown report into clean, natural spoken sentences.
    Strips headers, bullets, table pipes, links, and emojis.
    """
    if not text:
        return "Hello Baju, I have no update to read right now."

    # Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Remove markdown links, keep text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Remove table formatting
    cleaned = re.sub(r"\|[^\n]+\|", "", cleaned)
    # Remove markdown headers and blockquotes
    cleaned = re.sub(r"^[#>*\-\+]\s+", "", cleaned, flags=re.MULTILINE)
    # Remove emojis and special symbol noise
    cleaned = re.sub(r"[^\w\s.,!?:;'\-\"()]", " ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) > max_chars:
        # Cut at last complete sentence within limit
        cut = cleaned[:max_chars]
        last_period = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if last_period > 100:
            cleaned = cut[:last_period + 1]
        else:
            cleaned = cut + "."

    return cleaned


def generate_audio_bytes(text: str, slow: bool = False) -> Optional[bytes]:
    """Generate spoken MP3 audio bytes using gTTS at zero cost.
    Returns bytes or None if offline/unreachable.
    """
    speech_text = clean_markdown_for_speech(text)
    if not speech_text:
        speech_text = "Good day Baju! Here is your daily briefing from Aura."

    try:
        from gtts import gTTS
        tts = gTTS(text=speech_text, lang="en", tld="com", slow=slow)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None
