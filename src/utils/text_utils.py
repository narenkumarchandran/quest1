"""
utils/text_utils.py
────────────────────
Text normalization and cleaning helpers shared across pipeline modules.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
      - Unicode → ASCII where possible
      - Lowercase
      - Strip punctuation except apostrophes
      - Collapse whitespace
    """
    # Unicode normalization (handle curly quotes, em-dashes, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = text.lower()

    # Remove punctuation except apostrophes (keep contractions intact)
    text = re.sub(r"[^\w\s']", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_ocr_text(raw: str) -> str:
    """
    Extra cleanup for raw OCR output which may contain:
      - Pipe characters mistaken for 'I' or 'l'
      - Stray numbers from frame artifacts
      - Newlines within a subtitle block
    """
    # Join multi-line OCR result into single line
    text = raw.replace("\n", " ").replace("|", "I")

    # Remove isolated single digits (common OCR noise)
    text = re.sub(r"(?<!\w)\d(?!\w)", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def srt_time_to_seconds(time_str: str) -> float:
    """
    Convert SRT timestamp '00:10:21,400' → 621.4 seconds.
    Also handles VTT format '00:10:21.400'.
    """
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(time_str)


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert 621.4 → '00:10:21.400'
    Negative times: -0.8 -> '-00:00:00.800'
    """
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{sign}{h:02d}:{m:02d}:{s:06.3f}"
