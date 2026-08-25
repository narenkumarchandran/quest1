import re
import unicodedata

def normalize_text(text: str) -> str:
    
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
    
    # Join multi-line OCR result into single line
    text = raw.replace("\n", " ").replace("|", "I")

    # Remove isolated single digits (common OCR noise)
    text = re.sub(r"(?<!\w)\d(?!\w)", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def srt_time_to_seconds(time_str: str) -> float:
    
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
    
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{sign}{h:02d}:{m:02d}:{s:06.3f}"
