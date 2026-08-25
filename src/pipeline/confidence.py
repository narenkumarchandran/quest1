"""
pipeline/confidence.py
───────────────────────
Calculates an aggregate confidence score for the final result
based on the signals collected from the pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ConfidenceResult:
    score: int          # 0-100
    level: str          # "HIGH", "MEDIUM", "LOW"
    signals: Dict[str, Any]

def calculate_confidence(
    source_type: str,         # "subtitle", "ocr", "whisper", "fusion"
    visual_match_score: float,
    spoken_match_score: float,
    timestamp_agreement: bool = False,
    consecutive_frames: bool = False,
) -> ConfidenceResult:
    """
    Design a confidence score based on signals:
      - subtitle exact match:     +50
      - ocr similarity >= 90:     +30 (or +20 if 75-89)
      - whisper similarity >= 85: +15
      - timestamp agreement:      +10
      - consecutive frames:       +10
    """
    score = 0
    signals = {}

    if source_type == "subtitle":
        score += 50
        signals["subtitle_match"] = True
    else:
        signals["subtitle_match"] = False

    if visual_match_score >= 90.0:
        score += 30
        signals["ocr_strong"] = True
    elif visual_match_score >= 75.0:
        score += 20
        signals["ocr_moderate"] = True
    else:
        signals["ocr_strong"] = False
        signals["ocr_moderate"] = False

    if spoken_match_score >= 85.0:
        score += 15
        signals["spoken_strong"] = True
    else:
        signals["spoken_strong"] = False

    if timestamp_agreement:
        score += 10
        signals["timestamp_agreement"] = True
    else:
        signals["timestamp_agreement"] = False
        
    if consecutive_frames:
        score += 10
        signals["consecutive_frames"] = True
    else:
         signals["consecutive_frames"] = False

    # Cap at 100
    score = min(100, score)

    # If it was a perfect subtitle match but nothing else, it's highly confident
    # because subtitles are ground truth.
    if source_type == "subtitle" and visual_match_score == 0 and spoken_match_score == 0:
        score = 95
        
    # If it's a perfect visual match, it's highly confident.
    if visual_match_score >= 95.0:
        score = max(score, 95)

    if score >= 80:
        level = "HIGH"
    elif score >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    return ConfidenceResult(score=score, level=level, signals=signals)
