"""
pipeline/matcher.py
────────────────────
Text matching logic using RapidFuzz.

This module is the single source of truth for all similarity comparisons
across the pipeline (subtitle search, OCR results, Whisper transcripts).

Why token_set_ratio as primary scorer?
  - Handles word reordering (Whisper sometimes flips words)
  - Handles partial transcript errors
  - Ignores duplicate words from OCR stuttering
  - More forgiving of punctuation differences
"""

from rapidfuzz import fuzz
from utils.text_utils import normalize_text
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    score: float             # 0–100 similarity score
    matched_text: str        # The candidate text that was matched
    normalized_target: str   # Normalized target for reference
    is_match: bool           # True if score >= threshold


def fuzzy_score(candidate: str, target: str) -> float:
    """
    Compute fuzzy similarity between candidate and target text.

    Uses a blend of three scorers:
    1. token_set_ratio       — handles word reordering & OCR stuttering
    2. partial_ratio         — rewards substring containment (e.g., target inside a
                               longer OCR blob)
    3. partial_token_set_ratio — handles filler words like "uh.. um.." prepended to
                                 the real dialogue (e.g., OCR reads
                                 "uh um Im the type of person" but target is
                                 "I'm the type of person")

    Returns a float 0–100.
    """
    norm_candidate = normalize_text(candidate)
    norm_target = normalize_text(target)

    if not norm_target:
        return 0.0

    # Primary: token set ratio — best for dialogue matching
    score_tsr = fuzz.token_set_ratio(norm_candidate, norm_target)

    # Secondary: partial ratio — useful when candidate contains the target
    # as a substring (e.g., screen text with surrounding content)
    score_partial = fuzz.partial_ratio(norm_candidate, norm_target)

    # Use the best of the two. We slightly penalize partial_ratio so a full 
    # token_set_ratio match always wins over a substring match.
    combined = max(score_tsr, score_partial * 0.95)

    # --- REMAINDER PENALTY ---
    # Prevents false positives where one crucial word changes (e.g. "blue pill" vs "red pill")
    # which otherwise score high on token_set_ratio.
    c_words = norm_candidate.split()
    t_words = norm_target.split()
    
    intersection = set(c_words) & set(t_words)
    c_rem = [w for w in c_words if w not in intersection]
    t_rem = [w for w in t_words if w not in intersection]
    
    # Only penalize if BOTH have remainders (not a strict subset/superset)
    # and they actually share some words.
    if c_rem and t_rem and intersection:
        scores = []
        for tw in t_rem:
            scores.append(max([fuzz.ratio(tw, cw) for cw in c_rem] + [0]))
            
        avg_rem_sim = sum(scores) / len(scores) if scores else 100.0
        
        # If the remainder words don't match well (e.g., completely different words, not just OCR typos)
        if avg_rem_sim < 60.0:
            penalty = (60.0 - avg_rem_sim) * 0.5
            combined -= penalty

    return round(max(0.0, combined), 2)


def is_match(candidate: str, target: str, threshold: float = 75.0) -> MatchResult:
    """
    Determine if a candidate text matches the target above the threshold.

    Args:
        candidate:   The text to evaluate (from OCR, Whisper, or subtitles).
        target:      The target dialogue we are searching for.
        threshold:   Minimum similarity score to be considered a match (0–100).

    Returns:
        MatchResult with score, matched text, and boolean flag.
    """
    score = fuzzy_score(candidate, target)
    return MatchResult(
        score=score,
        matched_text=candidate,
        normalized_target=normalize_text(target),
        is_match=score >= threshold,
    )


def find_best_candidate(
    candidates: List[Tuple[float, str]],    # List of (timestamp_sec, text)
    target: str,
    threshold: float = 75.0,
) -> Optional[Tuple[float, MatchResult]]:
    """
    Find the best-matching candidate from a list of (timestamp, text) pairs.

    Returns:
        (timestamp_sec, MatchResult) for the best match, or None if no match
        exceeds the threshold.
    """
    best_score = -1.0
    best_ts = None
    best_result = None

    for timestamp_sec, text in candidates:
        result = is_match(text, target, threshold)
        if result.score > best_score:
            best_score = result.score
            best_ts = timestamp_sec
            best_result = result

    if best_result and best_result.is_match:
        return best_ts, best_result

    return None


def check_contains_target(candidate: str, target: str, threshold: float = 75.0) -> bool:
    """
    Quick boolean check — does the candidate contain the target dialogue?
    Used for fast filtering during coarse scan.
    """
    return fuzzy_score(candidate, target) >= threshold
