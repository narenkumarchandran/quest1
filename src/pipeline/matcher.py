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
    
    norm_candidate = normalize_text(candidate)
    norm_target = normalize_text(target)

    if not norm_target:
        return 0.0

    # Primary: token set ratio — best for dialogue matching
    score_tsr = fuzz.token_set_ratio(norm_candidate, norm_target)

    score_partial = fuzz.partial_ratio(norm_candidate, norm_target)

    combined = max(score_tsr, score_partial * 0.95)

    c_words = norm_candidate.split()
    t_words = norm_target.split()
    
    intersection = set(c_words) & set(t_words)
    c_rem = [w for w in c_words if w not in intersection]
    t_rem = [w for w in t_words if w not in intersection]
    
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
    
    return fuzzy_score(candidate, target) >= threshold
