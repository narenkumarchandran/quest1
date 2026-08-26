import pytest
# pyrefly: ignore [missing-import]
from pipeline.matcher import fuzzy_score, is_match, find_best_candidate, check_contains_target, MatchResult

def test_fuzzy_score():
    # Exact match
    assert fuzzy_score("hello world", "hello world") == 100.0
    
    # Partial match
    score = fuzzy_score("this is a hello world test", "hello world")
    assert score >= 90.0
    
    # Case insensitivity
    assert fuzzy_score("HeLLo WoRLd", "hello world") == 100.0
    
    # Punctuation differences
    assert fuzzy_score("hello, world!", "hello world") == 100.0
    
    # No match
    assert fuzzy_score("completely different", "hello world") < 50.0

def test_is_match():
    # Test match above threshold
    result = is_match("the quick brown fox", "quick brown", threshold=75.0)
    assert result.is_match == True
    assert result.score >= 75.0
    assert result.matched_text == "the quick brown fox"
    
    # Test match below threshold
    result = is_match("the slow white dog", "quick brown", threshold=75.0)
    assert result.is_match == False
    assert result.score < 75.0

def test_find_best_candidate():
    candidates = [
        (10.5, "random dialogue here"),
        (15.2, "this is exactly what we are looking for"),
        (20.0, "something else completely")
    ]
    
    target = "what we are looking for"
    
    # Best candidate should be the second one
    result = find_best_candidate(candidates, target, threshold=80.0)
    assert result is not None
    
    best_ts, best_result = result
    assert best_ts == 15.2
    assert best_result.is_match == True
    assert best_result.matched_text == "this is exactly what we are looking for"
    
    # No candidate should match if threshold is too high and candidates are wrong
    result_none = find_best_candidate(candidates, "nobody expects the spanish inquisition", threshold=80.0)
    assert result_none is None

def test_check_contains_target():
    assert check_contains_target("it contains the target string", "target string", threshold=75.0) == True
    assert check_contains_target("it does not contain it", "target string", threshold=75.0) == False
