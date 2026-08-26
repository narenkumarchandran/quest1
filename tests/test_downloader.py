import pytest

# pyrefly: ignore [missing-import]
from pipeline.downloader import _extract_video_id, _safe_dirname

def test_extract_video_id():
    # Test standard YouTube URL
    assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    # Test YouTube short URL
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    # Test YouTube shorts URL
    assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    # Test YouTube embed
    assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    # Test Vimeo
    assert _extract_video_id("https://vimeo.com/123456789") == "123456789"
    
    # Test Dailymotion
    assert _extract_video_id("https://www.dailymotion.com/video/x7tg3l1") == "x7tg3l1"

def test_safe_dirname():
    assert _safe_dirname("Standard Title") == "Standard Title"
    assert _safe_dirname("Title / With \\ Invalid : Characters *") == "Title _ With _ Invalid _ Characters _"
    assert _safe_dirname("  Leading and trailing  ") == "Leading and trailing"
