import pytest
# pyrefly: ignore [missing-import]
from utils.text_utils import normalize_text, clean_ocr_text, srt_time_to_seconds, seconds_to_timestamp

def test_normalize_text():
    # Test basic normalization
    assert normalize_text("Hello World!") == "hello world"
    
    # Test unicode and punctuation removal
    assert normalize_text("Café, jalapeño... 'quote'") == "cafe jalapeno 'quote'"
    
    # Test multiple spaces
    assert normalize_text("  This    has   spaces  ") == "this has spaces"

def test_clean_ocr_text():
    # Test newline replacement
    assert clean_ocr_text("Word One\nWord Two") == "Word One Word Two"
    
    # Test pipe replacement
    assert clean_ocr_text("Word | Another") == "Word I Another"
    
    # Test isolated digits
    assert clean_ocr_text("This is 1 test 2") == "This is test"
    
    # Test collapsing whitespace
    assert clean_ocr_text("Too   much     space") == "Too much space"

def test_srt_time_to_seconds():
    # Test standard format HH:MM:SS.mmm
    assert srt_time_to_seconds("01:02:03.500") == 3723.5
    
    # Test comma format HH:MM:SS,mmm
    assert srt_time_to_seconds("00:01:30,250") == 90.25
    
    # Test MM:SS format
    assert srt_time_to_seconds("05:15.5") == 315.5
    
    # Test just seconds
    assert srt_time_to_seconds("45.5") == 45.5

def test_seconds_to_timestamp():
    # Test standard output
    assert seconds_to_timestamp(3723.5) == "01:02:03.500"
    
    # Test under an hour
    assert seconds_to_timestamp(90.25) == "00:01:30.250"
    
    # Test negative
    assert seconds_to_timestamp(-15.5) == "-00:00:15.500"
