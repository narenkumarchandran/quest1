import pytest
from unittest.mock import patch, MagicMock
import json
# pyrefly: ignore [missing-import]
from pipeline.inspector import inspect_video

# Sample ffprobe JSON output
MOCK_FFPROBE_OUTPUT = {
    "streams": [
        {
            "index": 0,
            "codec_name": "h264",
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
            "tags": {"language": "und"}
        },
        {
            "index": 1,
            "codec_name": "aac",
            "codec_type": "audio",
            "tags": {"language": "eng"}
        },
        {
            "index": 2,
            "codec_name": "subrip",
            "codec_type": "subtitle",
            "tags": {"language": "eng"}
        }
    ],
    "format": {
        "duration": "120.5"
    }
}

@patch('subprocess.run')
def test_inspect_video(mock_run):
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = json.dumps(MOCK_FFPROBE_OUTPUT)
    mock_run.return_value = mock_process

    # Call function
    result = inspect_video("fake_video.mp4")

    # Assertions
    assert result.duration_sec == 120.5
    assert result.width == 1920
    assert result.height == 1080
    assert round(result.fps, 2) == 29.97
    
    assert len(result.video_streams) == 1
    assert result.video_streams[0].codec_name == "h264"
    
    assert len(result.audio_streams) == 1
    assert result.audio_streams[0].codec_name == "aac"
    
    assert len(result.subtitle_streams) == 1
    assert result.has_subtitle_stream == True
    assert result.subtitle_streams[0].language == "eng"

@patch('subprocess.run')
def test_inspect_video_ffprobe_error(mock_run):
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stderr = "ffprobe error message"
    mock_run.return_value = mock_process

    with pytest.raises(RuntimeError) as exc_info:
        inspect_video("fake_video.mp4")
    assert "ffprobe failed: ffprobe error message" in str(exc_info.value)
