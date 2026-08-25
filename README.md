# Hybrid Video Dialogue Detector

A highly optimized, multi-modal pipeline designed to find the exact frame and timestamp where a specific target dialogue appears in a video. It is built to seamlessly handle multiple sources of dialogue:
- **Embedded Subtitles** (`.srt`, `.vtt`)
- **Spoken Audio** (via GPU-accelerated `faster-whisper`)
- **Visual Scene Text** (via `EasyOCR`)

## How It Works (The Pipeline)

The pipeline prioritizes speed and accuracy by following a strict fallback chain:

1. **Fast Text Search (Subtitles):** Extracts embedded subtitle tracks using `ffmpeg` and fuzzy-matches the target dialogue directly against the text. If found, it skips heavy processing.
2. **Audio-First Fallback (Whisper):** If no subtitles exist, the audio track is extracted and transcribed using `faster-whisper`. It uses a highly memory-optimized, chunked pipeline with greedy decoding to safely process long videos on constrained VRAM (e.g., 4GB GPUs).
3. **Coarse Visual Scan (OCR):** If visual text is expected (or if audio/subtitles provide a candidate timestamp), a 1.0 FPS coarse OCR scan is performed using `EasyOCR` to narrow down the exact visual appearance.
4. **Fine Temporal Refinement:** Once a candidate window (±2 seconds) is isolated, the system scans at full FPS to pinpoint the exact, earliest frame where the *complete* dialogue is visually present above a similarity threshold.

## Requirements

- Python 3.8+
- `ffmpeg` installed and available in your system's PATH.
- (Optional but highly recommended) NVIDIA GPU with CUDA support for Whisper and EasyOCR acceleration.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/narenkumarchandran/quest1.git
   cd quest1
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: Depending on your CUDA version, you may need to install a specific version of PyTorch manually for full GPU acceleration).*

## Usage

Run the pipeline from the project root by targeting `src/main.py`.

```bash
python src/main.py --url "<VIDEO_URL>" --target "<TARGET_DIALOGUE>" [OPTIONS]
```

### Examples

**Basic Run (CPU):**
```bash
python src/main.py --url "https://youtu.be/iXZ1jeTCU-o" --target "I'm the type of person"
```

**GPU Accelerated:**
```bash
python src/main.py --url "https://youtu.be/iXZ1jeTCU-o" --target "I'm the type of person" --gpu
```

### Options
- `--url`: The URL of the video (YouTube, OK.ru, etc. supported via `yt-dlp`).
- `--target`: The target dialogue phrase you want to locate.
- `--gpu`: Enable CUDA acceleration for `faster-whisper` and `EasyOCR` (significantly faster).
- `--threshold`: The fuzzy match confidence threshold (default: `75.0`).
- `--fast-mode`: Skip the final visual OCR refinement if the dialogue was matched via spoken audio.
- `--disable-subs`: Force the pipeline to ignore fast subtitle matching and test the Whisper/OCR fallback pathways.
- `--output`: Custom output directory (default is `./output`).

## Output

Results are exported as a structured JSON object upon completion, including the exact timestamp, similarity score, match source, and a confidence assessment. 

```json
{
  "status": "success",
  "detection_type": "spoken_dialogue",
  "timestamp": "00:02:08.740",
  "frame_number": null,
  "dialogue_text": "I'm the type of person",
  "similarity_score": 100.0,
  "confidence": {
    "score": 65,
    "level": "MEDIUM",
    "signals": {
      "subtitle_match": false,
      "spoken_strong": true,
      ...
    }
  },
  "source": "whisper",
  "video_dir": "output\\iXZ1jeTCU-o"
}
```
