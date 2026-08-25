"""
pipeline/ocr_engine.py
───────────────────────
GPU-accelerated OCR using EasyOCR to scan video frames for burned-in text.
Implements the "coarse" part of the coarse-to-fine architecture by scanning
at a low framerate (e.g., 1 FPS) to find candidate regions.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console
import cv2

try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import easyocr
except ImportError:
    pass

from pipeline.matcher import is_match, MatchResult
from utils.video_utils import iter_frames_sampled
from utils.text_utils import clean_ocr_text, seconds_to_timestamp

console = Console()


@dataclass
class OCRCandidate:
    timestamp_sec: float
    frame_number: int
    text: str
    match_result: MatchResult
    bounding_boxes: list


def coarse_ocr_scan(
    video_path: str,
    target_dialogue: str,
    langs: List[str] = ["en"],
    sample_fps: float = 1.0,
    threshold: float = 75.0,
    gpu: bool = True,
    segment_start_sec: float = 0.0,
) -> Optional[OCRCandidate]:
    """
    Scan the video at low FPS to find the first candidate region containing
    the target dialogue.

    Args:
        video_path:         Path to the video file.
        target_dialogue:    Text to search for.
        langs:              Languages for EasyOCR (e.g., ["en", "ru"]).
        sample_fps:         Frames per second to sample (default 1.0).
        threshold:          Minimum fuzzy match score (0-100).
        gpu:                Whether to use GPU acceleration.
        segment_start_sec:  Offset (seconds) of this clip's start in the original
                            video. All returned timestamps will have this added so
                            they reflect the real position in the source video.

    Returns:
        OCRCandidate if found, else None.
    """
    console.print(
        f"[bold cyan]> Starting coarse OCR scan ({sample_fps} FPS)...[/bold cyan]"
    )

    try:
        reader = easyocr.Reader(langs, gpu=gpu)
    except NameError:
        console.print("[red]x easyocr not installed or imported.[/red]")
        return None
    except Exception as e:
        console.print(f"[red]x EasyOCR init failed:[/red] {e}")
        return None

    best_candidate = None
    best_score = -1.0

    frame_generator = iter_frames_sampled(video_path, sample_fps=sample_fps)

    try:
        for frame_num, timestamp, frame_rgb in frame_generator:
            # Resize frame if it's too large (e.g., 4K) to prevent OOM/segfaults
            h, w = frame_rgb.shape[:2]
            if w > 640:
                scale = 640 / w
                ocr_frame = cv2.resize(frame_rgb, (640, int(h * scale)))
            else:
                ocr_frame = frame_rgb

            # EasyOCR expects BGR or RGB numpy arrays, or file paths.
            # We pass the RGB array directly.
            results = reader.readtext(ocr_frame, detail=1)

            # Assemble all text in the frame
            # results is a list of tuples: (bbox, text, confidence)
            frame_texts = []
            bboxes = []
            for bbox, text, conf in results:
                cleaned = clean_ocr_text(text)
                if cleaned:
                    frame_texts.append(cleaned)
                    bboxes.append(bbox)

            if not frame_texts:
                continue

            full_frame_text = " ".join(frame_texts)
            match = is_match(full_frame_text, target_dialogue, threshold)

            if match.is_match:
                # Convert clip-relative timestamp to original-video timestamp
                original_timestamp = timestamp + segment_start_sec
                console.print(
                    f"[green]+ Visual text candidate found at "
                    f"{seconds_to_timestamp(original_timestamp)}[/green] "
                    f"(clip offset: {seconds_to_timestamp(timestamp)}, "
                    f"original: {seconds_to_timestamp(original_timestamp)}, "
                    f"Score: {match.score:.1f}%)"
                )
                return OCRCandidate(
                    # Store the ORIGINAL video timestamp, not the clip-relative one
                    timestamp_sec=original_timestamp,
                    frame_number=frame_num,
                    # Store only the target text so dialogue_text in the JSON is clean
                    text=target_dialogue,
                    match_result=match,
                    bounding_boxes=bboxes,
                )
    except KeyboardInterrupt:
        console.print("\n[yellow]! Coarse scan interrupted by user.[/yellow]")

    console.print("[yellow]- No visual text matched during coarse scan.[/yellow]")
    return None
