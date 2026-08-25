"""
pipeline/frame_finder.py
─────────────────────────
The "fine" part of the coarse-to-fine architecture.
Takes a candidate timestamp (from coarse OCR, Whisper, or subtitles)
and scans every frame in a small temporal window (e.g. ±2 seconds)
to find the EXACT FIRST FRAME where the complete target text appears.
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
from utils.video_utils import iter_frames_full, save_frame
from utils.text_utils import clean_ocr_text, seconds_to_timestamp

console = Console()


@dataclass
class ExactFrameResult:
    frame_number: int
    timestamp_sec: float
    text: str
    match_result: MatchResult
    frame_image_path: str
    is_visual: bool           # True if text is visually present, False if spoken-only


def find_exact_frame(
    video_path: str,
    target_dialogue: str,
    candidate_timestamp_sec: float,
    source_type: str,         # "ocr", "whisper", "subtitle"
    window_sec: float = 2.0,
    threshold: float = 85.0,  # Stricter threshold for exact frame
    langs: List[str] = ["en"],
    gpu: bool = True,
    output_dir: str = "output",
    verify_visual: bool = True,
    segment_start_sec: float = 0.0,
) -> Optional[ExactFrameResult]:
    """
    Scan every frame in [candidate - window, candidate + window] to find the
    first frame where the complete target text is visually present.

    candidate_timestamp_sec is the ORIGINAL-VIDEO timestamp (already offset).
    segment_start_sec is the offset of the clip in the original video — it is
    subtracted to get the clip-relative seek position for cv2, then added back
    to produce original-video timestamps in all returned results.

    If source_type == "whisper" and no visual text is found, fallback to
    returning the spoken timestamp (is_visual=False).
    """
    if not verify_visual and source_type in ["whisper", "subtitle"]:
        console.print(
            f"[bold cyan]> Skipping visual OCR refinement (fast mode). Extracting frame at {seconds_to_timestamp(candidate_timestamp_sec)}...[/bold cyan]"
        )
        return _spoken_fallback(video_path, candidate_timestamp_sec, target_dialogue, source_type, output_dir)

    # candidate_timestamp_sec is already in original-video time.
    # Convert to clip-relative time for seeking inside the video file.
    clip_candidate = candidate_timestamp_sec - segment_start_sec
    start_sec = max(0.0, clip_candidate - window_sec)
    end_sec = clip_candidate + window_sec

    console.print(
        f"[bold cyan]> Temporal refinement (full FPS scan):[/bold cyan] "
        f"[{seconds_to_timestamp(max(0.0, candidate_timestamp_sec - window_sec))} - "
        f"{seconds_to_timestamp(candidate_timestamp_sec + window_sec)}] (original video time)"
    )

    try:
        reader = easyocr.Reader(langs, gpu=gpu)
    except Exception as e:
        console.print(f"[red]x EasyOCR init failed for refinement:[/red] {e}")
        return None

    frame_generator = iter_frames_full(video_path, start_sec, end_sec)

    best_match_frame = None
    best_score = -1.0

    # Definition of exact frame:
    # "The target frame is the earliest frame in which the complete target dialogue
    # is visually detectable with confidence above the configured threshold."

    try:
        for frame_num, timestamp, frame_rgb in frame_generator:
            h, w = frame_rgb.shape[:2]
            if w > 640:
                scale = 640 / w
                ocr_frame = cv2.resize(frame_rgb, (640, int(h * scale)))
            else:
                ocr_frame = frame_rgb

            results = reader.readtext(ocr_frame, detail=0) # detail=0 returns just text list
            
            if not results:
                continue

            # Join all text blocks in the frame
            full_text = " ".join([clean_ocr_text(t) for t in results if clean_ocr_text(t)])
            
            if not full_text:
                continue

            match = is_match(full_text, target_dialogue, threshold)

            # Since we want the EARLIEST frame with COMPLETE text,
            # as soon as we cross the high threshold, we return it.
            if match.is_match:
                # Convert clip-relative timestamp back to original-video timestamp
                original_ts = timestamp + segment_start_sec
                img_path = save_frame(frame_rgb, f"{output_dir}/frame_{frame_num}.png")
                console.print(
                    f"[bold green]+ EXACT VISUAL FRAME FOUND:[/bold green] "
                    f"Frame {frame_num} at {seconds_to_timestamp(original_ts)} (original video)"
                )
                return ExactFrameResult(
                    frame_number=frame_num,
                    timestamp_sec=original_ts,
                    # Only store the clean target text, not the noisy OCR dump
                    text=target_dialogue,
                    match_result=match,
                    frame_image_path=img_path,
                    is_visual=True,
                )
            
            # Keep track of the best partial match just in case we don't cross threshold
            if match.score > best_score:
                best_score = match.score
                best_match_frame = (frame_num, timestamp, full_text, match, frame_rgb)

    except KeyboardInterrupt:
        pass

    # If we get here, no frame crossed the strict visual threshold.
    
    # Fallback 1: Was there a partial visual match that was pretty good?
    if best_match_frame and best_match_frame[3].score > 60.0: # relaxed threshold
        f_num, ts, text, match, f_rgb = best_match_frame
        original_ts = ts + segment_start_sec
        img_path = save_frame(f_rgb, f"{output_dir}/frame_{f_num}_partial.png")
        console.print(
            f"[yellow]! Only partial visual match found (Score: {match.score:.1f}%).[/yellow] "
            f"Using frame {f_num} at {seconds_to_timestamp(original_ts)} (original video)."
        )
        return ExactFrameResult(
            frame_number=f_num,
            timestamp_sec=original_ts,
            text=target_dialogue,
            match_result=match,
            frame_image_path=img_path,
            is_visual=True,
        )

    # Fallback 2: Spoken-only dialogue (Case 2 from requirements)
    if source_type in ["whisper", "subtitle"]:
        console.print(
            "[yellow]! No visual text detected in refinement window.[/yellow]\n"
            "This is likely spoken-only dialogue or embedded scene text that OCR missed. "
            "Falling back to candidate timestamp."
        )
        return _spoken_fallback(video_path, candidate_timestamp_sec, target_dialogue, source_type, output_dir)

    console.print("[red]x Refinement failed to find visual or spoken target.[/red]")
    return None


def _spoken_fallback(video_path, candidate_timestamp_sec, target_dialogue, source_type, output_dir):
    from utils.video_utils import extract_frame_at
    try:
        f_rgb = extract_frame_at(video_path, candidate_timestamp_sec)
        img_path = save_frame(f_rgb, f"{output_dir}/frame_spoken_fallback.png")
    except Exception:
        img_path = ""

    # Dummy match result to represent the spoken match
    spoken_match = MatchResult(
        score=100.0 if source_type == "subtitle" else 85.0, # assumed
        matched_text=target_dialogue,
        normalized_target=target_dialogue,
        is_match=True
    )

    return ExactFrameResult(
        frame_number=-1, # Unknown exact frame number for spoken
        timestamp_sec=candidate_timestamp_sec,
        text=target_dialogue, # The spoken text
        match_result=spoken_match,
        frame_image_path=img_path,
        is_visual=False,
    )
