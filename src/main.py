"""
main.py
────────
CLI entry point for the hybrid video dialogue detection pipeline.
Ties together downloader, inspector, subtitles, audio, and OCR modules.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from rich.console import Console

from pipeline.downloader import (
    download_subtitles_only,
    download_audio_only,
    download_video_segment,
    download_full_video_fallback,
    _get_video_dir,
)
from pipeline.inspector import inspect_video
from pipeline.subtitle_extractor import run_subtitle_search
from pipeline.audio_transcriber import transcribe_and_search
from pipeline.ocr_engine import coarse_ocr_scan
from pipeline.frame_finder import find_exact_frame
from utils.text_utils import seconds_to_timestamp

import re

console = Console()


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert a dialogue string into a safe filename slug.
    e.g. "Can I say something?" -> "can_i_say_something"
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # remove punctuation
    text = re.sub(r"[\s-]+", "_", text)    # spaces/dashes -> underscore
    return text[:max_len].strip("_")


def _get_fps_fallback(video_path: str, url: str) -> float:
    """Attempt to get the video FPS from the local file or via yt-dlp."""
    if video_path and os.path.exists(video_path):
        try:
            from utils.video_utils import get_video_info
            info = get_video_info(video_path)
            if info.get("fps"):
                return float(info["fps"])
        except Exception:
            pass

    try:
        import subprocess
        result = subprocess.run(
            ["yt-dlp", "--print", "%(fps)s", "--no-download", url],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            fps_str = result.stdout.strip()
            if fps_str and fps_str != "None":
                return float(fps_str)
    except Exception:
        pass

    return 25.0


def main():
    # Anchor the default output directory to the project root (one level up from src/)
    project_root = Path(__file__).resolve().parent.parent
    default_output = str(project_root / "output")

    parser = argparse.ArgumentParser(description="Hybrid Video Dialogue Detector")
    parser.add_argument("--url", required=True, help="Video URL to analyze")
    parser.add_argument("--target", required=True, help="Target dialogue to find")
    parser.add_argument("--output", default=default_output, help="Output directory")
    parser.add_argument("--threshold", type=float, default=75.0, help="Fuzzy match threshold")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--fast-mode", action="store_true", help="Skip visual OCR refinement for spoken dialogue (faster on CPU)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for OCR and Whisper")
    parser.add_argument("--disable-subs", action="store_true", help="Ignore subtitles to force testing of Whisper and OCR fallbacks")

    args = parser.parse_args()

    url = args.url
    target_dialogue = args.target
    out_dir = args.output
    threshold = args.threshold

    os.makedirs(out_dir, exist_ok=True)

    console.print(f"[bold magenta]=== Video Dialogue Detection Pipeline ===[/bold magenta]")
    console.print(f"Target: '{target_dialogue}'\n")

    candidate_timestamp = None
    source_type = None
    visual_match_score = 0.0
    spoken_match_score = 0.0
    video_path = None

    # ── Step 0: Resolve per-movie directory ────────────────────────────────────
    video_dir = str(_get_video_dir(url, out_dir))


    # ── Step 1: Subtitle Search ────────────────────────────────────────────────
    if not args.disable_subs:
        sub_dl = download_subtitles_only(url, video_dir=Path(video_dir))
        if sub_dl.has_external_subs:
            sub_result = run_subtitle_search(
                None, [], sub_dl.subtitle_paths, target_dialogue, video_dir, threshold
            )
            if sub_result.found:
                candidate_timestamp = sub_result.entry.start_sec
                source_type = "subtitle"
                spoken_match_score = sub_result.match_result.score

    # ── Step 2: Audio-First (Whisper) ──────────────────────────────────────────
    if candidate_timestamp is None:
        console.print("\n[bold yellow]No subtitles matched. Proceeding with Audio-First approach...[/bold yellow]\n")

        audio_dl = download_audio_only(url, video_dir=video_dir)
        if audio_dl.audio_path:
            audio_candidate = transcribe_and_search(
                audio_dl.audio_path, target_dialogue,
                model_size=args.whisper_model, threshold=threshold,
                output_dir=video_dir, device="cuda" if args.gpu else "cpu"
            )
            if audio_candidate:
                spoken_match_score = audio_candidate.match_result.score
                candidate_timestamp = audio_candidate.start_sec
                source_type = "whisper"
        else:
            console.print("[red]Audio download failed.[/red]")

    # ── Step 3: Targeted Segment Download + OCR ────────────────────────────────
    segment_start_sec = None  # offset of the downloaded clip inside the original video
    
    # If we found it in the subtitles, we trust it and skip the heavy OCR/video pipeline.
    if candidate_timestamp is not None and source_type != "subtitle":
        console.print("\n[bold cyan]> Target found! Downloading targeted video segment for OCR...[/bold cyan]")

        segment_start_sec = max(0, candidate_timestamp - 10.0)
        end_sec = candidate_timestamp + 10.0

        vid_dl = download_video_segment(url, segment_start_sec, end_sec, video_dir=video_dir)
        video_path = vid_dl.video_path

        if video_path:
            console.print("\n[bold cyan]> Verifying visual presence in video segment...[/bold cyan]")
            ocr_candidate = coarse_ocr_scan(
                video_path, target_dialogue, sample_fps=1.0, threshold=threshold, gpu=args.gpu,
                segment_start_sec=segment_start_sec,
            )
            if ocr_candidate:
                visual_match_score = ocr_candidate.match_result.score
                source_type = "fusion"
                # *** DO NOT override candidate_timestamp here ***
                console.print(
                    f"[dim]  (OCR confirmed visual at {ocr_candidate.timestamp_sec:.1f}s, "
                    f"keeping Whisper anchor at {candidate_timestamp:.1f}s)[/dim]"
                )

    elif candidate_timestamp is None:
        # ── Step 4: Full Video OCR Fallback ───────────────────────────────────
        console.print("\n[bold yellow]Target not found in audio. Falling back to Full Video OCR scan...[/bold yellow]\n")

        vid_dl = download_full_video_fallback(url, video_dir=video_dir)
        video_path = vid_dl.video_path

        if video_path:
            # Full-video fallback: segment_start_sec = 0 (timestamps are already absolute)
            ocr_candidate = coarse_ocr_scan(
                video_path, target_dialogue, sample_fps=1.0, threshold=threshold, gpu=args.gpu,
                segment_start_sec=0.0,
            )
            if ocr_candidate:
                visual_match_score = ocr_candidate.match_result.score
                candidate_timestamp = ocr_candidate.timestamp_sec
                source_type = "ocr"
                segment_start_sec = 0.0

    # ── Step 5: Temporal Refinement ────────────────────────────────────────────
    if candidate_timestamp is not None and video_path is not None:
        # Word-level timestamps from Whisper are accurate to ~±0.1s.
        # Keep the window tight at ±2s to avoid false OCR positives on nearby frames.
        refinement_window = 2.0
        exact_frame_result = find_exact_frame(
            video_path, target_dialogue, candidate_timestamp, source_type=source_type,
            window_sec=refinement_window, output_dir=video_dir, gpu=args.gpu,
            verify_visual=not args.fast_mode,
            segment_start_sec=segment_start_sec if segment_start_sec is not None else 0.0
        )

        if exact_frame_result:
            if exact_frame_result.is_visual:
                visual_match_score = max(visual_match_score, exact_frame_result.match_result.score)

            output_data = {
                "status": "success",
                "detection_type": "visual_text" if exact_frame_result.is_visual else "spoken_dialogue",
                "timestamp": seconds_to_timestamp(exact_frame_result.timestamp_sec),
                "frame_number": exact_frame_result.frame_number if exact_frame_result.frame_number >= 0 else int(exact_frame_result.timestamp_sec * _get_fps_fallback(video_path, url)),
                "dialogue_text": exact_frame_result.text,
                "similarity_score": exact_frame_result.match_result.score,
                "frame_image_path": exact_frame_result.frame_image_path,
                "tool_used": "whisper, ocr" if source_type == "fusion" else source_type,
                "video_dir": video_dir,
            }

            result_path = os.path.join(video_dir, f"{_slugify(target_dialogue)}.json")
            with open(result_path, "w") as f:
                json.dump(output_data, f, indent=2)

            console.print("\n[bold magenta]=== Final Result ===[/bold magenta]")
            console.print(json.dumps(output_data, indent=2))

        else:
            console.print("[red]Refinement failed to lock onto a frame.[/red]")

    elif candidate_timestamp is not None and video_path is None:
        # ── Target found in audio/subtitle but OCR was skipped or failed ──────────
        # Emit the result directly.
        console.print(
            f"\n[bold yellow]> Skipping OCR refinement. "
            f"Using {source_type.title()} timestamp directly.[/bold yellow]"
        )
        output_data = {
            "status": "success",
            "detection_type": "spoken_dialogue",
            "timestamp": seconds_to_timestamp(candidate_timestamp),
            "frame_number": int(candidate_timestamp * _get_fps_fallback(video_path, url)),
            "dialogue_text": target_dialogue,
            "similarity_score": spoken_match_score,
            "frame_image_path": None,
            "tool_used": "whisper, ocr" if source_type == "fusion" else source_type,
            "video_dir": video_dir,
        }
        result_path = os.path.join(video_dir, f"{_slugify(target_dialogue)}.json")
        with open(result_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print("\n[bold magenta]=== Final Result ===[/bold magenta]")
        console.print(json.dumps(output_data, indent=2))

    else:

        console.print("[red]Pipeline failed to find any candidate matches for the target dialogue.[/red]")
        output_data = {"status": "failed", "reason": "No matches found in subtitles, audio, or coarse OCR."}
        result_path = os.path.join(video_dir, f"{_slugify(target_dialogue)}.json")
        with open(result_path, "w") as f:
            json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()
