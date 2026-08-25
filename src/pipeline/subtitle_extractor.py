"""
pipeline/subtitle_extractor.py
────────────────────────────────
Handles both embedded subtitle streams (via ffmpeg) and
external subtitle files downloaded by yt-dlp (SRT/VTT).

Detection priority:
  1. Embedded subtitle stream in video container  → extract with ffmpeg
  2. External .srt/.vtt file from yt-dlp download → parse directly
  3. Neither found                                 → return None (fall through to OCR/Whisper)

This is the FASTEST path in the pipeline — text-based subtitle search
requires no frame decoding and is nearly instant even on long videos.
"""

import subprocess
import re
import srt
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console

from pipeline.matcher import is_match, MatchResult
from utils.text_utils import srt_time_to_seconds, seconds_to_timestamp

console = Console()


@dataclass
class SubtitleEntry:
    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class SubtitleSearchResult:
    found: bool
    entry: Optional[SubtitleEntry] = None
    match_result: Optional[MatchResult] = None
    source: str = ""          # "embedded_stream" | "external_file"
    subtitle_file_path: str = ""


def extract_embedded_subtitles(
    video_path: str,
    stream_index: int = 0,
    output_dir: str = "output",
) -> Optional[str]:
    """
    Use ffmpeg to extract an embedded subtitle stream to an SRT file.

    Args:
        video_path:    Path to the video file.
        stream_index:  Index of the subtitle stream (usually 0).
        output_dir:    Where to save the extracted .srt file.

    Returns:
        Path to the extracted .srt file, or None on failure.
    """
    out_path = Path(output_dir) / "extracted_subtitle.srt"

    console.print(
        f"[bold cyan]> Extracting embedded subtitle stream #{stream_index}...[/bold cyan]"
    )

    cmd = [
        "ffmpeg",
        "-y",               # overwrite output
        "-v", "error",
        "-i", str(video_path),
        "-map", f"0:s:{stream_index}",   # select subtitle stream
        "-an", "-vn",        # skip audio & video — fast
        "-c:s", "srt",
        str(out_path),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    if result.returncode != 0 or not out_path.exists():
        console.print(f"[red]x ffmpeg subtitle extraction failed:[/red] {result.stderr}")
        return None

    console.print(f"[green]+ Subtitle extracted to:[/green] {out_path}")
    return str(out_path)


def parse_srt_file(srt_path: str) -> List[SubtitleEntry]:
    """
    Parse an SRT file into a list of SubtitleEntry objects.
    Handles both standard SRT and VTT (converted to SRT).
    """
    text = Path(srt_path).read_text(encoding="utf-8", errors="replace")

    # If this looks like a VTT file, pre-process it to SRT format
    if srt_path.endswith(".vtt") or text.strip().startswith("WEBVTT"):
        text = _vtt_to_srt(text)

    entries = []
    try:
        for sub in srt.parse(text):
            entries.append(
                SubtitleEntry(
                    index=sub.index,
                    start_sec=sub.start.total_seconds(),
                    end_sec=sub.end.total_seconds(),
                    text=sub.content.strip(),
                )
            )
    except Exception as e:
        console.print(f"[yellow]! SRT parse warning:[/yellow] {e}")

    return entries


def _vtt_to_srt(vtt_text: str) -> str:
    """
    Convert WebVTT format to SRT format for uniform parsing.
    Handles the main differences: WEBVTT header, dot vs comma in timestamps.
    """
    lines = vtt_text.splitlines()
    # Remove WEBVTT header and metadata
    content_lines = []
    skip = True
    for line in lines:
        if skip and "-->" in line:
            skip = False
        if not skip:
            content_lines.append(line)

    text = "\n".join(content_lines)

    # VTT uses dots in timestamps (00:00:01.000), SRT uses commas (00:00:01,000)
    text = re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d{3})", r"\1,\2", text)

    # Add sequential indices if missing (VTT doesn't require them)
    lines_out = []
    block_num = 1
    for block in text.strip().split("\n\n"):
        if "-->" in block:
            block_lines = block.strip().splitlines()
            # If first line is a number, keep it; otherwise prepend one
            if not block_lines[0].strip().isdigit():
                block = f"{block_num}\n{block}"
                block_num += 1
        lines_out.append(block)

    return "\n\n".join(lines_out)


def search_subtitles(
    subtitle_entries: List[SubtitleEntry],
    target_dialogue: str,
    threshold: float = 75.0,
) -> Optional[SubtitleEntry]:
    """
    Search parsed subtitle entries for the target dialogue.

    Returns the first entry that matches above the threshold, or None.
    """
    for entry in subtitle_entries:
        result = is_match(entry.text, target_dialogue, threshold)
        if result.is_match:
            return entry
    return None


def run_subtitle_search(
    video_path: str,
    subtitle_stream_indices: List[int],
    external_subtitle_paths: List[str],
    target_dialogue: str,
    output_dir: str = "output",
    threshold: float = 75.0,
) -> SubtitleSearchResult:
    """
    Main entry point for subtitle-based search.

    Priority:
      1. Embedded subtitle streams (extracted via ffmpeg)
      2. External subtitle files (from yt-dlp)

    Args:
        video_path:              Path to the downloaded video.
        subtitle_stream_indices: Indices of embedded subtitle streams (from inspector).
        external_subtitle_paths: Paths to .srt/.vtt files from yt-dlp.
        target_dialogue:         The dialogue line to search for.
        output_dir:              Where to write extracted subtitle files.
        threshold:               Minimum fuzzy match score (0–100).

    Returns:
        SubtitleSearchResult — found=True if a match was found.
    """
    # ── 1. Try embedded subtitle streams ─────────────────────────────────────
    for stream_idx in subtitle_stream_indices:
        srt_path = extract_embedded_subtitles(video_path, stream_idx, output_dir)
        if srt_path:
            entries = parse_srt_file(srt_path)
            hit = search_subtitles(entries, target_dialogue, threshold)
            if hit:
                match_res = is_match(hit.text, target_dialogue, threshold)
                console.print(
                    f"[bold green]+ Target found in embedded subtitle stream![/bold green] "
                    f"Score: {match_res.score:.1f}%  "
                    f"Time: {seconds_to_timestamp(hit.start_sec)}"
                )
                return SubtitleSearchResult(
                    found=True,
                    entry=hit,
                    match_result=match_res,
                    source="embedded_stream",
                    subtitle_file_path=srt_path,
                )

    # ── 2. Try external subtitle files ────────────────────────────────────────
    for sub_path in external_subtitle_paths:
        console.print(
            f"[bold cyan]> Searching external subtitle file:[/bold cyan] "
            f"{Path(sub_path).name}"
        )
        entries = parse_srt_file(sub_path)
        hit = search_subtitles(entries, target_dialogue, threshold)
        if hit:
            match_res = is_match(hit.text, target_dialogue, threshold)
            console.print(
                f"[bold green]+ Target found in external subtitle file![/bold green] "
                f"Score: {match_res.score:.1f}%  "
                f"Time: {seconds_to_timestamp(hit.start_sec)}"
            )
            return SubtitleSearchResult(
                found=True,
                entry=hit,
                match_result=match_res,
                source="external_file",
                subtitle_file_path=sub_path,
            )

    console.print(
        "[yellow]- Target dialogue not found in any subtitle file.[/yellow]"
    )
    return SubtitleSearchResult(found=False)
