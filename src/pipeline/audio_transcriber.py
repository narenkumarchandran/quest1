import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console

try:
    from faster_whisper import WhisperModel
except ImportError:
    pass

from pipeline.matcher import is_match, MatchResult
from utils.text_utils import seconds_to_timestamp

console = Console()

@dataclass
class AudioCandidate:
    start_sec: float
    end_sec: float
    text: str
    match_result: MatchResult

# Audio file extensions that can be fed directly to Whisper (no ffmpeg extraction needed)
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}

def extract_audio(media_path: str, output_dir: str = "output") -> str:
    
    src = Path(media_path)
    out_path = Path(output_dir) / "extracted_audio.wav"

    # If the source IS already the target output file, Whisper can use it as-is
    if src.resolve() == out_path.resolve():
        console.print(f"[dim]> Audio already at correct path, skipping extraction.[/dim]")
        return str(src)

    # If the source is already a WAV, just resample/remux — don't re-extract video
    if src.suffix.lower() in _AUDIO_EXTENSIONS:
        console.print("[bold cyan]> Resampling audio for Whisper...[/bold cyan]")
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_path),
        ]
    else:
        console.print("[bold cyan]> Extracting audio for Whisper...[/bold cyan]")
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-vn",                 # skip video stream
            "-ac", "1",            # mono
            "-ar", "16000",        # 16kHz
            "-c:a", "pcm_s16le",   # raw PCM
            str(out_path),
        ]

    subprocess.run(cmd, check=True)
    return str(out_path)

def _get_audio_duration(audio_path: str) -> float:
    
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def _extract_chunk(audio_path: str, start_sec: float, duration_sec: float, out_path: str) -> bool:
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start_sec), "-t", str(duration_sec),
            "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            out_path,
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def transcribe_and_search(
    media_path: str,
    target_dialogue: str,
    model_size: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
    threshold: float = 75.0,
    output_dir: str = "output",
    chunk_minutes: float = 5.0,
) -> Optional[AudioCandidate]:
    
    try:
        audio_path = extract_audio(media_path, output_dir)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]x Audio extraction failed:[/red] {e}")
        return None

    duration = _get_audio_duration(audio_path)
    chunk_sec = chunk_minutes * 60.0

    console.print(
        f"[bold cyan]> Running Whisper ASR ({model_size} on {device})...[/bold cyan]"
    )
    if duration > chunk_sec:
        num_chunks = int(duration / chunk_sec) + 1
        console.print(
            f"  [dim]Audio is {duration/60:.1f} min — processing in {num_chunks} × {chunk_minutes:.0f}-min chunks[/dim]"
        )

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except NameError:
        console.print("[red]x faster-whisper not installed or imported.[/red]")
        return None
    except Exception as e:
        console.print(f"[red]x Whisper model load failed (try device='cpu'):[/red] {e}")
        return None

    best_candidate = None
    best_score = -1.0
    chunk_path = str(Path(output_dir) / "_whisper_chunk.wav")

    # Process audio in fixed-size chunks to avoid OOM on long files
    chunk_start = 0.0
    while chunk_start < max(duration, 1.0):
        actual_duration = min(chunk_sec, duration - chunk_start) if duration > 0 else chunk_sec

        # Extract this chunk using ffmpeg
        if duration > chunk_sec:
            ok = _extract_chunk(audio_path, chunk_start, actual_duration, chunk_path)
            if not ok:
                chunk_start += chunk_sec
                continue
            active_path = chunk_path
        else:
            # File is short enough to process whole
            active_path = audio_path

        segments, info = model.transcribe(
            active_path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        if chunk_start == 0.0:
            console.print(
                f"  [dim]Detected language '{info.language}' "
                f"with probability {info.language_probability:.2f}[/dim]"
            )

        for segment in segments:
            text = segment.text.strip()
            console.print(f"  [dim]Transcribed: '{text}'[/dim]")
            match = is_match(text, target_dialogue, threshold)

            if match.is_match and match.score > best_score:
                # Offset by chunk start to get original-video time
                start_sec = segment.start + chunk_start
                end_sec = segment.end + chunk_start

                best_candidate = AudioCandidate(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    match_result=match,
                )
                best_score = match.score

        if best_candidate and best_score >= threshold:
            break

        if duration <= chunk_sec:
            break  
        chunk_start += chunk_sec

    # Cleanup temp chunk file
    try:
        Path(chunk_path).unlink(missing_ok=True)
    except Exception:
        pass

    if best_candidate:
        console.print(
            f"[bold green]+ Target spoken dialogue found![/bold green] "
            f"Score: {best_candidate.match_result.score:.1f}%  "
            f"Time: {seconds_to_timestamp(best_candidate.start_sec)}"
        )
    else:
        console.print("[yellow]- Target dialogue not heard in audio.[/yellow]")

    # Force memory cleanup before OCR runs
    del model
    import gc
    gc.collect()

    return best_candidate
