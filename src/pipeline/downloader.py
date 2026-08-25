import subprocess
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console

console = Console()

@dataclass
class DownloadResult:
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    subtitle_paths: List[str] = field(default_factory=list)
    platform_info: dict = field(default_factory=dict)
    has_external_subs: bool = False
    video_dir: Optional[str] = None   # per-movie subfolder

_YTDLP_BASE = [
    "yt-dlp",
    "--force-ipv4",
    "--no-check-certificates",
    "--js-runtimes", "node",
    "--no-playlist",
    "--extractor-args", "youtube:player_client=android,web",
]

def _safe_dirname(title: str) -> str:
    
    return re.sub(r'[\\/*?:"<>|\uff1a]', "_", title).strip()

def _extract_video_id(url: str) -> str:
    
    patterns = [
        # YouTube
        r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([A-Za-z0-9_-]{11})",
        # ok.ru / odnoklassniki
        r"ok\.ru/video/(\d+)",
        # Vimeo
        r"vimeo\.com/(\d+)",
        # Dailymotion
        r"dailymotion\.com/video/([A-Za-z0-9]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)

    # Generic fallback: last non-empty path segment or ?v= query param
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    segment = parsed.path.rstrip("/").split("/")[-1]
    return segment or re.sub(r"[^A-Za-z0-9_-]", "_", url)[-40:]

def _get_video_dir(url: str, base_output_dir: str) -> Path:
    
    video_id = _extract_video_id(url)
    video_dir = Path(base_output_dir) / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # Write / update a meta.json so the folder is human-identifiable
    meta_path = video_dir / "meta.json"
    meta = {"url": url, "video_id": video_id, "title": None}

    # If we already have a stored title, keep it
    if meta_path.exists():
        try:
            existing = json.load(open(meta_path, encoding="utf-8"))
            meta["title"] = existing.get("title")
        except Exception:
            pass

    # Try to fetch title in the background (best-effort, don't fail if it errors)
    if not meta["title"]:
        try:
            result = subprocess.run(
                _YTDLP_BASE + ["--print", "%(title)s", "--no-download",
                               "--quiet", "--no-warnings", url],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10,
            )
            title = result.stdout.strip().splitlines()[0] if result.returncode == 0 else ""
            if title and title != video_id:
                meta["title"] = title
        except Exception:
            pass

    try:
        import json as _json
        meta_path.write_text(_json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    display_name = meta["title"] or video_id
    console.print(f"[dim]> Working directory: {video_dir}  ({display_name})[/dim]")
    return video_dir

def _find_existing_mp4(out_dir: Path, exclude_suffixes=("_segment",)) -> Optional[str]:
    
    candidates = sorted(out_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in candidates:
        if not any(f.stem.endswith(s) for s in exclude_suffixes):
            return str(f)
    return None

def _find_existing_audio(out_dir: Path) -> Optional[str]:
    
    for pattern in ["extracted_audio.wav", "*_audio.wav"]:
        files = sorted(out_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            return str(files[0])
    return None

# ── Public Download Functions ─────────────────────────────────────────────────

def download_subtitles_only(url: str, video_dir: Path) -> DownloadResult:
    
    console.print(f"[bold cyan]> Checking and downloading subtitles into:[/bold cyan] {video_dir.name}/")

    cmd = _YTDLP_BASE + [
        "--skip-download", "--write-sub", "--write-auto-sub",
        "--sub-format", "srt/vtt/best",
        "--sub-langs", "en,en-US,en-GB",
        "-o", str(video_dir / "%(title)s.%(ext)s"),
        url
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    subtitle_paths = []
    for ext in ["*.srt", "*.vtt", "*.ass", "*.ssa"]:
        subtitle_paths.extend(video_dir.glob(ext))
    subtitle_paths = [str(p) for p in subtitle_paths]

    if subtitle_paths:
        console.print(f"[green]+ Subtitles found:[/green] " + ", ".join(Path(p).name for p in subtitle_paths))
    else:
        console.print("[yellow]- No subtitle files downloaded.[/yellow]")

    return DownloadResult(
        subtitle_paths=subtitle_paths,
        has_external_subs=bool(subtitle_paths),
        video_dir=str(video_dir)
    )

def download_audio_only(url: str, video_dir: str) -> DownloadResult:
    
    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reuse existing audio
    existing_audio = _find_existing_audio(out_dir)
    if existing_audio:
        console.print(f"[green]> Reusing existing audio:[/green] {Path(existing_audio).name}")
        return DownloadResult(audio_path=existing_audio, video_dir=str(out_dir))

    # 2. Extract from existing full video via ffmpeg (zero network)
    existing_video = _find_existing_mp4(out_dir)
    if existing_video:
        console.print(f"[green]> Extracting audio from existing video:[/green] {Path(existing_video).name}")
        audio_out = str(out_dir / "extracted_audio.wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-i", existing_video,
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                audio_out
            ], check=True)
            console.print(f"[green]+ Audio extracted:[/green] extracted_audio.wav")
            return DownloadResult(audio_path=audio_out, video_dir=str(out_dir))
        except subprocess.CalledProcessError:
            console.print("[yellow]ffmpeg extraction failed. Downloading audio from network...[/yellow]")

    # 3. Download audio track from network
    output_template = str(out_dir / "%(title)s_audio.%(ext)s")
    cmd = _YTDLP_BASE + [
        "-f", "bestaudio[ext=m4a]/bestaudio/worst",
        "--extract-audio", "--audio-format", "wav",
        "--quiet", "--no-warnings",
        "-o", output_template, url
    ]
    with console.status("[bold cyan]Downloading audio track...[/bold cyan]", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Show the real error from yt-dlp so the user knows what went wrong
        err = (result.stderr or result.stdout or "").strip()
        if err:
            console.print(f"[red]x yt-dlp error:[/red] {err}")
        console.print("[red]x Audio download failed. Cannot proceed without audio.[/red]")
        return DownloadResult(audio_path=None, video_dir=str(out_dir))

    # Find the downloaded .wav
    audio_files = sorted(out_dir.glob("*_audio.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not audio_files:
        audio_files = sorted(out_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not audio_files:
        raise RuntimeError(f"Audio download succeeded but no audio file found in {out_dir}")

    audio_path = str(audio_files[0])
    console.print(f"[green]+ Audio saved:[/green] {Path(audio_path).name}")
    return DownloadResult(audio_path=audio_path, video_dir=str(out_dir))

def download_video_segment(url: str, start_sec: float, end_sec: float, video_dir: str) -> DownloadResult:
    
    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse existing full video
    existing_video = _find_existing_mp4(out_dir)
    if existing_video:
        console.print(f"[green]> Using already-downloaded video:[/green] {Path(existing_video).name}")
        return DownloadResult(video_path=existing_video, video_dir=str(out_dir))

    def fmt_time(sec):
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    start_str = fmt_time(start_sec)
    end_str = fmt_time(end_sec)
    output_template = str(out_dir / "%(title)s_segment.%(ext)s")

    cmd = _YTDLP_BASE + [
        "-f", "bestvideo[vcodec^=avc]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start_str}-{end_str}",
        "--quiet", "--no-warnings",
        "-o", output_template, url
    ]

    with console.status(f"[bold cyan]Downloading video segment [{start_str} - {end_str}]...[/bold cyan]", spinner="dots"):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        except subprocess.TimeoutExpired:
            console.print("[yellow]! Segment download timed out. Skipping visual OCR.[/yellow]")
            return DownloadResult(video_path=None, video_dir=str(out_dir))

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        reason = f": {err[:200]}" if err else ""
        console.print(f"[yellow]! Segment download failed (site may not support range cuts){reason}[/yellow]")
        console.print("[dim]  Skipping visual OCR — result will use Whisper timestamp.[/dim]")
        return DownloadResult(video_path=None, video_dir=str(out_dir))

    mp4_files = sorted(out_dir.glob("*_segment.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not mp4_files:
        files = sorted(out_dir.glob("*_segment.*"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError("Segment download succeeded but no file found.")
        video_path = str(files[0])
    else:
        video_path = str(mp4_files[0])

    console.print(f"[green]+ Segment saved:[/green] {Path(video_path).name}")
    return DownloadResult(video_path=video_path, video_dir=str(out_dir))

def download_full_video_fallback(url: str, video_dir: str) -> DownloadResult:
    
    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_video = _find_existing_mp4(out_dir)
    if existing_video:
        console.print(f"[green]> Using already-downloaded video:[/green] {Path(existing_video).name}")
        return DownloadResult(video_path=existing_video, video_dir=str(out_dir))

    output_template = str(out_dir / "%(title)s.%(ext)s")
    output_template = str(out_dir / "%(title)s.%(ext)s")
    cmd = _YTDLP_BASE + [
        "-f", "bestvideo[vcodec^=avc]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", output_template, url
    ]
    with console.status("[bold yellow]Downloading FULL video (fallback)...[/bold yellow]", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError("yt-dlp failed to download full video.")

    mp4_files = sorted(out_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not mp4_files:
        raise RuntimeError(f"Full video download succeeded but no .mp4 found in {out_dir}")
    video_path = str(mp4_files[0])
    console.print(f"[green]+ Full video saved:[/green] {Path(video_path).name}")
    return DownloadResult(video_path=video_path, video_dir=str(out_dir))
