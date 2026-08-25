import subprocess
import json
from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console

console = Console()

@dataclass
class StreamInfo:
    index: int
    codec_type: str       # "video" | "audio" | "subtitle"
    codec_name: str
    language: Optional[str] = None
    extra: dict = field(default_factory=dict)

@dataclass
class InspectionResult:
    video_streams: List[StreamInfo] = field(default_factory=list)
    audio_streams: List[StreamInfo] = field(default_factory=list)
    subtitle_streams: List[StreamInfo] = field(default_factory=list)
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def has_subtitle_stream(self) -> bool:
        return len(self.subtitle_streams) > 0

def inspect_video(video_path: str) -> InspectionResult:
    
    console.print(f"[bold cyan]> Inspecting video streams:[/bold cyan] {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    streams_raw = data.get("streams", [])
    fmt = data.get("format", {})

    inspection = InspectionResult()
    inspection.duration_sec = float(fmt.get("duration", 0.0))

    for s in streams_raw:
        codec_type = s.get("codec_type", "unknown")
        lang = s.get("tags", {}).get("language", None)

        stream_info = StreamInfo(
            index=s.get("index", -1),
            codec_type=codec_type,
            codec_name=s.get("codec_name", "unknown"),
            language=lang,
            extra=s,
        )

        if codec_type == "video":
            inspection.video_streams.append(stream_info)
            # Parse FPS from avg_frame_rate (e.g. "30000/1001")
            if inspection.fps == 0.0:
                raw_fps = s.get("avg_frame_rate", "0/1")
                try:
                    num, den = raw_fps.split("/")
                    if int(den) > 0:
                        inspection.fps = int(num) / int(den)
                except (ValueError, ZeroDivisionError):
                    pass
            inspection.width = s.get("width", 0)
            inspection.height = s.get("height", 0)

        elif codec_type == "audio":
            inspection.audio_streams.append(stream_info)

        elif codec_type == "subtitle":
            inspection.subtitle_streams.append(stream_info)

    # Report
    console.print(
        f"  [dim]Resolution:[/dim] {inspection.width}×{inspection.height}  "
        f"[dim]FPS:[/dim] {inspection.fps:.2f}  "
        f"[dim]Duration:[/dim] {inspection.duration_sec:.1f}s"
    )

    if inspection.has_subtitle_stream:
        langs = [s.language or "unknown" for s in inspection.subtitle_streams]
        console.print(
            f"  [green]+ Embedded subtitle stream(s) found:[/green] "
            + ", ".join(f"stream #{s.index} [{s.language or '?'}]"
                        for s in inspection.subtitle_streams)
        )
    else:
        console.print("  [yellow]- No embedded subtitle streams in container.[/yellow]")

    return inspection
