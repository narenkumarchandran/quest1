"""
utils/video_utils.py
─────────────────────
OpenCV helpers for frame extraction used across pipeline modules.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple


def get_video_info(video_path: str) -> dict:
    """
    Return basic video metadata: fps, total_frames, duration_sec, width, height.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def extract_frame_at(video_path: str, second: float) -> np.ndarray:
    """
    Extract a single frame at the given timestamp (seconds).
    Returns the frame as an RGB numpy array.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame at {second:.3f}s from {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_frame_by_number(video_path: str, frame_number: int) -> np.ndarray:
    """
    Extract a specific frame by its 0-indexed frame number.
    Returns the frame as an RGB numpy array.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame #{frame_number} from {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_frames_sampled(
    video_path: str,
    sample_fps: float = 1.0,
    start_sec: float = 0.0,
    end_sec: float = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Yield (frame_number, timestamp_sec, frame_rgb) at a reduced frame rate.

    Args:
        video_path:  Path to the video file.
        sample_fps:  Frames to yield per second (default 1 FPS for coarse scan).
        start_sec:   Start of the region to scan.
        end_sec:     End of the region (None = entire video).

    Yields:
        (frame_number, timestamp_sec, rgb_frame)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps

    if end_sec is None:
        end_sec = duration

    # How many native frames to skip between samples
    frame_step = max(1, int(native_fps / sample_fps))

    # Jump to start position
    start_frame = int(start_sec * native_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / native_fps
        if timestamp > end_sec:
            break

        yield current_frame, timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Skip ahead
        current_frame += frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    cap.release()


def iter_frames_full(
    video_path: str,
    start_sec: float,
    end_sec: float,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Yield every frame between start_sec and end_sec (full FPS scan).
    Used in the temporal refinement window.

    Yields:
        (frame_number, timestamp_sec, rgb_frame)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int(start_sec * native_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / native_fps
        if timestamp > end_sec:
            break

        yield current_frame, timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame += 1

    cap.release()


def save_frame(frame_rgb: np.ndarray, path: str) -> str:
    """
    Save an RGB frame as PNG and return the path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)
    return str(output_path)
