"""
FastAPI backend for the Hybrid Video Dialogue Detector.

Endpoints
---------
GET  /gpu-info              -> { gpu: bool, name: str }
POST /run                   -> { job_id: str }
GET  /stream/{job_id}       -> SSE stream of stdout lines
GET  /result/{job_id}       -> final JSON result
DELETE /run/{job_id}        -> kill the running subprocess
GET  /output/{path:path}    -> serve extracted frames / clips
GET  /                      -> serve the SPA (frontend/index.html)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # project root
FRONTEND_DIR = ROOT / "frontend"
OUTPUT_DIR = ROOT / "output"
MAIN_SCRIPT = ROOT / "src" / "main.py"

app = FastAPI(title="Video Dialogue Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted frames / clips
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Serve frontend static assets (css, js, etc.) -- index is handled manually below
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# -- In-memory job registry ----------------------------------------------------
# { job_id: { "process": Popen, "video_id": str | None, "lines": [str] } }
_jobs: dict[str, dict] = {}


# -- Models --------------------------------------------------------------------
class RunRequest(BaseModel):
    url: str
    target: str
    gpu: bool = False
    fast_mode: bool = False
    threshold: float = 75.0
    cookies: bool = False
    disable_subs: bool = False
    whisper_model: str = "base"


# -- Helpers -------------------------------------------------------------------

def _detect_gpu() -> dict:
    """Return GPU info using nvidia-smi or torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            name = result.stdout.strip().splitlines()[0]
            return {"gpu": True, "name": name}
    except Exception:
        pass

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return {"gpu": True, "name": name}
    except Exception:
        pass

    return {"gpu": False, "name": "No GPU detected"}


def _build_cmd(req: RunRequest, out_dir: str) -> list[str]:
    cmd = [
        sys.executable, str(MAIN_SCRIPT),
        "--url", req.url,
        "--target", req.target,
        "--output", out_dir,
        "--threshold", str(req.threshold),
        "--whisper-model", req.whisper_model,
    ]
    if req.gpu:
        cmd.append("--gpu")
    if req.fast_mode:
        cmd.append("--fast-mode")
    if req.cookies:
        cmd.append("--cookies")
    if req.disable_subs:
        cmd.append("--disable-subs")
    return cmd


async def _stream_process(job_id: str) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted lines from a subprocess."""
    job = _jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'line': '[ERROR] Job not found'})}\n\n"
        return

    process: subprocess.Popen = job["process"]
    loop = asyncio.get_event_loop()

    def _read_line():
        return process.stdout.readline()

    while True:
        line = await loop.run_in_executor(None, _read_line)
        if not line:
            break
        text = line.rstrip()
        job["lines"].append(text)

        # Heuristic: detect video_id from output path mentions in logs
        if job["video_id"] is None:
            for part in text.split():
                candidate = Path(part.strip("'\""))
                try:
                    if candidate.parent == OUTPUT_DIR and candidate.is_dir():
                        job["video_id"] = candidate.name
                        break
                except Exception:
                    pass

        yield f"data: {json.dumps({'line': text})}\n\n"
        await asyncio.sleep(0)  # yield to event loop

    process.wait()
    exit_code = process.returncode

    result_json = _find_result_json(job)
    job["result"] = result_json
    job["exit_code"] = exit_code

    yield f"data: {json.dumps({'done': True, 'exit_code': exit_code, 'result': result_json})}\n\n"


def _find_result_json(job: dict) -> dict | None:
    """Find the JSON result written by this job.

    Strategy: scan every *.json under the output tree and return the one
    with the highest mtime that is ALSO newer than the job's start time.
    This works even when the video directory has a custom name.
    """
    start_time: float = job.get("start_time", 0.0)

    best: Path | None = None
    best_mtime: float = 0.0

    for json_file in OUTPUT_DIR.rglob("*.json"):
        try:
            mtime = json_file.stat().st_mtime
        except OSError:
            continue
        # Only consider files written after this job started
        if mtime > start_time and mtime > best_mtime:
            best_mtime = mtime
            best = json_file

    if best:
        try:
            with open(best, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return None


# -- Endpoints -----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/gpu-info")
async def gpu_info():
    return JSONResponse(_detect_gpu())


@app.post("/run")
async def run_pipeline(req: RunRequest):
    job_id = str(uuid.uuid4())
    out_dir = str(OUTPUT_DIR)

    cmd = _build_cmd(req, out_dir)

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    _jobs[job_id] = {
        "process": process,
        "video_id": None,
        "lines": [],
        "result": None,
        "exit_code": None,
        "start_time": time.time(),
    }

    return JSONResponse({"job_id": job_id})


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    from fastapi.responses import StreamingResponse

    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        _stream_process(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.get("result")
    if result is None:
        result = _find_result_json(job)
        job["result"] = result

    if result is None:
        raise HTTPException(status_code=404, detail="Result not ready yet")

    return JSONResponse(result)


@app.delete("/run/{job_id}")
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    process: subprocess.Popen = job["process"]
    if process.poll() is None:
        process.kill()

    del _jobs[job_id]
    return JSONResponse({"status": "cancelled"})


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(Path(__file__).parent),
    )
