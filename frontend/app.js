/**
 * Video Dialogue Detector — SPA Logic
 *
 * Flow:
 *  1. On load: GET /gpu-info → set mode toggle default
 *  2. Form submit: POST /run → open EventSource /stream/{job_id}
 *  3. Parse SSE to advance progress bar
 *  4. On done event: render result screen (success or failure)
 *  5. Cancel: DELETE /run/{job_id}
 */

"use strict";

/* ── Constants ─────────────────────────────────────────────────────────────── */
const API = "";   // same origin; FastAPI serves both the SPA and the API

// Map log keywords → { step, pct, label }
const PROGRESS_STAGES = [
  { pattern: /subtitle/i,                step: "Step 1 / 5", pct: 15, label: "Searching subtitle tracks…" },
  { pattern: /audio|whisper|transcrib/i, step: "Step 2 / 5", pct: 35, label: "Transcribing audio with Whisper…" },
  { pattern: /segment|download.*video/i, step: "Step 3 / 5", pct: 55, label: "Downloading targeted video segment…" },
  { pattern: /ocr|visual/i,              step: "Step 4 / 5", pct: 75, label: "Running visual OCR scan…" },
  { pattern: /refinement|exact|frame/i,  step: "Step 5 / 5", pct: 90, label: "Formatting the output…" },
];

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const screens = {
  input:      $("screen-input"),
  processing: $("screen-processing"),
  result:     $("screen-result"),
  failure:    $("screen-failure"),
};

/* ── Screen switching ──────────────────────────────────────────────────────── */
function showScreen(name) {
  Object.entries(screens).forEach(([key, el]) => {
    el.classList.toggle("screen--active", key === name);
  });
}

/* ── GPU Detection ─────────────────────────────────────────────────────────── */
let hasGpu = false;

async function detectGPU() {
  const badge = $("gpu-badge");
  const text  = $("gpu-badge-text");
  try {
    const res  = await fetch(`${API}/gpu-info`);
    const data = await res.json();
    hasGpu = !!data.gpu;

    if (data.gpu) {
      badge.className = "gpu-badge gpu-badge--detected";
      text.textContent = `GPU: ${data.name} ✓`;
      // Pre-select "gpu" mode when GPU is present
      setMode("gpu");
    } else {
      badge.className = "gpu-badge gpu-badge--none";
      text.textContent = "No GPU detected — Without GPU mode recommended";
      // Pre-select "cpu" mode on CPU
      setMode("cpu");
    }
  } catch {
    hasGpu = false;
    badge.className = "gpu-badge gpu-badge--none";
    text.textContent = "GPU detection unavailable";
  }
}

/* ── Mode toggle ───────────────────────────────────────────────────────────── */
let currentMode = "gpu";

function setMode(mode) {
  currentMode = mode;
  $("mode-gpu").classList.toggle("mode-btn--active", mode === "gpu");
  $("mode-cpu").classList.toggle("mode-btn--active", mode === "cpu");
}

$("mode-gpu").addEventListener("click",   () => setMode("gpu"));
$("mode-cpu").addEventListener("click", () => setMode("cpu"));

/* ── Threshold slider ──────────────────────────────────────────────────────── */
$("input-threshold").addEventListener("input", e => {
  $("threshold-display").textContent = `${e.target.value}%`;
});

/* ── Progress helpers ──────────────────────────────────────────────────────── */
function setProgress(pct, step, label) {
  $("progress-fill").style.width = `${pct}%`;
  $("progress-pct").textContent  = `${pct}%`;
  $("progress-step").textContent = step;
  $("processing-step-label").textContent = label;
}

/* ── State ─────────────────────────────────────────────────────────────────── */
let activeJobId = null;
let eventSource = null;

/* ── Submit ─────────────────────────────────────────────────────────────────── */
$("run-form").addEventListener("submit", async e => {
  e.preventDefault();

  const url    = $("input-url").value.trim();
  const target = $("input-target").value.trim();
  if (!url || !target) return;

  if (currentMode === "gpu" && !hasGpu) {
    alert("Warning: You do not have a GPU detected. Please select the 'Without GPU' mode instead.");
    return;
  }

  // Initialise
  setProgress(5, "Starting…", "Initialising pipeline");

  showScreen("processing");

  const body = {
    url,
    target,
    gpu:          currentMode === "gpu",
    fast_mode:    currentMode === "gpu",
    threshold:    parseFloat($("input-threshold").value),
    cookies:      $("input-cookies").checked,
    disable_subs: $("input-disable-subs").checked,
    whisper_model: $("input-whisper-model").value,
  };

  let jobId;
  try {
    const res = await fetch(`${API}/run`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    jobId = data.job_id;
    activeJobId = jobId;
  } catch (err) {
    showFailure("Failed to start the pipeline. Is the server running?");
    return;
  }

  // Open SSE stream
  eventSource = new EventSource(`${API}/stream/${jobId}`);

  eventSource.onmessage = ev => {
    let payload;
    try { payload = JSON.parse(ev.data); } catch { return; }

    if (payload.line !== undefined) {
      // Advance progress bar based on keywords
      for (const stage of PROGRESS_STAGES) {
        if (stage.pattern.test(payload.line)) {
          setProgress(stage.pct, stage.step, stage.label);
          break;
        }
      }
    }

    if (payload.done) {
      eventSource.close();
      eventSource = null;
      activeJobId = null;
      setProgress(100, "Complete", "Pipeline finished.");

      if (payload.result && payload.result.status === "success") {
        renderResult(payload.result);
      } else if (payload.result && payload.result.status === "failed") {
        showFailure(payload.result.reason || "No matches found.");
      } else {
        showFailure("Pipeline finished but no result JSON was produced.");
      }
    }
  };

  eventSource.onerror = () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    // Try fetching result anyway (stream may have closed cleanly)
    if (jobId) {
      fetch(`${API}/result/${jobId}`)
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data && data.status === "success") renderResult(data);
          else showFailure(data?.reason || "Stream closed unexpectedly.");
        })
        .catch(() => showFailure("Lost connection to the server."));
    }
  };
});

/* ── Cancel ─────────────────────────────────────────────────────────────────── */
$("cancel-btn").addEventListener("click", async () => {
  if (eventSource) { eventSource.close(); eventSource = null; }
  if (activeJobId) {
    try { await fetch(`${API}/run/${activeJobId}`, { method: "DELETE" }); } catch {}
    activeJobId = null;
  }
  showScreen("input");
});

/* ── Run Again / Retry ──────────────────────────────────────────────────────── */
$("again-btn").addEventListener("click",  () => showScreen("input"));
$("retry-btn").addEventListener("click",  () => showScreen("input"));

/* ── Render result (success) ────────────────────────────────────────────────── */
function renderResult(data) {
  // Detection type banner
  const isVisual = data.detection_type === "visual_text";
  $("result-detection-type").textContent =
    isVisual ? "Visual Text — dialogue found on screen" : "Spoken Dialogue — matched via audio";

  // Timestamp
  $("result-timestamp").textContent = data.timestamp ?? "—";

  // Frame number
  $("result-frame-no").textContent =
    data.frame_number != null ? `#${data.frame_number}` : "—";

  // Matched text
  $("result-text").textContent = data.dialogue_text ?? "—";

  // Score
  const score = data.similarity_score;
  $("result-score").textContent = score != null ? `${score.toFixed(1)}%` : "—";

  // Method
  $("result-method").textContent = data.tool_used ?? data.source ?? "—";

  // Frame image
  const imgEl   = $("result-frame-img");
  const noFrame = $("result-no-frame");
  const imgPath = data.frame_image_path;

  if (imgPath && imgPath !== "" && imgPath !== "null") {
    // Convert local absolute path to a URL served by FastAPI's /output mount
    const relPath = extractRelativeOutputPath(imgPath);
    imgEl.src = relPath ? `/output/${relPath}` : "";
    imgEl.style.display = imgEl.src ? "block" : "none";
    noFrame.style.display = imgEl.src ? "none" : "grid";
  } else {
    imgEl.style.display = "none";
    noFrame.style.display = "grid";
  }

  // Download button
  const dlBtn = $("download-btn");
  if (imgPath && imgPath !== "") {
    const relPath = extractRelativeOutputPath(imgPath);
    dlBtn.style.display = "inline-flex";
    dlBtn.onclick = () => {
      const a = document.createElement("a");
      a.href = `/output/${relPath}`;
      a.download = relPath?.split(/[\\/]/).pop() ?? "frame.png";
      a.click();
    };
  } else {
    dlBtn.style.display = "none";
  }

  showScreen("result");
}

/**
 * Convert an absolute local path like
 *   C:\Users\...\output\video123\frame_200.png
 * to a relative path like
 *   video123/frame_200.png
 * so we can build /output/video123/frame_200.png
 */
function extractRelativeOutputPath(absPath) {
  if (!absPath) return null;
  // Normalise separators
  const normalized = absPath.replace(/\\/g, "/");
  const marker = "/output/";
  const idx = normalized.indexOf(marker);
  if (idx !== -1) return normalized.slice(idx + marker.length);
  // Fallback: just the last two segments
  const parts = normalized.split("/");
  return parts.slice(-2).join("/");
}

/* ── Render failure ─────────────────────────────────────────────────────────── */
function showFailure(reason) {
  $("failure-reason").textContent = reason || "";
  showScreen("failure");
}

/* ── Init ───────────────────────────────────────────────────────────────────── */
detectGPU();
