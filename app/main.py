from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import StreamingResponse

APP_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUT_DIR = DATA_DIR / "out"
JOBS_DIR = DATA_DIR / "jobs"
METRICS_PATH = DATA_DIR / "metrics.json"

WHISPER_BIN = os.environ.get("WHISPER_BIN", "/opt/whisper.cpp/build/bin/whisper-cli")
MODEL_PATH = os.environ.get("WHISPER_MODEL", "/models/ggml-base.en.bin")

SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}$")

app = FastAPI(title="Whisper Transcription Service")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_upload(dest: Path, up: UploadFile) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ffprobe_duration_seconds(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        ).decode("utf-8", "ignore").strip()
        return float(out)
    except Exception:
        return None


def load_metrics() -> dict:
    try:
        return json.loads(METRICS_PATH.read_text("utf-8"))
    except Exception:
        return {"rtf_avg": 0.0, "samples": 0}


def save_metrics(m: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(m, indent=2), "utf-8")


@dataclass
class Job:
    job_id: str
    name: str
    created_at: float
    stage: str = "queued"
    error: str = ""
    audio_path: str = ""
    audio_seconds: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0


JOBS: dict[str, Job] = {}
JOB_QUEUE: queue.Queue[str] = queue.Queue()


def job_log_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.log"


def job_write_log(job_id: str, line: str) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with job_log_path(job_id).open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def job_eta_seconds(job: Job) -> float | None:
    # Estimate total time from historical real-time factor (RTF = processing_seconds / audio_seconds)
    if job.audio_seconds <= 0:
        return None
    m = load_metrics()
    rtf = float(m.get("rtf_avg") or 0.0)
    if rtf <= 0:
        return None
    total = job.audio_seconds * rtf
    if job.started_at:
        elapsed = time.time() - job.started_at
        return max(0.0, total - elapsed)
    return total


def worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if not job:
            continue
        try:
            job.started_at = time.time()
            job.stage = "converting"
            job_write_log(job_id, f"Stage: {job.stage}")

            src_path = Path(job.audio_path)
            wav_path = UPLOADS_DIR / f"{job.name}.wav"
            cmd_ff = [
                "ffmpeg",
                "-y",
                "-i",
                str(src_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ]
            job_write_log(job_id, "Running ffmpeg...")
            subprocess.check_call(cmd_ff)

            job.stage = "transcribing"
            job_write_log(job_id, f"Stage: {job.stage}")

            out_base = OUT_DIR / job.name
            out_txt = OUT_DIR / f"{job.name}.txt"
            out_srt = OUT_DIR / f"{job.name}.srt"
            out_vtt = OUT_DIR / f"{job.name}.vtt"

            cmd_wh = [
                WHISPER_BIN,
                "-m",
                MODEL_PATH,
                "-f",
                str(wav_path),
                "-otxt",
                "-osrt",
                "-ovtt",
                "-of",
                str(out_base),
            ]
            job_write_log(job_id, "Running whisper-cli...")

            # capture stdout/stderr to log
            p = subprocess.Popen(cmd_wh, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert p.stdout is not None
            for line in p.stdout:
                job_write_log(job_id, line.rstrip())
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"whisper-cli exited with {rc}")

            if not (out_txt.exists() or out_srt.exists() or out_vtt.exists()):
                raise RuntimeError("transcription output not found")

            job.stage = "done"
            job.finished_at = time.time()
            job_write_log(job_id, "Stage: done")

            # update RTF average
            if job.audio_seconds > 0 and job.started_at > 0:
                elapsed = job.finished_at - job.started_at
                rtf_sample = elapsed / job.audio_seconds
                m = load_metrics()
                n = int(m.get("samples") or 0)
                prev = float(m.get("rtf_avg") or 0.0)
                new = (prev * n + rtf_sample) / (n + 1)
                m["rtf_avg"] = new
                m["samples"] = n + 1
                save_metrics(m)

        except Exception as e:
            job.stage = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.finished_at = time.time()
            job_write_log(job_id, f"ERROR: {job.error}")
        finally:
            JOB_QUEUE.task_done()


threading.Thread(target=worker_loop, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    outs = []
    if OUT_DIR.exists():
        for p in sorted(OUT_DIR.glob("*.vtt"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            outs.append({"name": p.name})
    return TEMPLATES.TemplateResponse(request, "index.html", {"outs": outs, "model": MODEL_PATH})


@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    name: str = Form(default=""),
):
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found in container")

    job_name = (name or "").strip() or f"job_{now_stamp()}"
    if not SAFE_NAME_RE.match(job_name):
        raise HTTPException(400, "Name must be filename-safe (letters/numbers/_/-)")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_ext = Path(audio.filename or "audio.mp3").suffix or ".mp3"
    src_path = UPLOADS_DIR / f"{job_name}{src_ext}"
    save_upload(src_path, audio)

    # create job
    job_id = uuid.uuid4().hex
    dur = ffprobe_duration_seconds(src_path) or 0.0
    job = Job(job_id=job_id, name=job_name, created_at=time.time(), audio_path=str(src_path), audio_seconds=dur)
    JOBS[job_id] = job

    eta = job_eta_seconds(job)
    if eta is not None:
        job_write_log(job_id, f"Estimated time remaining: ~{int(eta)}s")

    JOB_QUEUE.put(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404)

    # outputs (when done)
    outputs = []
    for ext in ("vtt", "srt", "txt"):
        p = OUT_DIR / f"{job.name}.{ext}"
        if p.exists():
            outputs.append(p.name)

    return TEMPLATES.TemplateResponse(
        request,
        "job.html",
        {
            "job": job,
            "eta": job_eta_seconds(job),
            "outputs": outputs,
        },
    )


@app.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)

    def gen():
        log_path = job_log_path(job_id)
        # send existing content first
        pos = 0
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
                pos = f.tell()
            for line in data.splitlines():
                yield f"data: {line}\n\n"

        # stream new lines
        while True:
            time.sleep(0.5)
            if log_path.exists():
                with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
                if new:
                    for line in new.splitlines():
                        yield f"data: {line}\n\n"

            job = JOBS.get(job_id)
            if not job:
                break
            if job.stage in ("done", "error"):
                # final heartbeat with status
                yield f"data: __STATUS__ {job.stage}\n\n"
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/out/{filename}")
def get_out(filename: str):
    p = OUT_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404)

    # basic type
    mt = "text/plain"
    if filename.endswith(".vtt"):
        mt = "text/vtt"
    elif filename.endswith(".srt"):
        mt = "application/x-subrip"

    return FileResponse(str(p), media_type=mt, filename=p.name)
