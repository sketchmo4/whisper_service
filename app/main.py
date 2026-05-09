from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

APP_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUT_DIR = DATA_DIR / "out"

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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    jobs = []
    if OUT_DIR.exists():
        for p in sorted(OUT_DIR.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            jobs.append({"name": p.name, "mtime": int(p.stat().st_mtime)})
    return TEMPLATES.TemplateResponse(request, "index.html", {"jobs": jobs, "model": MODEL_PATH})


@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    name: str = Form(default=""),
):
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found in container")

    job = (name or "").strip() or f"job_{now_stamp()}"
    if not SAFE_NAME_RE.match(job):
        raise HTTPException(400, "Name must be filename-safe (letters/numbers/_/-)")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_ext = Path(audio.filename or "audio.mp3").suffix or ".mp3"
    src_path = UPLOADS_DIR / f"{job}{src_ext}"
    save_upload(src_path, audio)

    # convert to 16k mono wav for whisper.cpp
    wav_path = UPLOADS_DIR / f"{job}.wav"
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
    subprocess.check_call(cmd_ff)

    out_txt = OUT_DIR / f"{job}.txt"

    # whisper-cli writes output via -of (without extension)
    out_base = OUT_DIR / job

    # produce multiple timestamped outputs
    # -otxt: plain text
    # -osrt: subtitles with timestamps
    # -ovtt: webvtt with timestamps
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
    subprocess.check_call(cmd_wh)

    out_srt = OUT_DIR / f"{job}.srt"
    out_vtt = OUT_DIR / f"{job}.vtt"
    if not out_txt.exists() and not out_srt.exists() and not out_vtt.exists():
        raise HTTPException(500, "transcription output not found")

    # prefer timestamped output
    if out_vtt.exists():
        return RedirectResponse(url=f"/out/{out_vtt.name}", status_code=303)
    if out_srt.exists():
        return RedirectResponse(url=f"/out/{out_srt.name}", status_code=303)
    return RedirectResponse(url=f"/out/{out_txt.name}", status_code=303)


@app.get("/out/{filename}")
def get_out(filename: str):
    p = OUT_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="text/plain", filename=p.name)
