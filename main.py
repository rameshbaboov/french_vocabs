from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from urllib.parse import quote

from docx import Document  # for DOCX preview

from config_manager import load_config, save_config
from job_manager import start_job, stop_current_job, get_current_job, get_log_tail

import subprocess
import os

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _safe_join(base: Path, rel_path: str) -> Path:
    full = (base / rel_path).resolve()
    if not str(full).startswith(str(base.resolve())):
        raise ValueError("Invalid path")
    return full


def list_output_files(config: Dict[str, Any]):
    words_base = (BASE_DIR / config.get("words_outdir", "out_french")).resolve()
    sent_base = (BASE_DIR / config.get("sent_output_dir", "out_sentences")).resolve()

    words_files: List[Dict[str, Any]] = []
    if words_base.exists():
        for path in sorted(words_base.rglob("*.txt")):
            stat = path.stat()
            rel = path.relative_to(words_base)
            words_files.append(
                {
                    "name": str(rel),
                    "rel_path": str(rel),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "mtime": datetime.fromtimestamp(stat.st_mtime),
                }
            )

    sent_files: List[Dict[str, Any]] = []
    if sent_base.exists():
        for path in sorted(sent_base.rglob("*.docx")):
            stat = path.stat()
            rel = path.relative_to(sent_base)
            sent_files.append(
                {
                    "name": str(rel),
                    "rel_path": str(rel),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "mtime": datetime.fromtimestamp(stat.st_mtime),
                }
            )

    return words_files, sent_files


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, message: str | None = None):
    config = load_config()
    current_job = get_current_job()
    words_files, sent_files = list_output_files(config)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "config": config,
            "current_job": current_job,
            "words_files": words_files,
            "sent_files": sent_files,
            "message": message,
        },
    )


@app.post("/config")
async def update_config(
    words_level: str = Form(...),
    words_model: str = Form(...),
    words_batch_size: int = Form(...),
    words_outdir: str = Form(...),
    sent_level: str = Form(...),
    sent_model: str = Form(...),
    sent_input_dir: str = Form(...),
    sent_output_dir: str = Form(...),
    ollama_model_name: str = Form(...),
):
    cfg = load_config()
    cfg["words_level"] = words_level
    cfg["words_model"] = words_model
    cfg["words_batch_size"] = int(words_batch_size)
    cfg["words_outdir"] = words_outdir

    cfg["sent_level"] = sent_level
    cfg["sent_model"] = sent_model
    cfg["sent_input_dir"] = sent_input_dir
    cfg["sent_output_dir"] = sent_output_dir

    cfg["ollama_model_name"] = ollama_model_name

    save_config(cfg)
    return RedirectResponse("/?message=" + quote("Config saved"), status_code=303)


@app.post("/run")
async def run_job(job_type: str = Form(...)):
    cfg = load_config()
    try:
        start_job(job_type, cfg)
        msg = f"Started {job_type} job"
    except Exception as exc:
        msg = f"Could not start job: {exc}"
    return RedirectResponse("/?message=" + quote(msg), status_code=303)


@app.post("/stop")
async def stop_job():
    stop_current_job()
    return RedirectResponse("/?message=" + quote("Job stopped"), status_code=303)


@app.get("/logs/current", response_class=JSONResponse)
async def current_log():
    job = get_current_job()
    running = bool(job and job.process.poll() is None)
    log_text = get_log_tail()
    return {
        "running": running,
        "job_id": job.id if job else None,
        "job_type": job.job_type if job else None,
        "log": log_text,
    }


@app.get("/preview", response_class=HTMLResponse)
async def preview_file(request: Request, type: str, path: str):
    cfg = load_config()
    if type == "words":
        base = (BASE_DIR / cfg.get("words_outdir", "out_french")).resolve()
    elif type == "sentences":
        base = (BASE_DIR / cfg.get("sent_output_dir", "out_sentences")).resolve()
    else:
        return HTMLResponse("Invalid type", status_code=400)

    try:
        full = _safe_join(base, path)
    except ValueError:
        return HTMLResponse("Invalid path", status_code=400)

    if not full.exists():
        return HTMLResponse("File not found", status_code=404)

    if full.suffix.lower() == ".docx":
        doc = Document(full)
        lines: List[str] = []
        for p in doc.paragraphs[:80]:
            text = p.text.strip()
            if text:
                lines.append(text)
        content = "\n".join(lines)
    else:
        content = full.read_text(encoding="utf-8", errors="ignore")

    return templates.TemplateResponse(
        "preview.html",
        {"request": request, "file_name": full.name, "content": content},
    )


@app.get("/download")
async def download_file(type: str, path: str):
    cfg = load_config()
    if type == "words":
        base = (BASE_DIR / cfg.get("words_outdir", "out_french")).resolve()
    elif type == "sentences":
        base = (BASE_DIR / cfg.get("sent_output_dir", "out_sentences")).resolve()
    else:
        return HTMLResponse("Invalid type", status_code=400)

    try:
        full = _safe_join(base, path)
    except ValueError:
        return HTMLResponse("Invalid path", status_code=400)

    if not full.exists():
        return HTMLResponse("File not found", status_code=404)

    return FileResponse(full, filename=full.name)


@app.post("/ollama/stop-model")
async def ollama_stop_model():
    cfg = load_config()
    model = cfg.get("ollama_model_name", "gemma2:2b")
    try:
        subprocess.run(["ollama", "stop", model], check=False)
        msg = f"Requested stop for model {model}"
    except Exception as exc:
        msg = f"Error stopping model: {exc}"
    return RedirectResponse("/?message=" + quote(msg), status_code=303)


@app.post("/ollama/restart")
async def ollama_restart():
    try:
        subprocess.run(["pkill", "-f", "ollama"], check=False)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(BASE_DIR),
        )
        msg = "Requested Ollama restart"
    except Exception as exc:
        msg = f"Error restarting Ollama: {exc}"
    return RedirectResponse("/?message=" + quote(msg), status_code=303)
