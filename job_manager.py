from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Job:
    id: str
    job_type: str
    log_path: Path
    started_at: datetime
    process: subprocess.Popen


_current_job: Optional[Job] = None


def start_job(job_type: str, config: Dict[str, Any]) -> Job:
    global _current_job

    if _current_job and _current_job.process.poll() is None:
        raise RuntimeError("A job is already running")

    log_dir = BASE_DIR / config.get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{job_type}_{ts}"
    log_path = log_dir / f"{job_id}.log"

    if job_type == "words":
        script = BASE_DIR / "generate_words.py"
        cmd = [
            sys.executable,
            str(script),
            "--level",
            config.get("words_level", "A1"),
            "--model",
            config.get("words_model", "gemma2:2b"),
            "--batch-size",
            str(config.get("words_batch_size", 50)),
            "--outdir",
            config.get("words_outdir", "out_french"),
        ]
    elif job_type == "sentences":
        script = BASE_DIR / "generate_sentences.py"
        cmd = [
            sys.executable,
            str(script),
            "--level",
            config.get("sent_level", "A1"),
            "--model",
            config.get("sent_model", "gemma2:2b"),
            "--input-dir",
            config.get("sent_input_dir", "out_french"),
            "--output-dir",
            config.get("sent_output_dir", "out_sentences"),
        ]
    else:
        raise ValueError(f"Unknown job_type: {job_type}")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] Starting job {job_id}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )

    _current_job = Job(
        id=job_id,
        job_type=job_type,
        log_path=log_path,
        started_at=datetime.now(),
        process=proc,
    )
    return _current_job


def stop_current_job() -> None:
    global _current_job
    if not _current_job:
        return
    if _current_job.process.poll() is None:
        try:
            _current_job.process.terminate()
            _current_job.process.wait(timeout=10)
        except Exception:
            pass
    _current_job = None


def get_current_job() -> Optional[Job]:
    return _current_job


def get_log_tail(max_lines: int = 200) -> str:
    job = _current_job
    if not job or not job.log_path.exists():
        return ""
    text = job.log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])
