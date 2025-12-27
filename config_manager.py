from pathlib import Path
import json
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "web_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "words_level": "A1",
    "words_model": "gemma2:2b",
    "words_batch_size": 50,
    "words_outdir": "out_french",

    "sent_level": "A1",
    "sent_model": "gemma2:2b",
    "sent_input_dir": "out_french",
    "sent_output_dir": "out_sentences",

    "log_dir": "logs",
    "ollama_model_name": "gemma2:2b",
}


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            merged = {**DEFAULT_CONFIG, **data}
            return merged
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
