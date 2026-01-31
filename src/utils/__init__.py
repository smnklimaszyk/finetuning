"""
Utility Functions
"""

import logging
from pathlib import Path
import json
import torch
import os


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Richtet Logging ein."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Reduziere HuggingFace verbose Output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    # Setze HuggingFace Verbosity auf Warnings only
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
    os.environ["DATASETS_VERBOSITY"] = "warning"


def get_device() -> str:
    """Bestimmt bestes verfügbares Device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_json(data: dict, path: Path):
    """Speichert Dict als JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Lädt JSON als Dict."""
    with open(path, 'r') as f:
        return json.load(f)


__all__ = ['setup_logging', 'get_device', 'save_json', 'load_json']
