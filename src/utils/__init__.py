"""
Utility Functions
"""

import logging
from pathlib import Path
import json
import torch
import os
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Richtet Logging ein."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "app.log"), logging.StreamHandler()],
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
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Lädt JSON als Dict."""
    with open(path, "r") as f:
        return json.load(f)


def generate_cache_key(
    model_name: str,
    dataset_size: int,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generiert eindeutigen Cache-Key für Predictions.

    Args:
        model_name: Name des Modells
        dataset_size: Größe des Test-Datensatzes
        generation_config: Generation-Parameter (temperature, max_tokens, etc.)

    Returns:
        Eindeutiger Hash-String
    """
    # Erstelle String mit allen relevanten Parametern
    cache_parts = [
        model_name.replace("/", "_"),
        str(dataset_size),
    ]

    if generation_config:
        # Sortiere für konsistente Hashes
        config_str = json.dumps(generation_config, sort_keys=True)
        cache_parts.append(config_str)

    # Generiere Hash
    cache_string = "|".join(cache_parts)
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()[:12]

    # Format: ModelName_hash
    model_short = model_name.split("/")[-1].replace(".", "_")
    return f"{model_short}_{cache_hash}"


def get_cached_predictions(
    cache_key: str, cache_dir: Path, max_age_days: Optional[int] = None
) -> Optional[Dict]:
    """
    Lädt gecachte Predictions falls verfügbar.

    Args:
        cache_key: Cache-Identifikator
        cache_dir: Verzeichnis für Cache-Dateien
        max_age_days: Maximales Alter in Tagen (None = unbegrenzt)

    Returns:
        Dictionary mit gecachten Daten oder None
    """
    cache_file = cache_dir / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    # Prüfe Alter der Cache-Datei
    if max_age_days is not None:
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - file_time
        if age > timedelta(days=max_age_days):
            logging.info(
                f"⏰ Cache expired (age: {age.days} days, max: {max_age_days})"
            )
            return None

    try:
        return load_json(cache_file)
    except Exception as e:
        logging.warning(f"⚠️  Failed to load cache: {e}")
        return None


def save_predictions_cache(
    cache_key: str,
    cache_dir: Path,
    predictions: list,
    references: list,
    metrics: Dict,
    metadata: Dict,
) -> Path:
    """
    Speichert Predictions im Cache.

    Args:
        cache_key: Cache-Identifikator
        cache_dir: Verzeichnis für Cache-Dateien
        predictions: Liste der Vorhersagen
        references: Liste der Referenz-Labels
        metrics: Berechnete Metriken
        metadata: Zusätzliche Metadaten (Modellname, Config, etc.)

    Returns:
        Pfad zur Cache-Datei
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"

    cache_data = {
        "metadata": {
            **metadata,
            "timestamp": datetime.now().isoformat(),
            "cache_version": "1.0",
        },
        "predictions": predictions,
        "references": references,
        "metrics": metrics,
    }

    save_json(cache_data, cache_file)
    return cache_file


def clear_prediction_cache(cache_dir: Path, older_than_days: Optional[int] = None):
    """
    Löscht Prediction-Cache.

    Args:
        cache_dir: Cache-Verzeichnis
        older_than_days: Optional - nur Dateien älter als X Tage löschen
    """
    if not cache_dir.exists():
        return

    count = 0
    for cache_file in cache_dir.glob("*.json"):
        should_delete = True

        if older_than_days is not None:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age = datetime.now() - file_time
            should_delete = age > timedelta(days=older_than_days)

        if should_delete:
            cache_file.unlink()
            count += 1

    logging.info(f"🗑️  Deleted {count} cache files")


def log_gpu_memory(prefix: str = ""):
    """
    Loggt aktuellen GPU Memory Status.

    Args:
        prefix: Optional prefix für Log-Nachricht (z.B. "Before training")

    Warum wichtig: Hilft bei Debugging von OOM-Fehlern und Memory-Leaks.
    Zeigt allocated (tatsächlich genutzt) vs. reserved (von PyTorch reserviert).
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

    msg = f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
    if prefix:
        msg = f"{prefix} - {msg}"

    logging.info(msg)


def aggressive_memory_cleanup():
    """
    Führt aggressive Memory-Cleanup durch.

    Warum wichtig: Bei sequentiellem Laden mehrerer großer Modelle
    kann Memory-Fragmentierung auftreten. Diese Funktion stellt sicher,
    dass zwischen Modellen wirklich alles freigegeben wird.

    Nutzung:
        model.unload()
        aggressive_memory_cleanup()
        # Nun ist Memory bereit für nächstes Modell
    """
    import gc

    # Python Garbage Collection
    gc.collect()

    # CUDA Cleanup (wenn verfügbar)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Gibt cached memory frei
        torch.cuda.synchronize()  # Wartet auf alle CUDA operations
        torch.cuda.ipc_collect()  # Sammelt inter-process memory

    logging.debug("Aggressive memory cleanup completed")


def get_gpu_memory_stats() -> Dict[str, float]:
    """
    Gibt detaillierte GPU Memory Statistiken zurück.

    Returns:
        Dict mit Memory-Statistiken in GB, oder leeres Dict wenn keine GPU

    Beispiel:
        {
            'allocated_gb': 5.2,
            'reserved_gb': 6.0,
            'free_gb': 10.8,
            'total_gb': 16.0
        }
    """
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }


__all__ = [
    "setup_logging",
    "get_device",
    "save_json",
    "load_json",
    "generate_cache_key",
    "get_cached_predictions",
    "save_predictions_cache",
    "clear_prediction_cache",
    "log_gpu_memory",
    "aggressive_memory_cleanup",
    "get_gpu_memory_stats",
]
