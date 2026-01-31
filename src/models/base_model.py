"""
Base Model Interface

Definiert die abstrakte Basisklasse für alle Modell-Wrapper.
Dies ermöglicht einheitliche Schnittstellen für verschiedene Modelltypen.

Warum wichtig: Durch eine gemeinsame Schnittstelle können wir verschiedene
Modelle (LLM, SLM, finetuned) auf die gleiche Weise verwenden.
Das vereinfacht Evaluation und macht den Code wartbarer.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstrakte Basisklasse für alle Modelle.

    Warum Abstract Base Class:
    - Erzwingt konsistente Schnittstelle für alle Modell-Implementierungen
    - Macht Codebase wartbarer und testbarer
    - Ermöglicht Polymorphismus (verschiedene Modelle austauschbar nutzen)

    Design Pattern: Strategy Pattern.
    """

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float16"):
        """
        Initialisiert das Basis-Modell.

        Args:
            model_name: Name/Pfad des Modells
            device: Device für Inference ("cuda", "cpu", "mps")
            dtype: Datentyp für Modell ("float32", "float16", "bfloat16")
        """
        self.model_name = model_name
        self.device = device
        self.dtype = self._get_torch_dtype(dtype)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")

    @staticmethod
    def _get_torch_dtype(dtype_str: str) -> torch.dtype:
        """
        Konvertiert String zu PyTorch dtype.

        Warum verschiedene dtypes:
        - float32: Volle Präzision, aber hoher Memory-Verbrauch
        - float16: Halber Memory-Verbrauch, minimaler Qualitätsverlust
        - bfloat16: Wie float16 aber besserer Wertebereich (für neuere GPUs)
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float16)

    @abstractmethod
    def load(self) -> None:
        """
        Lädt das Modell und den Tokenizer.

        Muss von jeder Subclass implementiert werden.
        """
        pass

    @abstractmethod
    def predict(
        self, input_text: Union[str, List[str]], **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Generiert Predictions für Input-Text(e).

        Args:
            input_text: Einzelner Text oder Liste von Texten
            **generation_kwargs: Zusätzliche Parameter für Generation

        Returns:
            Generierter Text oder Liste von Texten
        """
        pass

    @abstractmethod
    def predict_batch(
        self, input_texts: List[str], batch_size: int = 8, **generation_kwargs
    ) -> List[str]:
        """
        Generiert Predictions für Batch von Inputs (effizienter).

        Args:
            input_texts: Liste von Input-Texten
            batch_size: Batch-Größe für Verarbeitung
            **generation_kwargs: Generation-Parameter

        Returns:
            Liste von generierten Texten
        """
        pass

    def unload(self) -> None:
        """Entlädt Modell aus dem Memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clean up CUDA Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info(f"Model {self.model_name} unloaded")

    def get_model_info(self) -> Dict:
        """
        Gibt Informationen über das Modell zurück.

        Returns:
            Dict mit Modell-Metadaten
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__,
        }

        # Füge Modellgröße hinzu wenn geladen
        if self.is_loaded and self.model is not None:
            try:
                # Zähle Parameter
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )

                info["total_parameters"] = total_params
                info["trainable_parameters"] = trainable_params
                info["parameters_in_millions"] = round(total_params / 1e6, 2)

                # Schätze Memory-Verbrauch
                param_memory_mb = (total_params * self.dtype.itemsize) / (1024**2)
                info["estimated_memory_mb"] = round(param_memory_mb, 2)

            except Exception as e:
                logger.warning(f"Could not compute model info: {e}")

        return info

    def save(self, save_path: Path) -> None:
        """
        Speichert Modell und Tokenizer.

        Args:
            save_path: Pfad zum Speichern
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before saving")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_path}")

        # Speichere Modell
        self.model.save_pretrained(str(save_path))

        # Speichere Tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info("Model saved successfully")

    def __repr__(self) -> str:
        """String-Repräsentation des Modells."""
        return f"{self.__class__.__name__}(model_name={self.model_name}, device={self.device})"

    def __enter__(self):
        """Context Manager Support: Lädt Modell beim Eintreten."""
        if not self.is_loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support: Entlädt Modell beim Verlassen."""
        self.unload()


class ModelMetrics:
    """
    Hilfsklasse zum Tracking von Modell-Metriken während Inference.

    Warum wichtig: Performance-Metriken (Latenz, Throughput) sind
    wichtig für Produktions-Entscheidungen.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Setzt alle Metriken zurück."""
        self.total_samples = 0
        self.total_time = 0.0
        self.total_tokens_generated = 0
        self.latencies = []

    def update(
        self, n_samples: int, time_taken: float, n_tokens: Optional[int] = None
    ) -> None:
        """
        Updated Metriken mit neuen Measurements.

        Args:
            n_samples: Anzahl verarbeiteter Samples
            time_taken: Benötigte Zeit in Sekunden
            n_tokens: Anzahl generierter Tokens
        """
        self.total_samples += n_samples
        self.total_time += time_taken
        self.latencies.append(time_taken / n_samples if n_samples > 0 else 0)

        if n_tokens is not None:
            self.total_tokens_generated += n_tokens

    def get_metrics(self) -> Dict:
        """
        Berechnet aggregierte Metriken.

        Returns:
            Dict mit verschiedenen Performance-Metriken
        """
        if self.total_samples == 0:
            return {}

        metrics = {
            "total_samples": self.total_samples,
            "total_time_seconds": round(self.total_time, 2),
            "average_latency_seconds": round(self.total_time / self.total_samples, 3),
            "throughput_samples_per_second": (
                round(self.total_samples / self.total_time, 2)
                if self.total_time > 0
                else 0
            ),
        }

        if self.total_tokens_generated > 0:
            metrics["total_tokens_generated"] = self.total_tokens_generated
            metrics["tokens_per_second"] = (
                round(self.total_tokens_generated / self.total_time, 2)
                if self.total_time > 0
                else 0
            )

        # Latency-Statistiken
        if self.latencies:
            import numpy as np

            metrics["latency_p50_seconds"] = round(np.percentile(self.latencies, 50), 3)
            metrics["latency_p95_seconds"] = round(np.percentile(self.latencies, 95), 3)
            metrics["latency_p99_seconds"] = round(np.percentile(self.latencies, 99), 3)

        return metrics


if __name__ == "__main__":
    # Test ModelMetrics
    metrics = ModelMetrics()

    # Simuliere einige Predictions
    import time

    for i in range(10):
        start = time.time()
        time.sleep(0.1)  # Simuliere Verarbeitung
        elapsed = time.time() - start
        metrics.update(n_samples=1, time_taken=elapsed, n_tokens=50)

    print("Metrics:", metrics.get_metrics())
