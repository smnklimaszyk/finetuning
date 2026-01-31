"""
Base Configuration Module

Dieses Modul definiert die Basiskonfiguration für das gesamte Projekt.
Es folgt dem MLOps Best Practice Ansatz mit:
- Zentralisierter Konfiguration
- Environment-spezifischen Overrides
- Type-Safe Configuration mit Pydantic
- Versionierung der Konfiguration
"""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import os


class PathConfig(BaseModel):
    """
    Konfiguration für alle Pfade im Projekt.

    Warum wichtig: Zentralisierte Pfadverwaltung ermöglicht einfache
    Anpassung bei unterschiedlichen Deployment-Umgebungen.
    """

    # Root-Verzeichnis des Projekts (src/config -> src -> project_root)
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    # Daten-Verzeichnisse (im Projekt-Root, nicht in src/)
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data"
    )
    raw_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "raw"
    )
    processed_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "data"
        / "processed"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "cache"
    )

    # Modell-Verzeichnisse
    models_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "models"
    )
    checkpoints_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "checkpoints"
    )
    finetuned_models_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "models"
        / "finetuned"
    )

    # Output-Verzeichnisse
    outputs_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "outputs"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "outputs" / "logs"
    )
    metrics_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "outputs"
        / "metrics"
    )
    plots_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "outputs"
        / "plots"
    )
    reports_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "outputs"
        / "reports"
    )

    # Experiment Tracking
    experiments_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "experiments"
    )
    mlflow_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "experiments"
        / "mlruns"
    )

    class Config:
        arbitrary_types_allowed = True

    def create_directories(self) -> None:
        """
        Erstellt alle konfigurierten Verzeichnisse.

        Warum wichtig: Stellt sicher, dass alle benötigten Ordner existieren,
        bevor das Training startet. Verhindert FileNotFound-Fehler.
        """
        for _, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    """
    Konfiguration für Datenverarbeitung und -loading.

    Warum wichtig: Reproduzierbarkeit und konsistente Datenverarbeitung
    über verschiedene Experimente hinweg.
    """

    # Dataset-Informationen
    dataset_name: str = "Ahmad0067/MedSynth"
    dataset_split_seed: int = 42  # Für reproduzierbare Splits

    # Train/Val/Test Split Ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Datenverarbeitungs-Parameter
    max_sequence_length: int = 512  # Maximum Token-Länge
    truncation: bool = True
    padding: str = "max_length"

    # Batch Processing
    batch_size: int = 8
    num_workers: int = 1  # Auf 1 gesetzt für macOS Kompatibilität (Fork-Probleme mit multiprocessing)
    prefetch_factor: int = 2

    # Caching (Speed-up für wiederholte Läufe)
    use_cache: bool = True
    cache_batch_size: int = 1000

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    def validate_ratios(cls, v):
        """Validiert, dass Ratios zwischen 0 und 1 liegen."""
        if not 0 < v < 1:
            raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v

    @field_validator("test_ratio")
    def validate_sum(cls, v, values):
        """Validiert, dass alle Ratios zusammen 1.0 ergeben."""
        total = values.get("train_ratio", 0) + values.get("val_ratio", 0) + v
        if not abs(total - 1.0) < 0.001:  # Floating-point tolerance
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        return v


class LLMModelConfig(BaseModel):
    """
    Konfiguration für ein einzelnes LLM.
    
    Ermöglicht die Definition mehrerer LLMs mit individuellen Einstellungen
    für den Modellvergleich.
    """
    name: str  # HuggingFace Modellname
    description: str = ""  # Kurze Beschreibung des Modells
    load_in_4bit: bool = True  # Quantisierung für weniger Memory
    load_in_8bit: bool = False  # Alternative Quantisierung
    dtype: str = "float16"  # Datentyp für Berechnungen


class ModelConfig(BaseModel):
    """
    Konfiguration für Modellarchitektur und -parameter.

    Warum wichtig: Zentrale Definition aller Modell-Hyperparameter
    ermöglicht einfaches Experiment-Tracking und Reproduzierbarkeit.
    """

    # Baseline LLMs (Liste von Modellen für Vergleichsevaluation)
    # Jedes Modell hat seine eigene Konfiguration für Flexibilität
    baseline_llms: List[LLMModelConfig] = Field(
        default=[
            LLMModelConfig(
                name="Qwen/Qwen2.5-3B-Instruct",
                description="3B Parameter - kompaktes aber leistungsstarkes Modell",
                load_in_4bit=True,
            ),
            LLMModelConfig(
                name="mistralai/Mistral-7B-Instruct-v0.3",
                description="7B Parameter - größere Referenz für Vergleich",
                load_in_4bit=True,
            ),
        ],
        description="Liste der Baseline-LLMs mit individuellen Konfigurationen"
    )

    # Small Language Model (für Finetuning)
    slm_name: str = "microsoft/Phi-3-mini-4k-instruct"  # Kompaktes Modell
    slm_device: str = "cuda"
    slm_dtype: str = "float16"
    slm_load_in_8bit: bool = False
    slm_load_in_4bit: bool = True

    # Tokenizer Settings
    tokenizer_padding_side: str = "right"
    tokenizer_truncation_side: str = "right"
    add_special_tokens: bool = True

    # Generation Parameters (für Inference)
    max_new_tokens: int = 256
    temperature: float = 0.7  # Kontrolliert Kreativität (niedrig = deterministischer)
    top_p: float = 0.9  # Nucleus Sampling
    top_k: int = 50  # Top-K Sampling
    repetition_penalty: float = 1.1  # Verhindert Wiederholungen
    do_sample: bool = True  # Aktiviert Sampling statt greedy decoding


class TrainingConfig(BaseModel):
    """
    Konfiguration für das Finetuning-Training.

    Warum wichtig: Diese Hyperparameter bestimmen die Qualität und
    Geschwindigkeit des Trainings. Jeder Parameter hat einen Einfluss
    auf das Endergebnis.
    """

    # Training Hyperparameters
    num_epochs: int = 3  # Anzahl kompletter Durchläufe durch den Datensatz
    learning_rate: float = 2e-5  # Wichtigster Hyperparameter - bestimmt Schrittgröße
    warmup_steps: int = (
        500  # Langsames Ansteigen der LR zu Beginn (stabilisiert Training)
    )
    weight_decay: float = 0.01  # L2-Regularisierung (verhindert Overfitting)

    # Optimizer Settings
    optimizer: str = "adamw"  # AdamW ist State-of-the-Art für Transformer
    adam_beta1: float = 0.9  # Momentum für ersten Moment
    adam_beta2: float = 0.999  # Momentum für zweiten Moment
    adam_epsilon: float = 1e-8  # Numerische Stabilität

    # Learning Rate Scheduler
    lr_scheduler_type: str = "cosine"  # Alternativen: "linear", "constant"
    num_warmup_steps: int = 500

    # Batch Sizes und Gradient Accumulation
    per_device_train_batch_size: int = 4  # Batch Size pro GPU/Device
    per_device_eval_batch_size: int = 8  # Kann größer sein, da kein Backprop
    gradient_accumulation_steps: int = 4  # Simuliert größere Batch Size
    # Effektive Batch Size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

    # Mixed Precision Training (spart Memory und beschleunigt Training)
    fp16: bool = False  # Für NVIDIA GPUs (vor Ampere)
    bf16: bool = True  # Für neuere GPUs (Ampere+) - bessere numerische Stabilität

    # Gradient Clipping (verhindert explodierende Gradienten)
    max_grad_norm: float = 1.0

    # Logging und Evaluation
    logging_steps: int = 10  # Log alle X Steps
    eval_steps: int = 100  # Evaluate alle X Steps
    save_steps: int = 500  # Checkpoint alle X Steps
    save_total_limit: int = 3  # Maximal 3 Checkpoints behalten (spart Speicher)

    # Evaluation Strategy
    evaluation_strategy: str = "steps"  # Alternativen: "epoch", "no"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True  # Lädt bestes Modell nach Training
    metric_for_best_model: str = "eval_loss"  # Metrik für "bestes" Modell
    greater_is_better: bool = False  # False für Loss (niedriger ist besser)

    # Early Stopping (stoppt Training wenn keine Verbesserung mehr)
    early_stopping_patience: int = 3  # Stoppt nach 3 Evaluationen ohne Verbesserung
    early_stopping_threshold: float = 0.001  # Minimale Verbesserung

    # LoRA (Parameter-Efficient Fine-Tuning)
    use_lora: bool = True  # Aktiviert LoRA (nur kleine Adapter-Weights trainieren)
    lora_r: int = 16  # Rank der LoRA-Matrizen (8-32 typisch, höher = mehr Kapazität)
    lora_alpha: int = 32  # Scaling-Faktor (typisch 2*r)
    lora_dropout: float = 0.05  # Dropout für Regularisierung
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]  # Welche Layer

    # Checkpointing
    gradient_checkpointing: bool = True  # Spart Memory (trade-off: langsamer)

    # Reproducibility
    seed: int = 42  # Für reproduzierbare Ergebnisse

    @field_validator("learning_rate")
    def validate_lr(cls, v):
        """Validiert, dass Learning Rate in sinnvollem Bereich liegt."""
        if not 1e-6 <= v <= 1e-3:
            raise ValueError(f"Learning rate should be between 1e-6 and 1e-3, got {v}")
        return v


class EvaluationConfig(BaseModel):
    """
    Konfiguration für Model Evaluation und Metriken.

    Warum wichtig: Definiert wie wir Modellqualität messen.
    Unterschiedliche Metriken geben unterschiedliche Einblicke.
    """

    # Metriken
    metrics: List[str] = [
        "accuracy",  # Gesamtgenauigkeit
        "precision",  # Wie viele positive Predictions waren korrekt?
        "recall",  # Wie viele tatsächliche Positive wurden gefunden?
        "f1",  # Harmonisches Mittel von Precision und Recall
        "confusion_matrix",  # Detaillierte Fehleranalyse
        "classification_report",  # Umfassender Report
    ]

    # ICD-10 spezifische Evaluierung
    evaluate_top_k: List[int] = [
        1,
        3,
        5,
    ]  # Top-K Accuracy (ist richtige Diagnose in Top-5?)
    icd_level_evaluation: bool = (
        True  # Evaluiert auf verschiedenen ICD-Hierarchie-Ebenen
    )

    # Inferenz-Settings
    eval_batch_size: int = 16
    max_eval_samples: Optional[int] = None  # Limitiert Eval-Set (für schnelle Tests)

    # Performance-Metriken
    measure_latency: bool = True  # Misst Antwortzeit pro Sample
    measure_throughput: bool = True  # Misst Samples/Sekunde
    measure_memory: bool = True  # Misst GPU/CPU Memory Usage

    # Reporting
    generate_plots: bool = True  # Erstellt Visualisierungen
    save_predictions: bool = True  # Speichert alle Predictions für Analyse
    save_errors: bool = True  # Speichert nur falsche Predictions für Error-Analysis


class LoggingConfig(BaseModel):
    """
    Konfiguration für Logging und Experiment Tracking.

    Warum wichtig: Gutes Logging ist essentiell für Debugging und
    das Nachvollziehen von Experimenten.
    """

    # Console Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # File Logging
    log_to_file: bool = True
    log_file_name: str = "training.log"

    # MLflow Tracking
    use_mlflow: bool = True
    mlflow_experiment_name: str = "medical-diagnosis-finetuning"
    mlflow_tracking_uri: Optional[str] = None  # Default: file-based in experiments/

    # Weights & Biases (alternatives Tracking-Tool)
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "outputs/tensorboard"


class ExperimentConfig(BaseModel):
    """
    Konfiguration für ein spezifisches Experiment.

    Warum wichtig: Ermöglicht das Ausführen verschiedener Experimente
    mit unterschiedlichen Konfigurationen und deren Vergleich.
    """

    experiment_name: str = "baseline_experiment"
    experiment_description: str = "Initial baseline experiment"
    tags: List[str] = []

    # Welche Modelle sollen verglichen werden?
    run_baseline_llm: bool = True
    run_baseline_slm: bool = True  # SLM ohne Finetuning
    run_finetuned_slm: bool = True

    # Experiment Reproducibility
    seed: int = 42
    deterministic: bool = (
        True  # Macht CUDA deterministisch (langsamer, aber reproduzierbar)
    )


class Config(BaseModel):
    """
    Haupt-Konfigurationsklasse die alle Sub-Konfigurationen vereint.

    Diese Klasse ist der zentrale Einstiegspunkt für alle Konfigurationen.
    """

    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    class Config:
        arbitrary_types_allowed = True

    def save(self, path: Path) -> None:
        """Speichert Konfiguration als JSON für Reproduzierbarkeit."""
        import json

        with open(path, "w") as f:
            # Custom JSON encoder für Path objects
            json.dump(self.dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Lädt Konfiguration aus JSON-Datei."""
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def setup(self) -> None:
        """
        Initialisiert alle Verzeichnisse und Basis-Einstellungen.

        Diese Methode sollte zu Beginn jedes Experiments aufgerufen werden.
        """
        # Erstelle alle Verzeichnisse
        self.paths.create_directories()

        # Setze Seeds für Reproduzierbarkeit
        self._set_seeds()

        # Konfiguriere CUDA für Determinismus
        if self.experiment.deterministic:
            self._set_deterministic()

    def _set_seeds(self) -> None:
        """Setzt alle Random Seeds für Reproduzierbarkeit."""
        import random
        import numpy as np
        import torch

        seed = self.experiment.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _set_deterministic(self) -> None:
        """Konfiguriert PyTorch für deterministisches Verhalten."""
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Dies kann Training verlangsamen, garantiert aber Reproduzierbarkeit


def get_config() -> Config:
    """
    Factory-Funktion zum Erstellen der Default-Konfiguration.

    Returns:
        Config: Vollständige Konfiguration mit allen Defaults

    Verwendung:
        config = get_config()
        config.training.learning_rate = 1e-4  # Override einzelner Werte
        config.setup()  # Initialisierung
    """
    return Config()


# Für direkten Import
__all__ = [
    "Config",
    "PathConfig",
    "DataConfig",
    "LLMModelConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "get_config",
]
