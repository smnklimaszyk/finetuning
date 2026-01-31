"""
Data Processor Module

Dieses Modul verarbeitet Rohdaten in ein für das Training geeignetes Format.
Es führt Tokenisierung, Formatierung und Splitting durch.

Warum wichtig: Das richtige Datenformat ist entscheidend für effektives Training.
Verschiedene Modelle erwarten unterschiedliche Input-Formate (Chat-Templates, etc.).
"""

from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
import logging
from pathlib import Path
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MedicalDialogProcessor:
    """
    Verarbeitet medizinische Dialoge für das Training.

    Diese Klasse ist verantwortlich für:
    1. Formatierung der Dialoge in ein einheitliches Format
    2. Tokenisierung mit dem gewählten Tokenizer
    3. Anwendung von Chat-Templates (für Instruct-Modelle)
    4. Padding und Truncation

    Warum Chat-Templates: Moderne Instruct-Modelle erwarten spezielle
    Formatierungen wie "<|im_start|>user\n..." für optimale Performance.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialisiert den Processor.

        Args:
            tokenizer: HuggingFace Tokenizer
            max_length: Maximale Sequenzlänge
            padding: Padding-Strategie ("max_length", "longest", "do_not_pad")
            truncation: Ob zu lange Sequenzen gekürzt werden sollen
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Stelle sicher, dass Tokenizer ein pad_token hat
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")

    def format_dialog_for_training(
        self, conversation: str, icd_code: str, include_system_prompt: bool = True
    ) -> str:
        """
        Formatiert einen Dialog in das Training-Format.

        Args:
            conversation: Der Arzt-Patienten-Dialog
            icd_code: Der zugehörige ICD-10 Code
            include_system_prompt: Ob System-Prompt hinzugefügt werden soll

        Returns:
            str: Formatierter Text für Training
        """
        # System Prompt definiert die Aufgabe des Modells
        system_prompt = """You are a medical assistance system that supports doctors in making diagnoses. 
Your task is to suggest the appropriate ICD-10 diagnosis code based on a doctor-patient dialogue.
Respond only with the ICD-10 code, without further explanation."""

        # User Prompt enthält den Dialog
        user_prompt = f"""Analyze the following doctor-patient dialogue and determine the appropriate ICD-10 code:

{conversation}

ICD-10 Code:"""

        # Assistant Response ist der erwartete Output
        assistant_response = icd_code

        # Formatiere als Chat (kompatibel mit Chat-Modellen)
        if include_system_prompt:
            formatted_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: {assistant_response}"
        else:
            formatted_text = f"User: {user_prompt}\n\nAssistant: {assistant_response}"

        return formatted_text

    def format_dialog_for_inference(
        self, conversation: str, include_system_prompt: bool = True
    ) -> str:
        """
        Formatiert einen Dialog für Inference (ohne ICD-Code).

        Args:
            conversation: Der Arzt-Patienten-Dialog
            include_system_prompt: Ob System-Prompt hinzugefügt werden soll

        Returns:
            str: Formatierter Text für Inference

        Warum getrennte Funktion: Zur Inference haben wir den ICD-Code
        noch nicht und müssen daher anders formatieren.
        """
        system_prompt = """You are a medical assistance system that supports doctors in making diagnoses. 
Your task is to suggest the appropriate ICD-10 diagnosis code based on a doctor-patient dialogue.
Respond only with the ICD-10 code, without further explanation."""

        user_prompt = f"""Analyze the following doctor-patient dialogue and determine the appropriate ICD-10 code:

{conversation}

ICD-10 Code:"""

        if include_system_prompt:
            formatted_text = (
                f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            )
        else:
            formatted_text = f"User: {user_prompt}\n\nAssistant:"

        return formatted_text

    def apply_chat_template(
        self, conversation: str, icd_code: Optional[str] = None
    ) -> str:
        """
        Wendet das Chat-Template des Tokenizers an (falls vorhanden).

        Args:
            conversation: Dialog-Text
            icd_code: ICD-Code (None für Inference)

        Returns:
            str: Formatierter Text mit Chat-Template

        Warum Chat-Templates: Viele moderne Modelle (Llama, Phi, etc.)
        sind auf spezifische Chat-Formate trainiert. Die Verwendung des
        richtigen Templates verbessert die Performance deutlich.
        """
        # Prüfe ob Tokenizer Chat-Template unterstützt
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            # Baue Messages-Liste (Standard Chat-Format)
            messages = [
                {
                    "role": "system",
                    "content": "You are a medical assitance system for diagnotic support.",
                },
                {
                    "role": "user",
                    "content": f"Analyze this dialogue and determine the ICD-10 code:\n\n{conversation}",
                },
            ]

            # Füge Assistant-Response hinzu wenn ICD-Code vorhanden (Training)
            if icd_code is not None:
                messages.append({"role": "assistant", "content": icd_code})

            # Wende Template an
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=(icd_code is None),  # Für Inference
                )
                return formatted_text
            except Exception as e:
                logger.warning(f"Could not apply chat template: {e}. Using fallback.")

        # Fallback: Nutze eigenes Format
        if icd_code is not None:
            return self.format_dialog_for_training(conversation, icd_code)
        else:
            return self.format_dialog_for_inference(conversation)

    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenisiert einen Batch von Beispielen.

        Args:
            examples: Dict mit Listen von Dialogen und ICD-Codes

        Returns:
            Dict: Tokenisierte Inputs mit input_ids, attention_mask, labels

        Warum batched: Effizienter als einzelne Samples zu tokenisieren.
        HuggingFace Datasets nutzt diese Funktion mit map() für schnelle
        Verarbeitung großer Datensätze.
        """
        # Finde die richtigen Feld-Namen im Dataset
        # MedSynth Dataset hat: 'Dialogue', 'ICD10', 'ICD10_desc', ' Note'
        text_field = None
        label_field = None

        for field in ["Dialogue", "dialogue", "conversation", "text", "input"]:
            if field in examples:
                text_field = field
                break

        for field in ["ICD10", "icd10", "icd_code", "ICD_CODE", "label", "diagnosis"]:
            if field in examples:
                label_field = field
                break

        if not text_field or not label_field:
            raise ValueError(
                f"Could not find text and label fields. Available: {examples.keys()}"
            )

        # Formatiere alle Beispiele
        formatted_texts = []
        for conversation, icd_code in zip(examples[text_field], examples[label_field]):
            formatted_text = self.apply_chat_template(conversation, icd_code)
            formatted_texts.append(formatted_text)

        # Tokenisiere den Batch
        tokenized = self.tokenizer(
            formatted_texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=None,  # Wir wollen Listen, keine Tensoren
        )

        # Für Causal LM: labels = input_ids (Modell lernt, nächstes Token vorherzusagen)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def process_dataset(
        self, dataset: Dataset, num_proc: int = 4, batch_size: int = 1000
    ) -> Dataset:
        """
        Verarbeitet den kompletten Datensatz.

        Args:
            dataset: Rohdatensatz
            num_proc: Anzahl paralleler Prozesse
            batch_size: Batch-Größe für Verarbeitung

        Returns:
            Dataset: Verarbeiteter und tokenisierter Datensatz

        Warum parallelisiert: Tokenisierung kann langsam sein.
        Mit num_proc > 1 nutzen wir mehrere CPU-Kerne für Beschleunigung.
        """
        logger.info(f"Processing dataset with {num_proc} processes...")

        # Entferne Spalten, die wir nicht mehr brauchen (spart Memory)
        columns_to_remove = [
            col
            for col in dataset.column_names
            if col not in ["input_ids", "attention_mask", "labels"]
        ]

        # Wende Tokenisierung auf ganzen Datensatz an
        processed_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            desc="Tokenizing dataset",
        )

        logger.info(f"Dataset processed. Size: {len(processed_dataset)}")

        return processed_dataset


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """
    Splittet Datensatz in Train/Validation/Test.

    Args:
        dataset: Zu splittender Datensatz
        train_ratio: Anteil für Training
        val_ratio: Anteil für Validation
        test_ratio: Anteil für Test
        seed: Random Seed für Reproduzierbarkeit

    Returns:
        Dict mit 'train', 'validation', 'test' Datasets

    Warum 3 Splits:
    - Train: Zum Lernen der Parameter
    - Validation: Zum Tunen der Hyperparameter und Early Stopping
    - Test: Finale Evaluation (darf nie ins Training fließen!)

    Typische Ratios: 70/15/15 oder 80/10/10
    """
    logger.info(
        f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}"
    )

    # Validiere Ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Erster Split: Train vs Rest
    train_test = dataset.train_test_split(test_size=(1 - train_ratio), seed=seed)

    # Zweiter Split: Validation vs Test
    # val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_test = train_test["test"].train_test_split(
        test_size=(test_ratio / (val_ratio + test_ratio)), seed=seed
    )

    splits = {
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }

    logger.info(
        f"Split sizes - Train: {len(splits['train'])}, "
        f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}"
    )

    return splits


def save_processed_dataset(
    dataset: Dataset, save_path: Path, format: str = "arrow"
) -> None:
    """
    Speichert verarbeiteten Datensatz auf Disk.

    Args:
        dataset: Zu speichernder Datensatz
        save_path: Speicherpfad
        format: Format ("arrow", "parquet", "csv")

    Warum speichern: Tokenisierung ist langsam. Einmal verarbeiten,
    dann wiederverwenden spart viel Zeit bei mehreren Experimenten.
    """
    logger.info(f"Saving dataset to {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    if format == "arrow":
        dataset.save_to_disk(str(save_path))
    elif format == "parquet":
        dataset.to_parquet(str(save_path / "dataset.parquet"))
    elif format == "csv":
        dataset.to_csv(str(save_path / "dataset.csv"))
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info("Dataset saved successfully")


def load_processed_dataset(load_path: Path) -> Dataset:
    """
    Lädt verarbeiteten Datensatz von Disk.

    Args:
        load_path: Ladepfad

    Returns:
        Dataset: Geladener Datensatz
    """
    from datasets import load_from_disk

    logger.info(f"Loading dataset from {load_path}")
    dataset = load_from_disk(str(load_path))
    logger.info(f"Dataset loaded. Size: {len(dataset)}")

    return dataset


if __name__ == "__main__":
    # Test-Code
    logging.basicConfig(level=logging.INFO)

    # Teste mit einem Beispiel-Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    processor = MedicalDialogProcessor(tokenizer, max_length=512)

    # Beispiel-Dialog
    example_conversation = """Patient: I've had a bad headache and fever for 3 days.
Arzt: How high is your fever?
Patient: Yesterday it was 39 degrees.
Arzt: Do you have any other symptoms?
Patient: Yes, mythroat hurts too."""

    example_icd = "J06.9"  # Akute Infektion der oberen Atemwege

    # Formatiere für Training
    formatted = processor.format_dialog_for_training(example_conversation, example_icd)
    print("=" * 50)
    print("FORMATTED FOR TRAINING:")
    print("=" * 50)
    print(formatted)

    # Formatiere für Inference
    formatted_inf = processor.format_dialog_for_inference(example_conversation)
    print("\n" + "=" * 50)
    print("FORMATTED FOR INFERENCE:")
    print("=" * 50)
    print(formatted_inf)
