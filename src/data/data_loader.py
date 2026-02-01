"""
Data Loader Module für MedSynth Dataset

Dieses Modul lädt und verarbeitet den MedSynth-Datensatz von HuggingFace.
Der Datensatz enthält synthetische Arzt-Patienten-Dialoge mit ICD-10 Codes.

Warum wichtig: Saubere Datenverarbeitung ist die Grundlage für gutes Training.
Schlechte Daten führen zu schlechten Modellen ("Garbage In, Garbage Out").
"""

from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
import logging
import json
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


class MedSynthDataLoader:
    """
    Lädt und verarbeitet den MedSynth-Datensatz.

    Der MedSynth-Datensatz enthält:
    - Arzt-Patienten-Dialoge (synthetisch generiert)
    - Chief Complaints (Hauptbeschwerde des Patienten)
    - ICD-10 Diagnose-Codes
    - Zusätzliche medizinische Informationen
    """

    def __init__(
        self,
        dataset_name: str = "Ahmad0067/MedSynth",
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """
        Initialisiert den DataLoader.

        Args:
            dataset_name: Name des HuggingFace-Datensatzes
            cache_dir: Verzeichnis für gecachte Daten
            use_cache: Ob Caching verwendet werden soll
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.dataset: Optional[Dataset] = None

        logger.info(f"Initializing MedSynthDataLoader for dataset: {dataset_name}")

    def load(self) -> Dataset:
        """
        Lädt den Datensatz von HuggingFace.

        Returns:
            Dataset: Geladener HuggingFace Dataset

        Warum HuggingFace Datasets:
        - Effizientes Memory-Management (Memory-Mapping)
        - Automatisches Caching
        - Viele eingebaute Transformationen
        - Standardisierte Schnittstelle
        """
        logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            # Lade Datensatz von HuggingFace Hub
            # cache_dir: Lokales Verzeichnis für Downloads
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

            # Wenn DatasetDict, nimm ersten Split (meist 'train')
            if isinstance(self.dataset, DatasetDict):
                # Viele Datasets haben nur einen Split oder 'train'
                if "train" in self.dataset:
                    self.dataset = self.dataset["train"]
                else:
                    # Nimm ersten verfügbaren Split
                    first_split = list(self.dataset.keys())[0]
                    logger.warning(f"No 'train' split found, using '{first_split}'")
                    self.dataset = self.dataset[first_split]

            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset)}")
            logger.info(f"Dataset features: {self.dataset.features}")

            return self.dataset

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def get_sample(self, idx: int = 0) -> Dict:
        """
        Holt ein einzelnes Sample für Inspektion.

        Args:
            idx: Index des gewünschten Samples

        Returns:
            Dict: Einzelnes Datenbeispiel
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        return self.dataset[idx]

    def get_statistics(self) -> Dict:
        """
        Berechnet Statistiken über den Datensatz.

        Returns:
            Dict: Verschiedene Statistiken (Größe, Durchschnittslängen, etc.)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        logger.info("Computing dataset statistics...")

        stats = {
            "total_samples": len(self.dataset),
            "features": list(self.dataset.features.keys()),
        }

        # Beispiel-Analyse: Länge der Dialoge
        if (
            "conversation" in self.dataset.features
            or "dialogue" in self.dataset.features
        ):
            # Finde das richtige Feld für den Dialog
            dialog_field = None
            for field in ["conversation", "dialogue", "text", "input"]:
                if field in self.dataset.features:
                    dialog_field = field
                    break

            if dialog_field:
                # Berechne Längenstatistiken
                lengths = []
                for sample in tqdm(self.dataset, desc="Computing lengths"):
                    text = sample[dialog_field]
                    if isinstance(text, str):
                        lengths.append(len(text.split()))
                    elif isinstance(text, list):
                        # Falls Dialog als Liste von Turns
                        lengths.append(
                            sum(
                                len(turn.split())
                                for turn in text
                                if isinstance(turn, str)
                            )
                        )

                stats["dialog_lengths"] = {
                    "mean": sum(lengths) / len(lengths) if lengths else 0,
                    "min": min(lengths) if lengths else 0,
                    "max": max(lengths) if lengths else 0,
                    "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
                }

        # ICD-10 Code Analyse
        if "icd_code" in self.dataset.features or "ICD_CODE" in self.dataset.features:
            icd_field = (
                "icd_code" if "icd_code" in self.dataset.features else "ICD_CODE"
            )
            icd_codes = [sample[icd_field] for sample in self.dataset]
            unique_codes = set(icd_codes)

            stats["icd_codes"] = {
                "total_unique": len(unique_codes),
                "samples_per_code": (
                    len(self.dataset) / len(unique_codes) if unique_codes else 0
                ),
            }

        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        return stats

    def preview(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Zeigt Preview der ersten n Samples als DataFrame.

        Args:
            n_samples: Anzahl zu zeigender Samples

        Returns:
            pd.DataFrame: Preview der Daten
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Konvertiere zu DataFrame für bessere Darstellung
        samples = [self.dataset[i] for i in range(min(n_samples, len(self.dataset)))]
        df = pd.DataFrame(samples)

        return df

    def validate_dataset(self) -> bool:
        """
        Validiert, dass der Datensatz die erwartete Struktur hat.

        Returns:
            bool: True wenn valid, sonst False
        """
        if self.dataset is None:
            logger.error("Dataset not loaded")
            return False

        required_fields = []  # Wird gefüllt basierend auf dem tatsächlichen Schema

        # Prüfe, dass mindestens ein Text-Feld und ein Label-Feld existiert
        features = list(self.dataset.features.keys())

        has_text_field = any(
            field in features for field in ["conversation", "dialogue", "text", "input"]
        )
        has_label_field = any(
            field in features
            for field in ["icd_code", "ICD_CODE", "label", "diagnosis"]
        )

        if not has_text_field:
            logger.error("No text field found in dataset")
            return False

        if not has_label_field:
            logger.error("No label field found in dataset")
            return False

        logger.info("Dataset validation passed")
        return True


def create_data_loader(
    dataset_name: str, cache_dir: Optional[Path] = None, use_cache: bool = True
) -> MedSynthDataLoader:
    """
    Factory-Funktion zum Erstellen eines DataLoaders.

    Args:
        dataset_name: Name des zu ladenden Datensatzes
        cache_dir: Cache-Verzeichnis
        use_cache: Ob Caching verwendet werden soll

    Returns:
        MedSynthDataLoader: Initialisierter DataLoader

    Verwendung:
        loader = create_data_loader("Ahmad0067/MedSynth")
        dataset = loader.load()
    """
    return MedSynthDataLoader(
        dataset_name=dataset_name, cache_dir=cache_dir, use_cache=use_cache
    )


if __name__ == "__main__":
    # Test-Code für lokale Entwicklung
    logging.basicConfig(level=logging.INFO)

    loader = create_data_loader("Ahmad0067/MedSynth")
    dataset = loader.load()

    print("\n" + "=" * 50)
    print("DATASET PREVIEW")
    print("=" * 50)
    print(loader.preview(3))

    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(json.dumps(loader.get_statistics(), indent=2))
