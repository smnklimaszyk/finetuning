# Medical Diagnosis Model Finetuning Tests

import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfig:
    """Tests für das Konfigurationssystem."""

    def test_config_creation(self):
        """Test ob Config erstellt werden kann."""
        from config import get_config

        config = get_config()
        assert config is not None

    def test_config_defaults(self):
        """Test ob Defaults korrekt gesetzt sind."""
        from config import get_config

        config = get_config()

        assert config.data.dataset_name == "Ahmad0067/MedSynth"
        assert config.training.learning_rate == 2e-5
        assert config.training.use_lora == True

    def test_split_ratios_sum(self):
        """Test ob Split-Ratios sich zu 1.0 summieren."""
        from config import get_config

        config = get_config()

        total = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
        assert abs(total - 1.0) < 0.001


class TestDataProcessor:
    """Tests für die Datenverarbeitung."""

    def test_format_dialog_for_training(self):
        """Test Dialog-Formatierung für Training."""
        from data import MedicalDialogProcessor
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Lightweight for testing
        processor = MedicalDialogProcessor(tokenizer, max_length=128)

        conversation = "Patient: Ich habe Kopfschmerzen."
        icd_code = "G43.9"

        formatted = processor.format_dialog_for_training(conversation, icd_code)

        assert "Patient" in formatted
        assert "G43.9" in formatted
        assert "ICD-10" in formatted

    def test_format_dialog_for_inference(self):
        """Test Dialog-Formatierung für Inference."""
        from data import MedicalDialogProcessor
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        processor = MedicalDialogProcessor(tokenizer, max_length=128)

        conversation = "Patient: Ich habe Kopfschmerzen."

        formatted = processor.format_dialog_for_inference(conversation)

        assert "Patient" in formatted
        assert formatted.endswith(":")  # Sollte auf Assistant-Prompt enden


class TestEvaluation:
    """Tests für Evaluation-Metriken."""

    def test_exact_match_accuracy(self):
        """Test Exact Match Berechnung."""
        from evaluation import MedicalDiagnosisEvaluator

        evaluator = MedicalDiagnosisEvaluator()

        predictions = ["J06.9", "I10", "G43.9"]
        references = ["J06.9", "I10", "G40.9"]  # Letztes ist falsch

        evaluator.add_batch(predictions, references)
        accuracy = evaluator.compute_exact_match_accuracy()

        assert accuracy == pytest.approx(2 / 3)

    def test_prefix_match_accuracy(self):
        """Test Prefix Match Berechnung."""
        from evaluation import MedicalDiagnosisEvaluator

        evaluator = MedicalDiagnosisEvaluator()

        predictions = ["J06.9", "I10", "G43.9"]
        references = ["J06.1", "I11", "G43.1"]  # Alle 3-char prefixes gleich

        evaluator.add_batch(predictions, references)
        accuracy = evaluator.compute_prefix_match_accuracy(prefix_length=3)

        assert accuracy == pytest.approx(3 / 3)


class TestModelBase:
    """Tests für Basis-Modell-Funktionalität."""

    def test_model_metrics(self):
        """Test ModelMetrics Klasse."""
        from models.base_model import ModelMetrics

        metrics = ModelMetrics()

        metrics.update(n_samples=10, time_taken=1.0, n_tokens=100)
        metrics.update(n_samples=10, time_taken=1.0, n_tokens=100)

        result = metrics.get_metrics()

        assert result["total_samples"] == 20
        assert result["total_time_seconds"] == 2.0
        assert result["throughput_samples_per_second"] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
