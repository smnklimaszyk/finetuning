"""
Test Custom Medical Interviews

Dieses Script testet trainierte Modelle auf selbst erstellten Arzt-Patienten-Dialogen.
Lädt alle verfügbaren Modelle (LLMs + finetuned SLMs) und vergleicht ihre Predictions.

Verwendung:
    python test_custom_interviews.py
    python test_custom_interviews.py --interviews-dir custom_interviews/
    python test_custom_interviews.py --interview custom_interviews/fall_001.json

JSON Format:
    {
        "name": "Interview Name",
        "expected_icd10": "J06.9",
        "messages": [
            {"role": "doctor", "message": "Was führt Sie zu mir?"},
            {"role": "patient", "message": "Ich habe Halsschmerzen..."},
            ...
        ]
    }
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_config
from models import LLMModel, SLMModel
from data import MedicalDialogProcessor
from utils import get_device, setup_logging, log_gpu_memory, aggressive_memory_cleanup
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)


class CustomInterviewTester:
    """
    Testet Modelle auf benutzerdefinierten Interviews.
    """

    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.results = []

        # Initialize processor (using first SLM's tokenizer)
        first_slm = config.model.slm_models[0]
        self.tokenizer = AutoTokenizer.from_pretrained(first_slm.name)
        self.processor = MedicalDialogProcessor(
            tokenizer=self.tokenizer,
            max_length=config.data.max_sequence_length,
            padding=config.data.padding,
            truncation=config.data.truncation,
        )

    def load_interview(self, json_path: Path) -> Dict:
        """
        Lädt ein Interview aus einer JSON-Datei.

        Expected format:
        {
            "name": "Interview Name",
            "expected_icd10": "J06.9",
            "messages": [
                {"role": "doctor", "message": "..."},
                {"role": "patient", "message": "..."}
            ]
        }
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        required_fields = ["name", "messages"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in {json_path}")

        if not isinstance(data["messages"], list) or len(data["messages"]) == 0:
            raise ValueError(f"'messages' must be a non-empty list in {json_path}")

        return data

    def convert_interview_to_dialog(self, interview: Dict) -> str:
        """
        Konvertiert Interview-Format zu Dialog-String.

        Input:
            {"messages": [{"role": "doctor", "message": "..."}, ...]}

        Output:
            "Doctor: ...\nPatient: ...\n..."
        """
        dialog_lines = []
        for msg in interview["messages"]:
            role = msg.get("role", "unknown").capitalize()
            message = msg.get("message", "")
            dialog_lines.append(f"{role}: {message}")

        return "\n".join(dialog_lines)

    def test_llm(self, llm_config, interview: Dict) -> Dict:
        """
        Testet ein LLM auf dem Interview.
        """
        model_short_name = llm_config.name.split("/")[-1]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing LLM: {model_short_name} ({llm_config.size})")
        logger.info(f"{'='*60}")

        log_gpu_memory(f"Before loading {model_short_name}")

        # Load model
        llm = LLMModel(
            model_name=llm_config.name,
            device=self.device,
            load_in_4bit=llm_config.load_in_4bit,
            load_in_8bit=llm_config.load_in_8bit,
        )
        llm.load()

        # Convert to dialog
        dialog = self.convert_interview_to_dialog(interview)

        # Format for inference
        formatted_input = self.processor.format_dialog_for_inference(dialog)

        # Generate prediction
        prediction_raw = llm.predict(
            formatted_input,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
        )

        # Extract ICD code
        predicted_icd = llm.extract_icd_code(prediction_raw)

        # Unload
        llm.unload()
        aggressive_memory_cleanup()

        result = {
            "model_name": model_short_name,
            "model_type": "LLM",
            "model_size": llm_config.size,
            "training_status": "untrained",
            "predicted_icd10": predicted_icd,
            "raw_output": prediction_raw,
        }

        logger.info(f"Prediction: {predicted_icd}")
        return result

    def test_finetuned_slm(self, slm_config, interview: Dict) -> Dict:
        """
        Testet ein finetuned SLM auf dem Interview.
        """
        model_short_name = slm_config.name.split("/")[-1]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Finetuned SLM: {model_short_name} ({slm_config.size})")
        logger.info(f"{'='*60}")

        # Find finetuned model path
        model_path = self.config.paths.finetuned_models_dir / model_short_name / "final_model"

        if not model_path.exists():
            logger.warning(f"Finetuned model not found at {model_path}")
            logger.warning("Skipping this model. Run training first!")
            return None

        log_gpu_memory(f"Before loading {model_short_name}")

        # Load finetuned model
        slm = SLMModel(
            model_name=str(model_path),
            device=self.device,
            load_in_4bit=False,
            is_finetuned=True,
        )
        slm.load()

        # Convert to dialog
        dialog = self.convert_interview_to_dialog(interview)

        # Format for inference
        formatted_input = self.processor.format_dialog_for_inference(dialog)

        # Generate prediction
        prediction_raw = slm.predict(
            formatted_input,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
        )

        # Extract ICD code
        predicted_icd = slm.extract_icd_code(prediction_raw)

        # Unload
        slm.unload()
        aggressive_memory_cleanup()

        result = {
            "model_name": model_short_name,
            "model_type": "SLM",
            "model_size": slm_config.size,
            "training_status": "finetuned",
            "predicted_icd10": predicted_icd,
            "raw_output": prediction_raw,
        }

        logger.info(f"Prediction: {predicted_icd}")
        return result

    def test_interview(self, interview: Dict) -> Dict:
        """
        Testet alle Modelle auf einem Interview.
        """
        interview_name = interview.get("name", "Unknown")
        expected_icd = interview.get("expected_icd10", "Unknown")

        print("\n" + "=" * 80)
        print(f"TESTING INTERVIEW: {interview_name}")
        print(f"Expected ICD-10: {expected_icd}")
        print("=" * 80)

        # Show dialog
        dialog = self.convert_interview_to_dialog(interview)
        print(f"\nDialog Preview:")
        print("-" * 80)
        print(dialog[:500] + "..." if len(dialog) > 500 else dialog)
        print("-" * 80)

        results = {
            "interview_name": interview_name,
            "expected_icd10": expected_icd,
            "dialog": dialog,
            "predictions": [],
        }

        # Test all LLMs
        print("\n" + "=" * 80)
        print("PHASE 1: Testing Large Language Models (Untrained)")
        print("=" * 80)

        for llm_config in self.config.model.llm_models:
            try:
                result = self.test_llm(llm_config, interview)
                results["predictions"].append(result)
            except Exception as e:
                logger.error(f"Error testing LLM {llm_config.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Test all finetuned SLMs
        print("\n" + "=" * 80)
        print("PHASE 2: Testing Small Language Models (Finetuned)")
        print("=" * 80)

        for slm_config in self.config.model.slm_models:
            try:
                result = self.test_finetuned_slm(slm_config, interview)
                if result:  # Only add if model was found
                    results["predictions"].append(result)
            except Exception as e:
                logger.error(f"Error testing SLM {slm_config.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Print summary
        self.print_results_summary(results)

        return results

    def print_results_summary(self, results: Dict):
        """
        Druckt Zusammenfassung der Ergebnisse.
        """
        print("\n" + "=" * 80)
        print(f"RESULTS SUMMARY: {results['interview_name']}")
        print("=" * 80)
        print(f"\nExpected ICD-10: {results['expected_icd10']}")
        print("\nModel Predictions:")
        print("-" * 80)

        for pred in results["predictions"]:
            model_info = f"{pred['model_name']} ({pred['model_size']}, {pred['training_status']})"
            prediction = pred["predicted_icd10"]

            # Check if correct
            is_correct = prediction.upper() == results["expected_icd10"].upper()
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"

            print(f"{model_info:50} → {prediction:15} {status}")

        print("-" * 80)

        # Count correct predictions
        correct_count = sum(
            1
            for pred in results["predictions"]
            if pred["predicted_icd10"].upper() == results["expected_icd10"].upper()
        )
        total_count = len(results["predictions"])

        print(f"\nCorrect Predictions: {correct_count}/{total_count}")
        print("=" * 80)

    def save_results(self, all_results: List[Dict], output_path: Path):
        """
        Speichert Ergebnisse als JSON.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ Results saved to: {output_path}")


def find_interview_files(interviews_dir: Path) -> List[Path]:
    """
    Findet alle JSON-Dateien im Interviews-Verzeichnis.
    """
    if not interviews_dir.exists():
        raise ValueError(f"Interviews directory not found: {interviews_dir}")

    json_files = list(interviews_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {interviews_dir}")

    return sorted(json_files)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test models on custom medical interviews"
    )
    parser.add_argument(
        "--interviews-dir",
        type=str,
        default="custom_interviews",
        help="Directory containing interview JSON files",
    )
    parser.add_argument(
        "--interview",
        type=str,
        default=None,
        help="Path to single interview JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/custom_interview_results.json",
        help="Output path for results JSON",
    )
    return parser.parse_args()


def main():
    """Main testing pipeline."""
    args = parse_args()

    # Load config
    config = get_config()
    config.setup()

    # Setup logging
    setup_logging(config.paths.logs_dir, "INFO")

    logger.info("=" * 80)
    logger.info("CUSTOM INTERVIEW TESTING")
    logger.info("=" * 80)
    logger.info(f"Device: {get_device()}")

    # Initialize tester
    tester = CustomInterviewTester(config)

    # Find interview files
    if args.interview:
        # Single interview
        interview_files = [Path(args.interview)]
    else:
        # All interviews in directory
        interviews_dir = Path(args.interviews_dir)
        interview_files = find_interview_files(interviews_dir)

    logger.info(f"Found {len(interview_files)} interview(s) to test")

    # Test all interviews
    all_results = []
    for interview_path in interview_files:
        try:
            logger.info(f"\nLoading interview: {interview_path}")
            interview = tester.load_interview(interview_path)
            results = tester.test_interview(interview)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error processing {interview_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Save results
    output_path = Path(args.output)
    tester.save_results(all_results, output_path)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Tested {len(all_results)} interview(s)")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
