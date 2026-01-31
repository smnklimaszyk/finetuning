"""
Main Pipeline für Medical Diagnosis Model Finetuning

Dieses Script orchestriert den gesamten ML-Pipeline:
1. Daten laden und verarbeiten
2. Baseline LLM evaluieren
3. Baseline SLM evaluieren
4. SLM finetunen
5. Finetuned SLM evaluieren
6. Ergebnisse vergleichen und visualisieren

Verwendung:
    python main.py --experiment baseline
    python main.py --experiment full --skip-training
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_config
from data import create_data_loader, MedicalDialogProcessor, split_dataset
from models import LLMModel, SLMModel
from training import FineTuner
from evaluation import evaluate_model, plot_model_comparison, create_results_report
from utils import setup_logging, get_device, save_json
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical Diagnosis Model Training")
    parser.add_argument(
        "--experiment",
        type=str,
        default="full",
        choices=["baseline", "training", "full"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing model",
    )
    parser.add_argument(
        "--model-llm", type=str, default=None, help="Override LLM model name"
    )
    parser.add_argument(
        "--model-slm", type=str, default=None, help="Override SLM model name"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def load_and_prepare_data(config):
    """
    Schritt 1: Daten laden und vorbereiten.

    Returns:
        Tuple von (train_dataset, val_dataset, test_dataset, processor)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("=" * 60)

    # Lade Datensatz
    data_loader = create_data_loader(
        dataset_name=config.data.dataset_name,
        cache_dir=config.paths.cache_dir,
        use_cache=config.data.use_cache,
    )

    dataset = data_loader.load()

    # Zeige Statistiken
    stats = data_loader.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    # Split Dataset
    splits = split_dataset(
        dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.dataset_split_seed,
    )

    # Initialisiere Processor
    tokenizer = AutoTokenizer.from_pretrained(config.model.slm_name)
    processor = MedicalDialogProcessor(
        tokenizer=tokenizer,
        max_length=config.data.max_sequence_length,
        padding=config.data.padding,
        truncation=config.data.truncation,
    )

    # Verarbeite Datasets
    logger.info("Processing datasets...")
    train_dataset = processor.process_dataset(
        splits["train"], num_proc=config.data.num_workers
    )
    val_dataset = processor.process_dataset(
        splits["validation"], num_proc=config.data.num_workers
    )
    test_dataset = splits["test"]  # Rohdaten für Evaluation

    logger.info(f"Data preparation complete!")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset, processor


def evaluate_baseline_llm(config, test_dataset, processor, llm_config=None):
    """
    Schritt 2: Evaluiere Baseline LLM.
    
    Args:
        config: Experiment-Konfiguration
        test_dataset: Test-Datensatz für Evaluation
        processor: Datenverarbeitungs-Pipeline
        llm_config: Optional - LLMModelConfig mit Modelleinstellungen
    
    Returns:
        Dictionary mit Evaluationsmetriken
    """
    # Fallback auf erstes konfiguriertes LLM wenn keine Config übergeben
    if llm_config is None:
        llm_config = config.model.baseline_llms[0]
    
    model_short_name = llm_config.name.split("/")[-1]
    
    logger.info("=" * 60)
    logger.info(f"STEP 2: Evaluating Baseline LLM: {model_short_name}")
    logger.info("=" * 60)

    llm = LLMModel(
        model_name=llm_config.name,
        device=get_device(),
        load_in_4bit=llm_config.load_in_4bit,
        load_in_8bit=llm_config.load_in_8bit,
    )

    llm.load()

    metrics = evaluate_model(
        model=llm,
        test_dataset=test_dataset,
        processor=processor,
        batch_size=config.evaluation.eval_batch_size,
    )

    llm.unload()

    logger.info(f"Baseline LLM ({model_short_name}) Results: {metrics}")
    return metrics


def evaluate_baseline_slm(config, test_dataset, processor):
    """Schritt 3: Evaluiere Baseline SLM (ohne Finetuning)."""
    logger.info("=" * 60)
    logger.info("STEP 3: Evaluating Baseline SLM")
    logger.info("=" * 60)

    slm = SLMModel(
        model_name=config.model.slm_name,
        device=get_device(),
        load_in_4bit=config.model.slm_load_in_4bit,
        is_finetuned=False,
    )

    slm.load()

    metrics = evaluate_model(
        model=slm,
        test_dataset=test_dataset,
        processor=processor,
        batch_size=config.evaluation.eval_batch_size,
    )

    slm.unload()

    logger.info(f"Baseline SLM Results: {metrics}")
    return metrics


def train_slm(config, train_dataset, val_dataset):
    """Schritt 4: Trainiere SLM mit LoRA."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training SLM with LoRA")
    logger.info("=" * 60)

    trainer = FineTuner(config)
    trainer.setup_model_and_tokenizer()
    trainer.setup_lora()

    metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=config.paths.finetuned_models_dir / "training",
    )

    logger.info(f"Training complete! Metrics: {metrics}")
    return metrics


def evaluate_finetuned_slm(config, test_dataset, processor):
    """Schritt 5: Evaluiere finetuned SLM."""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating Finetuned SLM")
    logger.info("=" * 60)

    # Lade finetuned model
    model_path = config.paths.finetuned_models_dir / "final_model"

    slm = SLMModel(
        model_name=str(model_path),
        device=get_device(),
        load_in_4bit=False,  # Finetuned model already optimized
        is_finetuned=True,
    )

    slm.load()

    metrics = evaluate_model(
        model=slm,
        test_dataset=test_dataset,
        processor=processor,
        batch_size=config.evaluation.eval_batch_size,
    )

    slm.unload()

    logger.info(f"Finetuned SLM Results: {metrics}")
    return metrics


def compare_and_visualize(results, config):
    """Schritt 6: Vergleiche Ergebnisse und erstelle Visualisierungen."""
    logger.info("=" * 60)
    logger.info("STEP 6: Comparing Results and Creating Visualizations")
    logger.info("=" * 60)

    # Speichere Ergebnisse
    results_path = config.paths.reports_dir / "results.json"
    save_json(results, results_path)
    logger.info(f"Results saved to {results_path}")

    # Erstelle Plots
    if config.evaluation.generate_plots:
        plot_model_comparison(results=results, save_path=config.paths.plots_dir)
        logger.info(f"Plots saved to {config.paths.plots_dir}")

    # Erstelle Report
    report_path = create_results_report(
        results=results, save_path=config.paths.reports_dir
    )
    logger.info(f"HTML report created: {report_path}")

    # Zeige Zusammenfassung
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics.get('exact_match_accuracy', 0):.4f}")
        print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")


def main():
    """Haupt-Pipeline."""
    args = parse_args()

    # Lade Konfiguration
    config = get_config()

    # Override mit CLI args
    if args.model_llm:
        # CLI Override: Ersetze alle LLMs mit dem angegebenen Modell
        from config.base_config import LLMModelConfig
        config.model.baseline_llms = [LLMModelConfig(name=args.model_llm, description="CLI Override")]
    if args.model_slm:
        config.model.slm_name = args.model_slm

    # Setup
    config.setup()
    setup_logging(config.paths.logs_dir, config.logging.log_level)

    logger.info("Starting Medical Diagnosis Model Pipeline")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Device: {get_device()}")

    # Step 1: Load Data
    train_dataset, val_dataset, test_dataset, processor = load_and_prepare_data(config)

    results = {}

    # Step 2-3: Baseline Evaluation
    if args.experiment in ["baseline", "full"]:
        # Evaluiere alle konfigurierten Baseline LLMs
        if config.experiment.run_baseline_llm:
            for llm_config in config.model.baseline_llms:
                model_short_name = llm_config.name.split("/")[-1]
                result_key = f"Baseline_LLM_{model_short_name}"
                try:
                    logger.info(f"\n>>> Evaluating LLM: {llm_config.name}")
                    logger.info(f"    Description: {llm_config.description}")
                    results[result_key] = evaluate_baseline_llm(
                        config, test_dataset, processor, llm_config=llm_config
                    )
                except Exception as e:
                    logger.error(f"Error evaluating Baseline LLM ({llm_config.name}): {e}")
                    # Versuche mit dem nächsten Modell fortzufahren
                    continue

        if config.experiment.run_baseline_slm:
            try:
                results["Baseline_SLM"] = evaluate_baseline_slm(
                    config, test_dataset, processor
                )
            except Exception as e:
                logger.error(f"Error evaluating Baseline SLM: {e}")

    # Step 4-5: Training and Finetuned Evaluation
    if args.experiment in ["training", "full"]:
        if not args.skip_training and config.experiment.run_finetuned_slm:
            try:
                train_metrics = train_slm(config, train_dataset, val_dataset)
            except Exception as e:
                logger.error(f"Error during training: {e}")

        if config.experiment.run_finetuned_slm:
            try:
                results["Finetuned_SLM"] = evaluate_finetuned_slm(
                    config, test_dataset, processor
                )
            except Exception as e:
                logger.error(f"Error evaluating Finetuned SLM: {e}")

    # Step 6: Compare and Visualize
    if results:
        compare_and_visualize(results, config)
    else:
        logger.warning("No results to compare")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
