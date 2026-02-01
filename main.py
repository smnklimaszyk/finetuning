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
from utils import (
    setup_logging,
    get_device,
    save_json,
    log_gpu_memory,
    aggressive_memory_cleanup,
)
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
        "--force-recompute",
        action="store_true",
        help="Force recompute predictions (ignore cache)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear prediction cache before running",
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


def evaluate_llm(config, test_dataset, processor, llm_config):
    """
    Evaluiert ein großes LLM (zero-shot, ohne Finetuning).

    Args:
        config: Experiment-Konfiguration
        test_dataset: Test-Datensatz für Evaluation
        processor: Datenverarbeitungs-Pipeline
        llm_config: ModelInstanceConfig mit Modelleinstellungen

    Returns:
        Dictionary mit Evaluationsmetriken
    """
    model_short_name = llm_config.name.split("/")[-1]

    logger.info("=" * 60)
    logger.info(f"Evaluating LLM (untrained): {model_short_name} ({llm_config.size})")
    logger.info(f"Description: {llm_config.description}")
    logger.info("=" * 60)

    # Log memory before loading
    log_gpu_memory("Before LLM load")

    llm = LLMModel(
        model_name=llm_config.name,
        device=get_device(),
        load_in_4bit=llm_config.load_in_4bit,
        load_in_8bit=llm_config.load_in_8bit,
    )

    llm.load()
    log_gpu_memory("After LLM load")

    metrics = evaluate_model(
        model=llm,
        test_dataset=test_dataset,
        processor=processor,
        batch_size=config.evaluation.eval_batch_size,
        config=config,
    )

    llm.unload()
    log_gpu_memory("After LLM unload")

    logger.info(f"LLM ({model_short_name}) Results: {metrics}")
    return metrics


def train_and_evaluate_slm(
    config, train_dataset, val_dataset, test_dataset, processor, slm_config
):
    """
    Trainiert und evaluiert ein kleines SLM.

    Args:
        config: Experiment-Konfiguration
        train_dataset: Training-Datensatz
        val_dataset: Validation-Datensatz
        test_dataset: Test-Datensatz
        processor: Datenverarbeitungs-Pipeline
        slm_config: ModelInstanceConfig mit Modelleinstellungen

    Returns:
        Tuple von (training_metrics, evaluation_metrics)
    """
    model_short_name = slm_config.name.split("/")[-1]

    logger.info("=" * 60)
    logger.info(f"Training SLM: {model_short_name} ({slm_config.size})")
    logger.info(f"Description: {slm_config.description}")
    logger.info("=" * 60)

    # Log memory before training
    log_gpu_memory("Before SLM training")

    # Erstelle modell-spezifisches Output-Verzeichnis
    model_output_dir = config.paths.finetuned_models_dir / model_short_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Temporär: Update config.model.slm_name für Trainer
    # (Trainer nutzt noch config.model.slm_name)
    original_slm_name = getattr(config.model, "slm_name", None)
    config.model.slm_name = slm_config.name

    # Training
    trainer = FineTuner(config)
    trainer.setup_model_and_tokenizer()
    trainer.setup_lora()

    training_metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=model_output_dir / "training",
    )

    logger.info(f"Training complete for {model_short_name}!")

    # Restore original config
    if original_slm_name:
        config.model.slm_name = original_slm_name

    # Cleanup nach Training
    del trainer
    aggressive_memory_cleanup()
    log_gpu_memory("After SLM training cleanup")

    # Evaluation
    logger.info("=" * 60)
    logger.info(f"Evaluating finetuned SLM: {model_short_name}")
    logger.info("=" * 60)

    model_path = model_output_dir / "final_model"

    slm = SLMModel(
        model_name=str(model_path),
        device=get_device(),
        load_in_4bit=False,
        is_finetuned=True,
    )

    slm.load()
    log_gpu_memory("After SLM load for evaluation")

    evaluation_metrics = evaluate_model(
        model=slm,
        test_dataset=test_dataset,
        processor=processor,
        batch_size=config.evaluation.eval_batch_size,
        config=config,
    )

    slm.unload()
    log_gpu_memory("After SLM evaluation")

    logger.info(f"Finetuned SLM ({model_short_name}) Results: {evaluation_metrics}")

    return training_metrics, evaluation_metrics




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
        from config.base_config import ModelInstanceConfig

        config.model.llm_models = [
            ModelInstanceConfig(
                name=args.model_llm, size="unknown", description="CLI Override"
            )
        ]
    if args.model_slm:
        # CLI Override: Ersetze alle SLMs mit dem angegebenen Modell
        from config.base_config import ModelInstanceConfig

        config.model.slm_models = [
            ModelInstanceConfig(
                name=args.model_slm, size="unknown", description="CLI Override"
            )
        ]

    # Handle force recompute flag
    if args.force_recompute:
        config.evaluation.force_recompute = True
        logger.info("🔄 Force recompute enabled - will regenerate all predictions")

    # Handle cache clearing
    if args.clear_cache:
        from utils import clear_prediction_cache

        logger.info("🗑️  Clearing prediction cache...")
        clear_prediction_cache(config.paths.predictions_cache_dir)

    # Setup
    config.setup()
    setup_logging(config.paths.logs_dir, config.logging.log_level)

    logger.info("Starting Medical Diagnosis Model Pipeline")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Device: {get_device()}")
    logger.info(
        f"Prediction Cache: {'Enabled' if config.evaluation.use_prediction_cache else 'Disabled'}"
    )

    # Step 1: Load Data
    train_dataset, val_dataset, test_dataset, processor = load_and_prepare_data(config)

    results = {}

    # === NEW PIPELINE ===
    # Step 2: Evaluate LLMs (Large, Untrained Models)
    if args.experiment in ["baseline", "full"]:
        if config.experiment.run_llm_evaluation:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: Evaluating Large Language Models (Zero-Shot)")
            logger.info("=" * 60)

            for llm_config in config.model.llm_models:
                model_short_name = llm_config.name.split("/")[-1]
                result_key = f"LLM_{model_short_name}_untrained"

                try:
                    results[result_key] = evaluate_llm(
                        config, test_dataset, processor, llm_config
                    )

                    # Add metadata to results
                    results[result_key]["model_type"] = "LLM"
                    results[result_key]["model_size"] = llm_config.size
                    results[result_key]["training_status"] = "untrained"

                except Exception as e:
                    logger.error(f"Error evaluating LLM ({llm_config.name}): {e}")
                    import traceback

                    logger.error(traceback.format_exc())

                finally:
                    # Aggressive cleanup between models
                    aggressive_memory_cleanup()

    # Step 3: Train and Evaluate SLMs (Small, Finetuned Models)
    if args.experiment in ["training", "full"]:
        if config.experiment.run_slm_finetuning and config.experiment.run_slm_evaluation:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: Training and Evaluating Small Language Models")
            logger.info("=" * 60)

            for slm_config in config.model.slm_models:
                model_short_name = slm_config.name.split("/")[-1]
                result_key = f"SLM_{model_short_name}_finetuned"

                try:
                    if not args.skip_training:
                        train_metrics, eval_metrics = train_and_evaluate_slm(
                            config,
                            train_dataset,
                            val_dataset,
                            test_dataset,
                            processor,
                            slm_config,
                        )

                        # Store results
                        results[result_key] = eval_metrics
                        results[result_key]["model_type"] = "SLM"
                        results[result_key]["model_size"] = slm_config.size
                        results[result_key]["training_status"] = "finetuned"
                        results[result_key]["training_metrics"] = train_metrics
                    else:
                        logger.info(f"Skipping training for {model_short_name} (--skip-training)")

                except Exception as e:
                    logger.error(f"Error with SLM ({slm_config.name}): {e}")
                    import traceback

                    logger.error(traceback.format_exc())

                finally:
                    # Aggressive cleanup between models
                    aggressive_memory_cleanup()

    # Step 6: Compare and Visualize
    if results:
        compare_and_visualize(results, config)
    else:
        logger.warning("No results to compare")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
