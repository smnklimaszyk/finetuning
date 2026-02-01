"""
Evaluation Metrics für medizinische Diagnose-Modelle.
"""

from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MedicalDiagnosisEvaluator:
    """
    Evaluator für ICD-10 Code Predictions.
    Berechnet verschiedene Metriken für Modell-Vergleich.
    """
    
    def __init__(self):
        self.predictions = []
        self.references = []
        self.metrics = {}
    
    def add_batch(self, predictions: List[str], references: List[str]):
        """Fügt einen Batch von Predictions hinzu."""
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute_exact_match_accuracy(self) -> float:
        """Berechnet Exact Match Accuracy."""
        if not self.predictions or not self.references:
            return 0.0
        
        matches = sum(1 for pred, ref in zip(self.predictions, self.references) 
                     if pred.strip().upper() == ref.strip().upper())
        return matches / len(self.predictions)
    
    def compute_prefix_match_accuracy(self, prefix_length: int = 3) -> float:
        """
        Berechnet Prefix Match Accuracy.
        Nützlich weil ICD-10 hierarchisch ist (J06.9 -> J06 -> J).
        """
        if not self.predictions or not self.references:
            return 0.0
        
        matches = sum(1 for pred, ref in zip(self.predictions, self.references)
                     if pred[:prefix_length].upper() == ref[:prefix_length].upper())
        return matches / len(self.predictions)
    
    def compute_top_k_accuracy(self, k: int = 5) -> float:
        """
        Top-K Accuracy (wenn Modell mehrere Vorschläge macht).
        Hier vereinfacht für single predictions.
        """
        # Für single prediction ist Top-1 gleich Exact Match
        if k == 1:
            return self.compute_exact_match_accuracy()
        # Für echte Top-K bräuchten wir Confidence Scores
        return self.compute_exact_match_accuracy()
    
    def compute_classification_metrics(self) -> Dict:
        """Berechnet Precision, Recall, F1."""
        if not self.predictions or not self.references:
            return {}
        
        # Für multi-class classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.references,
            self.predictions,
            average='weighted',
            zero_division=0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def compute_all_metrics(self) -> Dict:
        """Berechnet alle Metriken."""
        logger.info(f"Computing metrics for {len(self.predictions)} predictions...")
        
        metrics = {
            'n_samples': len(self.predictions),
            'exact_match_accuracy': self.compute_exact_match_accuracy(),
            'prefix_match_3': self.compute_prefix_match_accuracy(3),
            'prefix_match_1': self.compute_prefix_match_accuracy(1),  # Nur Hauptkategorie
        }
        
        # Classification metrics
        class_metrics = self.compute_classification_metrics()
        metrics.update(class_metrics)
        
        # Top-K Accuracies
        for k in [1, 3, 5]:
            metrics[f'top_{k}_accuracy'] = self.compute_top_k_accuracy(k)
        
        self.metrics = metrics
        logger.info(f"Metrics computed: {metrics}")
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Gibt detaillierten Classification Report zurück."""
        if not self.predictions or not self.references:
            return "No predictions available"
        
        return classification_report(self.references, self.predictions, zero_division=0)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Gibt Confusion Matrix zurück."""
        if not self.predictions or not self.references:
            return np.array([])
        
        return confusion_matrix(self.references, self.predictions)
    
    def reset(self):
        """Setzt Evaluator zurück."""
        self.predictions = []
        self.references = []
        self.metrics = {}


def evaluate_model(
    model,
    test_dataset,
    processor,
    batch_size: int = 8,
    config = None
) -> Dict:
    """
    Evaluiert ein Modell auf Test-Datensatz mit Smart Caching.
    
    Args:
        model: Modell-Wrapper (LLMModel oder SLMModel)
        test_dataset: Test-Datensatz
        processor: Data Processor für Formatierung
        batch_size: Batch-Größe
        config: Optional - Config für Caching-Einstellungen
    
    Returns:
        Dict mit Metriken
    """
    from utils import (
        generate_cache_key, 
        get_cached_predictions, 
        save_predictions_cache
    )
    
    logger.info("Starting model evaluation...")
    
    # Finde Feld-Namen (MedSynth hat: 'Dialogue', 'ICD10', 'ICD10_desc', ' Note')
    text_field = None
    label_field = None
    for field in ['Dialogue', 'dialogue', 'conversation', 'text', 'input']:
        if field in test_dataset.column_names:
            text_field = field
            break
    for field in ['ICD10', 'icd10', 'icd_code', 'ICD_CODE', 'label', 'diagnosis']:
        if field in test_dataset.column_names:
            label_field = field
            break
    
    if not text_field or not label_field:
        raise ValueError(f"Could not find required fields in dataset")
    
    # Prepare inputs and references
    conversations = [sample[text_field] for sample in test_dataset]
    references = [sample[label_field] for sample in test_dataset]
    
    # === SMART CACHING SYSTEM ===
    use_cache = config and hasattr(config.evaluation, 'use_prediction_cache') and config.evaluation.use_prediction_cache
    force_recompute = config and hasattr(config.evaluation, 'force_recompute') and config.evaluation.force_recompute
    
    if use_cache and not force_recompute:
        # Generate cache key
        model_name = getattr(model, 'model_name', 'unknown_model')
        generation_config = {
            'max_new_tokens': 50,
            'temperature': 0.1,
            'do_sample': False,
            'batch_size': batch_size
        }
        cache_key = generate_cache_key(model_name, len(test_dataset), generation_config)
        
        # Try to load from cache
        cache_dir = config.paths.predictions_cache_dir
        max_age = config.evaluation.cache_max_age_days if hasattr(config.evaluation, 'cache_max_age_days') else None
        cached_data = get_cached_predictions(cache_key, cache_dir, max_age)
        
        if cached_data:
            logger.info(f"✅ Loaded predictions from cache: {cache_key}")
            logger.info(f"   Cached at: {cached_data['metadata'].get('timestamp', 'unknown')}")
            logger.info(f"   Predictions: {len(cached_data['predictions'])}")
            return cached_data['metrics']
        else:
            logger.info(f"🔄 No valid cache found - generating predictions...")
    elif force_recompute:
        logger.info(f"🔄 Force recompute enabled - ignoring cache...")
    
    # Format for inference
    formatted_inputs = [
        processor.format_dialog_for_inference(conv)
        for conv in conversations
    ]
    
    # Generate predictions
    import time
    start_time = time.time()
    
    predictions_raw = model.predict_batch(
        formatted_inputs,
        batch_size=batch_size,
        max_new_tokens=50,  # ICD codes sind kurz
        temperature=0.1,  # Niedrig für deterministische Vorhersagen
        do_sample=False  # Greedy decoding für Evaluation
    )
    
    eval_time = time.time() - start_time
    
    # Extract ICD codes
    predictions = [model.extract_icd_code(pred) for pred in predictions_raw]
    
    # Compute metrics
    evaluator = MedicalDiagnosisEvaluator()
    evaluator.add_batch(predictions, references)
    metrics = evaluator.compute_all_metrics()
    
    # Add model metrics
    if hasattr(model, 'metrics'):
        performance_metrics = model.metrics.get_metrics()
        metrics.update({'performance': performance_metrics})
    
    # === SAVE TO CACHE ===
    if use_cache and config:
        model_name = getattr(model, 'model_name', 'unknown_model')
        generation_config = {
            'max_new_tokens': 50,
            'temperature': 0.1,
            'do_sample': False,
            'batch_size': batch_size
        }
        cache_key = generate_cache_key(model_name, len(test_dataset), generation_config)
        
        metadata = {
            'model_name': model_name,
            'dataset_size': len(test_dataset),
            'generation_config': generation_config,
            'evaluation_time_seconds': eval_time
        }
        
        cache_file = save_predictions_cache(
            cache_key,
            config.paths.predictions_cache_dir,
            predictions,
            references,
            metrics,
            metadata
        )
        logger.info(f"💾 Predictions cached to: {cache_file.name}")
    
    return metrics
