"""
Training Module für Finetuning.
Verwendet HuggingFace Trainer mit LoRA für parameter-effizientes Training.
"""

from typing import Optional, Dict
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FineTuner:
    """
    Klasse zum Finetuning von Language Models mit LoRA.

    LoRA (Low-Rank Adaptation): Trainiert nur kleine Adapter-Matrizen
    statt des ganzen Modells. Spart viel Memory und ist schneller.
    """

    def __init__(self, config, model_name: Optional[str] = None):
        """
        Initialisiert FineTuner.

        Args:
            config: Experiment-Konfiguration
            model_name: Optional - Überschreibt config.model.slm_name
                       Nützlich für Training mehrerer Modelle
        """
        self.config = config
        self.model_name = model_name or getattr(config.model, 'slm_name', None)

        if self.model_name is None:
            raise ValueError("model_name must be provided either via parameter or config.model.slm_name")

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model_and_tokenizer(self):
        """Lädt Basis-Modell und Tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading kwargs with optimizations
        model_kwargs = {
            "device_map": "auto"
        }
        
        # Use BF16/FP16 based on config
        if self.config.training.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.training.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        # Add Flash Attention 2 if specified in config (RTX 5090 optimization)
        if hasattr(self.config.model, 'attn_implementation'):
            model_kwargs["attn_implementation"] = self.config.model.attn_implementation
            logger.info(f"✅ Using {self.config.model.attn_implementation} for 2-3x attention speedup")
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Aktiviere Gradient Checkpointing für Memory-Effizienz (if enabled)
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("⚠️  Gradient checkpointing enabled - slower but saves memory")
        else:
            logger.info("✅ Gradient checkpointing disabled - using full speed mode")
        
        logger.info("Model and tokenizer loaded")
    
    def setup_lora(self):
        """Konfiguriert LoRA für das Modell."""
        if not self.config.training.use_lora:
            logger.info("LoRA disabled, training full model")
            return
        
        logger.info("Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config.training.lora_r,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            target_modules=self.config.training.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Apply torch.compile for graph optimization (RTX 5090 optimization)
        if hasattr(self.config.training, 'torch_compile') and self.config.training.torch_compile:
            logger.info("✅ Applying torch.compile for 20-40% speedup...")
            try:
                self.model = torch.compile(self.model)
                logger.info("✅ torch.compile successfully applied!")
            except Exception as e:
                logger.warning(f"⚠️  torch.compile failed: {e}. Continuing without compilation.")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Führt Training durch.
        
        Returns:
            Dict mit Training-Metriken
        """
        if output_dir is None:
            output_dir = self.config.paths.finetuned_models_dir / "checkpoint"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            optim=self.config.training.optimizer,  # Uses fused optimizer from config
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy=self.config.training.evaluation_strategy,
            save_strategy=self.config.training.save_strategy,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            report_to="tensorboard" if self.config.logging.use_tensorboard else "none",
            logging_dir=str(self.config.paths.logs_dir),
            seed=self.config.training.seed,
            max_grad_norm=self.config.training.max_grad_norm,
        )
        
        logger.info("✅ Using optimizer: {}".format(self.config.training.optimizer))
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping_patience,
                early_stopping_threshold=self.config.training.early_stopping_threshold
            )]
        )
        
        # Start Training
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Speichere finales Modell
        final_model_path = self.config.paths.finetuned_models_dir / "final_model"
        self.trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"Training completed. Model saved to {final_model_path}")
        
        return train_result.metrics
    
    def save_model(self, save_path: Path):
        """Speichert trainiertes Modell."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info(f"Model saved to {save_path}")
