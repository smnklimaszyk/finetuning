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
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_model_and_tokenizer(self):
        """Lädt Basis-Modell und Tokenizer."""
        logger.info(f"Loading model: {self.config.model.slm_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.slm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.slm_name,
            torch_dtype=torch.float16 if self.config.training.fp16 else torch.bfloat16,
            device_map="auto"
        )
        
        # Aktiviere Gradient Checkpointing für Memory-Effizienz
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
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
