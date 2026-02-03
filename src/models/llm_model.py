"""
LLM Model Wrapper

Wrapper für Large Language Models (Baseline).
Unterstützt verschiedene LLMs von HuggingFace.

Warum wichtig: LLMs dienen als Baseline-Vergleich.
Wir testen ob ein spezialisiertes kleines Modell ein großes
generalistisches Modell schlagen kann.
"""

from typing import List, Union, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base_model import BaseModel, ModelMetrics
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LLMModel(BaseModel):
    """Wrapper für Large Language Models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attention: bool = False,
        trust_remote_code: bool = False,
    ):
        """
        Initialisiert das Large Language Model.

        Args:
            model_name: HuggingFace Modell-ID
            device: Target device
            dtype: Datentyp
            load_in_8bit: Ob 8-bit Quantisierung verwendet werden soll
            load_in_4bit: Ob 4-bit Quantisierung verwendet werden soll
            use_flash_attention: Ob Flash Attention 2 aktiviert werden soll
            trust_remote_code: Ob Custom Code ausgeführt werden darf
        """
        super().__init__(model_name, device, dtype)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        self.trust_remote_code = trust_remote_code
        self.metrics = ModelMetrics()

    def load(self) -> None:
        """Lädt das LLM und den Tokenizer."""
        logger.info(f"Loading LLM: {self.model_name}")

        # Konfiguriere Quantisierung
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Lade Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )

        # Wichtig: Für Decoder-Only Modelle muss padding links sein!
        # Sonst gibt es Probleme bei der Generierung mit gepadten Batches
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        # Lade das Modell
        model_kwargs = {
            "dtype": self.dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": self.trust_remote_code,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if self.use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash_attention_2 for attention")
            except ImportError:
                logger.warning("flash_attn not installed, using default attention")
                # PyTorch will use SDPA by default if available

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        self.model.eval()
        self.is_loaded = True
        logger.info("Model loaded successfully")

    def predict(
        self,
        input_text: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        **generation_kwargs,
    ) -> Union[str, List[str]]:
        """Generiert Prediction(s) für Input-Text(e)."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load() first.")

        is_single = isinstance(input_text, str)
        texts = [input_text] if is_single else input_text

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self.model.device)

        start_time = time.time()

        # Generation-Parameter vorbereiten
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        sampling_params = {"temperature", "top_p", "top_k"}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        # Filtere Sampling-Parameter aus generation_kwargs wenn do_sample=False
        filtered_kwargs = {
            k: v
            for k, v in generation_kwargs.items()
            if do_sample or k not in sampling_params
        }
        gen_kwargs.update(filtered_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        decoded = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        elapsed = time.time() - start_time
        n_tokens = generated_tokens.numel()
        self.metrics.update(len(texts), elapsed, n_tokens)

        return decoded[0] if is_single else decoded

    def predict_batch(
        self,
        input_texts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
        **generation_kwargs,
    ) -> List[str]:
        """Batched Prediction für bessere Effizienz."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load() first.")

        all_predictions = []
        n_batches = (len(input_texts) + batch_size - 1) // batch_size

        iterator = range(0, len(input_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Generating predictions")

        for i in iterator:
            batch = input_texts[i : i + batch_size]
            predictions = self.predict(batch, **generation_kwargs)

            if isinstance(predictions, str):
                predictions = [predictions]

            all_predictions.extend(predictions)

        return all_predictions

    def extract_icd_code(self, generated_text: str) -> str:
        """
        Extrahiert ICD-10 Code aus generiertem Text.

        Unterstützt beide Formate:
        - Standard ICD-10: A12.34, J06.9, M25.562 (mit Punkt)
        - MedSynth Format: A1234, J069, M25562 (ohne Punkt)

        Gibt immer das Format OHNE Punkt zurück für konsistente Vergleiche.
        """
        import re

        text = generated_text.strip().upper()

        # Pattern 1: Standard ICD-10 mit Punkt (z.B. H60.9, M25.562, J06.9)
        # Erlaubt 1-4 Ziffern nach dem Punkt
        pattern_with_dot = r"\b([A-Z]\d{2})\.(\d{1,4})\b"
        matches = re.findall(pattern_with_dot, text)
        if matches:
            # Kombiniere ohne Punkt: H60.9 -> H609
            return matches[0][0] + matches[0][1]

        # Pattern 2: MedSynth Format ohne Punkt (z.B. H609, M25562, B9562)
        # Letter + 2-6 Ziffern
        pattern_no_dot = r"\b([A-Z]\d{2,6})\b"
        matches = re.findall(pattern_no_dot, text)
        if matches:
            return matches[0]

        # Pattern 3: Nur Hauptkategorie (z.B. A12, J06)
        pattern_short = r"\b([A-Z]\d{2})\b"
        matches = re.findall(pattern_short, text)
        if matches:
            return matches[0]

        # Fallback: Erste Zeile, bereinigt
        lines = text.split("\n")
        if lines:
            first_line = lines[0].strip()
            # Entferne Punkt falls vorhanden
            return first_line.replace(".", "")[:10]

        return text.replace(".", "")[:10]
