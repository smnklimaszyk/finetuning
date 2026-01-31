"""
SLM Model Wrapper für Small Language Models.
Unterstützt sowohl vortrainierte als auch finetuned Modelle.
"""

from .llm_model import LLMModel
import logging

logger = logging.getLogger(__name__)


class SLMModel(LLMModel):
    """
    Small Language Model Wrapper.
    Erbt von LLMModel, da Funktionalität identisch ist.
    Trennung für Klarheit in Experimenten.
    """

    def __init__(self, model_name: str, is_finetuned: bool = False, **kwargs):
        super().__init__(model_name, **kwargs)
        self.is_finetuned = is_finetuned
        logger.info(f"Initialized SLM (finetuned={is_finetuned}): {model_name}")
