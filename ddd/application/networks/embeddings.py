from functools import cached_property
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer

from ddd.settings import settings
from .base import SingletonMeta

class EmbedingModelSingleton(metaclass=SingletonMeta):
    """A singleton class that provides a pre-trained transformer model for generating embeddings of input text."""

    def __init__(
            self, 
            model_id:str = settings.TEXT_EMBEDDING_MODEL_ID,
            device:str = settings.RAG_MODEL_DEVICE,
            cache_dir: Optional[Path]  = None
            ) -> None:

        self._model_id = model_id
        self._device = device
        
        self._model = SentenceTransformer(
            self._model_id,
            device=self._device,
            cache_folder=str(cache_dir) if cache_dir else None
        )
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self._model_id
    
    @cached_property
    def embedding_size(self) -> int:
        return self._model.encode(" ").shape[0]
    
    @property
    def max_input_length(self) -> int:
        return self._model.max_seq_length
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._model.tokenizer
    
    def __call__(
            self,
            input_text: str | list[str],
            to_list: bool = True,
        ) -> NDArray[np.float32] | list[float] | list[list[float]]:
        
        try:
            embeddings = self._model.encode(input_text)
        except Exception:
            logger.error(f"Error generating embeddings for input text: {input_text} with model {self._model_id}")
            return [] if to_list else np.array([])
        
        if to_list:
            embeddings = embeddings.tolist()
        
        return embeddings


class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    def __init__(
            self,
            model_id: str = settings.CROSS_ENCODER_MODEL_ID,
            device: str = settings.RAG_MODEL_DEVICE,
        ) -> None:
        
        self._model_id = model_id
        self._device = device

        self._model = CrossEncoder(self._model_id, device=self._device)
        self._model.eval()
    
    def __call__(self, pairs: list[tuple[str, str]], to_list: bool = True) -> NDArray[np.float32] | list[float]:
        scores = self._model.predict(pairs)
        if to_list:
            scores = scores.tolist()
        
        return scores 
