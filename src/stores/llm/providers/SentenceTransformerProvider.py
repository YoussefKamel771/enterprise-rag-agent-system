# stores/llm/providers/SentenceTransformerProvider.py

from ..LLMInterface import LLMInterface
import numpy as np
import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer

class SentenceTransformerProvider(LLMInterface):
    def __init__(self, device: str = "cuda",
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 defaul_generation_temperature: float = 0.1):

        # Kept for interface/config parity with OpenAIProvider, unused for embedding-only provider
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.defaul_generation_temperature = defaul_generation_temperature

        self.device = device

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None
        self.full_dim = None

        self.model = None

        self.logger = logging.getLogger("uvicorn")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.model = SentenceTransformer(model_id, device=self.device, truncate_dim=embedding_size)
        self.full_dim = self.model.get_embedding_dimension()

        if embedding_size > self.full_dim:
            self.logger.error(
                f"embedding_size ({embedding_size}) > model dim ({self.full_dim})"
            )
            raise ValueError(
                f"embedding_size ({embedding_size}) > model dim ({self.full_dim})"
            )

        self.embedding_size = embedding_size
        self.logger.info(
            f"Loaded {model_id} | full={self.full_dim} -> using {self.embedding_size} (MRL)"
        )

    

    def embed_text(self, text: Union[str, List[str]], 
                   document_type: str = None,
                   batch_size: int = 64) -> list:
        if not self.model:
            self.logger.error("Embedding model is not set.")
            return None

        if not self.embedding_model_id or not self.embedding_size:
            self.logger.error("Embedding model ID / size is not set.")
            return None

        # Some models benefit from different prompts for query vs document
        if document_type == "query":
            if isinstance(text, str):
                text = f"query: {text}"
            else:
                text = [f"query: {t}" for t in text]
                
        
        embeddings = self.model.encode(
            text,
            normalize_embeddings=True,   
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )


        if isinstance(text, str):
            return [embeddings.tolist()]
        return embeddings.tolist()

    def set_generation_model(self, model_id: str):
            # This provider is embedding-only; generation is not supported.
            self.logger.warning(
                "SentenceTransformerProvider does not support text generation. "
                f"Ignoring set_generation_model('{model_id}')."
            )
            self.generation_model_id = model_id
            
    def generate_text(self, prompt: str, chat_history: list = [],
                           max_output_tokens: int = None, temperature: float = None) -> str:
            self.logger.error("SentenceTransformerProvider does not support generate_text.")
            return None
        
    def process_text(self, prompt: str):
        return None

    def construct_prompt(self, prompt: str, role: str) -> str:
        return None