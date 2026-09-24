from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging
from typing import List, Union
import numpy as np
class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, api_url: str = None, 
                 default_input_max_characters: int = 1000, 
                 default_generation_max_output_tokens: int = 1000,
                 defaul_generation_temperature : float = 0.1):

        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.defaul_generation_temperature = defaul_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = OpenAIEnums

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url if self.api_url and len(self.api_url) else None
        )

        self.logger = logging.getLogger("uvicorn")

    def set_generation_model(self, model_id : str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id : str, embedding_size : int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(self, prompt : str, chat_history : list = None,
                       max_output_tokens : int = None, temperature : float = None):

        if chat_history is None:
            chat_history = []

        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens  else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.defaul_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt, role=OpenAIEnums.USER.value)
                
        )

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature 
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message.content:
            self.logger.error("Error while generating text with OpenAI")
            return None
         
        return response.choices[0].message.content


    def embed_text(
    self,
    text: Union[str, List[str]],
    document_type: str = None,
    batch_size: int = 100,
    ) -> list:

        if not self.embedding_model_id or not self.embedding_size:
            self.logger.error(
                "Embedding model ID / size is not set."
            )
            return None

        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False

        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=text,
        )

        # Keep API ordering explicit
        raw_embeddings = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]

        # 768 -> 256
        embeddings = np.asarray(
            raw_embeddings,
            dtype=np.float32
        )

        embeddings = embeddings[:, :self.embedding_size]

        # Normalize entire batch
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        # Avoid division by zero
        embeddings = embeddings / np.clip(
            norms,
            a_min=1e-12,
            a_max=None
        )

        embeddings = embeddings.tolist()

        if single_input:
            return embeddings[0]

        return embeddings

    def process_text(self, prompt: str):
        return prompt[:self.default_input_max_characters].strip().lower()

    def construct_prompt(self, prompt: str, role: str) -> str:
        return {
            "role": role,
            "content": prompt
        }
        
    def normalize_embedding(self, embedding: list) -> list:
        embedding = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding

        return embedding / norm


