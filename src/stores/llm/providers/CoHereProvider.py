from ..LLMInterface import LLMInterface
from ..LLMEnums import COHEREEnums, DocumentTypeEnums
import cohere
import logging
from typing import List, Union

class CoHereProvider(LLMInterface):
    def __init__(self, api_key: str, 
                 default_input_max_characters: int = 1000, 
                 default_generation_max_output_tokens: int = 1000,
                 defaul_generation_temperature : float = 0.1):

        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.defaul_generation_temperature = defaul_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = COHEREEnums

        self.client = cohere.Client(api_key=self.api_key)

        self.logger = logging.getLogger("uvicorn")

    def set_generation_model(self, model_id : str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id : str, embedding_size : int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, prompt: str):
            return prompt[:self.default_input_max_characters].strip().lower()

    def generate_text(self, prompt : str, chat_history : list = [],
                           max_output_tokens : int = None, temperature : float = None):
    
            if not self.client:
                self.logger.error("Cohere client is not initialized.")
                return None
    
            if not self.generation_model_id:
                self.logger.error("Generation model ID is not set.")
                return None
    
            max_output_tokens = max_output_tokens if max_output_tokens  else self.default_generation_max_output_tokens
            temperature = temperature if temperature else self.defaul_generation_temperature
    
    
            response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history,
                message=self.process_text(prompt),
                max_tokens=max_output_tokens,
                temperature=temperature 
            )
    
            if not response or not response.text :
                self.logger.error("Error while generating text with Cohere")
                return None
             
            return response.text

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
            if not self.client:
                self.logger.error("Cohere client is not initialized.")
                return None

            if isinstance(text, str):
                text = [text]
    
            if not self.embedding_model_id:
                self.logger.error("Embedding model ID for Cohere is not set.")
                return None

            input_type = COHEREEnums.DOCUMENT.value
            if document_type == DocumentTypeEnums.QUERY.value:
                input_type = COHEREEnums.QUERY.value

    
            response = self.client.embed(
                model=self.embedding_model_id,
                texts=[self.process_text(t) for t in text],
                input_type=input_type,
                embedding_types=['float']
            )   
    
            if not response or not response.embeddings or not response.embeddings.float:
                self.logger.error("Error while embedding text with CoHere")
                return None
    

            return response.embeddings.float

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": prompt
        }