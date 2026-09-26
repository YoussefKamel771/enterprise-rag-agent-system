from abc import ABC, abstractmethod
from typing import List
from models import RetrievedDocument, RetrievalResult

class RerankerInterface(ABC):

    @abstractmethod
    def set_reranker_model(self, model_id: str):
        pass

    @abstractmethod
    async def rerank(self, query: str, results: RetrievalResult,
                      top_n: int = None) -> RetrievalResult:
        pass