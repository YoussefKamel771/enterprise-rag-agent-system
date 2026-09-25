from abc import ABC, abstractmethod
from typing import List
from models import RetrievedDocument

class RerankerInterface(ABC):

    @abstractmethod
    def set_reranker_model(self, model_id: str):
        pass

    @abstractmethod
    async def rerank(self, query: str, documents: List[RetrievedDocument],
                      top_n: int = None) -> List[RetrievedDocument]:
        pass