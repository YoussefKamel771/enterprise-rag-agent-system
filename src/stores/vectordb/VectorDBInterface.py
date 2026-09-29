from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from models import RetrievedDocument
from dataclasses import dataclass, field
@dataclass
class MetadataFilter:
    """Backend-agnostic filter. Each provider translates it to its native query.
    `date_after` / `created_*` style ranges use gte/lte (ISO-8601 strings compare correctly)."""
    equals: Dict[str, Any] = field(default_factory=dict)         # field == value
    in_: Dict[str, List[Any]] = field(default_factory=dict)      # field IN (values)
    gte: Dict[str, Any] = field(default_factory=dict)            # field >= value
    lte: Dict[str, Any] = field(default_factory=dict)            # field <= value
    
    def is_empty(self) -> bool:
        return not (self.equals or self.in_ or self.gte or self.lte)


class VectorDBInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def is_collection_exists(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def list_all_collections(self) -> List:
        pass

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, 
                          embedding_size: int, 
                          do_reset: bool = False) -> bool:
        pass    

    @abstractmethod
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        pass

    @abstractmethod
    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        pass

    @abstractmethod
    def search_by_vector(self, collection_name: str, vector: list, limit: int) -> List[RetrievedDocument]:
        pass
    
    @abstractmethod
    def search_hybrid(self, collection_name: str, query_text: str, query_vector: list,
                    limit: int, return_debug: bool,
                    filters: Optional[MetadataFilter] = None,
                    exclude_chunk_ids: Optional[list[int]] = None,) -> List[RetrievedDocument]:
        pass
