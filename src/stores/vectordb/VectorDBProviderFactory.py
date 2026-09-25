
from .providers import QdrantDBProvider, PGVectorProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController
from sqlalchemy.orm import sessionmaker

class VectorDBProviderFactory:
    def __init__(self, config: dict, db_client: sessionmaker=None):
        self.config = config
        self.controller = BaseController()
        self.db_client = db_client

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            qdrant_db_client  = self.controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)
            
            return QdrantDBProvider(
                db_path=qdrant_db_client ,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                sparse_model_id=self.config.SPARSE_MODEL_ID,
                default_vector_size=self.config.EMBEDDING_MODEL_SIZE,
            )

        if provider == VectorDBEnums.PGVECTOR.value:
            return PGVectorProvider(
                db_client=self.db_client,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                default_vector_size=self.config.EMBEDDING_MODEL_SIZE,
                index_threshold=self.config.VECTOR_DB_PGVEC_INDEX_THRESHOLD,
                fts_language=self.config.VECTOR_DB_PGVEC_FTS_LANGUAGE,
                rrf_k=self.config.VECTOR_DB_PGVEC_RRF_K,
            )
        

        return None
