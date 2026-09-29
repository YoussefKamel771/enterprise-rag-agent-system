from qdrant_client import models, AsyncQdrantClient 
from ..VectorDBInterface import VectorDBInterface, MetadataFilter
from ..VectorDBEnums import DistanceMethodEnums
from models import RetrievedDocument
import logging
from typing import List, Optional

class QdrantDBProvider(VectorDBInterface):
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    _TOP_LEVEL_FIELDS = {"source_type", "doc_id", "chunk_id"}
    
    def __init__(self, db_path: str, distance_method: str,
                 sparse_model_id: str = "Qdrant/bm25",
                 default_vector_size: int = 786):

        self.db_path = db_path
        self.distance_method = None
        self.client = None
        self.sparse_model_id = sparse_model_id
        self.default_vector_size = default_vector_size

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger("uvicorn")

    async def connect(self):
        self.client = AsyncQdrantClient(path=self.db_path)
        self.logger.info("Connected to Qdrant database.")

    async def disconnect(self):
        if self.client:
            await self.client.close()
            self.logger.info("Disconnected from Qdrant database.")

    async def is_collection_exists(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self) -> List:
        collections = await self.client.get_collections()
        return [collection.name for collection in collections.collections]

    async def get_collection_info(self, collection_name: str) -> dict:
        collection_info = await self.client.get_collection(collection_name=collection_name)
        return collection_info.model_dump()

    async def delete_collection(self, collection_name: str) -> bool:
        if await self.is_collection_exists(collection_name):
            await self.client.delete_collection(collection_name=collection_name)
            self.logger.info(f"Collection '{collection_name}' deleted.")
            return True
        return False

    async def create_collection(self, collection_name: str, 
                                embedding_size: int, 
                                do_reset: bool = False) -> bool:
        
        if await self.is_collection_exists(collection_name):
            if do_reset:
                await self.delete_collection(collection_name)
            else:
                self.logger.warning(f"Collection '{collection_name}' already exists.")
                return False

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=embedding_size, distance=self.distance_method
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        self.logger.info(
            f"Collection '{collection_name}' created with dense size {embedding_size} "
            f"and IDF-weighted sparse field '{self.SPARSE_VECTOR_NAME}'."
        )
        return True
    
    def _build_point(self, id_, text: str, vector: list, metadata: dict = None):
        metadata = metadata or {}
        source_type = metadata.get("source_type")
        return models.PointStruct(
            id=id_,
            vector={
                self.DENSE_VECTOR_NAME: vector,
                self.SPARSE_VECTOR_NAME: models.Document(text=text, model=self.sparse_model_id),
            },
            payload={
                "text": text,
                "metadata": metadata,
                "chunk_id": id_,
                "doc_id": metadata.get("doc_id"),
                # lower-cased at write AND at filter time so "Slack" == "slack"
                "source_type": source_type.lower() if isinstance(source_type, str) else source_type,
            },
        )

    async def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        try:
            await self.client.upsert(
                collection_name=collection_name,
                points=[self._build_point(record_id, text, vector, metadata)],
            )
            self.logger.info(f"Inserted record with ID '{record_id}' into collection '{collection_name}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to insert record into collection '{collection_name}': {e}")
            return False

    async def insert_many(self, collection_name: str, texts: list,
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Can not insert new records to non-existed collection: {collection_name}")
            return False

        if not metadata:
            metadata = [None] * len(texts)

        if not record_ids:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_points = [
                self._build_point(id_, txt, vec, meta)
                for txt, vec, meta, id_ in zip(batch_texts, batch_vectors, batch_metadata, batch_record_ids)
            ]

            try:
                await self.client.upsert(
                    collection_name=collection_name,
                    records=batch_points
                )
                self.logger.info(f"Inserted batch of {len(batch_points)} records into collection '{collection_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to insert batch into collection '{collection_name}': {e}")
                return False
        return True

    async def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        results = await self.client.query_points(
            collection_name=collection_name,
            query_vector=vector,
            using=self.DENSE_VECTOR_NAME,
            limit=limit
        )
        points = results.points
        if not points or len(points) == 0:
            return None
        
        return [
            RetrievedDocument(**{
                "score": point.score,
                "text": point.payload["text"],
            })
            for point in points
        ]
        
    def _payload_key(self, key: str) -> str:
        return key if key in self._TOP_LEVEL_FIELDS else f"metadata.{key}"

    @staticmethod
    def _norm(key: str, value):
        return value.lower() if key == "source_type" and isinstance(value, str) else value

    def _build_qdrant_filter(self, filters: Optional[MetadataFilter],
                             exclude_chunk_ids: Optional[List[int]]) -> Optional[models.Filter]:
        must, must_not = [], []

        if filters:
            for key, value in filters.equals.items():
                must.append(models.FieldCondition(
                    key=self._payload_key(key), match=models.MatchValue(value=self._norm(key, value))))
            for key, values in filters.in_.items():
                if values:
                    must.append(models.FieldCondition(
                        key=self._payload_key(key),
                        match=models.MatchAny(any=[self._norm(key, v) for v in values])))
            for key in set(filters.gte) | set(filters.lte):       # merge gte+lte per key
                lo, hi = filters.gte.get(key), filters.lte.get(key)
                sample = lo if lo is not None else hi
                rng = (models.DatetimeRange(gte=lo, lte=hi) if isinstance(sample, str)
                       else models.Range(gte=lo, lte=hi))
                must.append(models.FieldCondition(key=self._payload_key(key), range=rng))

        if exclude_chunk_ids:
            # point ids ARE chunk ids (NLPController passes record_ids=chunk ids)
            must_not.append(models.HasIdCondition(has_id=[int(x) for x in exclude_chunk_ids]))

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)
        
    async def search_hybrid(self, collection_name: str, query_text: str, query_vector: list,
                            limit: int = 10, offset: int = 0,
                            filters: Optional[MetadataFilter] = None,
                            exclude_chunk_ids: Optional[List[int]] = None,
                            return_debug: bool = False):
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Can not hybrid-search a non-existed collection: {collection_name}")
            return None

        qfilter = self._build_qdrant_filter(filters, exclude_chunk_ids)
        fetch = limit + offset          # each leg must reach past the offset for paging to work

        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                # the filter must be repeated inside EACH prefetch -- a top-level
                # query_filter does not propagate into prefetch stages
                models.Prefetch(query=query_vector, using=self.DENSE_VECTOR_NAME,
                                filter=qfilter, limit=fetch),
                models.Prefetch(query=models.Document(text=query_text, model=self.sparse_model_id),
                                using=self.SPARSE_VECTOR_NAME, filter=qfilter, limit=fetch),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            offset=offset,
        )

        points = results.points
        if not points:
            return None

        docs = [
            RetrievedDocument(
                chunk_id=int(p.id),
                doc_id=p.payload.get("doc_id") or str(p.id),
                source_type=p.payload.get("source_type"),
                text=p.payload["text"],
                score=p.score,
                metadata=p.payload.get("metadata") or {},
            )
            for p in points
        ]
        # Qdrant fuses server-side, so per-leg rows aren't available for debug output
        return {"results": docs, "dense": [], "lexical": []} if return_debug else docs
