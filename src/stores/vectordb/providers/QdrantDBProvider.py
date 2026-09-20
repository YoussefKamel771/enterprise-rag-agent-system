from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from models import RetrievedDocument
import logging
from typing import List

class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_path: str, distance_method: str):

        self.db_path = db_path
        self.distance_method = None
        self.client = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger("uvicorn")

    async def connect(self):
        self.client = QdrantClient(path=self.db_path)
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
            vectors_config=models.VectorParams(size=embedding_size, distance=self.distance_method)
        )
        self.logger.info(f"Collection '{collection_name}' created with embedding size {embedding_size}.")
        return True

    async def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        try:
            await self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=[record_id],
                        vector=vector,
                        payload={"text": text, "metadata": metadata if metadata else {}}
                    )
                ]
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

            batch_records = [
                models.Record(
                    id=id,
                    vector=vec,
                    payload={"text": txt, "metadata": meta if meta else {}}
                )
                for txt, vec, meta, id in zip(batch_texts, batch_vectors, batch_metadata, batch_record_ids)
            ]

            try:
                await self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                )
                self.logger.info(f"Inserted batch of {len(batch_records)} records into collection '{collection_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to insert batch into collection '{collection_name}': {e}")
                return False
        return True

    async def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        search_results = await self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit
        )
        if not search_results or len(search_results) == 0:
            return None
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
            })
            for result in search_results
        ]
