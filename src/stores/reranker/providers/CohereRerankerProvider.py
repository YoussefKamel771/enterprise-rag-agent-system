from stores.reranker.RerankerInterface import RerankerInterface
from models import RetrievedDocument
import cohere
import asyncio
import logging
from typing import List


class CohereRerankerProvider(RerankerInterface):
    def __init__(self, api_key: str, default_top_n: int = 10):

        self.api_key = api_key
        self.default_top_n = default_top_n
        self.reranker_model_id = None

        self.client = cohere.Client(api_key=self.api_key)

        self.logger = logging.getLogger("uvicorn")

    def set_reranker_model(self, model_id: str):
        self.reranker_model_id = model_id

    async def rerank(self, query: str, documents: List[RetrievedDocument],
                      top_n: int = None) -> List[RetrievedDocument]:

        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return documents

        if not self.reranker_model_id:
            self.logger.error("Reranker model ID is not set.")
            return documents

        if not documents or len(documents) == 0:
            return documents

        top_n = top_n if top_n else self.default_top_n
        top_n = min(top_n, len(documents))
        doc_texts = [doc.text for doc in documents]

        try:
            # cohere.Client.rerank is a blocking call; run it off the event loop
            # so it doesn't stall other concurrent requests.
            response = await asyncio.to_thread(
                self.client.rerank,
                model=self.reranker_model_id,
                query=query,
                documents=doc_texts,
                top_n=top_n,
            )
        except Exception as e:
            self.logger.error(f"Error while reranking with Cohere: {e}")
            return documents[:top_n]

        if not response or not response.results:
            self.logger.error("No rerank results returned from Cohere.")
            return documents[:top_n]

        return [
            RetrievedDocument(
                text=documents[result.index].text,
                score=result.relevance_score,
            )
            for result in response.results
        ]