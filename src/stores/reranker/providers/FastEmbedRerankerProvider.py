from ..RerankerInterface import RerankerInterface
from models import RetrievedDocument
import asyncio
import logging
from typing import List
from fastembed.rerank.cross_encoder import TextCrossEncoder

class FastEmbedRerankerProvider(RerankerInterface):
    """
    Free, local cross-encoder reranker via FastEmbed's TextCrossEncoder —
    same library already pulled in as the qdrant-client[fastembed] extra
    for BM25 sparse vectors, so if you're on Qdrant this adds zero new
    dependencies. Runs on-device, no API key/rate limits.

    NOTE: FastEmbed's reranking API has changed across versions — verify
    `from fastembed.rerank.cross_encoder import TextCrossEncoder` and the
    shape of `.rerank(...)`'s return value against your installed version
    before relying on this in production.
    """

    def __init__(self, default_top_n: int = 10):
        self.default_top_n = default_top_n
        self.reranker_model_id = None
        self._model = None
        self.logger = logging.getLogger("uvicorn")

    def set_reranker_model(self, model_id: str):
        self.reranker_model_id = model_id
        self._model = TextCrossEncoder(model_name=model_id)

    async def rerank(self, query: str, documents: List[RetrievedDocument],
                     top_n: int = None) -> List[RetrievedDocument]:

        if not self._model:
            self.logger.error("Reranker model is not loaded. Call set_reranker_model first.")
            return documents

        if not documents or len(documents) == 0:
            return documents

        top_n = min(top_n or self.default_top_n, len(documents))
        doc_texts = [doc.text for doc in documents]

        def _run_rerank():
            return list(self._model.rerank(query, doc_texts))

        scores = await asyncio.to_thread(_run_rerank)

        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            RetrievedDocument(text=doc.text, score=float(score))
            for doc, score in scored
        ]