from ..RerankerInterface import RerankerInterface
from models import RetrievedDocument
import asyncio
import logging
from typing import List
from sentence_transformers import CrossEncoder
import math


class SentenceTransformerRerankerProvider(RerankerInterface):
    """
    Free, local cross-encoder reranker via sentence-transformers. Runs
    entirely on-device (CPU is fine for small models like MiniLM) — no API
    key, no rate limits, no network call at inference time. Model weights
    download from the Hugging Face Hub the first time set_reranker_model
    is called.
    """

    def __init__(self, default_top_n: int = 10, device="cuda"):
        self.default_top_n = default_top_n
        self.reranker_model_id = None
        self._model = None
        self.device = device
        self.logger = logging.getLogger("uvicorn")

    def set_reranker_model(self, model_id: str):
        self.reranker_model_id = model_id
        self._model = CrossEncoder(model_id, device=self.device)

    async def rerank(self, query: str, documents: List[RetrievedDocument],
                     top_n: int = None) -> List[RetrievedDocument]:

        if not self._model:
            self.logger.error("Reranker model is not loaded. Call set_reranker_model first.")
            return documents

        if not documents or len(documents) == 0:
            return documents

        top_n = min(top_n or self.default_top_n, len(documents))
        pairs = [(query, doc.text) for doc in documents]

        # CrossEncoder.predict is a blocking, CPU-bound call — keep it off
        # the event loop.
        scores = await asyncio.to_thread(self._model.predict, pairs)
        scores = [1 / (1 + math.exp(-s)) for s in scores]  # optional: squash logits to (0, 1)

        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            RetrievedDocument(text=doc.text, score=float(score))
            for doc, score in scored
        ]