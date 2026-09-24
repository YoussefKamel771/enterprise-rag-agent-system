from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0
    page_size: Optional[int] = 1000
    embedding_batch_size: Optional[int] = 32
    document_set: Optional[str] = None,
    document_set_kwargs: Optional[dict] = None

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
    