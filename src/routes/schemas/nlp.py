from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0
    page_size: Optional[int] = 1000
    embedding_batch_size: Optional[int] = 64

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
    