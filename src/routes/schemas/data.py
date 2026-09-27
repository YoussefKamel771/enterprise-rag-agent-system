from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ProcessRequest(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 500
    delete_existing: bool = False
    skip_chunked: bool = False
    document_set: Optional[str] = None
    document_set_kwargs: Optional[Dict[str, Any]] = None
     