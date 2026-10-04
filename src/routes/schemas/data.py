from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ProcessRequest(BaseModel):
    batch_size: int = 500
    delete_existing: bool = False
    skip_chunked: bool = False
    document_set: Optional[str] = None
    document_set_kwargs: Optional[Dict[str, Any]] = None
     
class PipelineRequest(BaseModel):
    # chunking params
    batch_size: int = 500
    delete_existing: bool = False
    skip_chunked: bool = False
    document_set: Optional[str] = None
    document_set_kwargs: Optional[Dict[str, Any]] = None
    # indexing params
    do_reset: bool = False
    page_size: int = 1000
    embedding_batch_size: int = 16