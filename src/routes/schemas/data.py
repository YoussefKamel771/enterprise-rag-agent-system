from pydantic import BaseModel, Field
from typing import List, Optional

class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: Optional[int] = 100
    chunk_overlap: Optional[int] = 20
    do_reset: Optional[int] = 0
     