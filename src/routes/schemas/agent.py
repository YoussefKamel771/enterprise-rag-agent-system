from pydantic import BaseModel
from typing import Optional

class AgentAnswerRequest(BaseModel):
    text: str
    thread_id: Optional[str] = None