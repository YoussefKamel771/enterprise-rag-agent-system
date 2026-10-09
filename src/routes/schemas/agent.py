from pydantic import BaseModel
from typing import Optional


class AgentAnswerRequest(BaseModel):
    text: str
    user_id: Optional[str] = None      # scopes thread + long-term memory
    thread_id: Optional[str] = None    # omit to start a new conversation