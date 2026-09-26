from .base import SQLAlchemyBase
from sqlalchemy import Column, Integer, DateTime, func, String, ForeignKey, UniqueConstraint, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from pydantic import BaseModel
import uuid
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass

class DataChunk(SQLAlchemyBase):

    __tablename__ = "chunks"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)

    chunk_text = Column(Text, nullable=False)
    chunk_order = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, nullable=True)
    chunk_strategy = Column(String, nullable=True)
    
    chunk_char_count = Column(Integer, nullable=True)
    chunk_token_count = Column(Integer, nullable=True)

    chunk_project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    chunk_asset_id = Column(String, 
                            ForeignKey("assets.asset_id", ondelete="CASCADE"),
                            nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    project = relationship("Project", back_populates="chunks")
    asset = relationship("Asset", back_populates="chunks")

    __table_args__ = (
        Index('ix_chunk_project_id', chunk_project_id),
        Index('ix_chunk_asset_id', chunk_asset_id),      
    )

class RetrievedDocument(BaseModel):
    chunk_id: int
    doc_id: str                              # = chunk_asset_id, the ground-truth doc identifier
    text: str
    score: float
    rank: int
    
    source_type: Optional[str] = None
    parent_doc_id: Optional[str] = None      # for small-to-big / parent-document retrieval later
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    
@dataclass
class RetrievalResult:
    documents: list[RetrievedDocument]
    dense_debug: Optional[list[dict]] = None
    lexical_debug: Optional[list[dict]] = None
    error: Optional[str] = None          # None == success

    @property
    def ok(self) -> bool:
        return self.error is None