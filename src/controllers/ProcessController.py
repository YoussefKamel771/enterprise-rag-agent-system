from pathlib import Path
from typing import Set
 
import pandas as pd
 
from models.db_schemas.supportRag.schemas.dataChunk import DataChunk
 
from .BaseController import BaseController
from .ChunkingStrategies import ChunkerFactory, token_length
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: dict

class ProcessController(BaseController):
    def __init__(self, project_id: str, logger=None):
        super().__init__()

        self.project_id = project_id
        self.logger = logger
    
    # ------------------------------------------------------------------
    # Bulk dataset chunking (EnterpriseRAG-Bench and similar datasets)
    # ------------------------------------------------------------------

    def build_enterprise_rag_chunks(self, asset, content: str,
                                    source_type: str = None):
        """
        Chunk a single already-ingested Asset's content (re-derived from the
        source parquet) into DataChunk records ready for bulk insert.

        Extra metadata (strategy/char_count/token_count) is stored in
        chunk_metadata since DataChunk has no dedicated columns for it.
        """
        

        chunker = ChunkerFactory.get_chunker(source=source_type)
        pieces = chunker.chunk(content)


        return [
            DataChunk(
                chunk_text=piece,
                chunk_order=order + 1,
                chunk_strategy="Source-aware",
                chunk_metadata={
                    "doc_id": asset.asset_id,
                    "source_type":asset.asset_source_type,
                    "title":asset.asset_name,
                    "chunk_index": order+1,
                    },
                chunk_char_count=len(piece),
                chunk_token_count=token_length(piece),
                chunk_project_id=asset.asset_project_id,
                chunk_asset_id=asset.asset_id,
            )
            for order, piece in enumerate(pieces)
        ]   
        
    
