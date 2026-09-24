import uuid

from models.enums.AssetTypeEnum import AssetTypeEnum

from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models import ResponseSignal
from models.db_schemas import Asset
from typing import List, Optional, Iterator, Dict, Set
import pyarrow.parquet as pq
import os
from pathlib import Path
import pandas as pd


def clean_text(text: str) -> str:
    """
    Strip NUL bytes (and other characters Postgres TEXT/VARCHAR columns
    reject) from text pulled out of the parquet dataset. Postgres's UTF8
    encoding does not allow embedded \\x00 even though it's valid UTF-8,
    so asyncpg raises CharacterNotInRepertoireError on insert otherwise.
    """
    if not text:
        return text
    return text.replace("\x00", "")

class DataController(BaseController):
    
    def __init__(self):
        super().__init__()
        self.questions_path = os.path.join("EnterpriseRAG-Bench", "data", "questions", "test.parquet")
        self.documents_path = os.path.join("EnterpriseRAG-Bench", "data", "documents", "test.parquet")
    
    # ------------------------------------------------------------------
    # Bulk parquet ingestion (EnterpriseRAG-Bench and similar datasets)
    # ------------------------------------------------------------------

    def get_parquet_row_count(self, parquet_path: str) -> int:
        return pq.ParquetFile(parquet_path).metadata.num_rows

    def iter_parquet_batches(self, parquet_path: str, columns: List[str],
                             batch_size: int = 500) -> Iterator[List[Dict]]:
        """Stream a parquet file in batches, yielding one list-of-dicts per batch."""
        parquet_file = pq.ParquetFile(parquet_path, memory_map=False, buffer_size=64 * 1024 * 1024)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield record_batch.to_pylist()

    def build_enterprise_rag_asset(self, row: dict, project_id: int,
                                   dataset_name: str = "EnterpriseRAG-Bench") -> Optional[Asset]:
        """
        Build an Asset record from a single EnterpriseRAG-Bench parquet row.
        Returns None if the row has no doc_id (nothing to key it by).

        Note: asset_id is left to autoincrement as usual; doc_id/source_type/title
        are kept in asset_config since Asset has no dedicated columns for them.
        """
        doc_id = row.get("doc_id")
        if not doc_id:
            return None

        source_type = row.get("source_type")
        title = clean_text(row.get("title") or "")
        content = clean_text(row.get("content") or "")

        return Asset(
                asset_project_id=project_id,
                asset_uuid=uuid.uuid4(),
                asset_id=doc_id,
                asset_source_type=source_type,
                asset_name=title,
                asset_type=AssetTypeEnum.FILE.value,
                asset_size=len(content),
                asset_config={
                    "dataset": "EnterpriseRAG-Bench",
                },
            )
    
    # ------------------------------------------------------------------
    # Gold-document helpers (EnterpriseRAG-Bench evaluation subsets)
    # ------------------------------------------------------------------
    
    def load_gold_doc_ids(self) -> Set[str]:
        """
        Load all unique expected_doc_ids from a questions parquet file.
        Returns a set of strings (dsid_...).
        """
        if not self.questions_path:
            raise ValueError("questions_path is required to load gold doc ids")
        if not Path(self.questions_path).exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_path}")
    
        df = pd.read_parquet(self.questions_path, columns=["expected_doc_ids"])
        gold: Set[str] = set()
        for ids in df["expected_doc_ids"]:
            if ids is None:
                continue
            # handle both list and numpy array
            for doc_id in ids:
                if doc_id:
                    gold.add(str(doc_id))
        return gold
    
    def filter_gold_asset_ids(self, asset_ids: Set[str]) -> Set[str]:
        """
        Intersect a set of asset ids with the gold document ids referenced
        by the questions in questions_path. Raises the same errors as
        load_gold_doc_ids if questions_path is missing/invalid.
        """
        gold_ids = self.load_gold_doc_ids(self.questions_path)
        return asset_ids & gold_ids