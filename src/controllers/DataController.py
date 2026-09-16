import uuid

from models.enums.AssetTypeEnum import AssetTypeEnum

from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models import ResponseSignal
from models.db_schemas import Asset
from typing import List, Optional, Iterator, Dict
import pyarrow.parquet as pq
import re
import os

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
        self.size_scale = 1048576 # convert MB to bytes


    def validate_uploaded_file(self, file: UploadFile):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value

        return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value
    
    def generate_unique_file_path(self, orig_file_name: str, project_id: str):
        cleaned_file_name = self.get_clean_file_name(orig_file_name)
        project_path = ProjectController().get_project_path(project_id=project_id)
        unique_suffix = self.generate_random_string()
        new_file_path = os.path.join(
                                    project_path, 
                                    f"{unique_suffix}_{cleaned_file_name}")
        
        while os.path.exists(new_file_path):
            unique_suffix = self.generate_random_string()
            new_file_path = os.path.join(
                                    project_path, 
                                    f"{unique_suffix}_{cleaned_file_name}")
        return new_file_path, f"{unique_suffix}_{cleaned_file_name}"
    
    def get_clean_file_name(self, orig_file_name: str):

        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())

        # replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name
    
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
    