import tiktoken

from models.db_schemas.supportRag.schemas.dataChunk import DataChunk

from .BaseController import BaseController
from .ProjectController import ProjectController
from models import ProcessingEnum
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .ChunkingStrategies import ChunkerFactory, token_length
import os
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: dict

class ProcessController(BaseController):
    def __init__(self, project_id: str, logger=None):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.logger = logger

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]
    
    def get_file_loader(self, file_id: str):
        file_extension = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)

        # self.logger.info(f"Attempting to load file with extension: {file_extension}")
        if not os.path.exists(file_path):
            return None

        if file_extension == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        
        if file_extension == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path) 
        
        return None
    
    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)

        if not loader:
            return None

        return loader.load() # Returns a list of Document objects, each with page_content and metadata attributes.
    

    def process_file_content(self, file_content: list, file_id: str,
                             chunk_size: int = 1000, chunk_overlap: int = 200):
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
        #                                                chunk_overlap=chunk_overlap,
        #                                                length_function=len)
        
        file_content_texts = [doc.page_content for doc in file_content]
        file_content_metadatas = [doc.metadata for doc in file_content]

        # chunks = text_splitter.create_documents(
        #     file_content_texts, 
        #     metadatas=file_content_metadatas
        # )
        chunks = self.process_simpler_splitter(
            texts=file_content_texts,
            metadatas=file_content_metadatas,
            chunk_size=chunk_size,
        )
        return chunks

    def process_simpler_splitter(self, texts: List[str], metadatas: List[dict], chunk_size: int, splitter_tag: str="\n"):
        
        full_text = " ".join(texts)

        # split by splitter_tag
        lines = [ doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip()) > 1 ]

        chunks = []
        current_chunk = ""

        for line in lines:
            current_chunk += line + splitter_tag
            if len(current_chunk) >= chunk_size:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={}
                ))

                current_chunk = ""

        if len(current_chunk) >= 0:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={}
            ))

        return chunks
    
    # ------------------------------------------------------------------
    # Bulk dataset chunking (EnterpriseRAG-Bench and similar datasets)
    # ------------------------------------------------------------------

    def build_enterprise_rag_chunks(self, asset, content: str, chunk_size: int = 1000,
                                    chunk_overlap: int = 100, strategy: str = "recursive"):
        """
        Chunk a single already-ingested Asset's content (re-derived from the
        source parquet) into DataChunk records ready for bulk insert.

        Extra metadata (strategy/char_count/token_count) is stored in
        chunk_metadata since DataChunk has no dedicated columns for it.
        """
        chunker = ChunkerFactory.get_chunker(strategy, 
                                             chunk_size=chunk_size, 
                                             chunk_overlap=chunk_overlap, 
                                             length_function=token_length)
        pieces = chunker.chunk(content)


        return [
            DataChunk(
                chunk_text=piece,
                chunk_order=order + 1,
                chunk_strategy=strategy,
                chunk_char_count=len(piece),
                chunk_token_count=token_length(piece),
                chunk_project_id=asset.asset_project_id,
                chunk_asset_id=asset.asset_id,
            )
            for order, piece in enumerate(pieces)
        ]   
