from .BaseController import BaseController
from .ProjectController import ProjectController
from models import ProcessingEnum
from models.db_schemas import DataChunk
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from typing import List, Optional
from dataclasses import dataclass
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    """
    Return the number of tokens in a text string.

    cl100k_base is commonly used for OpenAI-style tokenization and is
    sufficient for establishing a consistent chunking baseline.
    """
    return len(ENCODING.encode(text, disallowed_special=()))

# --------------------------------------------------------------------
# Pluggable chunking strategies (used for bulk/dataset ingestion paths)
# --------------------------------------------------------------------

class BaseChunker:
    strategy_name: str = "base"

    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, length_function=len):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return self._splitter.split_text(text)


class ChunkerFactory:
    _strategies = {
        "recursive": RecursiveChunker,
    }

    @classmethod
    def get_chunker(cls, strategy: str, **kwargs) -> BaseChunker:
        if strategy not in cls._strategies:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. Available: {list(cls._strategies)}"
            )
        return cls._strategies[strategy](**kwargs)