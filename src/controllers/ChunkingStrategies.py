from typing import List, Optional, Callable, Dict, Any
import re
import tiktoken
from helpers.config import get_settings

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

ENCODING = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    """Consistent token count (cl100k_base)."""
    if not text:
        return 0
    return len(ENCODING.encode(text, disallowed_special=()))


# --------------------------------------------------------------------
# Base + factory
# --------------------------------------------------------------------

class BaseChunker:
    strategy_name: str = "base"
    settings = get_settings()

    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        length_function: Callable[[str], int] = token_length,
    ):
        self.chunk_size = chunk_size if chunk_size is not None else self.settings.RECURSIVE_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.RECURSIVE_CHUNK_OVERLAP
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return self._splitter.split_text(text)


# --------------------------------------------------------------------
# Source-aware chunkers
# --------------------------------------------------------------------

class FirefliesChunker(BaseChunker):
    strategy_name = "fireflies"

    def __init__(self, length_function: Callable[[str], int] = token_length):
        self.chunk_size = self.settings.FIREFLIES_CHUNK_SIZE
        self.chunk_overlap = self.settings.FIREFLIES_CHUNK_OVERLAP
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        chunks: List[str] = []
        remaining = text

        summary_match = re.search(
            r"(?is)^(summary\s*:?\s*)(.*?)(?=\n\s*(transcript|topics|action_items)\s*:|$)",
            text,
        )
        if summary_match:
            summary = summary_match.group(0).strip()
            if summary:
                chunks.append(summary)
            remaining = text[summary_match.end():].strip()

        parts = re.split(r"(?=\[\d{1,2}:\d{2}(?::\d{2})?\])", remaining)
        parts = [p.strip() for p in parts if p and p.strip()]

        if not parts:
            return chunks or self._fallback.chunk(text)

        current: List[str] = []
        current_len = 0
        for part in parts:
            part_len = self.length_function(part)
            if current and current_len + part_len > self.chunk_size:
                chunks.append("\n".join(current))
                if self.chunk_overlap > 0 and current:
                    current = [current[-1]]
                    current_len = self.length_function(current[0])
                else:
                    current = []
                    current_len = 0
            current.append(part)
            current_len += part_len

        if current:
            chunks.append("\n".join(current))

        final: List[str] = []
        for c in chunks:
            if self.length_function(c) > self.chunk_size * 1.5:
                final.extend(self._fallback.chunk(c))
            else:
                final.append(c)
        return final


class ConfluenceChunker(BaseChunker):
    strategy_name = "confluence"

    def __init__(self, length_function: Callable[[str], int] = token_length):
        self.chunk_size = self.settings.CONFLUENCE_CHUNK_SIZE
        self.chunk_overlap = self.settings.CONFLUENCE_CHUNK_OVERLAP
        self.length_function = length_function
        self._headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        self._fallback = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        if not re.search(r"(?m)^#{1,3}\s+\S", text) and not re.search(
            r"(?m)^(Overview|Goal and scope|Purpose|Scope|Terminology|Background)\s*$",
            text,
        ):
            return self._fallback.chunk(text)

        try:
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self._headers)
            docs = md_splitter.split_text(text)
            sections = []
            for d in docs:
                header_bits = [f"{k}: {v}" for k, v in (d.metadata or {}).items()]
                prefix = (" | ".join(header_bits) + "\n") if header_bits else ""
                sections.append(prefix + d.page_content)
        except Exception:
            return self._fallback.chunk(text)

        out: List[str] = []
        for sec in sections:
            if self.length_function(sec) <= self.chunk_size:
                out.append(sec)
            else:
                out.extend(self._fallback.chunk(sec))
        return out or self._fallback.chunk(text)

class SlackChunker(BaseChunker):
    strategy_name = "slack"

    MSG_RE = re.compile(
        r"(?m)^([A-Za-z0-9_./\-]+(?:\s+[A-Za-z0-9_./\-]+)*):\s*(.*?)(?=^\S[^:\n]*:|\Z)",
        re.DOTALL,
    )

    def __init__(self, length_function: Callable[[str], int] = token_length):
        self.chunk_size = self.settings.SLACK_CHUNK_SIZE
        self.chunk_overlap = self.settings.SLACK_CHUNK_OVERLAP
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        messages = [f"{m.group(1)}: {m.group(2).strip()}" for m in self.MSG_RE.finditer(text)]

        if len(messages) < 2:
            return self._fallback.chunk(text)

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for msg in messages:
            msg_len = self.length_function(msg)
            if current and current_len + msg_len > self.chunk_size:
                chunks.append("\n".join(current))
                if self.chunk_overlap > 0 and current:
                    current = [current[-1]]
                    current_len = self.length_function(current[0])
                else:
                    current = []
                    current_len = 0
            current.append(msg)
            current_len += msg_len + 1

        if current:
            chunks.append("\n".join(current))
        return chunks


class TicketChunker(BaseChunker):
    strategy_name = "ticket"

    def __init__(self, length_function: Callable[[str], int] = token_length):
        self.chunk_size = self.settings.TICKET_CHUNK_SIZE
        self.chunk_overlap = self.settings.TICKET_CHUNK_OVERLAP
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        blocks = re.split(
            r"(?i)(?=\n(?:description|comments|acceptance_criteria|body)\s*:)",
            text,
        )
        blocks = [b.strip() for b in blocks if b and b.strip()]

        if len(blocks) <= 1:
            return self._fallback.chunk(text)

        out: List[str] = []
        for block in blocks:
            if self.length_function(block) <= self.chunk_size:
                out.append(block)
            else:
                out.extend(self._fallback.chunk(block))
        return out


class GmailChunker(BaseChunker):
    strategy_name = "gmail"

    REPLY_BOUNDARY = re.compile(
        r"(?m)^(?:"
        r"On .+ wrote:|"
        r"From:\s*.+$|"
        r"-{2,}\s*Original Message\s*-{2,}"
        r")",
    )

    def __init__(self, length_function: Callable[[str], int] = token_length):
        self.chunk_size = self.settings.GMAIL_CHUNK_SIZE
        self.chunk_overlap = self.settings.GMAIL_CHUNK_OVERLAP
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        if self.length_function(text) <= self.chunk_size:
            return [text.strip()]

        parts = self.REPLY_BOUNDARY.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) <= 1:
            return self._fallback.chunk(text)

        out: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + "\n\n" + part).strip() if current else part
            if current and self.length_function(candidate) > self.chunk_size:
                out.append(current)
                current = part
            else:
                current = candidate
        if current:
            out.append(current)

        final: List[str] = []
        for c in out:
            if self.length_function(c) > self.chunk_size * 1.4:
                final.extend(self._fallback.chunk(c))
            else:
                final.append(c)
        return final


# --------------------------------------------------------------------
# Router & Factory
# --------------------------------------------------------------------

class ChunkerFactory:
    
    _strategies: Dict[str, Any] = {
        "recursive": RecursiveChunker,
        "fireflies": FirefliesChunker,
        "confluence": ConfluenceChunker,
        "slack": SlackChunker,
        "ticket": TicketChunker,
        "gmail": GmailChunker,
    }

    # Map external source names/aliases to target strategy names
    SOURCE_MAP: Dict[str, str] = {
        "fireflies": "fireflies",
        "confluence": "confluence",
        "slack": "slack",
        "jira": "ticket",
        "linear": "ticket",
        "github": "ticket",
        "gmail": "gmail",
        "google_drive": "confluence",
        "gdrive": "confluence",
        "hubspot": "recursive",
    }

    @classmethod
    def _normalize_source(cls, source: Optional[str]) -> str:
        """Normalizes inputs like 'Jira-Ticket' or 'Google Drive' to mapped strategies."""
        if not source:
            return "recursive"
        
        # Clean spacing, dashes, and lowercase
        clean_key = re.sub(r"[\s\-]+", "_", source.strip().lower())
        return cls.SOURCE_MAP.get(clean_key, "recursive")

    @classmethod
    def get_chunker(
        cls, 
        source: Optional[str] = None, 
        strategy: Optional[str] = None,
        length_function: Callable[[str], int] = token_length,
        **kwargs
    ) -> BaseChunker:
        """
        Returns a chunker instance by strategy name or source type.
        """
        target_strategy = strategy or cls._normalize_source(source)
        chunker_cls = cls._strategies.get(target_strategy, RecursiveChunker)
        return chunker_cls(length_function=length_function, **kwargs)

    @classmethod
    def chunk(
        cls, 
        text: str, 
        source: Optional[str] = None, 
        strategy: Optional[str] = None, 
        **kwargs
    ) -> List[str]:
        """
        Direct one-shot helper to select chunker and split text in one line.
        """
        chunker = cls.get_chunker(source=source, strategy=strategy, **kwargs)
        return chunker.chunk(text)