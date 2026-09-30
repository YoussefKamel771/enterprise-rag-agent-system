from typing import List, Optional, Callable
import re
import tiktoken

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

    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        length_function: Callable[[str], int] = token_length,
    ):
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


# --------------------------------------------------------------------
# Source-aware chunkers
# --------------------------------------------------------------------

class FirefliesChunker(BaseChunker):
    """
    Prefer summary as its own chunk, then split transcript on
    timestamps [HH:MM] or speaker turns.
    """
    strategy_name = "fireflies"

    def __init__(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 80,
        length_function: Callable[[str], int] = token_length,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        chunks: List[str] = []
        remaining = text

        # Pull out leading summary if present
        summary_match = re.search(
            r"(?is)^(summary\s*:?\s*)(.*?)(?=\n\s*(transcript|topics|action_items)\s*:|$)",
            text,
        )
        if summary_match:
            summary = summary_match.group(0).strip()
            if summary:
                chunks.append(summary)
            remaining = text[summary_match.end():].strip()

        # Split transcript on [HH:MM] or [HH:MM:SS]
        parts = re.split(r"(?=\[\d{1,2}:\d{2}(?::\d{2})?\])", remaining)
        parts = [p.strip() for p in parts if p and p.strip()]

        if not parts:
            return chunks or self._fallback.chunk(text)

        # Pack consecutive turns up to chunk_size
        current: List[str] = []
        current_len = 0
        for part in parts:
            part_len = self.length_function(part)
            if current and current_len + part_len > self.chunk_size:
                chunks.append("\n".join(current))
                # simple overlap: keep last turn
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

        # Any leftover that is still huge → recursive fallback
        final: List[str] = []
        for c in chunks:
            if self.length_function(c) > self.chunk_size * 1.5:
                final.extend(self._fallback.chunk(c))
            else:
                final.append(c)
        return final


class ConfluenceChunker(BaseChunker):
    """Markdown / heading-aware splitter with recursive fallback."""
    strategy_name = "confluence"

    def __init__(
        self,
        chunk_size: int = 650,
        chunk_overlap: int = 80,
        length_function: Callable[[str], int] = token_length,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self._headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        self._fallback = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # Heuristic: only use header splitter if we actually see headings
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

        # Pack / re-split oversized sections
        out: List[str] = []
        for sec in sections:
            if self.length_function(sec) <= self.chunk_size:
                out.append(sec)
            else:
                out.extend(self._fallback.chunk(sec))
        return out or self._fallback.chunk(text)


class SlackChunker(BaseChunker):
    """Group consecutive messages; never split mid-message."""
    strategy_name = "slack"

    # "Name: message" or "user: message"
    MSG_RE = re.compile(
        r"(?m)^([A-Za-z0-9_./\-]+(?:\s+[A-Za-z0-9_./\-]+)*):\s*(.*?)(?=^\S[^:\n]*:|\Z)",
        re.DOTALL,
    )

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: Callable[[str], int] = token_length,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        messages = []
        for m in self.MSG_RE.finditer(text):
            messages.append(f"{m.group(1)}: {m.group(2).strip()}")

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
    """
    Jira / Linear style: keep description together, group comments.
    Also works reasonably for GitHub PR body + comments.
    """
    strategy_name = "ticket"

    def __init__(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 60,
        length_function: Callable[[str], int] = token_length,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # Split into description / comments / acceptance_criteria blocks if present
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
    """Prefer whole short emails; split long threads on reply boundaries."""
    strategy_name = "gmail"

    REPLY_BOUNDARY = re.compile(
        r"(?m)^(?:"
        r"On .+ wrote:|"
        r"From:\s*.+$|"
        r"-{2,}\s*Original Message\s*-{2,}"
        r")",
    )

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        length_function: Callable[[str], int] = token_length,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self._fallback = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
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

        # Re-split any still-oversized pieces
        final: List[str] = []
        for c in out:
            if self.length_function(c) > self.chunk_size * 1.4:
                final.extend(self._fallback.chunk(c))
            else:
                final.append(c)
        return final


class SourceAwareChunker(BaseChunker):
    """
    Routes to the best specialist based on source_type.
    Falls back to RecursiveChunker for unknown sources.
    """
    strategy_name = "source_aware"

    SOURCE_MAP = {
        "fireflies": "fireflies",
        "confluence": "confluence",
        "slack": "slack",
        "jira": "ticket",
        "linear": "ticket",
        "github": "ticket",
        "gmail": "gmail",
        "google_drive": "confluence",  # often structured / notes
        "hubspot": "recursive",        # short CRM notes – recursive is fine
    }

    def __init__(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 80,
        length_function: Callable[[str], int] = token_length,
        source_type: Optional[str] = None,
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.source_type = (source_type or "").lower().strip()
        self._kwargs = kwargs

    def chunk(self, text: str) -> List[str]:
        strategy = self.SOURCE_MAP.get(self.source_type, "recursive")
        chunker = ChunkerFactory.get_chunker(
            strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            **self._kwargs,
        )
        return chunker.chunk(text)


class ChunkerFactory:
    _strategies = {
        "recursive": RecursiveChunker,
        "fireflies": FirefliesChunker,
        "confluence": ConfluenceChunker,
        "slack": SlackChunker,
        "ticket": TicketChunker,
        "gmail": GmailChunker,
        "auto": SourceAwareChunker,  # alias
    }

    @classmethod
    def get_chunker(cls, strategy: str, **kwargs) -> BaseChunker:
        if strategy not in cls._strategies:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Available: {list(cls._strategies)}"
            )
        return cls._strategies[strategy](**kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())