"""Token counting for memory nodes. Uses tiktoken directly (NOT
controllers.ChunkingStrategies) to avoid a controllers <-> stores.agents
import cycle."""

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text) -> int:
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return len(_ENC.encode(text, disallowed_special=()))


def count_message_tokens(messages) -> int:
    return sum(count_tokens(getattr(m, "content", "")) + 4 for m in messages)