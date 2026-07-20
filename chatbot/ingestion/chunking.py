"""
Structure-aware chunking.

Retrieval quality is bounded by this file. Three rules drive the design:

1. **Never split at a blind offset.** Sections arrive from the loaders already
   carrying a heading path and page; packing happens *within* a heading group,
   so a chunk never straddles two unrelated topics.
2. **A chunk must stand alone.** The heading breadcrumb is prepended to the
   embedded text, so a chunk pulled out of context still says what it is
   about. The raw text is kept separate for display and citation.
3. **Re-ingesting must upsert, not duplicate.** ``chunk_id`` is a hash of
   source plus ordinal, so a changed file overwrites its own chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .loaders import Document, Section

TARGET_TOKENS = 512
OVERLAP_TOKENS = 64

TokenCounter = Callable[[str], int]

_WORD = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """
    Local token estimate, used for packing decisions.

    Packing asks "does this fit?" once per candidate section, so a real
    ``count_tokens`` call here would mean thousands of network round trips to
    ingest a single large PDF. This approximation runs offline and is
    consistently within ~10% of Gemini's tokenizer for English prose, which is
    enough to size a chunk. Pass a different ``counter`` to
    ``chunk_document`` if you need exact counts.
    """
    pieces = _WORD.findall(text)
    # Long words split into multiple tokens; short words and punctuation are one.
    return sum(max(1, (len(p) + 4) // 5) for p in pieces)


# Sentence boundary: terminal punctuation followed by whitespace and a capital
# or digit. Deliberately conservative -- a missed boundary just yields a
# slightly larger chunk, while a false one cuts a sentence in half.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    heading: str
    page: int | None
    text: str
    token_count: int

    @property
    def embed_text(self) -> str:
        """What actually gets embedded: breadcrumb + body."""
        prefix_parts = [self.title]
        if self.heading:
            prefix_parts.append(self.heading)
        return f"{' > '.join(prefix_parts)}\n\n{self.text}"

    def to_metadata(self) -> dict:
        """Chroma metadata values must be scalars, so ``page`` gets a sentinel."""
        return {
            "source": self.source,
            "title": self.title,
            "heading": self.heading,
            "page": self.page if self.page is not None else -1,
            "token_count": self.token_count,
        }


def make_chunk_id(source: str, index: int) -> str:
    """Stable across runs: same file + same ordinal -> same id."""
    digest = hashlib.sha256(f"{source}::{index}".encode("utf-8")).hexdigest()
    return digest[:32]


def _hard_split(text: str, target: int, counter: TokenCounter) -> list[str]:
    """
    Last resort for a single sentence that exceeds the whole budget (tables,
    minified data, PDFs whose sentence punctuation didn't survive extraction).
    Splits on word boundaries so no word is torn in half.
    """
    words = text.split()
    out: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if counter(" ".join(current)) >= target:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out or [text]


def _units_for(section: Section, target: int, counter: TokenCounter) -> list[str]:
    """Break a section into pieces that each fit the budget."""
    if counter(section.text) <= target:
        return [section.text]

    units: list[str] = []
    for sentence in split_sentences(section.text):
        if counter(sentence) > target:
            units.extend(_hard_split(sentence, target, counter))
        else:
            units.append(sentence)
    return units


def _overlap_tail(units: list[str], overlap: int, counter: TokenCounter) -> list[str]:
    """
    Trailing units of the finished chunk, to prepend to the next one.

    Overlap exists so a fact stated across a chunk boundary is retrievable
    from either side. Taken whole-unit (whole sentences) rather than by token
    count, so the carried text is always readable.
    """
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = counter(unit)
        if total + unit_tokens > overlap and tail:
            break
        tail.insert(0, unit)
        total += unit_tokens
    # Carrying the entire previous chunk forward would duplicate it wholesale.
    if len(tail) == len(units):
        tail = tail[1:]
    return tail


def _group_key(section: Section) -> tuple:
    """Sections pack together only when they share a heading."""
    return (section.heading_path,)


def chunk_document(
    document: Document,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    counter: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    """Pack a document's sections into overlapping, token-budgeted chunks."""
    chunks: list[Chunk] = []
    index = 0

    for key, group in _consecutive_groups(document.sections):
        heading = " > ".join(key[0])
        pending: list[str] = []
        pending_page: int | None = None

        def flush(carry_overlap: bool) -> None:
            nonlocal pending, index, pending_page
            if not pending:
                return
            text = " ".join(pending).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(document.source, index),
                        source=document.source,
                        title=document.title,
                        heading=heading,
                        page=pending_page,
                        text=text,
                        token_count=counter(text),
                    )
                )
                index += 1
            pending = (
                _overlap_tail(pending, overlap_tokens, counter) if carry_overlap else []
            )

        for section in group:
            if pending_page is None:
                pending_page = section.page
            for unit in _units_for(section, target_tokens, counter):
                candidate = " ".join(pending + [unit])
                if pending and counter(candidate) > target_tokens:
                    flush(carry_overlap=True)
                    if not pending:
                        pending_page = section.page
                pending.append(unit)

        flush(carry_overlap=False)
        pending_page = None

    return chunks


def _consecutive_groups(sections: Iterable[Section]):
    """Group consecutive sections sharing a heading, preserving order."""
    group: list[Section] = []
    key = None
    for section in sections:
        section_key = _group_key(section)
        if key is None:
            key = section_key
        elif section_key != key:
            yield key, group
            group, key = [], section_key
        group.append(section)
    if group:
        yield key, group


def chunk_documents(
    documents: Iterable[Document], **kwargs
) -> list[Chunk]:
    out: list[Chunk] = []
    for document in documents:
        out.extend(chunk_document(document, **kwargs))
    return out
