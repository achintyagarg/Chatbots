"""
Load source files into a normalized ``Document`` shape.

Every loader's job is to recover *structure* -- headings and page numbers --
not just text. Chunking depends on that structure to avoid splitting a
document at an arbitrary character offset, and retrieval depends on it to
cite where an answer came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class Section:
    """A contiguous run of text sharing one heading path and page."""

    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None

    @property
    def heading(self) -> str:
        """Human-readable breadcrumb, e.g. ``Setup > Installing``."""
        return " > ".join(self.heading_path)


@dataclass
class Document:
    source: str
    title: str
    sections: list[Section] = field(default_factory=list)


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, dropping whitespace-only fragments."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def load_markdown(path: Path, source: str) -> Document:
    """Split on ATX headings, tracking the heading stack down the document."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []
    title = path.stem

    def flush() -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if body:
            for para in _split_paragraphs(body):
                sections.append(Section(text=para, heading_path=tuple(stack)))
        buffer.clear()

    for line in raw.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1 and not sections and not stack:
                title = heading
            # Pop to the parent level, then push this heading.
            del stack[level - 1 :]
            stack.append(heading)
        else:
            buffer.append(line)

    flush()
    return Document(source=source, title=title, sections=sections)


def load_text(path: Path, source: str) -> Document:
    """Plain text has no structure to recover, so paragraphs are the unit."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections = [Section(text=p) for p in _split_paragraphs(raw)]
    return Document(source=source, title=path.stem, sections=sections)


# A heading in extracted PDF text: short, not sentence-punctuated, and either
# numbered ("3.1 Results") or Title/UPPER case. Extraction loses font size, so
# this is a heuristic -- when it misfires the text still lands in a section,
# just with a weaker breadcrumb.
_PDF_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^.!?]{2,79}$"
)


def _looks_like_heading(line: str, next_line: str | None = None) -> bool:
    """
    ``next_line`` disambiguates a heading from a wrapped line of body text.

    Without it, "Tier 2 support responds within 90 minutes during business"
    reads as a heading: short, capitalized, no terminal punctuation. What gives
    it away is the line after it starting lowercase ("hours and within..."),
    which only happens mid-sentence. Real headings are followed by a blank line
    or by a new capitalized sentence.
    """
    line = line.strip()
    if not (3 <= len(line) <= 80):
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    if len(line.split()) > 12:
        return False
    if next_line is not None and re.match(r"^\s*[a-z]", next_line):
        return False
    return bool(_PDF_HEADING.match(line))


def load_pdf(path: Path, source: str) -> Document:
    """Extract per page, recovering headings heuristically within each page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: list[Section] = []
    stack: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue  # scanned/image-only page; OCR is out of scope

        buffer: list[str] = []

        def flush(page_number: int = page_number) -> None:
            if not buffer:
                return
            body = "\n".join(buffer).strip()
            if body:
                for para in _split_paragraphs(body):
                    sections.append(
                        Section(
                            text=para,
                            heading_path=tuple(stack),
                            page=page_number,
                        )
                    )
            buffer.clear()

        lines = text.splitlines()
        for index, line in enumerate(lines):
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if _looks_like_heading(line, next_line):
                flush()
                stack.clear()  # flat structure; extraction can't give real depth
                stack.append(line.strip())
            else:
                buffer.append(line)
        flush()

    title = path.stem
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or title

    return Document(source=source, title=title, sections=sections)


def load_path(path: Path, root: Path | None = None) -> Document:
    """Dispatch on file extension. ``root`` makes ``source`` a relative path."""
    path = Path(path)
    source = str(path.relative_to(root)) if root else path.name
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path, source)
    if suffix in {".md", ".markdown"}:
        return load_markdown(path, source)
    if suffix == ".txt":
        return load_text(path, source)
    raise ValueError(f"Unsupported file type {suffix!r}: {path}")


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file under ``directory``, recursively."""
    directory = Path(directory)
    documents = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append(load_path(path, root=directory))
    return documents
