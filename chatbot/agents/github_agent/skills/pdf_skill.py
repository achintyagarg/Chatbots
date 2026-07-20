"""
PDF skill -- the reference implementation of the skill contract.

Read this file as the template for adding a capability: define the functions,
declare ``TOOLS`` and ``INSTRUCTION``, and the registry does the rest.

Text extraction reuses ``ingestion.loaders.load_pdf`` rather than calling pypdf
here. That keeps one definition of what a PDF's structure is, so an uploaded
PDF and a batch-ingested one chunk identically -- otherwise the two paths drift
and retrieval quality quietly depends on how a document happened to arrive.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from google.adk.tools.tool_context import ToolContext

from ingestion.chunking import chunk_document
from ingestion.embed import embed_documents
from ingestion.loaders import load_pdf
from ingestion.store import get_store

logger = logging.getLogger(__name__)

MAX_PREVIEW_CHARS = 4000


async def _load_pdf_artifact(filename: str, tool_context: ToolContext):
    """Fetch an uploaded PDF and parse it into a Document, or return an error."""
    try:
        part = await tool_context.load_artifact(filename)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Artifact load failed")
        return None, {"status": "error", "error": f"Could not load '{filename}': {exc}"}

    if part is None or part.inline_data is None:
        try:
            available = await tool_context.list_artifacts()
        except Exception:  # noqa: BLE001
            available = []
        return None, {
            "status": "not_found",
            "error": f"No uploaded file named '{filename}'.",
            "available_files": available,
        }

    # pypdf needs a real path, and the artifact only exists as bytes.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / (Path(filename).name or "upload.pdf")
        path.write_bytes(part.inline_data.data)
        try:
            document = load_pdf(path, source=filename)
        except Exception as exc:  # noqa: BLE001
            logger.exception("PDF parse failed")
            return None, {
                "status": "error",
                "error": f"Could not parse '{filename}' as a PDF: {exc}",
            }

    if not document.sections:
        return None, {
            "status": "empty",
            "message": (
                f"'{filename}' has no extractable text. It is most likely a "
                "scanned image PDF, which needs OCR that this skill does not do. "
                "Tell the user that rather than guessing at the contents."
            ),
        }

    return document, None


async def parse_pdf(filename: str, tool_context: ToolContext) -> dict[str, Any]:
    """Extract the text and structure of an uploaded PDF file.

    Use this when the user uploads a PDF and wants to discuss, summarize, or
    ask questions about it. This reads the document once for the current
    conversation; it does not make it permanently searchable. Use
    `ingest_pdf` for that.

    Args:
        filename: Name of the uploaded PDF, as shown in the conversation.
        tool_context: Provided by the runtime.

    Returns:
        A dict with the document title, page count, section headings, and the
        beginning of the text. Long documents are truncated -- use
        `ingest_pdf` then `search_corpus` to query the whole thing.
    """
    document, error = await _load_pdf_artifact(filename, tool_context)
    if error:
        return error

    pages = {s.page for s in document.sections if s.page is not None}
    headings: list[str] = []
    for section in document.sections:
        if section.heading and section.heading not in headings:
            headings.append(section.heading)

    full_text = "\n\n".join(s.text for s in document.sections)
    truncated = len(full_text) > MAX_PREVIEW_CHARS

    return {
        "status": "ok",
        "filename": filename,
        "title": document.title,
        "page_count": max(pages) if pages else 0,
        "section_count": len(document.sections),
        "headings": headings[:40],
        "text": full_text[:MAX_PREVIEW_CHARS],
        "truncated": truncated,
        "note": (
            "Text was truncated. Call ingest_pdf to index the whole document, "
            "then use search_corpus to answer questions about the rest."
            if truncated
            else None
        ),
    }


async def ingest_pdf(filename: str, tool_context: ToolContext) -> dict[str, Any]:
    """Add an uploaded PDF to the searchable document corpus, permanently.

    Use this when the user wants a document to be available for future
    questions, in this conversation and later ones. After ingesting, answer
    questions about it with `search_corpus` rather than from this tool's
    output.

    Args:
        filename: Name of the uploaded PDF, as shown in the conversation.
        tool_context: Provided by the runtime.

    Returns:
        A dict reporting how many chunks were indexed.
    """
    document, error = await _load_pdf_artifact(filename, tool_context)
    if error:
        return error

    chunks = chunk_document(document)
    if not chunks:
        return {
            "status": "empty",
            "message": f"'{filename}' produced no indexable chunks.",
        }

    try:
        store = get_store()
        # Re-ingesting a revised upload must replace the old version, not
        # stack a second copy that competes with it at query time.
        store.delete_source(document.source)
        embeddings = embed_documents([c.embed_text for c in chunks])
        indexed = store.upsert_chunks(chunks, embeddings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF ingestion failed")
        return {"status": "error", "error": f"Indexing failed: {exc}"}

    return {
        "status": "ok",
        "filename": filename,
        "title": document.title,
        "chunks_indexed": indexed,
        "message": (
            f"'{document.title}' is now searchable. Use search_corpus to answer "
            "questions about it."
        ),
    }


TOOLS = [parse_pdf, ingest_pdf]

INSTRUCTION = """
## PDF files

When the user uploads a PDF:
- To discuss it right now, call `parse_pdf`.
- To make it permanently searchable, call `ingest_pdf`, then use
  `search_corpus` for questions about its contents.

If a PDF has no extractable text it is a scanned image and cannot be read
here. Say so; do not infer its contents from the filename.
""".strip()
