"""
Corpus retrieval -- the second grounding source alongside live GitHub data.

The tool returns provenance (source, page, heading) *inside the payload*
rather than concatenating chunks into one blob of prose. That is what makes
the citation rule in the prompt enforceable: the model cannot cite a page
number it was never given, and a reader can trace any claim back to a file.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ingestion.embed import embed_query
from ingestion.store import get_store

logger = logging.getLogger(__name__)

MAX_RESULTS = 10

# Below this cosine similarity a "match" is noise -- the nearest neighbour of
# an unrelated query is still *a* neighbour. Filtering here is what lets the
# agent say "the corpus does not cover this" instead of confidently citing the
# least-irrelevant chunk it could find.
#
# Calibrated empirically, not guessed. Gemini embeddings have a high floor:
# completely unrelated queries still score ~0.45-0.57 against this corpus,
# while genuine matches score 0.62-0.84. An intuitive-looking threshold like
# 0.35 never fires and silently defeats the whole no-match path.
#
# Measured over 8 relevant and 10 unrelated queries:
#     relevant   min 0.625
#     unrelated  max 0.570
# 0.60 sits in that gap. The margin is narrow, so re-run the calibration if
# you change the embedding model, dimension count, or corpus domain:
# embed a set of known-relevant and known-irrelevant queries, take each
# one's top-1 similarity, and pick the midpoint between the two clusters.
MIN_SIMILARITY = float(os.getenv("CORPUS_MIN_SIMILARITY", "0.60"))


def search_corpus(query: str, k: int = 5) -> dict[str, Any]:
    """Search the ingested document corpus for passages relevant to a query.

    Use this for questions about specifications, guides, PDFs, and any other
    ingested documentation. The corpus is the only way to see these documents.
    Do not use it for live GitHub state such as issues, pull requests, or
    commits -- use the GitHub tools for those.

    Args:
        query: A natural-language description of the information you need.
            Full questions retrieve better than bare keywords.
        k: How many passages to return, from 1 to 10. Defaults to 5.

    Returns:
        A dict with a 'results' list. Each result carries the passage text
        plus the 'source' filename, 'page' number, and 'heading' needed to
        cite it. An empty list means the corpus does not cover the query, and
        you should say so rather than answering from memory.
    """
    query = (query or "").strip()
    if not query:
        return {"status": "error", "error": "query must not be empty", "results": []}

    k = max(1, min(int(k), MAX_RESULTS))

    try:
        store = get_store()
        if store.stats()["chunks"] == 0:
            return {
                "status": "empty_corpus",
                "results": [],
                "message": (
                    "No documents have been ingested yet. Tell the user the "
                    "corpus is empty and that documents can be added with "
                    "`python -m ingestion.cli ingest ./corpus`."
                ),
            }

        hits = store.query(embed_query(query), k=k)
    except Exception as exc:  # noqa: BLE001 - surface failure, never fall back to memory
        logger.exception("Corpus search failed")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "results": [],
            "message": "Corpus search failed. Report this instead of answering from memory.",
        }

    relevant = [h for h in hits if h["similarity"] >= MIN_SIMILARITY]

    if not relevant:
        return {
            "status": "no_match",
            "results": [],
            "message": (
                "No passage in the corpus is relevant to this query. Say the "
                "corpus does not cover it rather than guessing."
            ),
        }

    return {
        "status": "ok",
        "result_count": len(relevant),
        "results": [
            {
                "text": hit["text"],
                "source": hit["source"],
                "page": hit["page"],
                "heading": hit["heading"] or None,
                "similarity": hit["similarity"],
                "citation": _citation(hit),
            }
            for hit in relevant
        ],
    }


def _citation(hit: dict[str, Any]) -> str:
    """Pre-rendered citation string, so the model has no room to garble it."""
    parts = [hit["source"]]
    if hit.get("page"):
        parts.append(f"p.{hit['page']}")
    if hit.get("heading"):
        parts.append(f'"{hit["heading"]}"')
    return " ".join(parts)


def corpus_stats() -> dict[str, Any]:
    """Report what is currently in the document corpus.

    Use this when the user asks what documents you can see, or when a corpus
    search unexpectedly returns nothing and you want to check whether anything
    has been ingested at all.

    Returns:
        A dict with the number of indexed documents and chunks, and the list
        of source filenames.
    """
    try:
        stats = get_store().stats()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Corpus stats failed")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "ok",
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "sources": stats["sources"],
    }
