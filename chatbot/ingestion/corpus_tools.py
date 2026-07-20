"""
Corpus retrieval tools, parametrized by collection.

Each agent gets its own Chroma collection ("corpus" for the GitHub agent,
"finance" for the finance agent) so retrieval never crosses domains: a
finance query must miss cleanly rather than surface the least-irrelevant
GitHub spec chunk, because cross-domain hits do not error -- they quietly
degrade both agents.

``make_corpus_tools(collection)`` returns [search_corpus, corpus_stats]
functions closed over that collection's store. The functions keep their
names and docstrings because ADK builds the tool declarations the model
sees from exactly those.

The tools return provenance (source, page, heading) *inside the payload*
rather than concatenating chunks into one blob of prose. That is what makes
the citation rule in the prompts enforceable: the model cannot cite a page
number it was never given, and a reader can trace any claim back to a file.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .embed import embed_query
from .store import COLLECTION_NAME, get_store

logger = logging.getLogger(__name__)

MAX_RESULTS = 10

# Below this cosine similarity a "match" is noise -- the nearest neighbour of
# an unrelated query is still *a* neighbour. Filtering here is what lets the
# agent say "the corpus does not cover this" instead of confidently citing the
# least-irrelevant chunk it could find.
#
# Calibrated empirically, not guessed. Gemini embeddings have a high floor:
# completely unrelated queries still score ~0.45-0.57, while genuine matches
# score 0.62-0.84. An intuitive-looking threshold like 0.35 never fires and
# silently defeats the whole no-match path.
#
# The 0.60 default was measured on the GitHub agent's corpus (8 relevant
# queries, min 0.625; 10 unrelated, max 0.570). The margin is narrow and does
# NOT automatically transfer to other domains or embedding models -- calibrate
# per collection: embed known-relevant and known-irrelevant query sets, take
# each one's top-1 similarity, pick the midpoint between the clusters, and set
# the collection's env var (e.g. CORPUS_MIN_SIMILARITY_FINANCE).
DEFAULT_MIN_SIMILARITY = 0.60


def _min_similarity_for(collection: str) -> float:
    """Per-collection threshold: CORPUS_MIN_SIMILARITY_<NAME>, then global."""
    specific = os.getenv(f"CORPUS_MIN_SIMILARITY_{collection.upper()}")
    if specific:
        return float(specific)
    return float(os.getenv("CORPUS_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY)))


def _citation(hit: dict[str, Any]) -> str:
    """Pre-rendered citation string, so the model has no room to garble it."""
    parts = [hit["source"]]
    if hit.get("page"):
        parts.append(f"p.{hit['page']}")
    if hit.get("heading"):
        parts.append(f'"{hit["heading"]}"')
    return " ".join(parts)


def make_corpus_tools(
    collection: str = COLLECTION_NAME,
    corpus_description: str = "specifications, guides, PDFs, and other ingested documentation",
) -> list[Callable[..., dict[str, Any]]]:
    """
    Build [search_corpus, corpus_stats] bound to one collection.

    ``corpus_description`` is spliced into the search tool's docstring so each
    agent's model sees an accurate description of what its corpus holds.
    """
    min_similarity = _min_similarity_for(collection)

    def search_corpus(query: str, k: int = 5) -> dict[str, Any]:
        """Search the ingested document corpus for passages relevant to a query.

        Use this for questions answered by the ingested documents. The corpus
        is the only way to see them. Do not use it for live data available
        from other tools.

        Args:
            query: A natural-language description of the information you need.
                Full questions retrieve better than bare keywords.
            k: How many passages to return, from 1 to 10. Defaults to 5.

        Returns:
            A dict with a 'results' list. Each result carries the passage text
            plus the 'source' filename, 'page' number, and 'heading' needed to
            cite it. An empty list means the corpus does not cover the query,
            and you should say so rather than answering from memory.
        """
        query = (query or "").strip()
        if not query:
            return {"status": "error", "error": "query must not be empty", "results": []}

        k = max(1, min(int(k), MAX_RESULTS))

        try:
            store = get_store(collection)
            if store.stats()["chunks"] == 0:
                return {
                    "status": "empty_corpus",
                    "results": [],
                    "message": (
                        "No documents have been ingested yet. Tell the user the "
                        "corpus is empty and that documents can be added via the "
                        "UI drop zone or `python -m ingestion.cli ingest <dir> "
                        f"--collection {collection}`."
                    ),
                }

            hits = store.query(embed_query(query), k=k)
        except Exception as exc:  # noqa: BLE001 - surface failure, never fall back to memory
            logger.exception("Corpus search failed (collection=%s)", collection)
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "results": [],
                "message": "Corpus search failed. Report this instead of answering from memory.",
            }

        relevant = [h for h in hits if h["similarity"] >= min_similarity]

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

    def corpus_stats() -> dict[str, Any]:
        """Report what is currently in the document corpus.

        Use this when the user asks what documents you can see, or when a
        corpus search unexpectedly returns nothing and you want to check
        whether anything has been ingested at all.

        Returns:
            A dict with the number of indexed documents and chunks, and the
            list of source filenames.
        """
        try:
            stats = get_store(collection).stats()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Corpus stats failed (collection=%s)", collection)
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        return {
            "status": "ok",
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "sources": stats["sources"],
        }

    # The model routes on the docstring, so name the actual corpus contents.
    search_corpus.__doc__ = search_corpus.__doc__.replace(
        "questions answered by the ingested documents",
        f"questions about {corpus_description}",
    )

    return [search_corpus, corpus_stats]
