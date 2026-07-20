"""
Corpus retrieval for the GitHub agent -- the second grounding source
alongside live GitHub data.

The implementation lives in ingestion/corpus_tools.py, shared with the
finance agent; this module just binds it to the GitHub agent's collection.
"""

from __future__ import annotations

from ingestion.corpus_tools import make_corpus_tools

search_corpus, corpus_stats = make_corpus_tools(
    collection="corpus",
    corpus_description=(
        "specifications, guides, PDFs, and other ingested project "
        "documentation. Do not use it for live GitHub state such as issues, "
        "pull requests, or commits -- use the GitHub tools for those"
    ),
)
