"""
The Chroma-backed corpus store.

This module is the single owner of the collection name, distance metric, and
on-disk location. Both the ingestion CLI and the runtime ``search_corpus``
tool import ``CorpusStore`` from here, so the way chunks are written can never
drift from the way they are read -- a mismatch that would degrade retrieval
quietly instead of failing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings

from .chunking import Chunk

DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".chroma"
COLLECTION_NAME = "corpus"


class CorpusStore:
    def __init__(
        self,
        path: str | Path | None = None,
        collection: str = COLLECTION_NAME,
    ):
        """
        One store instance per collection. Collections isolate agents from each
        other: the GitHub agent's Zarnex spec must never surface in a finance
        query and vice versa -- cross-domain hits would not error, they would
        just quietly degrade retrieval for both agents.
        """
        self.path = Path(path or os.getenv("CHROMA_PATH") or DEFAULT_PATH)
        self.collection_name = collection
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # Embeddings are normalized in embed.py, so cosine is the right
            # metric and equivalent to inner product here.
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self, chunks: Iterable[Chunk], embeddings: list[list[float]]
    ) -> int:
        """
        Write chunks by their stable ``chunk_id``.

        Upsert rather than add: re-ingesting an edited file overwrites its own
        chunks instead of piling up duplicates that all match the same query.
        """
        chunks = list(chunks)
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        """
        Drop every chunk from one file.

        Needed because a file that *shrinks* on re-ingest leaves its old
        tail chunks behind -- their ids are never revisited by the upsert.
        """
        self._collection.delete(where={"source": source})

    def query(self, embedding: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Nearest chunks, each carrying the provenance needed to cite it."""
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        hits = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            page = metadata.get("page", -1)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "source": metadata.get("source", "unknown"),
                    "title": metadata.get("title", ""),
                    "heading": metadata.get("heading", ""),
                    "page": None if page == -1 else page,
                    # Chroma returns cosine distance; similarity reads better
                    # in tool output the model has to reason about.
                    "similarity": round(1.0 - float(distance), 4),
                }
            )
        return hits

    def stats(self) -> dict[str, Any]:
        count = self._collection.count()
        sources: set[str] = set()
        if count:
            got = self._collection.get(include=["metadatas"], limit=count)
            sources = {m.get("source", "unknown") for m in got["metadatas"]}
        return {
            "chunks": count,
            "documents": len(sources),
            "sources": sorted(sources),
            "path": str(self.path),
        }

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_stores: dict[str, CorpusStore] = {}


def get_store(collection: str = COLLECTION_NAME) -> CorpusStore:
    """Process-wide cache, one store per collection; Chroma clients are not
    cheap to rebuild and all collections share one on-disk client path."""
    if collection not in _stores:
        _stores[collection] = CorpusStore(collection=collection)
    return _stores[collection]
