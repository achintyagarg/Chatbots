"""
Gemini embeddings for corpus chunks and queries.

Two details here are easy to get wrong and both silently degrade retrieval
rather than raising:

* **task_type must differ between indexing and querying.** Documents are
  embedded with ``RETRIEVAL_DOCUMENT`` and queries with ``RETRIEVAL_QUERY``.
  Using one type for both costs real accuracy without any visible error.
* **Reduced-dimension vectors are not unit length.** ``gemini-embedding-001``
  only normalizes its native 3072-dim output; asking for 768 returns vectors
  with norm ~0.59. Cosine similarity needs them normalized, so we do it here.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Sequence

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"

# 768 of the model's 3072 dimensions. Keeps the local index small and queries
# fast; the accuracy cost is minor at corpus sizes this store is built for.
EMBED_DIMS = 768

# The API accepts larger batches, but small ones fail and retry more cheaply.
BATCH_SIZE = 32

MAX_RETRIES = 5

# Substrings identifying errors that will never succeed on retry. Backing off
# five times on a missing API key wastes 31 seconds and buries the real cause
# under a wall of retry warnings.
FATAL_ERROR_MARKERS = (
    "no api key",
    "api key not valid",
    "api_key_invalid",
    "permission denied",
    "unauthenticated",
    "invalid argument",
)

_client: genai.Client | None = None


class EmbeddingError(RuntimeError):
    """Embedding could not be completed."""


def get_client() -> genai.Client:
    """Lazily built so importing this module never requires credentials."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


def _embed_batch(texts: Sequence[str], task_type: str) -> list[list[float]]:
    """One API call, with exponential backoff on transient failures."""
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBED_DIMS,
    )
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().models.embed_content(
                model=EMBED_MODEL,
                contents=list(texts),
                config=config,
            )
            return [_normalize(e.values) for e in response.embeddings]
        except Exception as exc:  # noqa: BLE001 - retry transport/quota errors only
            last_error = exc
            message = str(exc).lower()
            if any(marker in message for marker in FATAL_ERROR_MARKERS):
                raise EmbeddingError(
                    f"Embedding failed and will not be retried: {exc}"
                ) from exc

            delay = 2**attempt
            logger.warning(
                "Embedding batch failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1,
                MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)

    raise EmbeddingError(
        f"Embedding failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def embed_documents(texts: Sequence[str], progress: bool = False) -> list[list[float]]:
    """Embed corpus chunks for indexing."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(_embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        if progress:
            print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a user query for search."""
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]
