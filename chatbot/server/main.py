"""
FastAPI entry point.

    uvicorn server.main:app --reload    # run from the project root

Built on ADK's ``get_fast_api_app`` rather than hand-rolling the protocol. That
factory already provides ``/run_sse``, session CRUD, artifact upload, and --
critically -- the tool-confirmation round trip that human-in-the-loop depends
on. Re-implementing it would mean re-implementing the approval resume path,
which is the easiest part of this system to get subtly wrong.

Added on top: corpus ingestion routes and the static frontend.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from google.adk.cli.fast_api import get_fast_api_app  # noqa: E402

from ingestion.chunking import chunk_document  # noqa: E402
from ingestion.embed import embed_documents  # noqa: E402
from ingestion.loaders import SUPPORTED_SUFFIXES, load_path  # noqa: E402
from ingestion.store import get_store  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AGENTS_DIR = PROJECT_ROOT / "agents"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Which Chroma collection each agent reads. The frontend uses this to route
# uploads to the active agent's corpus; /api/ingest validates against it so a
# typo cannot silently create an orphan collection no agent ever reads.
AGENTS = {
    "github_agent": {
        "collection": "corpus",
        "label": "GitHub assistant",
    },
    "finance_agent": {
        "collection": "finance",
        "label": "Finance research",
    },
}


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Close every agent's MCP sessions on shutdown.

    Toolsets hold long-lived connections (HTTP to remote MCP servers, pipes to
    stdio ones). Without this the process leaks them on every reload, which
    under `--reload` means a growing pile of half-open connections. Each agent
    module exports `mcp_toolsets` for exactly this hook.
    """
    yield
    sys.path.insert(0, str(AGENTS_DIR))
    for agent_name in AGENTS:
        try:
            module = __import__(f"{agent_name}.agent", fromlist=["mcp_toolsets"])
            for toolset in getattr(module, "mcp_toolsets", []):
                await toolset.close()
            logger.info("Closed MCP toolsets for %s.", agent_name)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("Failed to close MCP toolsets for %s", agent_name)


app: FastAPI = get_fast_api_app(
    agents_dir=str(AGENTS_DIR),
    web=False,  # we serve our own frontend
    session_service_uri=os.getenv("SESSION_DB_URI", "sqlite:///./sessions.db"),
    allow_origins=["*"] if os.getenv("ALLOW_CORS") else None,
    lifespan=lifespan,
)


# --- Corpus routes -------------------------------------------------------


def _validate_collection(collection: str) -> str:
    known = {cfg["collection"] for cfg in AGENTS.values()}
    if collection not in known:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown collection '{collection}'. Known: {', '.join(sorted(known))}",
        )
    return collection


@app.get("/api/corpus/stats")
async def corpus_stats(collection: str = "corpus"):
    """What is currently indexed in one collection."""
    return get_store(_validate_collection(collection)).stats()


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), collection: str = Form("corpus")):
    """
    Upload one document straight into the corpus.

    Same pipeline as `python -m ingestion.cli ingest` -- chunk, embed, upsert --
    so a file added through the UI is indexed identically to a batch-ingested
    one. Divergence between the two paths would make retrieval quality depend
    on how a document happened to arrive.
    """
    collection = _validate_collection(collection)
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_bytes(payload)

        try:
            document = load_path(path, root=Path(tmpdir))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingest parse failed for %s", filename)
            raise HTTPException(
                status_code=400, detail=f"Could not parse '{filename}': {exc}"
            ) from exc

    chunks = chunk_document(document)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{filename}' produced no text. If it is a PDF it is most "
                "likely a scanned image, which needs OCR."
            ),
        )

    try:
        store = get_store(collection)
        store.delete_source(document.source)
        embeddings = embed_documents([c.embed_text for c in chunks])
        indexed = store.upsert_chunks(chunks, embeddings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest indexing failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    logger.info("Ingested %s -> %d chunks (collection=%s)", filename, indexed, collection)
    return {
        "status": "ok",
        "filename": filename,
        "title": document.title,
        "chunks_indexed": indexed,
        "collection": collection,
    }


@app.get("/api/config")
async def config():
    """Agent roster for the frontend: names, labels, and corpus collections."""
    return {
        "agents": [
            {"name": name, **cfg} for name, cfg in AGENTS.items()
        ],
        "default_agent": "github_agent",
    }


# --- Frontend ------------------------------------------------------------
# Mounted at /ui rather than / so it can never shadow an ADK API route.

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


@app.get("/")
async def index():
    return RedirectResponse(url="/ui/")
