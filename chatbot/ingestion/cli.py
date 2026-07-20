"""
Corpus ingestion CLI.

    python -m ingestion.cli ingest ./corpus
    python -m ingestion.cli ingest ./corpus_finance --collection finance
    python -m ingestion.cli stats --collection finance
    python -m ingestion.cli inspect --source spec.pdf --limit 3
    python -m ingestion.cli reset --collection finance

Every command targets one collection (default "corpus", the GitHub agent's).
The finance agent reads from "finance" -- ingesting into the wrong collection
does not error, the documents are just invisible to the agent you meant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .chunking import chunk_document
from .embed import embed_documents
from .loaders import SUPPORTED_SUFFIXES, load_directory, load_path
from .store import get_store

load_dotenv()


def cmd_ingest(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 1

    if target.is_dir():
        documents = load_directory(target)
    else:
        if target.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"error: unsupported file type {target.suffix}", file=sys.stderr)
            return 1
        documents = [load_path(target, root=target.parent)]

    if not documents:
        print(f"No supported files found in {target}.")
        print(f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        return 0

    store = get_store(args.collection)
    total_chunks = 0

    for document in documents:
        chunks = chunk_document(document)
        if not chunks:
            print(f"! {document.source}: no extractable text, skipped")
            continue

        print(f"* {document.source}: {len(document.sections)} sections -> {len(chunks)} chunks")

        # Clear first: a file that shrank would otherwise keep stale tail chunks.
        store.delete_source(document.source)
        embeddings = embed_documents([c.embed_text for c in chunks], progress=args.verbose)
        total_chunks += store.upsert_chunks(chunks, embeddings)

    print(f"\nIngested {total_chunks} chunks from {len(documents)} document(s).")
    print(f"Index: {store.path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    stats = get_store(args.collection).stats()
    print(f"Index:     {stats['path']}")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks:    {stats['chunks']}")
    for source in stats["sources"]:
        print(f"  - {source}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """
    Print stored chunks verbatim.

    Worth running after every ingest of a new file type: mid-sentence splits
    and empty chunks are obvious to a human eye and invisible in the stats.
    """
    store = get_store(args.collection)
    where = {"source": args.source} if args.source else None
    got = store._collection.get(  # noqa: SLF001 - debug helper
        where=where, include=["documents", "metadatas"], limit=args.limit
    )
    if not got["ids"]:
        print("No chunks matched.")
        return 0

    for chunk_id, document, metadata in zip(
        got["ids"], got["documents"], got["metadatas"]
    ):
        page = metadata.get("page", -1)
        page_label = "" if page == -1 else f" p.{page}"
        heading = metadata.get("heading") or "(no heading)"
        print("-" * 72)
        print(f"{metadata.get('source')}{page_label} | {heading} | {chunk_id[:12]}")
        print(f"tokens={metadata.get('token_count')}")
        print(document)
    print("-" * 72)
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        confirm = (
            input(f"Delete the entire {args.collection!r} collection? [y/N] ")
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Aborted.")
            return 0
    get_store(args.collection).reset()
    print(f"Collection {args.collection!r} cleared.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingestion.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and index a file or directory")
    p_ingest.add_argument("path", help="file or directory to ingest")
    p_ingest.add_argument("-v", "--verbose", action="store_true", help="show embedding progress")
    p_ingest.set_defaults(func=cmd_ingest)

    p_stats = sub.add_parser("stats", help="show what is currently indexed")
    p_stats.set_defaults(func=cmd_stats)

    p_inspect = sub.add_parser("inspect", help="print stored chunks verbatim")
    p_inspect.add_argument("--source", help="filter to one source file")
    p_inspect.add_argument("--limit", type=int, default=5)
    p_inspect.set_defaults(func=cmd_inspect)

    p_reset = sub.add_parser("reset", help="delete the entire index")
    p_reset.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    p_reset.set_defaults(func=cmd_reset)

    for sub_parser in (p_ingest, p_stats, p_inspect, p_reset):
        sub_parser.add_argument(
            "--collection",
            default="corpus",
            help='target collection (default "corpus"; the finance agent reads "finance")',
        )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
