"""
The user's local ticker watchlist -- the finance agent's HITL surface.

This agent deliberately has no brokerage connection and no trade tools, so
the watchlist is its only write action: a JSON file on the user's disk. It
still goes through ``require_confirmation`` because the point of the gate is
the habit, not the blast radius -- an agent that can silently edit *any*
persistent state the user relies on should pause for a human, and keeping the
one write path gated means adding a bigger one later starts from the right
default.

Reads are ungated, same split as the GitHub agent.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

WATCHLIST_PATH = Path(__file__).resolve().parents[3] / "data" / "watchlist.json"

# Uppercase ticker, optionally exchange-suffixed ("7203.T", "BRK-B").
_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,9}([.-][A-Z0-9]{1,4})?$")


def _load() -> dict[str, Any]:
    if not WATCHLIST_PATH.exists():
        return {"tickers": {}}
    try:
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        logger.exception("Watchlist file unreadable; treating as empty")
        return {"tickers": {}}


def _save(data: dict[str, Any]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _normalize(symbol: str) -> str | None:
    symbol = (symbol or "").strip().upper()
    return symbol if _TICKER.match(symbol) else None


def view_watchlist() -> dict[str, Any]:
    """Show the user's ticker watchlist.

    Returns:
        A dict with the list of watched tickers, each with the note and date
        it was added with. An empty list means nothing is watched yet.
    """
    tickers = _load()["tickers"]
    return {
        "status": "ok",
        "count": len(tickers),
        "tickers": [
            {"symbol": symbol, **info} for symbol, info in sorted(tickers.items())
        ],
    }


def add_to_watchlist(symbol: str, note: str = "") -> dict[str, Any]:
    """Add a ticker to the user's watchlist. Requires the user's approval.

    Propose this only when the user asks to track something. State the symbol
    and note you intend to save before calling.

    Args:
        symbol: The ticker symbol, e.g. 'NVDA' or 'BRK-B'.
        note: Optional one-line reason the user is watching it.

    Returns:
        The updated watchlist on success, or an error for invalid symbols.
    """
    normalized = _normalize(symbol)
    if not normalized:
        return {
            "status": "error",
            "error": f"'{symbol}' does not look like a ticker symbol.",
        }

    data = _load()
    already = normalized in data["tickers"]
    data["tickers"][normalized] = {
        "note": (note or "").strip()[:200],
        "added": date.today().isoformat(),
    }
    _save(data)
    return {
        "status": "ok",
        "message": (
            f"Updated note for {normalized}." if already else f"Added {normalized}."
        ),
        "count": len(data["tickers"]),
    }


def remove_from_watchlist(symbol: str) -> dict[str, Any]:
    """Remove a ticker from the user's watchlist. Requires the user's approval.

    Args:
        symbol: The ticker symbol to remove.

    Returns:
        Confirmation of removal, or an error if the symbol is not watched.
    """
    normalized = _normalize(symbol)
    if not normalized:
        return {
            "status": "error",
            "error": f"'{symbol}' does not look like a ticker symbol.",
        }

    data = _load()
    if normalized not in data["tickers"]:
        return {
            "status": "error",
            "error": f"{normalized} is not on the watchlist.",
        }

    del data["tickers"][normalized]
    _save(data)
    return {"status": "ok", "message": f"Removed {normalized}.", "count": len(data["tickers"])}


def _confirm_writes(**kwargs: Any) -> bool:
    """Every watchlist mutation gets a human decision; reads never do."""
    return True


view_watchlist_tool = FunctionTool(view_watchlist)
add_to_watchlist_tool = FunctionTool(add_to_watchlist, require_confirmation=_confirm_writes)
remove_from_watchlist_tool = FunctionTool(
    remove_from_watchlist, require_confirmation=_confirm_writes
)

WATCHLIST_TOOLS = [view_watchlist_tool, add_to_watchlist_tool, remove_from_watchlist_tool]
