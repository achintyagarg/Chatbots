"""
Finance agent assembly. Exports ``app`` (see github_agent/agent.py for why an
App rather than a bare root_agent).

Grounding sources, in the order the prompt routes them:

1. yfinance MCP over stdio -- quotes, history, company info, financials, news.
   Free and effectively unlimited, so it is the bulk source. Community server
   (yfmcp), pinned; treat it as best-effort and let errors surface honestly.
2. Alpha Vantage official remote MCP -- deeper fundamentals and curated news
   sentiment. Included only when ALPHAVANTAGE_API_KEY is set, because the free
   tier is ~25 requests/day: scarce, so the prompt reserves it for what
   yfinance cannot answer.
3. The user's research corpus (Chroma collection "finance").
4. Quant skills -- local computation over full price history, returning
   summary statistics only. Bulk data never transits model context.

No brokerage, no trade tools, by design. The only write surface is the local
watchlist, and it is approval-gated.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps.app import App, EventsCompactionConfig  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.adk.tools.mcp_tool import McpToolset  # noqa: E402
from google.adk.tools.mcp_tool.mcp_session_manager import (  # noqa: E402
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
from google.genai import types  # noqa: E402
from mcp import StdioServerParameters  # noqa: E402

from ingestion.corpus_tools import make_corpus_tools  # noqa: E402
from plugins import ObservabilityPlugin, SafetyPlugin  # noqa: E402

from .prompts import STATIC_INSTRUCTION  # noqa: E402
from .skills import load_skills  # noqa: E402
from .tools.watchlist import WATCHLIST_TOOLS  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

MODEL = os.getenv("CHATBOT_MODEL", "gemini-flash-lite-latest")

# Pinned: community stdio server, verified to expose the yfinance_* tools
# below. An unpinned version could rename tools and silently empty the filter.
YFINANCE_MCP_SPEC = "yfmcp@0.12.2"

mcp_toolsets: list[McpToolset] = []


# --- Grounding source 1: yfinance MCP (stdio) ----------------------------

yfinance_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=[YFINANCE_MCP_SPEC],
        ),
        timeout=30.0,  # first call includes uvx resolving the package
    ),
    # The server also offers screeners and options chains; keep the surface
    # small until there is a prompt-routing story for them.
    tool_filter=[
        "yfinance_get_ticker_info",
        "yfinance_get_price_history",
        "yfinance_get_ticker_news",
        "yfinance_get_financials",
        "yfinance_search",
    ],
)
mcp_toolsets.append(yfinance_toolset)


# --- Grounding source 2: Alpha Vantage official MCP (optional) -----------

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

if ALPHAVANTAGE_API_KEY:
    alphavantage_toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=f"https://mcp.alphavantage.co/mcp?apikey={ALPHAVANTAGE_API_KEY}",
        ),
        # Names follow Alpha Vantage function naming. If AV renames them the
        # filter yields no tools and the agent still works via yfinance --
        # check the server's tool list if these stop appearing in traces.
        tool_filter=[
            "GLOBAL_QUOTE",
            "COMPANY_OVERVIEW",
            "NEWS_SENTIMENT",
            "TIME_SERIES_DAILY",
        ],
    )
    mcp_toolsets.append(alphavantage_toolset)
else:
    alphavantage_toolset = None
    logger.info(
        "ALPHAVANTAGE_API_KEY not set; finance agent runs on yfinance only. "
        "Get a free key at https://www.alphavantage.co/support/#api-key for "
        "fundamentals and news sentiment."
    )


# --- Grounding source 3: the user's research corpus ----------------------

search_corpus, corpus_stats = make_corpus_tools(
    collection="finance",
    corpus_description=(
        "the user's own quant research notes, investing references, and "
        "ingested filings. Do not use it for live market data -- use the "
        "market tools for that"
    ),
)
corpus_tools = [FunctionTool(search_corpus), FunctionTool(corpus_stats)]


# --- Skills: quant analytics, discovered not hardcoded -------------------

skill_tools, skill_instructions = load_skills()

STATIC_TEXT = "\n\n".join([STATIC_INSTRUCTION, *skill_instructions])


root_agent = LlmAgent(
    model=MODEL,
    name="finance_research_assistant",
    description=(
        "Financial research assistant: live market data over MCP, the user's "
        "own research corpus, and locally computed quant analytics. Provides "
        "data and analysis, never buy/sell recommendations."
    ),
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=STATIC_TEXT)]
    ),
    # MUST stay empty. A non-empty `instruction` alongside `static_instruction`
    # is appended as a user message *after* the user's question, and the model
    # answers that instead. See agents/github_agent/prompts.py.
    instruction="",
    tools=[
        yfinance_toolset,
        *([alphavantage_toolset] if alphavantage_toolset else []),
        *corpus_tools,
        *WATCHLIST_TOOLS,
        *skill_tools,
    ],
)


app = App(
    name="finance_agent",
    root_agent=root_agent,
    plugins=[SafetyPlugin(), ObservabilityPlugin()],
    # Same free-tier rationale as the GitHub agent: caching 429s on every turn
    # with a free Gemini key, so it is opt-in.
    context_cache_config=(
        ContextCacheConfig(min_tokens=4096, ttl_seconds=600, cache_intervals=5)
        if os.getenv("ENABLE_CONTEXT_CACHE", "").lower() in ("1", "true", "yes")
        else None
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=6, overlap_size=2
    ),
)
