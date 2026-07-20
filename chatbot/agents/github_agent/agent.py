"""
Agent assembly -- the one file that wires every capability together.

Exports ``app``, an ``App`` rather than a bare ``root_agent``. ADK's agent
loader checks for ``app`` first and falls back to ``root_agent``, so exporting
the App is what gives plugins and context caching to *both* `adk web` and the
custom FastAPI server from a single definition. Exporting ``root_agent``
instead would silently drop the safety plugin under one of the two entry
points, which is exactly the kind of gap that is hard to notice.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# The agent is imported by ADK with `agents/` on sys.path, so the project root
# (which holds `ingestion/` and `plugins/`) would otherwise be unreachable.
# Making this explicit keeps the agent loadable regardless of the working
# directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps.app import App, EventsCompactionConfig  # noqa: E402
from google.adk.tools import FunctionTool, load_memory  # noqa: E402
from google.adk.tools.mcp_tool import McpToolset  # noqa: E402
from google.adk.tools.mcp_tool.mcp_session_manager import (  # noqa: E402
    StreamableHTTPConnectionParams,
)
from google.genai import types  # noqa: E402

from plugins import ObservabilityPlugin, SafetyPlugin  # noqa: E402

from .prompts import STATIC_INSTRUCTION  # noqa: E402
from .skills import load_skills  # noqa: E402
from .tools.corpus import corpus_stats, search_corpus  # noqa: E402
from .tools.github_write import WRITE_TOOLS  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Default to flash-lite: gemini-3.5-flash is stronger but its free-tier daily
# request cap is low enough that a handful of conversations exhausts it, which
# makes the project look broken on first run. Override in .env.
MODEL = os.getenv("CHATBOT_MODEL", "gemini-flash-lite-latest")

GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not GITHUB_PAT:
    raise RuntimeError(
        "GITHUB_PERSONAL_ACCESS_TOKEN is not set. Copy .env.example to .env "
        "and add a GitHub personal access token."
    )


# --- Grounding source 1: live GitHub, over MCP ---------------------------
#
# Read-only by design. Every write goes through the approval-gated
# FunctionTools in tools/github_write.py instead, so no MCP tool added to this
# server later can become an ungated write path.
github_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {GITHUB_PAT}"},
    ),
    tool_filter=[
        "get_repository",
        "list_issues",
        "get_issue",
        "get_issue_comments",
        "search_issues",
        "list_commits",
        "get_commit",
        "list_pull_requests",
        "get_pull_request",
        "list_releases",
        "get_file_contents",
        "search_code",
    ],
)

# Server lifespan closes these on shutdown; every agent module exports the list.
mcp_toolsets = [github_toolset]


# --- Grounding source 2: the local document corpus -----------------------
corpus_tools = [FunctionTool(search_corpus), FunctionTool(corpus_stats)]


# --- Skills: discovered, not hardcoded -----------------------------------
skill_tools, skill_instructions = load_skills()


# Skill guidance is standing instruction, not per-turn context, so it belongs
# in the static (system) prompt. Putting it in `instruction` would append it as
# a user message after the user's question -- see the note in prompts.py.
STATIC_TEXT = "\n\n".join([STATIC_INSTRUCTION, *skill_instructions])


root_agent = LlmAgent(
    model=MODEL,
    name="github_grounded_assistant",
    description=(
        "Answers questions about GitHub repositories using live API data, and "
        "about ingested documents using corpus retrieval. Writes require "
        "human approval."
    ),
    # Split deliberately: the static half is a stable, cacheable prefix; the
    # dynamic half carries per-session state. See prompts.py.
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=STATIC_TEXT)]
    ),
    # MUST stay empty. A non-empty `instruction` alongside `static_instruction`
    # is appended as a user message *after* the user's question, and the model
    # answers that instead. See the note at the bottom of prompts.py.
    instruction="",
    tools=[
        github_toolset,
        *corpus_tools,
        *WRITE_TOOLS,
        *skill_tools,
        load_memory,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,  # factual retrieval work; creativity is a liability here
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ],
    ),
)


app = App(
    name="github_agent",
    root_agent=root_agent,
    # Registered once here, so they apply to every agent, tool and model call.
    plugins=[SafetyPlugin(), ObservabilityPlugin()],
    # Off by default: the Gemini free tier allows zero cached-content storage
    # (limit=0), so enabling this there fails on every single turn with a 429.
    # The request still succeeds -- caching degrades gracefully -- but it logs
    # an error each time and buys nothing. Turn it on with a paid key, where
    # it does cut cost and latency on the ~3.5k-token static prefix.
    context_cache_config=(
        ContextCacheConfig(
            # Gemini enforces a hard 4096-token floor; lower values are no-ops.
            min_tokens=4096,
            ttl_seconds=600,
            cache_intervals=5,
        )
        if os.getenv("ENABLE_CONTEXT_CACHE", "").lower() in ("1", "true", "yes")
        else None
    ),
    events_compaction_config=EventsCompactionConfig(
        # Summarize every 10 invocations, keeping 2 of overlap so the boundary
        # between compacted and live history is not a hard cut.
        compaction_interval=10,
        overlap_size=2,
    ),
)
