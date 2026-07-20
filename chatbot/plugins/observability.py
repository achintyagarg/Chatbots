"""
Structured logging for every model and tool call.

An agent that retrieves before answering is only trustworthy if you can check
that it did. This plugin emits one JSON line per model call and per tool call
with latency and token counts, which is what turns "it seems grounded" into
something you can actually verify after the fact.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger("chatbot.trace")


def _emit(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


class ObservabilityPlugin(BasePlugin):
    def __init__(self, name: str = "observability"):
        super().__init__(name=name)
        # Keyed by id() of the context object, which is stable for the span of
        # one call and avoids assuming anything about concurrent invocations.
        self._model_started: dict[int, float] = {}
        self._tool_started: dict[int, float] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        _emit(
            "run.start",
            invocation_id=invocation_context.invocation_id,
            agent=invocation_context.agent.name,
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        self._model_started[id(callback_context)] = time.perf_counter()
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        started = self._model_started.pop(id(callback_context), None)
        usage = getattr(llm_response, "usage_metadata", None)
        _emit(
            "model.call",
            agent=callback_context.agent_name,
            latency_ms=round((time.perf_counter() - started) * 1000, 1) if started else None,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            response_tokens=getattr(usage, "candidates_token_count", None),
            # Non-zero here means context caching is doing its job.
            cached_tokens=getattr(usage, "cached_content_token_count", None),
        )
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> None:
        self._tool_started[id(tool_context)] = time.perf_counter()
        _emit("tool.start", tool=tool.name, args=_summarize_args(tool_args))
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        started = self._tool_started.pop(id(tool_context), None)
        _emit(
            "tool.end",
            tool=tool.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 1) if started else None,
            status=result.get("status") if isinstance(result, dict) else None,
            result_bytes=len(json.dumps(result, default=str)) if result else 0,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> None:
        self._tool_started.pop(id(tool_context), None)
        _emit("tool.error", tool=tool.name, error=f"{type(error).__name__}: {error}")
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> None:
        self._model_started.pop(id(callback_context), None)
        _emit("model.error", error=f"{type(error).__name__}: {error}")
        return None


def _summarize_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Truncate long values so a trace line stays readable."""
    out: dict[str, Any] = {}
    for key, value in tool_args.items():
        if isinstance(value, str) and len(value) > 120:
            out[key] = f"{value[:120]}... ({len(value)} chars)"
        else:
            out[key] = value
    return out
