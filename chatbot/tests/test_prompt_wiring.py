"""
Guards the prompt wiring invariant.

This exists because getting it wrong breaks the agent completely while looking
fine: the bot replies "I understand the instructions, I am ready to assist!"
to every question and never calls a tool. Nothing else in the suite catches it,
because every individual piece is correct in isolation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

pytest.importorskip("google.adk")

if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

pytestmark = pytest.mark.skipif(
    not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    reason="agent construction requires GITHUB_PERSONAL_ACCESS_TOKEN",
)


@pytest.fixture(scope="module")
def agent():
    from github_agent import app

    return app.root_agent


class TestInstructionPlacement:
    def test_instruction_is_empty_when_static_is_set(self, agent):
        """
        The invariant. ADK appends a non-empty `instruction` to `contents` as a
        user message *after* the user's question whenever `static_instruction`
        is also set, so the model answers the instruction instead of the user.

        An InstructionProvider callable does not rescue this: `agent.instruction`
        would be truthy, and ADK appends an empty user turn instead. The branch
        is only skipped when `instruction` is falsy.
        """
        if agent.static_instruction:
            assert not agent.instruction, (
                "instruction must be empty while static_instruction is set, "
                "or ADK appends it after the user's message. "
                "See the note at the bottom of prompts.py."
            )

    def test_static_instruction_carries_the_guidance(self, agent):
        text = agent.static_instruction.parts[0].text
        assert len(text) > 1000, "guidance should live in the static instruction"

    def test_static_instruction_includes_skill_fragments(self, agent):
        """Skill guidance is standing instruction and belongs in the system prompt."""
        text = agent.static_instruction.parts[0].text
        assert "PDF" in text, "pdf_skill's INSTRUCTION fragment is missing"

    def test_untrusted_markers_are_explained(self, agent):
        """
        The safety plugin wraps tool output in these markers. If the prompt does
        not explain them, the wrapping is decoration rather than a defense.
        """
        from plugins.safety import UNTRUSTED_OPEN

        assert UNTRUSTED_OPEN in agent.static_instruction.parts[0].text


class TestRequestOrdering:
    @pytest.mark.asyncio_compatible
    def test_user_question_is_the_last_message(self, agent):
        """Builds a real LlmRequest and checks what the model actually sees last."""
        import asyncio

        from google.adk.agents.invocation_context import InvocationContext
        from google.adk.flows.llm_flows.instructions import _build_instructions
        from google.adk.models import LlmRequest
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        async def build():
            service = InMemorySessionService()
            session = await service.create_session(app_name="t", user_id="u")
            ctx = InvocationContext(
                invocation_id="i",
                agent=agent,
                session=session,
                session_service=service,
            )
            request = LlmRequest()
            request.contents = [
                types.Content(
                    role="user", parts=[types.Part(text="SENTINEL_QUESTION")]
                )
            ]
            await _build_instructions(ctx, request)
            return request

        request = asyncio.run(build())
        last = " ".join(p.text or "" for p in request.contents[-1].parts)
        assert "SENTINEL_QUESTION" in last, (
            f"the model's most recent message is not the user's question but: {last[:200]!r}"
        )
