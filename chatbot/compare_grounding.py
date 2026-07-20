"""
Grounding demo: ask the SAME question to two agents and compare answers.

  grounded_agent    -> has the GitHub MCP toolset; must ground its answer in
                       a real tool call to the live GitHub API.
  ungrounded_agent  -> plain Gemini agent, no tools, answering only from
                       parametric / training knowledge.

Run:
    python compare_grounding.py
    python compare_grounding.py "how many open PRs does this repo have?"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.genai import types

load_dotenv()

GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
REPO_OWNER = os.getenv("DEMO_REPO_OWNER", "google")
REPO_NAME = os.getenv("DEMO_REPO_NAME", "adk-python")
MODEL = "gemini-flash-lite-latest"

if not GITHUB_PAT:
    raise RuntimeError(
        "GITHUB_PERSONAL_ACCESS_TOKEN is not set. Copy .env.example to .env "
        "and add a GitHub personal access token."
    )

DEFAULT_QUESTION = (
    f"Right now, how many open issues does the {REPO_OWNER}/{REPO_NAME} "
    "GitHub repository have, and what is the title and number of the most "
    "recently opened one? Be exact."
)


def build_grounded_agent(toolset):
    return LlmAgent(
        model=MODEL,
        name="grounded_agent",
        instruction=(
            "You are a GitHub analyst with access to LIVE GitHub data through "
            "MCP tools. ALWAYS call a tool to fetch current data before "
            "answering any question about a repo, issue, PR, or commit -- "
            "never answer from memory. Cite the exact numbers, titles, and "
            "timestamps you retrieved so the answer is verifiably grounded "
            "in real tool output."
        ),
        tools=[toolset],
    )


def build_ungrounded_agent():
    return LlmAgent(
        model=MODEL,
        name="ungrounded_agent",
        instruction=(
            "You are a helpful assistant answering questions about GitHub "
            "repositories. You have NO tools and NO access to live data -- "
            "answer using only what you already know from training."
        ),
    )


async def ask(agent, app_name, question):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=app_name, user_id="demo_user"
    )
    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=question)])

    final_text = None
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text
    return final_text


async def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {GITHUB_PAT}"},
        ),
        tool_filter=["get_repository", "list_issues", "search_issues"],
    )

    grounded_agent = build_grounded_agent(toolset)
    ungrounded_agent = build_ungrounded_agent()

    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    ungrounded_answer = await ask(ungrounded_agent, "ungrounded_app", question)
    print("\n--- UNGROUNDED (no tools, training knowledge only) ---\n")
    print(ungrounded_answer)

    grounded_answer = await ask(grounded_agent, "grounded_app", question)
    print("\n--- GROUNDED (GitHub MCP tool calls, live data) ---\n")
    print(grounded_answer)

    await toolset.close()


if __name__ == "__main__":
    asyncio.run(main())
