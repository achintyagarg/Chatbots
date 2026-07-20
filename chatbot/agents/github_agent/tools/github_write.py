"""
GitHub write actions, gated behind human approval.

Writes are implemented here as local FunctionTools rather than taken from the
MCP toolset, for one reason: ``require_confirmation`` and
``request_confirmation`` are FunctionTool mechanisms, so owning the function is
what makes the approval gate possible. The MCP toolset stays read-only, which
also means a newly added MCP tool can never become an ungated write path.

Two gate styles, chosen by what the approver needs to do:

* ``add_comment`` -- a yes/no decision, so ``require_confirmation`` with a
  callable predicate is enough.
* ``create_issue`` -- the approver usually wants to fix the title or trim the
  body before it goes out, so it uses ``request_confirmation`` with a payload
  that comes back edited.

Note that approval is the *second* gate. ``SafetyPlugin.before_tool_callback``
already rejected writes to non-allowlisted repositories before a human is ever
asked, so a human is only consulted about actions that are permitted at all.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
TIMEOUT = httpx.Timeout(20.0)


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _post(path: str, json_body: dict[str, Any]) -> dict[str, Any]:
    """POST to the GitHub API, converting failures into tool-readable results."""
    try:
        response = httpx.post(
            f"{GITHUB_API}{path}", headers=_headers(), json=json_body, timeout=TIMEOUT
        )
    except httpx.HTTPError as exc:
        logger.exception("GitHub request failed")
        return {"status": "error", "error": f"Network error: {exc}"}

    if response.status_code >= 400:
        # Surface GitHub's own message; "not found" here usually means the
        # token lacks write scope rather than a missing repository.
        try:
            detail = response.json().get("message", response.text)
        except ValueError:
            detail = response.text
        return {
            "status": "error",
            "http_status": response.status_code,
            "error": detail,
        }

    return {"status": "ok", "data": response.json()}


def _needs_confirmation(**kwargs: Any) -> bool:
    """
    Predicate for ``require_confirmation``.

    Always True: every write to a real repository gets a human decision. It is
    a callable rather than ``True`` so the policy has one obvious place to
    change -- e.g. auto-approving comments on a sandbox repo.
    """
    return True


def add_comment(
    owner: str, repo: str, issue_number: int, body: str
) -> dict[str, Any]:
    """Post a comment on an existing GitHub issue or pull request.

    This action is visible to everyone who can see the repository and requires
    human approval before it runs. Tell the user exactly what you intend to
    post before calling this.

    Args:
        owner: Repository owner, e.g. 'google'.
        repo: Repository name, e.g. 'adk-python'.
        issue_number: The issue or pull request number to comment on.
        body: The comment text, in GitHub-flavored markdown.

    Returns:
        A dict with the created comment's URL on success, or an error.
    """
    if not body.strip():
        return {"status": "error", "error": "Comment body must not be empty."}

    result = _post(
        f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body}
    )
    if result["status"] != "ok":
        return result

    return {
        "status": "ok",
        "message": f"Comment posted on {owner}/{repo}#{issue_number}.",
        "url": result["data"].get("html_url"),
    }


def create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Open a new issue on a GitHub repository.

    This creates something publicly visible and requires human approval. The
    approver can edit the title and body before it is created, so the values
    that actually get used may differ from what you proposed.

    Args:
        owner: Repository owner, e.g. 'google'.
        repo: Repository name, e.g. 'adk-python'.
        title: A short, specific issue title.
        body: The issue body in GitHub-flavored markdown.
        tool_context: Provided by the runtime.

    Returns:
        On the first call, a pending-approval status. After approval, the
        created issue's number and URL. If rejected, a rejected status.
    """
    confirmation = tool_context.tool_confirmation

    if confirmation is None:
        # First pass: describe the action and hand the approver an editable
        # copy of every field, then return. The runtime pauses here.
        tool_context.request_confirmation(
            hint=(
                f"Create a new issue on {owner}/{repo}?\n\n"
                f"Title: {title}\n\n{body}\n\n"
                "You can edit the title and body before approving."
            ),
            payload={"owner": owner, "repo": repo, "title": title, "body": body},
        )
        return {
            "status": "pending_approval",
            "message": (
                f"Waiting for human approval to open an issue on {owner}/{repo}. "
                "Do not claim the issue was created."
            ),
        }

    if not confirmation.confirmed:
        return {
            "status": "rejected",
            "message": (
                "The human rejected this issue. Do not retry and do not attempt "
                "another route to the same action."
            ),
        }

    # Approved. The payload is authoritative -- it carries the approver's edits.
    payload = confirmation.payload or {}
    final_owner = payload.get("owner", owner)
    final_repo = payload.get("repo", repo)
    final_title = payload.get("title", title)
    final_body = payload.get("body", body)

    if not str(final_title).strip():
        return {"status": "error", "error": "Issue title must not be empty."}

    result = _post(
        f"/repos/{final_owner}/{final_repo}/issues",
        {"title": final_title, "body": final_body},
    )
    if result["status"] != "ok":
        return result

    data = result["data"]
    edited = (final_title != title) or (final_body != body)
    return {
        "status": "ok",
        "message": (
            f"Opened issue #{data.get('number')} on {final_owner}/{final_repo}."
            + (" The approver edited it before it was created." if edited else "")
        ),
        "issue_number": data.get("number"),
        "url": data.get("html_url"),
        "title_used": final_title,
    }


# `create_issue` drives its own gate through request_confirmation, so it must
# NOT also set require_confirmation -- that would ask the human twice.
create_issue_tool = FunctionTool(create_issue)
add_comment_tool = FunctionTool(add_comment, require_confirmation=_needs_confirmation)

WRITE_TOOLS = [create_issue_tool, add_comment_tool]
