"""
Shared skill discovery, used by every agent's ``skills`` package.

A skill is a module in an agent's ``skills/`` directory that declares:

    TOOLS: list[BaseTool | Callable]   # required
    INSTRUCTION: str                   # optional prompt fragment

Discovery happens at import time, so adding a capability means dropping one
file into the agent's skills directory -- no edit to ``agent.py``, no
registration list to keep in sync. The agent's tool surface grows by addition
rather than by modification.

Modules prefixed with ``_`` are helpers, not skills, and are skipped silently.
A skill that fails to import is logged and skipped rather than taking down the
whole agent, since one broken optional capability should not make the chatbot
unbootable.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def load_skills_for(package: ModuleType) -> tuple[list[Any], list[str]]:
    """Discover every skill module in ``package``.

    Returns (tools, instruction fragments).
    """
    tools: list[Any] = []
    instructions: list[str] = []

    for module_info in sorted(
        pkgutil.iter_modules(package.__path__), key=lambda m: m.name
    ):
        if module_info.name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        except Exception:  # noqa: BLE001 - one bad skill must not break the agent
            logger.exception("Skill %r failed to import; skipping", module_info.name)
            continue

        skill_tools = getattr(module, "TOOLS", None)
        if not skill_tools:
            logger.warning("Skill %r declares no TOOLS; skipping", module_info.name)
            continue

        tools.extend(skill_tools)
        instruction = getattr(module, "INSTRUCTION", "")
        if instruction and instruction.strip():
            instructions.append(instruction.strip())

        logger.info("Loaded skill %r (%d tools)", module_info.name, len(skill_tools))

    return tools, instructions
