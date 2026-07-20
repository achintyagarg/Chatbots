"""
Skill registry.

A skill is a module in this package that declares:

    TOOLS: list[BaseTool | Callable]   # required
    INSTRUCTION: str                   # optional prompt fragment

``load_skills()`` discovers them at import time, so adding a capability means
dropping one file into this directory -- no edit to ``agent.py``, no
registration list to keep in sync. That is the point: the agent's tool surface
grows by addition rather than by modification.

A skill that fails to import is logged and skipped rather than taking down the
whole agent, since one broken optional capability should not make the chatbot
unbootable.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

logger = logging.getLogger(__name__)


def load_skills() -> tuple[list[Any], list[str]]:
    """Discover every skill module. Returns (tools, instruction fragments)."""
    tools: list[Any] = []
    instructions: list[str] = []

    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if module_info.name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"{__name__}.{module_info.name}")
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
