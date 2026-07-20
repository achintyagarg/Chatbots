"""Finance agent skills. See skills_registry.py at the project root."""

from __future__ import annotations

import sys

from skills_registry import load_skills_for


def load_skills():
    return load_skills_for(sys.modules[__name__])
