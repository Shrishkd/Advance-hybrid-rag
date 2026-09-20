"""Shared report helpers.

WHY THIS EXISTS: A REPORT IS TWO DOCUMENTS IN ONE FILE
-------------------------------------------------------
Every experiment report has a generated half (tables, numbers, run metadata)
and a hand-written half (the Decision section - the reasoning that justifies
a config value). Re-running the experiment must refresh the first and must
NOT touch the second.

This was learned the expensive way. The ANN lab got decision-preservation
after a formatting fix would have silently deleted its reasoning. The
retrieval lab then shipped without it, and a rerun destroyed a written D5
decision - the same bug, in a second place, because the fix lived in one
script instead of a shared function.

If a third report writer appears, it imports this.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "## Decision"
PLACEHOLDER = "_fill in_"


def decision_section(path: Path, default: list[str]) -> list[str]:
    """Return the existing hand-written Decision section, or `default`.

    A section still containing the placeholder counts as unwritten, so an
    untouched template is replaced rather than preserved forever.

    Args:
        path: the report file, which may not exist yet.
        default: template lines to use when there is nothing to preserve.

    Returns:
        Lines to append to the regenerated report.
    """
    if not path.exists():
        return default
    old = path.read_text(encoding="utf-8")
    if MARKER not in old:
        return default
    kept = old[old.index(MARKER):].rstrip()
    if PLACEHOLDER in kept:
        return default
    return kept.splitlines() + [""]
