"""Runtime helpers for layer-sweep pipelines."""

from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    """Find project root by searching upward for ``xsam/xsam``.

    Args:
        start_path: Start path.
    Returns:
        Resolved project root.
    """
    resolved = start_path.resolve()
    candidates = [resolved] + list(resolved.parents)
    for candidate in candidates:
        if (candidate / "xsam" / "xsam").exists():
            return candidate
    raise FileNotFoundError(f"Cannot locate project root from: {start_path}")


def ensure_project_paths(start_file: str) -> Path:
    """Ensure project paths are available in ``sys.path``.

    Args:
        start_file: ``__file__`` from caller.
    Returns:
        Project root path.
    """
    project_root = find_project_root(Path(start_file).resolve())
    xsam_pkg_root = project_root / "xsam"
    for path in [project_root, xsam_pkg_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return project_root


def resolve_path(project_root: Path, path_like: str) -> Path:
    """Resolve path from project root when input is relative.

    Args:
        project_root: Project root path.
        path_like: Absolute/relative path string.
    Returns:
        Absolute resolved path.
    """
    path = Path(path_like)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path
