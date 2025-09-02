from __future__ import annotations

"""Local import bootstrap for running CLI scripts directly.

Usage (in a script placed in the same directory):

    try:
        from srcs.utils.common import setup_logging
    except ModuleNotFoundError:
        import _path_bootstrap  # noqa: F401
        from srcs.utils.common import setup_logging

When a script is executed, only the script's
directory is on sys.path, so the project root (parent of `srcs`) is missing.
This module inserts the repo root once so absolute imports `srcs.*` work.
"""

import sys
from pathlib import Path


def _ensure_root():
    try:
        import srcs.utils.common

        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    try:
        import srcs  # noqa: F401  # imported for side-effect path validation
    except ModuleNotFoundError:
        pass


_ensure_root()
