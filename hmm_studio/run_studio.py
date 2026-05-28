"""Launcher: makes the local repo importable, then runs the GUI."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    sys.path.insert(0, str(repo_root))
    legacy_pkg = repo_root / "sticky-poisson-hmm-python"
    if legacy_pkg.exists():
        sys.path.insert(0, str(legacy_pkg))


def main() -> int:
    _bootstrap_path()
    from hmm_studio.app import run  # imported after sys.path is set
    return run()


if __name__ == "__main__":
    sys.exit(main())
