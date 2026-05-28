"""Persistent session state: last loaded data, last export folder, UI prefs.

Lives in ``%LOCALAPPDATA%\\HMM_Studio\\state.json`` on Windows (with sane
fall-backs elsewhere). The store is best-effort: any read/write error is
swallowed so a corrupted file never blocks the app.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _state_path() -> Path:
    base = (os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or os.path.expanduser("~"))
    return Path(base) / "HMM_Studio" / "state.json"


@dataclass
class SessionState:
    last_data_path: str = ""           # full path to last loaded file
    last_data_kind: str = ""           # 'counts' or 'continuous'
    last_bin_size: float = 0.05
    last_orientation: str = "auto"     # for non-CSV loaders
    last_export_dir: str = ""          # default folder for save dialogs
    workers: int | None = None         # remembered parallel worker count
    method_key: str = ""               # last used HMM method
    n_states: int = 3
    restarts: int = 5
    scan_min: int = 2
    scan_max: int = 6
    extra: dict[str, Any] = field(default_factory=dict)


def load() -> SessionState:
    path = _state_path()
    if not path.exists():
        return SessionState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = SessionState()
        for k, v in data.items():
            if hasattr(out, k):
                setattr(out, k, v)
        return out
    except Exception:
        return SessionState()


def save(state: SessionState) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    except Exception:
        pass


def update(**kwargs) -> SessionState:
    """Merge ``kwargs`` into the on-disk state and return the new state."""
    s = load()
    for k, v in kwargs.items():
        if hasattr(s, k):
            setattr(s, k, v)
        else:
            s.extra[k] = v
    save(s)
    return s
