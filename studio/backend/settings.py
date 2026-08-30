"""Machine-level settings that outlive a single project or session.

Just the Colab worker URL for now. It lives here rather than in a project
because it describes THIS MACHINE'S current connection to a GPU, not anything
about a particular video - every project on this laptop uses the same worker.

WHY IT IS STORED AT ALL. A cloudflared quick tunnel gets a new hostname every
time the notebook runs, so the URL changes constantly. Storing it means it is
typed once per Colab session instead of once per section, and the Voice page can
show whether the worker behind it is still alive.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import paths

DEFAULTS = {
    "colab_url": "",
    "colab_checked": None,      # ISO timestamp of the last successful health call
    "whisper_model": "large-v3",
}


def load() -> dict:
    """Current settings, with defaults filled in for anything missing."""
    out = dict(DEFAULTS)
    if paths.SETTINGS_FILE.exists():
        try:
            out.update(json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass                # a corrupt file is not worth failing startup over
    return out


def save(**changes) -> dict:
    """Update some settings and write them back. Unknown keys are ignored."""
    cur = load()
    for k, v in changes.items():
        if k in DEFAULTS and v is not None:
            cur[k] = v
    paths.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.SETTINGS_FILE.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return cur


def check_colab(url: str | None = None) -> dict:
    """Ask the worker whether it is alive, and which model it has loaded.

    Returns a plain dict rather than raising: "the worker is down" is the normal
    state most of the time (the runtime gets recycled, the tunnel dies), so it is
    something the page displays, not an error it has to handle.
    """
    url = (url or load()["colab_url"]).strip()
    if not url:
        return {"ok": False, "url": "", "error": "no worker URL set"}

    paths.ensure_importable()
    from pipeline.transcribe import RemoteWhisper

    try:
        info = RemoteWhisper(url).health()
    except Exception as e:
        return {"ok": False, "url": url,
                "error": f"{type(e).__name__}: {e}",
                "hint": "start the worker notebook and paste its CURRENT url - "
                        "a cloudflared quick tunnel changes hostname every run"}

    save(colab_checked=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return {"ok": True, "url": url, **info}
