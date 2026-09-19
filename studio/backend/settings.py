"""Machine-level settings that outlive a single project or session.

The Colab worker URLs and where renders run. They live here rather than in a
project because they describe THIS MACHINE'S current connections, not anything
about a particular video - every project on this laptop uses the same workers.

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
    # THE RENDER WORKER IS A SECOND, SEPARATE URL, and it has to be: the voice
    # worker wants a T4 and the render worker must be on a CPU runtime (see
    # studio/colab_render.py's header on what a GPU render costs). They are two
    # notebooks on two runtimes with two tunnels, so one field could not hold
    # both even if the roles were compatible.
    "colab_render_url": "",
    "render_target": "local",   # "local" | "colab"
    # LAST KNOWN STATE OF EACH WORKER, so a page can SAY something about them
    # without probing. A dead cloudflared tunnel takes ~20s to time out, and
    # the alternative - probing on every page load - makes every page load
    # take 20s once the runtime has gone away, which is most of the time.
    # {"ok": bool, "at": iso, "detail": str}
    "colab_last": None,
    "render_last": None,
    # Proxy for YouTube API calls (auth + upload), e.g. http://user:pass@host:port.
    # Empty = direct. Set it to the proxy the channel is managed through so the
    # uploads come from the same IP as the YouTube Studio sessions.
    "youtube_proxy": "",
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


def _remember(key: str, result: dict) -> dict:
    """Store a check's outcome so the UI can report it without re-probing.

    Returns `result` unchanged so a caller can `return _remember(...)` - the
    remembering is a side effect of answering, never a different answer.
    """
    if result.get("ok"):
        detail = (f"{result['cpus']} cores" if result.get("cpus")
                  else f"{result.get('gpu', '?')} · {result.get('model', '?')}")
    else:
        detail = str(result.get("error", "unreachable"))
    save(**{key: {
        "ok": bool(result.get("ok")),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": result.get("url", ""),
        "detail": detail[:200],
    }})
    return result


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
        return _remember("colab_last", {"ok": False, "url": url,
                "error": f"{type(e).__name__}: {e}",
                "hint": "start the worker notebook and paste its CURRENT url - "
                        "a cloudflared quick tunnel changes hostname every run"})

    save(colab_checked=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return _remember("colab_last", {"ok": True, "url": url, **info})


def check_render_worker(url: str | None = None) -> dict:
    """Is the RENDER worker up? Same contract as `check_colab`, different role.

    It also checks that the URL is not the voice worker's: the two are easy to
    paste into the wrong box - both are cloudflared hostnames printed by a
    Colab cell - and the failure that would follow (a bundle POSTed at a
    Whisper worker) is a 404 several minutes into a render rather than
    something the page could have said immediately.
    """
    url = (url or load()["colab_render_url"]).strip()
    if not url:
        return {"ok": False, "url": "", "error": "no render worker URL set"}

    try:
        import httpx
        r = httpx.get(f"{url.rstrip('/')}/health", timeout=20.0)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        return _remember("render_last", {"ok": False, "url": url,
                "error": f"{type(e).__name__}: {e}",
                "hint": "run studio/colab_render.py in a Colab CPU cell and "
                        "paste its CURRENT url - a cloudflared quick tunnel "
                        "changes hostname every run"})

    if info.get("role") != "render":
        return _remember("render_last", {"ok": False, "url": url,
                "error": f"that worker reports role={info.get('role')!r}",
                "hint": "this is the render worker's box - the voice worker's "
                        "URL goes on the Voice page. They are separate "
                        "runtimes: voice wants a T4, render must be CPU."})
    return _remember("render_last", {"ok": True, "url": url, **info})
