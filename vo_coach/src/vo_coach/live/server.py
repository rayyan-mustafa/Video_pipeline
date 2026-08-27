"""FastAPI app: serves the HUD and streams session state over a WebSocket.

DELIBERATELY LOCAL-ONLY. Binds to 127.0.0.1 by default. This process opens a
microphone and writes files; exposing it on a LAN interface would make both
available to anything on the network, and there is no authentication here
because a single-user local tool does not need any. Change the host only with
that trade understood.

The socket is one-directional in practice: state flows out at the HUD rate,
and the only thing flowing in is `stop`. Keeping the audio out of the socket is
what makes this file short - see the note at the top of `session.py`.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

# MODULE LEVEL, NOT INSIDE create_app - AND THAT IS LOAD-BEARING.
#
# This file uses `from __future__ import annotations`, so every annotation is a
# STRING at runtime. FastAPI resolves a websocket handler's parameter types by
# looking those strings up in the MODULE's globals. When `WebSocket` was
# imported inside `create_app` it was a local, invisible to that lookup, so
# FastAPI could not tell the parameter was a socket - it treated it as a query
# parameter, failed validation, and closed the connection before accepting it.
#
# The symptom was a bare `403 Forbidden` on the upgrade, with no traceback
# anywhere and a HUD frozen at 0:00 that looked exactly like a silent
# microphone. Plain Starlette routes were unaffected because they do no
# annotation resolution, which is why `/api/state` worked throughout.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..config import load_config
from .session import LiveSession

STATIC = Path(__file__).parent / "static"


def create_app(device=None, niche: str | None = None, save: bool = True,
               use_asr: bool = True, use_mentor: bool = True,
               script: str | None = None):
    cfg = load_config()

    @asynccontextmanager
    async def lifespan(app):
        yield
        # Stop on shutdown so Ctrl+C closes the wav and registers the take. A
        # session killed with the file still open leaves a take that exists on
        # disk and not in the database.
        s = app.state.session
        if s is not None:
            s.stop()

    app = FastAPI(title="vo-coach live", lifespan=lifespan)
    app.state.session = None

    def _start() -> LiveSession:
        s = LiveSession(cfg=cfg, device=device, niche=niche, save=save,
                        use_asr=use_asr, use_mentor=use_mentor,
                        script_lines=SCRIPT_LINES)
        s.start()
        app.state.session = s
        return s

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    def _parse_script() -> list[dict]:
        """Split the marked script into lines with their pause target, the
        words that carry them, and one plain-English note on how to say each.

        DIRECTION IS COMPUTED HERE, ONCE, BEFORE ANY AUDIO EXISTS. A note that
        arrives while the reader is mid-line is too late to use, and asking a
        model for one would put a network round trip in the moment where
        latency matters most. The shape of a sentence does not depend on how it
        was performed, so it can be worked out up front and shown instantly.
        """
        from ..timing.marking import BODY_SEPARATOR, delivery_note, emphasis_words

        if not script:
            return []
        body = script.split(BODY_SEPARATOR, 1)[-1]
        raw = [ln.strip() for ln in body.splitlines() if ln.strip()]
        lines = []
        for i, ln in enumerate(raw):
            pause = re.findall(r"\[(\d+\.\d+)\]", ln)
            plain = re.sub(r"\s*\[\d+\.\d+\]\s*", " ", ln).strip()
            lines.append({
                "text": re.sub(r"\s*\[\d+\.\d+\]\s*", " ⬚ ", ln).strip(),
                "pause": float(pause[-1]) if pause else None,
                "emphasis": emphasis_words(plain),
                "note": delivery_note(plain, is_last=(i == len(raw) - 1)),
            })
        return lines

    SCRIPT_LINES = _parse_script()

    @app.get("/api/script")
    def script_lines():
        """The marked script, split into lines with their pause targets.

        Sent once on load rather than in every state frame: the script does not
        change during a take, and repeating it four times a second would make
        the HUD payload grow with the length of the read.
        """
        return JSONResponse({"lines": SCRIPT_LINES})

    @app.get("/api/state")
    def state():
        s = app.state.session
        return JSONResponse(s.state.to_dict() if s else {"elapsed": 0})

    @app.post("/api/stop")
    def stop():
        s = app.state.session
        if s is None:
            return JSONResponse({"stopped": False})
        take_id = s.stop()
        app.state.session = None
        return JSONResponse({"stopped": True, "take_id": take_id,
                             "raw": s.state.recording_to,
                             "comped": s.state.comped,
                             "missing": s.state.comp_missing})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        if app.state.session is None:
            # A failed mic open (wrong device, no OS permission, device
            # already in use) must reach the HUD as text, not as a page stuck
            # silently at 0:00 with every metric showing "--" - that reads as
            # "not listening" when it is actually "never started".
            try:
                s = _start()
            except Exception as e:
                msg = "could not open the microphone: " + str(e)
                # PortAudio reports "device already in use" as a generic host
                # error, so the message alone points nowhere. On Windows the
                # overwhelmingly common cause is a previous `vo-coach live`
                # still holding the device - and mic PERMISSIONS are never the
                # cause here, because the browser is not what opens the mic.
                if "-9999" in str(e) or "Unanticipated host error" in str(e):
                    msg += ("\n\nThe device is almost certainly already open. "
                            "Another `vo-coach live` is probably still running "
                            "- close it (Ctrl+C in its terminal), then reload "
                            "this page. Browser microphone permissions are not "
                            "involved: this process opens the mic, not the "
                            "browser.")
                await sock.send_text(json.dumps(
                    {"elapsed": 0, "metrics": {}, "clauses": [], "notes": [],
                     "error": msg}))
                await sock.close()
                return
        else:
            s = app.state.session
        hz = float(cfg.get("live.hud_hz", 4.0))
        try:
            while True:
                # to_dict() is the one thing here that touches live state from
                # three threads. An exception raised inside it used to close
                # the socket with no message, which the HUD showed as a page
                # frozen at 0:00 - indistinguishable from "the mic is silent".
                try:
                    frame = json.dumps(s.state.to_dict())
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    frame = json.dumps({"elapsed": 0, "metrics": {},
                                        "clauses": [], "notes": [],
                                        "error": "HUD state failed: " + repr(e)})
                await sock.send_text(frame)
                await asyncio.sleep(1.0 / max(hz, 0.5))
        except WebSocketDisconnect:
            # The HUD closing is not the session ending - a dropped tab must not
            # throw away a take in progress. `stop` is explicit.
            pass

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, device=None,
          niche: str | None = None, save: bool = True, use_asr: bool = True,
          use_mentor: bool = True, script: str | None = None) -> None:
    import uvicorn

    app = create_app(device=device, niche=niche, save=save, use_asr=use_asr,
                     use_mentor=use_mentor, script=script)
    print(f"HUD:  http://{host}:{port}")
    if host == "127.0.0.1":
        # USE THE IP, NOT THE NAME. uvicorn binds IPv4 only, and on Windows
        # `localhost` resolves to the IPv6 loopback ::1 first - which nothing
        # is listening on, so the browser reports the page as unreachable while
        # the server sits there working. Editor-embedded browsers tend to dial
        # IPv4 directly and therefore work, which makes it look like a browser
        # problem rather than an address-family one.
        print("      ^ use this address, not http://localhost:"
              f"{port} - `localhost` may resolve to IPv6 (::1), "
              "which this server does not listen on.")
    print("  capture starts when the HUD connects; Ctrl+C or the Stop button "
          "ends the take.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
