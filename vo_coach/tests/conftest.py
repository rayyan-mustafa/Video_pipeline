"""Shared test guards.

Two things the whole suite must be protected from, both learned the hard way.

1. THE AUDIT LOG IS REAL DATA. It is append-only and lives in the project's
   data directory, so any test reaching the mentor client would write fake
   calls into the evidence base that answers "is this model good enough".

2. TESTS MUST NEVER OPEN THE MICROPHONE. `LiveSession.start()` opens a
   PortAudio input stream, and PortAudio does not fail quickly when the device
   is already in use - it BLOCKS. With a `vo-coach live` session running (which
   is exactly when someone is likely to run the tests), the suite hung for
   minutes at a time with no output and no failing test to point at. It also
   made timings depend on whether the machine happened to have a mic at all.

   So the stream is faked for every test. Nothing here needs real audio: the
   session tests drive `_on_audio` directly with synthetic blocks, which is
   both faster and far more controllable than talking at a microphone.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path, monkeypatch):
    from vo_coach.mentor import validate

    monkeypatch.setattr(validate, "AUDIT_LOG", tmp_path / "mentor_audit.jsonl")


class _FakeInputStream:
    """Accepts the same constructor as `sounddevice.InputStream` and does
    nothing. Tests feed audio by calling the session's callback themselves."""

    def __init__(self, *args, **kwargs):
        self.callback = kwargs.get("callback")
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.started = False


@pytest.fixture(autouse=True)
def _no_real_microphone(monkeypatch):
    import sounddevice as sd

    monkeypatch.setattr(sd, "InputStream", _FakeInputStream)
