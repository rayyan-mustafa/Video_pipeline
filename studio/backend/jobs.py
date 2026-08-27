"""Long-running pipeline stages as cancellable jobs with streamed output.

WHY SUBPROCESSES RATHER THAN IMPORTS. The pipeline's long stages - build_voice,
mux, run_script - are written as argparse `main()`s that print progress and raise
SystemExit. Calling them in-process would mean capturing stdout globally, and a
SystemExit would take the web server down with it. A subprocess gets isolation,
a real exit code, and line-by-line progress for free.

The short, pure functions (imageplan.scan_images, scenesplit.counts) are imported
directly instead - see paths.ensure_importable(). Spawning a process to ask how
many scenes a split produces would be absurd.

WHY OUTPUT IS KEPT IN MEMORY. A run prints a few hundred lines at most, and the
UI wants to show them both live (SSE) and on reload (GET). Keeping the lines in a
deque means a browser refresh mid-job does not lose the log, which a pure stream
would.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# A stage that prints more than this is malfunctioning; the cap stops a runaway
# loop from eating the server's memory.
MAX_LINES = 4000


@dataclass
class Job:
    id: str
    kind: str                      # "voice" | "plan" | "mux" | "script" | ...
    project: str
    argv: list[str]
    status: str = "running"        # running | done | failed | cancelled
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    _proc: asyncio.subprocess.Process | None = None
    _waiters: list[asyncio.Queue] = field(default_factory=list)

    def emit(self, line: str) -> None:
        self.lines.append(line)
        for q in list(self._waiters):
            q.put_nowait(line)

    def close(self) -> None:
        for q in list(self._waiters):
            q.put_nowait(None)      # sentinel: stream over
        self._waiters.clear()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "status": self.status,
            "returncode": self.returncode,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "lines": list(self.lines),
        }


class JobRegistry:
    """All jobs this server has run, newest last. Not persisted - a restart is a
    clean slate, which is correct: a job cannot survive the process that owns its
    subprocess anyway."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, project: str | None = None, limit: int = 50) -> list[dict]:
        js = sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)
        if project:
            js = [j for j in js if j.project == project]
        return [j.as_dict() for j in js[:limit]]

    def running(self, project: str, kind: str) -> Job | None:
        """A stage already in flight for this project. The UI uses this to refuse
        starting a second voice run over the first, which would have two Whisper
        processes writing the same scenes_timeline.json."""
        for j in self._jobs.values():
            if j.project == project and j.kind == kind and j.status == "running":
                return j
        return None

    async def start(self, kind: str, project: str, argv: list[str],
                    cwd: Path, env: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, project=project, argv=argv)
        self._jobs[job.id] = job

        job.emit(f"$ {' '.join(str(a) for a in argv)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *[str(a) for a in argv],
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,   # one interleaved stream
            )
        except OSError as e:
            job.status = "failed"
            job.returncode = -1
            job.finished = time.time()
            job.emit(f"could not start: {e}")
            job.close()
            return job

        job._proc = proc
        asyncio.create_task(self._pump(job, proc))
        return job

    async def _pump(self, job: Job, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                # errors="replace": the pipeline prints em-dashes and the Windows
                # console encoding is not always UTF-8. A decode error must not
                # kill a running build.
                job.emit(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        except Exception as e:                       # noqa: BLE001
            job.emit(f"[stream error] {e}")
        rc = await proc.wait()
        job.returncode = rc
        if job.status != "cancelled":
            job.status = "done" if rc == 0 else "failed"
        job.finished = time.time()
        job.emit(f"[{job.status}] exit {rc} after {job.as_dict()['elapsed']}s")
        job.close()

    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status != "running" or job._proc is None:
            return False
        job.status = "cancelled"
        try:
            job._proc.terminate()
        except ProcessLookupError:
            pass
        return True

    async def stream(self, job_id: str):
        """Yield SSE lines for a job: the backlog first, then live output.

        Replaying the backlog is what makes a browser refresh mid-run harmless.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(job.lines)
        if job.status == "running":
            job._waiters.append(q)

        for line in backlog:
            yield line
        if job.status != "running":
            return

        while True:
            line = await q.get()
            if line is None:
                return
            yield line


registry = JobRegistry()
_ids = itertools.count(1)
