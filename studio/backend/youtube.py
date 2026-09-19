"""Upload to YouTube: the channel connection, the thumbnail, and the upload job.

The upload itself is channels/youtube_upload.py, spawned as a job like every
other long stage - so it gets the same live log and cancel, and the CLI keeps
working without the Studio. This module only answers "can it run" and hands
over the arguments.

WHAT IS UPLOADED. The metadata is build/package.json, verbatim - the same
strings the Package page shows. The only choices made here are the ones the
package cannot make: which of the three titles, which video file, privacy,
schedule. Nothing is editable, for the same reason the Package page's fields
are not: an edit made here would not survive a re-run of the packager.

THE CREDENTIALS DIRECTORY (channels/.youtube) holds a refresh token, which is a
password for the channel. It is gitignored, and app.py blocks the directory
from /api/file - that endpoint serves .json, and a token is a .json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import UploadFile

from . import jobs, paths, settings

UPLOADER = paths.CHANNELS / "youtube_upload.py"
CRED_DIR = paths.CHANNELS / ".youtube"
CLIENT_SECRETS = CRED_DIR / "client_secrets.json"

THUMB_EXTS = {".jpg": "thumbnail.jpeg", ".jpeg": "thumbnail.jpeg", ".png": "thumbnail.png"}
THUMB_NAMES = ("thumbnail.jpeg", "thumbnail.jpg", "thumbnail.png")
THUMB_MAX_BYTES = 2 * 1024 * 1024
PRIVACY = ("private", "unlisted", "public")
CATEGORIES = {"27": "Education", "28": "Science & Technology", "24": "Entertainment"}


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def auth_status(channel: str) -> dict:
    """Whether this Studio channel can upload, and to which YouTube channel."""
    token = CRED_DIR / f"token_{channel}.json"
    return {
        "has_client_secrets": CLIENT_SECRETS.is_file(),
        "connected": token.is_file(),
        # Recorded by the auth job and refreshed on every upload. It is what
        # stops a video landing on the wrong Brand Account unnoticed.
        "identity": _read_json(CRED_DIR / f"channel_{channel}.json"),
    }


def _videos(proj) -> list[dict]:
    """Every .mp4 in the project folder, the likeliest final render first.

    The build writes <topic>.mp4, but a finished video is often renamed to its
    title before upload (chin's is "Why Do Humans Have a Chin.mp4"), and
    section previews sit alongside it. So the choice is offered rather than
    assumed: the canonical output if it exists, otherwise the largest file -
    a full render is always bigger than a section of it.
    """
    if not proj.dir.is_dir():
        return []
    files = [p for p in proj.dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    files.sort(key=lambda p: (p != proj.out, -p.stat().st_size))
    return [{
        "name": p.name,
        "abs_path": str(p),
        "mb": round(p.stat().st_size / 1e6, 1),
        "mtime": p.stat().st_mtime,
        "canonical": p == proj.out,
    } for p in files]


def _thumbnail(proj) -> dict | None:
    for name in THUMB_NAMES:
        for folder in (proj.dir, proj.build):
            p = folder / name
            if p.is_file():
                size = p.stat().st_size
                return {"name": p.name, "abs_path": str(p), "bytes": size,
                        "mtime": p.stat().st_mtime,
                        "ok": size <= THUMB_MAX_BYTES}
    return None


def status(proj) -> dict:
    record = _read_json(proj.build / "upload.json")
    pkg = proj.build / "package.json"
    thumb = _thumbnail(proj)
    videos = _videos(proj)
    auth = auth_status(proj.channel)

    blockers = []
    if not auth["connected"]:
        blockers.append("connect a YouTube channel")
    if not pkg.is_file():
        blockers.append("run the packager")
    if not videos:
        blockers.append(f"render the video (no .mp4 in {proj.dir.name}/)")
    if thumb and not thumb["ok"]:
        blockers.append("thumbnail is over YouTube's 2 MB limit")

    return {
        "auth": auth,
        "videos": videos,
        "thumbnail": thumb,
        "record": record,
        "blockers": blockers,
        "can_upload": not blockers,
        "privacy_choices": list(PRIVACY),
        "categories": CATEGORIES,
        "proxy": settings.load().get("youtube_proxy", ""),
    }


def _env() -> dict | None:
    """The job's environment: HTTPS_PROXY when a YouTube proxy is set.

    Routing the API calls through the proxy you manage the channel from keeps
    the upload on the same IP as your YouTube Studio sessions. localhost is
    excluded - the OAuth consent redirects there and must not leave the machine.
    """
    proxy = (settings.load().get("youtube_proxy") or "").strip()
    if not proxy:
        return None
    env = dict(os.environ)
    env.update(HTTPS_PROXY=proxy, HTTP_PROXY=proxy,
               NO_PROXY="localhost,127.0.0.1,::1")
    return env


async def save_thumbnail(proj, f: UploadFile) -> dict:
    """Save the uploaded image as the project's thumbnail.

    Any previous thumbnail is kept as thumbnail_prev.<ext>, not deleted - it
    was generated from a prompt and may be the one you end up wanting back.
    """
    name = Path(f.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in THUMB_EXTS:
        raise ValueError(f"thumbnail must be JPG or PNG, got {ext or 'no extension'}")

    data = await f.read()
    if len(data) > THUMB_MAX_BYTES:
        raise ValueError(f"{len(data) / 1e6:.1f} MB - YouTube rejects thumbnails over 2 MB")

    proj.dir.mkdir(parents=True, exist_ok=True)
    for old in THUMB_NAMES:
        p = proj.dir / old
        if p.is_file():
            # os.replace, not shutil.move: on Windows a rename onto an
            # existing thumbnail_prev fails, and the second swap would too.
            os.replace(p, proj.dir / f"thumbnail_prev{p.suffix}")
    (proj.dir / THUMB_EXTS[ext]).write_bytes(data)
    return status(proj)


async def save_client_secrets(f: UploadFile) -> dict:
    """Store the Google Cloud OAuth client JSON (a Desktop-app client)."""
    raw = await f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("that file is not JSON") from e
    if "installed" not in data:
        kind = next(iter(data), "unknown")
        raise ValueError(
            f"this is a {kind!r} OAuth client - create a 'Desktop app' client "
            f"instead; only that type can redirect to this machine")
    CRED_DIR.mkdir(parents=True, exist_ok=True)
    CLIENT_SECRETS.write_bytes(raw)
    return {"ok": True}


async def run_auth(channel: str):
    return await jobs.registry.start(
        kind="youtube-auth", project=f"youtube/{channel}",
        argv=[paths.PYTHON, UPLOADER, "auth", "--channel", channel],
        cwd=paths.CHANNELS, env=_env(),
    )


async def run_upload(proj, *, video: str, title: str | None, privacy: str,
                     publish_at: str | None, category: str, synthetic: bool,
                     force: bool):
    # A bare file name, resolved inside the project - never a path from the
    # browser, which could point anywhere on disk.
    names = {v["name"] for v in _videos(proj)}
    if video not in names:
        raise ValueError(f"{video!r} is not an .mp4 in this project's folder")
    if privacy not in PRIVACY:
        raise ValueError(f"privacy must be one of {PRIVACY}")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")

    argv = [paths.PYTHON, UPLOADER, "upload",
            "--project", f"{proj.channel}/{proj.topic}",
            "--video", video, "--privacy", privacy, "--category", category]
    if title:
        argv += ["--title", title]
    if publish_at:
        argv += ["--publish-at", publish_at]
    if synthetic:
        argv.append("--synthetic")
    if force:
        argv.append("--force")
    return await jobs.registry.start(
        kind="youtube-upload", project=f"{proj.channel}/{proj.topic}",
        argv=argv, cwd=paths.CHANNELS, env=_env(),
    )


async def run_thumbnail(proj):
    return await jobs.registry.start(
        kind="youtube-thumb", project=f"{proj.channel}/{proj.topic}",
        argv=[paths.PYTHON, UPLOADER, "thumbnail",
              "--project", f"{proj.channel}/{proj.topic}"],
        cwd=paths.CHANNELS, env=_env(),
    )
