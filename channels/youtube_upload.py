"""Upload a finished video to YouTube with its package: metadata + thumbnail.

    python youtube_upload.py auth     --channel wierdbiology
    python youtube_upload.py whoami   --channel wierdbiology
    python youtube_upload.py upload   --project wierdbiology/chin \
                                      --video "Why Do Humans Have a Chin.mp4" \
                                      --privacy private
    python youtube_upload.py thumbnail --project wierdbiology/chin

WHAT IT READS. Title, description and tags come from build/package.json - the
same file the Studio's Package page shows, written by youtube_packager.py. This
script never invents metadata; if the package is missing it refuses rather
than uploading a video called "chin".

WHERE THE CREDENTIALS LIVE. channels/.youtube/:

    client_secrets.json        the Google Cloud "Desktop app" OAuth client,
                               shared by every channel on this machine
    token_<channel>.json       one refresh token per YouTube channel
    channel_<channel>.json     which YouTube channel that token actually
                               belongs to (title + id), recorded at auth time

A token belongs to whichever Google account / Brand Account was picked on the
consent screen, which is not necessarily the one you meant. `auth` records the
channel it got and `upload` prints it before sending a byte, so a wrong pick is
visible before the video lands on the wrong channel.

The directory is gitignored and blocked from the Studio's /api/file endpoint -
a refresh token is a password.

ONE UPLOAD PER PROJECT. The result is written to build/upload.json, and a
second `upload` refuses while that file exists (pass --force to upload again).
videos.insert costs 1,600 of the default 10,000 daily quota units, and a
duplicate upload is a duplicate video on the channel - both worth one flag.

PROXY. Set HTTPS_PROXY in the environment and every API call goes through it
(google-auth uses requests, the API client uses httplib2; both read it). The
Studio sets it from its settings when a YouTube proxy is configured.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline.project import Project           # noqa: E402

# The Studio pipes this process's stdout into a log view, and on Windows the
# pipe's default encoding is the ANSI codepage - one curly quote in a title
# would kill the upload after the bytes were already sent.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CRED_DIR = HERE / ".youtube"
CLIENT_SECRETS = CRED_DIR / "client_secrets.json"

# upload: videos.insert and thumbnails.set (publishAt at insert time too).
# readonly: channels.list(mine) - the only way to know which channel a token
# belongs to. Nothing here edits or deletes existing videos, so neither
# youtube.force-ssl nor the full youtube scope is asked for.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# YouTube's own limits. The API rejects rather than truncates for most of
# these, and the rejection arrives after the upload has started.
TITLE_MAX = 100
DESC_MAX = 5000
TAGS_MAX = 500
THUMB_MAX_BYTES = 2 * 1024 * 1024
THUMB_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

PRIVACY = ("private", "unlisted", "public")

# Education. Science & Technology is 28; both fit this channel.
DEFAULT_CATEGORY = "27"

# Resumable-upload chunk. Smaller chunks mean finer progress and less to resend
# after a dropped connection; 8 MB is Google's suggested floor for throughput.
CHUNK = 8 * 1024 * 1024

# Transient failures worth retrying, per Google's upload sample.
RETRY_STATUS = {500, 502, 503, 504}
MAX_RETRIES = 8


def log(msg: str = "") -> None:
    print(msg, flush=True)


class UploadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
#  Credentials
# ---------------------------------------------------------------------------
def token_path(channel: str) -> Path:
    return CRED_DIR / f"token_{channel}.json"


def identity_path(channel: str) -> Path:
    return CRED_DIR / f"channel_{channel}.json"


def load_credentials(channel: str):
    """A valid Credentials for this channel, refreshed and re-saved if expired."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    tp = token_path(channel)
    if not tp.exists():
        raise UploadError(
            f"no YouTube token for channel {channel!r} at {tp} - connect the "
            f"channel first (Studio: Package > Connect YouTube, or "
            f"`python youtube_upload.py auth --channel {channel}`)")

    # Scopes are taken from the token itself, so a refresh never asks for more
    # than was granted - asking for more on refresh is an invalid_scope error.
    creds = Credentials.from_authorized_user_file(str(tp))
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            tp.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise UploadError(
                f"the token for {channel!r} is invalid and cannot be refreshed "
                f"- reconnect the channel")
    return creds


def client(channel: str):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=load_credentials(channel),
                 cache_discovery=False)


def fetch_identity(youtube) -> dict:
    """Which YouTube channel this token uploads to."""
    resp = youtube.channels().list(part="snippet", mine=True).execute()
    items = resp.get("items") or []
    if not items:
        raise UploadError(
            "this Google account has no YouTube channel - on the consent screen, "
            "pick the Brand Account that owns the channel, not the personal login")
    ch = items[0]
    return {
        "id": ch["id"],
        "title": ch["snippet"].get("title", ""),
        "custom_url": ch["snippet"].get("customUrl", ""),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def cmd_auth(args) -> int:
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CLIENT_SECRETS.exists():
        log(f"missing OAuth client file: {CLIENT_SECRETS}")
        log("Google Cloud Console > APIs & Services > enable 'YouTube Data API v3'")
        log("> Credentials > Create OAuth client ID > Desktop app > download JSON,")
        log("then drop it on the Package page (or save it at the path above).")
        return 2

    CRED_DIR.mkdir(parents=True, exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    log(f"opening the Google consent screen in your browser for {args.channel!r} ...")
    log("pick the account (or Brand Account) that OWNS the channel you upload to.")
    log("waiting up to 5 minutes for you to click Allow ...")
    # port=0: any free port. The redirect is to localhost, so this only works
    # on the machine whose browser does the consent - which, for a local
    # Studio, it is.
    creds = flow.run_local_server(
        port=0, open_browser=True, prompt="consent",
        access_type="offline", timeout_seconds=300,
        authorization_prompt_message="If no browser opened, visit: {url}",
        success_message="Connected. You can close this tab and go back to the Studio.",
    )
    if not getattr(creds, "refresh_token", None):
        log("Google returned no refresh token - remove this app's access at "
            "myaccount.google.com/permissions and connect again")
        return 1

    tp = token_path(args.channel)
    if tp.exists():
        tp.with_suffix(".json.bak").write_text(tp.read_text(encoding="utf-8"),
                                               encoding="utf-8")
    tp.write_text(creds.to_json(), encoding="utf-8")

    from googleapiclient.discovery import build
    ident = fetch_identity(build("youtube", "v3", credentials=creds,
                                 cache_discovery=False))
    identity_path(args.channel).write_text(json.dumps(ident, indent=2),
                                           encoding="utf-8")
    log(f"connected: {ident['title']} ({ident['custom_url'] or ident['id']})")
    log(f"token -> {tp}")
    return 0


def cmd_whoami(args) -> int:
    ident = fetch_identity(client(args.channel))
    identity_path(args.channel).write_text(json.dumps(ident, indent=2),
                                           encoding="utf-8")
    log(json.dumps(ident, indent=2))
    return 0


# ---------------------------------------------------------------------------
#  Metadata
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    # The API rejects any title or description containing < or > with a bare
    # "invalidDescription" - and the model occasionally writes "->".
    return text.replace("<", "‹").replace(">", "›").strip()


def fit_tags(tags: list[str], budget: int = TAGS_MAX) -> list[str]:
    """Drop tags from the tail until the list fits YouTube's character budget.

    The budget counts a comma between tags and two quote marks around any tag
    containing a space - that is how the API measures it, and a list that
    passes a naive len(",".join()) check can still be rejected.
    """
    out: list[str] = []
    used = 0
    seen: set[str] = set()
    for t in tags:
        t = _clean(t).replace(",", " ")
        if not t or t.lower() in seen:
            continue
        cost = len(t) + (2 if " " in t else 0) + (1 if out else 0)
        if used + cost > budget:
            break
        out.append(t)
        seen.add(t.lower())
        used += cost
    return out


def load_package(proj: Project) -> dict:
    p = proj.build / "package.json"
    if not p.is_file():
        raise UploadError(f"no package at {p} - run the packager first")
    return json.loads(p.read_text(encoding="utf-8"))


def find_thumbnail(proj: Project) -> Path | None:
    for name in ("thumbnail.jpeg", "thumbnail.jpg", "thumbnail.png"):
        for folder in (proj.dir, proj.build):
            if (folder / name).is_file():
                return folder / name
    return None


def check_thumbnail(path: Path) -> None:
    if path.suffix.lower() not in THUMB_MIME:
        raise UploadError(f"thumbnail must be JPG or PNG, got {path.suffix}")
    size = path.stat().st_size
    if size > THUMB_MAX_BYTES:
        raise UploadError(f"thumbnail is {size / 1e6:.1f} MB - YouTube's limit is 2 MB")


# ---------------------------------------------------------------------------
#  Upload
# ---------------------------------------------------------------------------
def _resumable(request, total: int) -> dict:
    """Drive a resumable upload to completion, retrying transient failures."""
    from googleapiclient.errors import HttpError

    response = None
    retries = 0
    last_pct = -1
    while response is None:
        try:
            status, response = request.next_chunk()
            if status is not None:
                pct = int(status.progress() * 100)
                if pct != last_pct:
                    log(f"  uploaded {pct:3d}%  ({status.resumable_progress / 1e6:.1f}"
                        f"/{total / 1e6:.1f} MB)")
                    last_pct = pct
            retries = 0
        except HttpError as e:
            if e.resp.status not in RETRY_STATUS:
                raise
            err = f"HTTP {e.resp.status}"
        except (ConnectionError, TimeoutError, OSError) as e:
            err = f"{type(e).__name__}: {e}"
        else:
            continue

        retries += 1
        if retries > MAX_RETRIES:
            raise UploadError(f"gave up after {MAX_RETRIES} retries ({err})")
        wait = random.uniform(0, 2 ** retries)
        log(f"  {err} - retry {retries}/{MAX_RETRIES} in {wait:.1f}s")
        time.sleep(wait)
    return response


def set_thumbnail(youtube, video_id: str, thumb: Path) -> None:
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(str(thumb), mimetype=THUMB_MIME[thumb.suffix.lower()],
                            resumable=False)
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def _thumbnail_error_hint(e: Exception) -> str:
    s = str(e)
    if "403" in s or "forbidden" in s.lower():
        return (" - custom thumbnails need a phone-verified channel "
                "(youtube.com/verify). The video is up; retry the thumbnail "
                "after verifying.")
    return ""


def record_path(proj: Project) -> Path:
    return proj.build / "upload.json"


def cmd_upload(args) -> int:
    from googleapiclient.http import MediaFileUpload

    channel, topic = args.project.split("/", 1)
    proj = Project(channel, topic, root=HERE)

    rec = record_path(proj)
    if rec.exists() and not args.force:
        prev = json.loads(rec.read_text(encoding="utf-8"))
        log(f"already uploaded: {prev.get('url')} ({prev.get('privacy')})")
        log("pass --force to upload a second copy")
        return 3

    video = Path(args.video)
    if not video.is_absolute():
        video = proj.dir / video
    if not video.is_file():
        raise UploadError(f"video not found: {video}")

    pkg = load_package(proj)
    title = _clean(args.title or pkg.get("best_title") or "")
    if not title:
        raise UploadError("package.json has no title")
    if len(title) > TITLE_MAX:
        raise UploadError(f"title is {len(title)} chars - YouTube's limit is {TITLE_MAX}")
    description = _clean(pkg.get("description") or "")[:DESC_MAX]
    tags = fit_tags(pkg.get("tag_list") or [])

    thumb = Path(args.thumbnail) if args.thumbnail else find_thumbnail(proj)
    if thumb is not None:
        if not thumb.is_absolute():
            thumb = proj.dir / thumb
        if not thumb.is_file():
            raise UploadError(f"thumbnail not found: {thumb}")
        check_thumbnail(thumb)

    status: dict = {
        "privacyStatus": args.privacy,
        "selfDeclaredMadeForKids": False,
        "containsSyntheticMedia": bool(args.synthetic),
    }
    if args.publish_at:
        # YouTube only honours publishAt on a private video - it is the flip
        # from private to public, so anything else is a contradiction.
        status["privacyStatus"] = "private"
        status["publishAt"] = args.publish_at

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": str(args.category),
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": status,
    }

    youtube = client(channel)
    ident = fetch_identity(youtube)
    identity_path(channel).write_text(json.dumps(ident, indent=2), encoding="utf-8")

    size = video.stat().st_size
    log(f"channel     : {ident['title']} ({ident['custom_url'] or ident['id']})")
    log(f"video       : {video.name}  ({size / 1e6:.1f} MB)")
    log(f"title       : {title}")
    log(f"description : {len(description)} chars")
    log(f"tags        : {len(tags)} of {len(pkg.get('tag_list') or [])}")
    log(f"thumbnail   : {thumb.name if thumb else '(none)'}")
    log(f"privacy     : {status['privacyStatus']}"
        + (f"  -> public at {args.publish_at}" if args.publish_at else ""))
    log(f"category    : {args.category}   synthetic media: {bool(args.synthetic)}")
    log("")

    media = MediaFileUpload(str(video), mimetype="video/mp4",
                            chunksize=CHUNK, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body,
                                      media_body=media)
    log("uploading ...")
    response = _resumable(request, size)
    video_id = response.get("id")
    if not video_id:
        raise UploadError(f"upload returned no video id: {response}")
    url = f"https://youtu.be/{video_id}"
    log(f"video uploaded: {url}")

    thumb_ok, thumb_err = False, None
    if thumb is not None:
        try:
            set_thumbnail(youtube, video_id, thumb)
            thumb_ok = True
            log("thumbnail set")
        except Exception as e:                              # noqa: BLE001
            thumb_err = f"{e}"[:400]
            log(f"thumbnail FAILED: {thumb_err}{_thumbnail_error_hint(e)}")

    record = {
        "video_id": video_id,
        "url": url,
        "studio_url": f"https://studio.youtube.com/video/{video_id}/edit",
        "channel": ident,
        "privacy": status["privacyStatus"],
        "publish_at": args.publish_at,
        "title": title,
        "video_file": video.name,
        "thumbnail_file": thumb.name if thumb else None,
        "thumbnail_set": thumb_ok,
        "thumbnail_error": thumb_err,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    proj.build.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"recorded -> {rec}")
    return 0 if (thumb is None or thumb_ok) else 4


def cmd_thumbnail(args) -> int:
    """Re-send the thumbnail for the video this project already uploaded."""
    channel, topic = args.project.split("/", 1)
    proj = Project(channel, topic, root=HERE)
    rec = record_path(proj)
    if not rec.exists():
        raise UploadError("this project has not been uploaded yet")
    record = json.loads(rec.read_text(encoding="utf-8"))

    thumb = Path(args.thumbnail) if args.thumbnail else find_thumbnail(proj)
    if thumb is None:
        raise UploadError("no thumbnail in the project folder")
    if not thumb.is_absolute():
        thumb = proj.dir / thumb
    check_thumbnail(thumb)

    try:
        set_thumbnail(client(channel), record["video_id"], thumb)
    except Exception as e:                                  # noqa: BLE001
        record["thumbnail_error"] = f"{e}"[:400]
        rec.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"thumbnail FAILED: {e}{_thumbnail_error_hint(e)}")
        return 4

    record.update(thumbnail_set=True, thumbnail_error=None, thumbnail_file=thumb.name)
    rec.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"thumbnail set on {record['url']}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("auth", help="connect a YouTube channel (opens a browser)")
    a.add_argument("--channel", required=True)
    a.set_defaults(fn=cmd_auth)

    w = sub.add_parser("whoami", help="which YouTube channel a token uploads to")
    w.add_argument("--channel", required=True)
    w.set_defaults(fn=cmd_whoami)

    u = sub.add_parser("upload", help="upload video + package metadata + thumbnail")
    u.add_argument("--project", required=True, help="<channel>/<topic>")
    u.add_argument("--video", required=True, help="file in the project folder, or a path")
    u.add_argument("--thumbnail", help="default: thumbnail.jpeg/.jpg/.png in the project")
    u.add_argument("--title", help="override package.json's best_title")
    u.add_argument("--privacy", choices=PRIVACY, default="private")
    u.add_argument("--publish-at", help="UTC ISO time, e.g. 2026-09-12T15:00:00Z "
                                        "(uploads private, goes public then)")
    u.add_argument("--category", default=DEFAULT_CATEGORY)
    u.add_argument("--synthetic", action="store_true",
                   help="declare realistic altered/synthetic content")
    u.add_argument("--force", action="store_true", help="upload even if already uploaded")
    u.set_defaults(fn=cmd_upload)

    t = sub.add_parser("thumbnail", help="retry the thumbnail on the uploaded video")
    t.add_argument("--project", required=True)
    t.add_argument("--thumbnail")
    t.set_defaults(fn=cmd_thumbnail)

    args = ap.parse_args()
    try:
        sys.exit(args.fn(args))
    except UploadError as e:
        log(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:                                  # noqa: BLE001
        # HttpError's str() carries the API's reason - quota, invalid title -
        # which is the only part of the traceback worth reading.
        log(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
