"""
The prop library: generated objects, reviewed once, reused forever.

=============================================================================
 WHY A LIBRARY AND NOT GENERATE-EVERY-TIME
=============================================================================
The obvious reason is cost, and it is the smaller one.

The real reason is CONSISTENCY. Generate a lung fresh on every frame that needs
one and video 3's lung is a different lung from video 40's -- which is exactly
the drift this project already paid to eliminate on the character, arriving
again through the props. A cached lung is the same lung forever.

That is the same argument as the palette lock and the PIL captions, and it is
the rule this pipeline keeps proving:

    the parts that look right in all 160 frames are the parts a model touches
    ONCE.

Review cost amortises the same way. A prop is looked at once, by a human, and
is then correct in every video that uses it -- instead of scanning 160 frames
per video hoping to catch the bad ones.

=============================================================================
 THE REVIEW GATE IS NOT OPTIONAL
=============================================================================
Caching amplifies whatever it stores. A wrong-looking MRI machine admitted once
appears in forty videos, and by then it is the channel's house style. So nothing
is readable by the compositor until a human has said yes: staged props are
written with `reviewed: false` and `lookup()` refuses to return them.

This is the one deliberately manual step in the whole pipeline. It is cheap --
a handful of props per video after the first dozen -- and it is the only thing
standing between one bad generation and a permanent defect.

=============================================================================
 KEYS, AND WHY THE PLANNER HAS TO EMIT THEM
=============================================================================
A cache needs a key. The scene planner used to emit only free English --
"a small round pebble sits in the corner" -- and you cannot key a store off
that: "lung", "lungs", "a human lung" and "the left lung" all have to reach one
file, and string-matching prose to find out is a guessing game that fails
quietly.

So plan_scene() now returns a `prop_key` alongside the prose, chosen from the
keys already in this library or declared new. known_keys() is what gets shown
to it. Normalisation here (slug()) is a safety net for when it improvises, not
the primary mechanism.
"""

from __future__ import annotations

import datetime
import json
import re

from PIL import Image

from . import compositor, paths

PROPS_DIRNAME = "props"
MANIFEST_NAME = "props/manifest.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_STOP = {"a", "an", "the", "one", "single", "of", "some", "this", "that"}

# A key must be a short, flat, lowercase slug. Anything else is the model
# improvising prose into the field, and prose keys do not collide correctly --
# which silently defeats the entire cache.
VALID_KEY = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def slug(text: str) -> str:
    """Normalise a noun phrase to a stable key.

    Deliberately crude: lowercase, drop articles, join with underscores, and
    strip a trailing plural 's'. It is a safety net for a model that returned
    "A pair of Lungs" instead of "lung" -- not a synonym engine. Two genuinely
    different words for one object ("belly", "stomach") are meant to be caught
    by the planner picking from known_keys(), not guessed at here.
    """
    words = [w for w in _SLUG_RE.sub(" ", (text or "").lower()).split() if w]
    words = [w for w in words if w not in _STOP] or words
    key = "_".join(words[:4])
    if len(key) > 3 and key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key[:40]


# --------------------------------------------------------------------------
#  the manifest
# --------------------------------------------------------------------------

def _empty() -> dict:
    return {"version": 1, "props": {}}


def load() -> dict:
    f = paths.p(MANIFEST_NAME)
    if not f.exists():
        return _empty()
    with open(f, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    m.setdefault("version", 1)
    m.setdefault("props", {})
    return m


def save(manifest: dict) -> None:
    d = paths.p(PROPS_DIRNAME)
    d.mkdir(parents=True, exist_ok=True)
    with open(paths.p(MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def known_keys(manifest: dict | None = None, reviewed_only: bool = True) -> list[str]:
    """The keys to SHOW THE PLANNER so it reuses instead of inventing.

    Reviewed-only by default: offering a pending key would have the planner
    build a video around a prop that may yet be rejected.
    """
    m = manifest or load()
    return sorted(k for k, v in m["props"].items()
                  if v.get("reviewed") or not reviewed_only)


def lookup(key: str, manifest: dict | None = None) -> dict | None:
    """A reviewed prop's entry, or None. Pending props are invisible on purpose."""
    m = manifest or load()
    entry = m["props"].get(key)
    if entry and entry.get("reviewed"):
        return entry
    return None


def path_for(key: str, manifest: dict | None = None):
    entry = lookup(key, manifest)
    return paths.p(f"{PROPS_DIRNAME}/{entry['file']}") if entry else None


# --------------------------------------------------------------------------
#  admission
# --------------------------------------------------------------------------

def stage(key: str, image: Image.Image, prompt: str = "",
          manifest: dict | None = None, palette_lock=None,
          tolerance: int | None = None, erode: int = 2) -> dict:
    """Cut a generated PNG out, measure it, and file it as UNREVIEWED.

    Runs the same cutout and anchor detection the compositor uses, so a prop is
    measured once here rather than on every paste. `assert_single_subject`
    guards the one failure that would be invisible until composition: a
    generation that came back with two objects, or with a stick figure standing
    next to the object, cannot be placed by a single ground anchor.

    `tolerance` switches cutout() to corner-colour matching. Measured on SDXL
    + the vector LoRA, EVERY generated prop came back on a flat COLOURED field
    rather than white, and the brightness test removes nothing on those -- so
    without this a prop is filed as a rectangle and looks correct in the
    manifest right up until it pastes over the wall 160 frames later.

    `erode` trims the boundary fringe that would otherwise paste as a bright
    halo. 2px by default: measured on the first merged frame, where the sofa
    and table both carried a visible white rim.

    THIS IS WHERE CUTOUT QUALITY IS DECIDED, ONCE. The compositor never re-cuts
    anything -- it pastes the RGBA filed here. That is what stops a variable
    cutout from becoming a random defect scattered across 160 frames: there is
    exactly one place it can go wrong, and cutout_report() now fails loudly
    here rather than letting it through.
    """
    if not VALID_KEY.match(key):
        raise ValueError(f"bad prop key {key!r} -- expected a short lowercase slug")
    m = manifest if manifest is not None else load()

    rgba = compositor.cutout(image, tolerance=tolerance)
    rep = compositor.cutout_report(rgba, original=image)
    if rep["verdict"].startswith("FAILED"):
        raise ValueError(
            f"prop {key!r}: {rep['verdict']} "
            f"(kept {rep['kept']:.1%}). "
            f"Pass tolerance= for a coloured background, or regenerate."
        )
    if rep["verdict"].startswith("HALO") and erode <= 0:
        raise ValueError(f"prop {key!r}: {rep['verdict']} (halo {rep['halo']:.0%})")
    if erode > 0:
        rgba = compositor.erode_alpha(rgba, erode)

    compositor.assert_single_subject(rgba, name=key)
    anc = compositor.anchors(rgba)
    if anc is None:
        raise ValueError(f"prop {key!r} has no visible subject after cutout")

    # Palette-lock ON ADMISSION, not at paste time. A prop enters the library
    # already on-palette, so every frame that uses it is consistent and the
    # per-frame lock has nothing left to move.
    if palette_lock is not None:
        rgb = palette_lock(rgba.convert("RGB"))
        rgba = Image.merge("RGBA", (*rgb.split(), rgba.split()[3]))

    d = paths.p(PROPS_DIRNAME)
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{key}.png"
    rgba.save(d / fname)

    w, h = rgba.size
    m["props"][key] = {
        "file": fname,
        "reviewed": False,
        "ground": round(anc["ground"] / h, 4),
        "width_px": w,
        "height_px": h,
        "nominal_h": 0.60,     # height relative to a standing figure; tune on review
        "prompt": prompt,
        "added": datetime.date.today().isoformat(),
    }
    save(m)
    return m["props"][key]


def admit(key: str, nominal_h: float | None = None,
          manifest: dict | None = None) -> None:
    """Mark a staged prop reviewed. After this the compositor can use it."""
    m = manifest if manifest is not None else load()
    if key not in m["props"]:
        raise KeyError(key)
    m["props"][key]["reviewed"] = True
    if nominal_h is not None:
        m["props"][key]["nominal_h"] = float(nominal_h)
    save(m)


def reject(key: str, manifest: dict | None = None) -> None:
    """Drop a staged prop and its file, so the next miss regenerates it."""
    m = manifest if manifest is not None else load()
    entry = m["props"].pop(key, None)
    if entry:
        f = paths.p(f"{PROPS_DIRNAME}/{entry['file']}")
        if f.exists():
            f.unlink()
    save(m)


def pending(manifest: dict | None = None) -> list[str]:
    """The review queue."""
    m = manifest or load()
    return sorted(k for k, v in m["props"].items() if not v.get("reviewed"))


# --------------------------------------------------------------------------
#  what a planned video still needs
# --------------------------------------------------------------------------

def misses(scenes: list[dict], manifest: dict | None = None) -> list[str]:
    """Prop keys this video needs that the library cannot serve yet.

    This is the generate list. It counts a PENDING prop as a miss, because a
    prop awaiting review cannot be drawn with -- but stage() will simply
    overwrite it rather than making a duplicate.
    """
    m = manifest or load()
    need = []
    for s in scenes:
        k = (s.get("prop_key") or "").strip()
        if k and k not in need and lookup(k, m) is None:
            need.append(k)
    return need


def coverage(scenes: list[dict], manifest: dict | None = None) -> dict:
    """How much of a planned video the library already covers. For the console."""
    m = manifest or load()
    want = [(s.get("prop_key") or "").strip() for s in scenes]
    want = [k for k in want if k]
    hit = sum(1 for k in want if lookup(k, m) is not None)
    return {"frames_needing_prop": len(want),
            "distinct": len(set(want)),
            "hits": hit,
            "misses": len(want) - hit,
            "missing_keys": misses(scenes, m)}
