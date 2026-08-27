"""
Build a multi-figure frame by PLACING assets, instead of asking the model for
a multi-figure scene.

=============================================================================
 WHY THIS EXISTS
=============================================================================
`two-character` shots render ONE figure on roughly 18 of 160 frames. That is
the largest known visual defect in the pipeline, and three rounds of prompt
work did not move it. The clause already says "two figures facing each other
... empty space between them" and the model still draws one.

This is the same situation the floor band and the line weight were in, and it
gets the same answer PLAN.md gave those: the parts that look right in all 160
frames are the parts the model does not touch. A layout is geometry. Geometry
is Python.

So: generate ONE figure at a time -- which the model is reliable at -- and
composite. Two figures become two pastes at known coordinates, and "the model
drew one figure instead of two" stops being a possible outcome.

=============================================================================
 WHY CUTOUT IS TRIVIAL HERE, AND WOULD NOT BE FOR PHOTOGRAPHS
=============================================================================
Three properties of this style remove the three hard parts of compositing:

  flat fills, no shading   -> no light direction to match between pieces
  no perspective           -> no foreshortening to reconcile
  no hands drawn           -> no contact points that must align

The third is why this works at all, and it is not an accident of this module:
config.CHARACTER_BIBLE describes limbs that simply END, and shots.py's
two-character clause already asserts "empty space between them". The prompt
design had ALREADY converged on no-contact staging, for its own reasons. This
module just stops paying the model to do the arithmetic.

CONTACT IS THE HARD LIMIT. Two independently generated figures cannot agree on
where a handshake lands. Frames that need contact must still be generated as
one image -- see `can_composite()`.

=============================================================================
 ORDER, AND THE SAME LESSON THE CAPTIONS TAUGHT
=============================================================================
finish.py records that captions once ran AFTER the palette lock and added ~230
off-palette colours from anti-aliased glyph edges.

Compositing has the identical hazard, one level up: boosting lines per-ASSET
and then pasting gives every figure its own outline weight, and the seams show.
So the order is

    1. paint background      (wall + floor, flat)
    2. paste assets          back to front
    3. finish.boost_lines    ONCE, on the whole composite
    4. palette lock          ONCE
    5. caption + re-lock     (unchanged, caller's job)

numpy and PIL only. No torch, so this stays importable on the CPU phase.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import config

# A pixel is "background" if every channel is at least this bright.
#
# NOT 255. The renders are JPEG, and JPEG ringing around a black stroke leaves
# the surrounding white at 246-253 rather than 255. Measured on the real
# strip: a threshold of 255 kept a grey halo around every figure, which then
# pasted onto the wall colour as a visible rectangle.
#
# config.WHITE_SNAP is 236 and is the palette lock's version of this same
# judgement; this is deliberately a little stricter, because here a wrong call
# punches a hole in the figure rather than just shifting a colour.
BG_WHITE_MIN = 240

# A pixel counts as part of the subject when its alpha is at least this.
#
# NOT `> 0`, and the test caught why. cutout() emits hard 0/255 alpha, so for a
# freshly cut asset any threshold behaves identically -- but scale_to_height()
# resamples with LANCZOS, which leaves a soft fringe of alpha 1..254 around
# every edge. Measuring with `> 0` counted that fringe, so a figure scaled to
# 100px measured back as 105 and every asset came out ~5% oversized, worse the
# more times it was rescaled.
ALPHA_MIN = 128


def cutout(img: Image.Image, white_min: int = BG_WHITE_MIN,
           tolerance: int | None = None) -> Image.Image:
    """Flat background -> transparent. Returns RGBA.

    `tolerance`, when given, switches the background test from "is this pixel
    BRIGHT?" to "is this pixel the SAME COLOUR as the corners?", matching
    within +/- tolerance per channel.

    WHY THIS EXISTS. The white test assumes STYLE_PREFIX got the white
    background it asked for. Measured against SDXL + the vector LoRA, it did
    not: every generated prop came back on a flat but COLOURED field -- pink,
    teal, slate. On those the brightness test marks nothing as background, the
    flood fill removes nothing, and the prop composites as a visible
    rectangle. Silently: alpha is all 255, no exception, and the frame just
    looks wrong.

    The fix is not a different threshold, it is a different question. The
    background is still FLAT and still connected to the border -- only its
    colour changed -- so matching the corner colour keeps every property the
    flood fill relies on while dropping the assumption that flat means white.

    Tolerance is not optional-with-a-default-value on purpose: passing it
    is a claim about the asset ("this one has a coloured field"), and
    defaulting it would silently change the behaviour of every existing
    white-background caller.

    No ML segmentation and no rembg. The background is a flat field by
    construction, so a per-pixel test is exact and costs nothing.

    Only the background CONNECTED TO THE EDGE is removed. A flood fill inward
    from the border, not a global test -- otherwise the white fill INSIDE the
    head, and the whites of the eyes, are punched transparent too and the
    figure pastes onto the wall with a hollow face. That reasoning is why the
    colour-match variant reuses this same fill rather than doing a global
    "every pixel of this colour" replace, which would punch holes in any prop
    whose fill happens to match its own background.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    if tolerance is None:
        bright = (rgb >= white_min).all(axis=-1)
    else:
        # Median of the four corners, not one corner: a single corner can land
        # on JPEG ringing or on a stray mark and drag the whole match off.
        corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
        bg = np.median(corners, axis=0).astype(np.int16)
        bright = (np.abs(rgb.astype(np.int16) - bg) <= tolerance).all(axis=-1)

    # Flood fill from the border across bright pixels only. Iterative dilation
    # against the bright mask -- a few dozen passes on a 640x360 frame, and it
    # keeps scipy optional so this module has the same dependency floor as
    # finish.py.
    outside = np.zeros((h, w), dtype=bool)
    outside[0, :] |= bright[0, :]
    outside[-1, :] |= bright[-1, :]
    outside[:, 0] |= bright[:, 0]
    outside[:, -1] |= bright[:, -1]

    while True:
        grown = outside.copy()
        grown[1:, :] |= outside[:-1, :]
        grown[:-1, :] |= outside[1:, :]
        grown[:, 1:] |= outside[:, :-1]
        grown[:, :-1] |= outside[:, 1:]
        grown &= bright
        if np.array_equal(grown, outside):
            break
        outside = grown

    alpha = np.where(outside, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def erode_alpha(rgba: Image.Image, px: int = 2) -> Image.Image:
    """Shrink the opaque region by `px` pixels. Returns RGBA.

    WHY: cutout() classifies a pixel as background or subject, but the pixels
    ON the boundary are a blend of both -- PNG/JPEG resampling and the
    generator's own anti-aliasing leave a 1-3px fringe carrying the BACKGROUND
    colour. Those survive the test (they are not close enough to the corner
    colour) and paste as a bright halo around the object, which is exactly
    what the first merged frame showed around the sofa and the table.

    Eroding is the right fix rather than widening the tolerance: a wider
    tolerance eats into the object's own fill wherever that fill resembles the
    background, which is how the table lost its legs. Erosion only ever
    removes boundary pixels, and the outline stroke is far thicker than the
    fringe, so nothing legible is lost.
    """
    if px <= 0:
        return rgba
    a = np.asarray(rgba).copy()
    mask = a[:, :, 3] >= ALPHA_MIN
    for _ in range(px):
        shrunk = mask.copy()
        shrunk[1:, :] &= mask[:-1, :]
        shrunk[:-1, :] &= mask[1:, :]
        shrunk[:, 1:] &= mask[:, :-1]
        shrunk[:, :-1] &= mask[:, 1:]
        mask = shrunk
    a[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(a, mode="RGBA")


def cutout_report(rgba: Image.Image, original: Image.Image | None = None) -> dict:
    """Measure whether a cutout actually worked. Cheap, and it has to be cheap
    because it runs on every prop at admission.

    Returns {"kept": fraction opaque, "halo": fraction of boundary pixels that
    still carry the background colour, "verdict": str}.

    THE FAILURE THIS CATCHES IS THE SILENT ONE. A cutout that removed nothing
    raises no exception -- alpha is simply all 255, anchors() happily measures
    the whole canvas, and the prop is filed looking fine. It only becomes
    visible 160 frames later as a rectangle of background pasted over the
    wall. Same class of bug as assert_single_subject(): a wrong answer that
    renders cleanly costs more than a crash.
    """
    a = np.asarray(rgba)
    if a.shape[-1] != 4:
        raise ValueError("cutout_report() needs RGBA")
    mask = a[:, :, 3] >= ALPHA_MIN
    total = mask.size
    kept = float(mask.sum()) / total

    halo = 0.0
    if original is not None and mask.any():
        rgb = np.asarray(original.convert("RGB"), dtype=np.int16)
        corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
        bg = np.median(corners, axis=0).astype(np.int16)
        inner = mask.copy()
        inner[1:, :] &= mask[:-1, :]
        inner[:-1, :] &= mask[1:, :]
        inner[:, 1:] &= mask[:, :-1]
        inner[:, :-1] &= mask[:, 1:]
        edge = mask & ~inner                       # the 1px rim of the subject
        if edge.any():
            near_bg = (np.abs(rgb - bg) <= 40).all(axis=-1)
            halo = float((edge & near_bg).sum()) / float(edge.sum())

    if kept >= 0.92:
        verdict = "FAILED: removed almost nothing -- background is not being matched"
    elif kept <= 0.02:
        verdict = "FAILED: removed almost everything -- tolerance ate the subject"
    elif halo > 0.35:
        verdict = "HALO: boundary still carries background colour -- erode_alpha()"
    else:
        verdict = "ok"
    return {"kept": kept, "halo": halo, "verdict": verdict}


def subject_spans(rgba: Image.Image, min_ratio: float = 0.10) -> list[tuple[int, int]]:
    """Column spans (x0, x1) of the asset's separate subjects, left to right.

    A "subject" is a group of ink that does not overlap another group in X.
    Grouping by COLUMN OVERLAP rather than by pixel connectivity is deliberate:
    a stick figure is not one connected blob -- the eyes float inside the head,
    and on some poses an arm stroke does not quite meet the body -- but every
    part of one figure does share a column range, and a separate prop does not.

    Used by assert_single_subject() below. Also useful on its own for splitting
    a two-figure render into two reusable assets.
    """
    a = np.asarray(rgba)
    if a.shape[-1] != 4:
        raise ValueError("subject_spans() needs RGBA -- run cutout() first")
    mask = a[:, :, 3] >= ALPHA_MIN
    if not mask.any():
        return []

    ink_cols = mask.any(axis=0)
    spans, start = [], None
    for x, on in enumerate(ink_cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            spans.append((start, x - 1))
            start = None
    if start is not None:
        spans.append((start, len(ink_cols) - 1))

    # Merge spans separated by a gap narrower than a stroke -- anti-aliasing
    # and JPEG ringing can drop a column to zero mid-figure.
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(list(s) if False else (s[0], s[1]))
    merged = [list(m) for m in merged]
    out = []
    for m in merged:
        if out and m[0] - out[-1][1] <= 3:
            out[-1][1] = m[1]
        else:
            out.append(m)

    widest = max(m[1] - m[0] for m in out)
    return [(m[0], m[1]) for m in out if (m[1] - m[0]) >= widest * min_ratio]


def assert_single_subject(rgba: Image.Image, name: str = "asset") -> None:
    """Raise if the asset holds more than one subject.

    THIS FAILS LOUDLY ON PURPOSE, and the first version of this module did the
    opposite -- it tried to GUESS which subject was the figure and silently
    measured the wrong one. On an asset holding a small figure beside a large
    brain it anchored on the brain, so scale_to_height() sized the frame to the
    brain and the figure came out visibly too small next to its neighbours.

    Nothing in the output looked broken enough to notice at a glance, which is
    the exact failure this project keeps writing down: a wrong answer that
    renders cleanly costs more than a crash. An asset is one subject. If it is
    not, the caller must split it -- see subject_spans() -- not hope.
    """
    spans = subject_spans(rgba)
    if len(spans) > 1:
        raise ValueError(
            f"{name}: found {len(spans)} subjects at column spans {spans}. "
            f"An asset must contain exactly one subject -- anchors() would "
            f"measure the bounding box of all of them and mis-scale the "
            f"figure. Split it with subject_spans() first."
        )


def anchors(rgba: Image.Image) -> dict | None:
    """Locate the points a layout needs, or None if the asset is empty.

    Returns pixel coordinates:
        top       highest ink   -- the crown of the head
        ground    lowest ink    -- where the figure meets the floor, or the
                                   seat if it is a sitting pose
        x_center  horizontal centre of the ink at the ground contact
        height    top..ground, the number scaling is normalised against

    x_center is measured at the GROUND, not over the whole figure, and that
    matters: a figure with one arm raised has its overall centroid pulled
    toward the raised arm, so centring on it makes the figure look like it is
    leaning. The feet are where it actually stands.
    """
    a = np.asarray(rgba)
    if a.shape[-1] != 4:
        raise ValueError("anchors() needs RGBA -- run cutout() first")
    mask = a[:, :, 3] >= ALPHA_MIN
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    top, ground = int(ys.min()), int(ys.max())
    foot_band = mask[max(ground - 4, top):ground + 1, :]
    cols = np.where(foot_band.any(axis=0))[0]
    x_center = int(cols.mean()) if cols.size else int(xs.mean())

    return {"top": top, "ground": ground,
            "x_center": x_center, "height": ground - top}


def scale_to_height(rgba: Image.Image, target_h: int) -> Image.Image:
    """Resize so the figure's top..ground span equals target_h.

    Scaling on the ANCHOR SPAN rather than the image height is the whole point.
    Two crops of the same character can carry different amounts of blank
    margin, so matching image heights leaves the actual figures different
    sizes -- which reads instantly as two different characters.
    """
    an = anchors(rgba)
    if an is None or an["height"] <= 0:
        return rgba
    factor = target_h / an["height"]
    new = (max(1, round(rgba.width * factor)), max(1, round(rgba.height * factor)))
    return rgba.resize(new, Image.LANCZOS)


def paint_background(size: tuple[int, int], wall: tuple, floor: tuple,
                     floor_y: float, edge_px: int | None = None) -> Image.Image:
    """Flat wall over flat floor, with the black rule between them.

    The code-owned equivalent of finish.draw_floor(), but for a composite there
    is no model-drawn horizon to detect -- the layout DECLARES where the floor
    is, and the figures are then placed onto that line. Deterministic in the
    direction that matters: the floor does not move between frames.
    """
    w, h = size
    edge_px = config.FLOOR_EDGE_PX if edge_px is None else edge_px
    y = max(0, min(h - 1, int(round(h * floor_y))))

    a = np.empty((h, w, 3), dtype=np.uint8)
    a[:y, :] = np.array(wall, dtype=np.uint8)
    a[y:, :] = np.array(floor, dtype=np.uint8)
    if edge_px > 0:
        y0 = max(0, y - edge_px // 2)
        a[y0:y0 + edge_px, :] = np.array(config.PALETTE[1], dtype=np.uint8)
    return Image.fromarray(a, mode="RGB")


def can_composite(scene: dict) -> bool:
    """False when the scene needs figures to TOUCH.

    The honest boundary of this module. Two separately generated figures have
    no way to agree on where a handshake meets, so a contact scene composites
    into two people reaching at nothing. Those frames must stay single-image
    generations and go through the existing redraw path.
    """
    return not bool(scene.get("contact"))


def composite(scene: dict, assets: dict[str, Image.Image],
              size: tuple[int, int] | None = None) -> Image.Image:
    """Render a layout dict to a finished frame.

    scene = {
      "background": {"wall": (r,g,b), "floor": (r,g,b), "floor_y": 0.72},
      "slots": [
        {"asset": "sofa",     "x": 0.30, "ground_y": 0.72, "h": 0.30, "z": 1},
        {"asset": "sitting",  "x": 0.22, "ground_y": 0.68, "h": 0.34, "z": 2},
        {"asset": "standing", "x": 0.70, "ground_y": 0.72, "h": 0.55, "z": 3},
      ],
    }

    x / ground_y / h are FRACTIONS of the frame, so one layout survives a
    resolution change. `h` is the figure's anchor span, not its bitmap height.
    Slots paint in ascending `z`, so a higher z is nearer the viewer.
    """
    size = size or (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    w, h = size
    bg = scene["background"]
    out = paint_background(size, bg["wall"], bg["floor"], bg["floor_y"]).convert("RGBA")

    for slot in sorted(scene["slots"], key=lambda s: s.get("z", 0)):
        art = assets[slot["asset"]]
        if art.mode != "RGBA":
            art = cutout(art)
        if "h" in slot:
            art = scale_to_height(art, int(round(h * slot["h"])))
        an = anchors(art)
        if an is None:
            continue
        # Place by ANCHOR, not by bitmap corner: line the figure's ground
        # contact up with the slot's ground_y, and its foot centre with x.
        px = int(round(w * slot["x"])) - an["x_center"]
        py = int(round(h * slot["ground_y"])) - an["ground"]
        out.alpha_composite(art, (px, py))

    return out.convert("RGB")
