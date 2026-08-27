"""
Deterministic post-processing of a drawn frame.

=============================================================================
 WHY THESE TWO THINGS LIVE HERE AND NOT IN THE PROMPT
=============================================================================
PLAN.md draws a line through this pipeline that has held up every time it was
tested: the parts that look right in all 160 frames are the parts FLUX does not
touch. The palette lock is code. The caption lettering is code. Both are
flawless in every frame. Everything left to the prompt varies.

Three rounds of prompt work (see config.CHARACTER_BIBLE / STYLE_PREFIX) fixed
the character -- the oval torso became a real single stroke, the dot eyes became
big pupils that read at thumbnail size. The same three rounds failed completely
on two attributes, across five phrasings each:

    the tan floor band      rendered as a thin line, an orange scribble, or
                            nothing, and differently on different frames
    the black line weight    stayed thin no matter how it was asked for

Both are geometric, both are identical on every frame, and both are trivial in
numpy. So they move here, and the floor comes OUT of the shot clause so FLUX
stops drawing a competing horizon under ours.

=============================================================================
 ORDER MATTERS, AND IT IS THE SAME LESSON AS THE CAPTIONS
=============================================================================
HANDOFF_NEXT.md records that captions were once drawn AFTER the palette lock,
and PIL's anti-aliased glyph edges added ~230 off-palette colours per frame.
Anything that paints must therefore be followed by a re-snap.

The order below is:

    1. line boost     dilate dark pixels   (before the lock: works on raw luma)
    2. floor band     paint the rectangle  (palette colour, but anti-aliasing
                                            on the edge rule still needs 3)
    3. palette lock   snap everything
    4. caption        PIL text
    5. palette lock   again -- glyph edges

No torch. numpy and PIL only, so this stays importable on the CPU phase.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from . import config


# =============================================================================
#  RESTROKE  --  why blur-and-threshold and not more dilation
# =============================================================================
#  boost_lines() below grows dark pixels outward. That makes the linework
#  THICKER and leaves it exactly as WOBBLY as it was, because dilation moves
#  every boundary point outward by the same amount -- including the boundary
#  points that are wobble. config.py already says this out loud:
#
#      "HONEST LIMIT: dilation fixes stroke WEIGHT, not stroke SMOOTHNESS."
#
#  and then concludes it is a generation problem. It is not. Wobble is
#  high-frequency noise on a contour, and the standard way to remove
#  high-frequency noise from a binary shape is to blur the mask and re-threshold
#  it. Large-scale shape survives the blur; boundary jitter averages away.
#
#  The same operation gives thickness for free: thresholding the blurred mask
#  BELOW 0.5 expands it, above 0.5 shrinks it. So one pass does both jobs that
#  five prompt phrasings failed at:
#
#      sigma   how much wobble is removed
#      thresh  how thick the result is  (lower = thicker)
#
#  This runs INSTEAD of boost_lines(), not after it -- stacking them would
#  dilate the already-expanded mask and close up the eyes.
# =============================================================================

def restroke(img: Image.Image, sigma: float | None = None,
             thresh: float | None = None, dark_max: int | None = None
             ) -> Image.Image:
    """Redraw the linework at a uniform, smoothed stroke width.

    Replaces the ink mask wholesale: everything that survives the smoothing is
    painted the palette's outline colour, everything that does not becomes
    white. That white assumption is why this must run FIRST, before
    paint_background() puts colour behind the drawing -- see apply_all().

    Returns the image unchanged when config.RESTROKE_ON is False.
    """
    sigma = config.RESTROKE_SIGMA if sigma is None else sigma
    thresh = config.RESTROKE_THRESH if thresh is None else thresh
    dark_max = config.LINE_DARK_MAX if dark_max is None else dark_max

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    ink = (luma < dark_max).astype(np.float32)
    if not ink.any():
        return img

    # DILATE TO A FLOOR *BEFORE* BLURRING. Blur-and-threshold on its own erases
    # thin strokes, and it does it silently: a 1-pixel line blurred at sigma=1.6
    # peaks around 0.25, which is under any sensible threshold, so the whole
    # stroke disappears. Measured -- a synthetic 1px line came back completely
    # blank. It only survived on a real frame because those strokes are already
    # 3-5px after the upscale, which is exactly the kind of accident that holds
    # until one frame has a fine detail in it and then quietly loses it.
    #
    # Dilating first guarantees every stroke is at least 2*px+1 wide going into
    # the blur, so the peak stays at 1.0 and the threshold only moves the EDGE.
    # Thickness therefore comes from the dilation (predictable, in pixels) and
    # smoothing from the blur -- one job each, instead of one threshold trying
    # to do both and failing at the extremes.
    grown = ink > 0.5
    for _ in range(max(0, config.RESTROKE_WIDTH_PX)):
        g = grown
        s = g.copy()
        s[1:, :] |= g[:-1, :]
        s[:-1, :] |= g[1:, :]
        s[:, 1:] |= g[:, :-1]
        s[:, :-1] |= g[:, 1:]
        grown = s

    # THE ORIGINAL INK IS A FLOOR, UNCONDITIONALLY.
    #
    # The dilation above raises the bar but does not remove the failure: blur
    # and threshold can ALWAYS erase a stroke if sigma outgrows it. Measured,
    # after the dilation floor was added -- a 1px stroke dilated to 3px still
    # vanished at sigma=3.0, because the peak lands near 0.40 and the threshold
    # is 0.5. Any width can be beaten by a large enough sigma.
    #
    # So the guarantee is made structural instead of numeric: whatever the blur
    # returns, the pixels that were ink stay ink. In normal use this changes
    # nothing at all -- dilate-then-blur yields a superset of the original, so
    # the union IS the blurred mask. It only engages in the pathological case,
    # and it turns "the linework silently disappeared" into "the linework was
    # not smoothed as much as asked", which is a survivable failure.
    soft = ndimage.gaussian_filter(grown.astype(np.float32), sigma=sigma)
    old_ink = ink > 0.5
    new_ink = (soft >= thresh) | old_ink

    rgb[old_ink & ~new_ink] = np.array((255, 255, 255), dtype=np.uint8)
    rgb[new_ink] = np.array(config.PALETTE[1], dtype=np.uint8)
    return Image.fromarray(rgb)


# =============================================================================
#  PAINT_BACKGROUND  --  the floor band's argument, applied to the whole wall
# =============================================================================
#  The reference channel's frames are full-bleed colour: a navy field behind a
#  sleeping figure, a beige room behind a sofa, flat green behind trees. Ours
#  are white, because _ROOM in shots.py asks for "pure white background wall".
#
#  Asking FLUX for a coloured background instead would reintroduce exactly the
#  per-frame inconsistency the floor band was moved into code to remove -- a
#  navy that is a different navy on every frame, and which the palette lock then
#  snaps somewhere unpredictable.
#
#  So it is painted here, for the same three reasons the floor is: it is a flat
#  region, it is identical on every frame that uses it, and it is trivial.
#
#  ONLY the region connected to the frame edge is filled. That is what keeps the
#  white INSIDE the character -- the head, the eyes -- white, which is the
#  reference channel's actual look: a white-and-black figure on a colour field.
#  A naive "replace all white" would flood the face too.
# =============================================================================

def paint_background(img: Image.Image, colour: tuple | None = None,
                     floor_y: int | None = None) -> Image.Image:
    """Flood the edge-connected white region with a flat background colour.

    `floor_y` protects the floor band: rows at or below it are left alone, so
    this can run either side of draw_floor() without eating it.
    """
    colour = config.BACKGROUND_COLOR if colour is None else colour
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    h, w = rgb.shape[:2]

    # "NOT LINEWORK", NOT "NEAR WHITE". The first version tested luma >
    # WHITE_SNAP, and any pixel between the two thresholds -- JPEG ringing, a
    # grey artifact, a soft edge -- was neither ink nor background, so it walled
    # the flood off. Measured on a real frame: a band of such pixels left a
    # white stripe across the wall that the fill could not cross.
    #
    # Testing for "not ink" instead leaves no third category. Coloured props
    # still survive because a prop is enclosed by its own black outline and is
    # therefore a different connected component; the floor survives via floor_y.
    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    near_white = luma >= config.LINE_DARK_MAX

    # PAD BY ONE WHITE PIXEL BEFORE LABELLING. Without this, any ink that
    # touches the frame edge cuts the background into pieces and only the piece
    # still touching an edge gets filled -- measured on a real frame: the wall
    # left of the figure stayed white while the wall right of it went beige,
    # because a horizon line reached both edges and split the region in two.
    #
    # Padding guarantees one connected ring around the whole drawing, so
    # "outside" is whatever that ring reaches, and the test becomes robust to
    # ink at the border rather than depending on it being absent.
    padded = np.pad(near_white, 1, mode="constant", constant_values=True)
    lbl, n = ndimage.label(padded)
    if n == 0:
        return img

    outside_label = lbl[0, 0]
    bg = (lbl == outside_label)[1:-1, 1:-1]
    if floor_y is not None:
        bg[max(0, floor_y):, :] = False

    rgb[bg] = np.array(colour, dtype=np.uint8)
    return Image.fromarray(rgb)


def boost_lines(img: Image.Image, px: int | None = None,
                dark_max: int | None = None) -> Image.Image:
    """Thicken the linework by growing dark pixels outward.

    A grey-scale dilation restricted to pixels already dark enough to be
    linework. Done with numpy shifts rather than a convolution so the package
    keeps numpy as its only numeric dependency.

    Runs BEFORE the palette lock, on the raw render, because that is where the
    line's own anti-aliased edge still exists -- dilating after the snap would
    grow hard-edged blocks instead.
    """
    px = config.LINE_BOOST_PX if px is None else px
    dark_max = config.LINE_DARK_MAX if dark_max is None else dark_max
    if px <= 0:
        return img

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    mask = luma < dark_max

    grown = mask.copy()
    for _ in range(px):
        g = grown
        shifted = g.copy()
        shifted[1:, :] |= g[:-1, :]      # down
        shifted[:-1, :] |= g[1:, :]      # up
        shifted[:, 1:] |= g[:, :-1]      # right
        shifted[:, :-1] |= g[:, 1:]      # left
        grown = shifted

    # Paint the newly-grown ring with the palette's own outline colour, so the
    # lock has nothing to move. PALETTE[1] is the near-black by construction.
    new = grown & ~mask
    out = rgb.copy()
    out[new] = np.array(config.PALETTE[1], dtype=np.uint8)
    return Image.fromarray(out)


def find_ground_line(img: Image.Image, search_from: float = 0.55,
                     min_span: float = 0.5) -> int | None:
    """Locate the horizon line FLUX drew, as a row index, or None.

    A FIXED band height cannot work. The first version used one, and it painted
    over the figure's FEET: the band top sat at 0.82 while the feet landed near
    0.84. Feet land at different heights on different frames -- a wide shot puts
    a small figure lower than a medium shot does -- so the band has to be placed
    relative to the drawing, not to the frame.

    The model reliably draws a long horizontal rule for the ground even when
    the prompt no longer asks for a floor. That rule is the best available
    anchor: the figure is standing ON it, so painting the band from there down
    puts the ground exactly where the character's feet already are.

    Detection is deliberately crude and cheap: in the lower part of the frame,
    find the row with the most dark pixels, and accept it only if it spans at
    least `min_span` of the width -- which a horizon does and a pair of feet,
    a leg or a caption does not.
    """
    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    y0 = int(h * search_from)
    dark = a[y0:, :] < config.LINE_DARK_MAX
    counts = dark.sum(axis=1)
    if counts.size == 0:
        return None
    best = int(counts.argmax())
    if counts[best] < w * min_span:
        return None
    line_y = y0 + best

    # THE FEET HANG BELOW THE HORIZON THE MODEL DREW.
    #
    # Anchoring straight to that rule still clipped the feet: FLUX draws the
    # ground as a line and then puts the shoes ON it, a few pixels lower. So
    # walk down from the rule and take the lowest ink that is plausibly still
    # the figure, and put the band under that. The window is bounded so a
    # caption or a prop lower in the frame cannot drag the floor to the bottom.
    window = min(h, line_y + int(h * config.FLOOR_FOOT_WINDOW))
    below = a[line_y:window, :] < config.LINE_DARK_MAX
    rows_with_ink = np.where(below.any(axis=1))[0]
    if rows_with_ink.size:
        line_y += int(rows_with_ink[-1]) + config.FLOOR_FOOT_MARGIN_PX
    return min(line_y, h - 1)


def draw_floor(img: Image.Image, top: float | None = None,
               colour: tuple | None = None, edge_px: int | None = None
               ) -> Image.Image:
    """Paint the tan floor band across the bottom of the frame.

    A flat rectangle plus a straight black rule along its top edge -- exactly
    what the reference channel has, and exactly what FLUX would not draw.

    The band's top edge is the detected ground line where one exists, so the
    character stands on the floor rather than being buried by it. config.
    FLOOR_TOP is only the fallback for frames with no usable horizon.

    Painted OVER whatever the model put down there: burying its wobbly rule is
    simpler and more reliable than talking it out of drawing one.
    """
    colour = config.FLOOR_COLOR if colour is None else colour
    edge_px = config.FLOOR_EDGE_PX if edge_px is None else edge_px

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    h, w = rgb.shape[:2]

    y = find_ground_line(img) if top is None else None
    if y is None:
        y = int(round(h * (config.FLOOR_TOP if top is None else top)))
    y = max(0, min(h - 1, y))

    rgb[y:, :] = np.array(colour, dtype=np.uint8)
    if edge_px > 0:
        y0 = max(0, y - edge_px // 2)
        rgb[y0:y0 + edge_px, :] = np.array(config.PALETTE[1], dtype=np.uint8)
    return Image.fromarray(rgb)


def apply_all(img: Image.Image) -> Image.Image:
    """The code-owned steps, in order, honouring the config switches.

    Callers still run the palette lock and captions themselves -- this only
    covers the parts that were moved out of the prompt.

    ORDER IS LOAD-BEARING, and each edge below is a real failure if reversed:

      1. restroke      must be FIRST. It resolves every pixel to ink or white,
                       so anything that paints colour has to come after it or
                       that colour gets thresholded as if it were linework.
      2. background    needs the ground line, so it is computed once here and
                       handed to both steps -- calling find_ground_line() twice
                       can give two answers, because step 2 changes the pixels
                       step 3 would measure.
      3. floor band    paints over the bottom, last, so it wins any overlap.

    restroke REPLACES boost_lines rather than following it: both expand the ink
    mask, and stacking them closes up the whites of the eyes (config.py's
    px=4 note is the same effect measured a different way).
    """
    if config.RESTROKE_ON:
        img = restroke(img)
    elif config.LINE_BOOST_ON:
        img = boost_lines(img)

    floor_y = find_ground_line(img) if config.FLOOR_BAND_ON else None

    if config.BACKGROUND_ON:
        img = paint_background(img, floor_y=floor_y)
    if config.FLOOR_BAND_ON:
        img = draw_floor(img, top=None if floor_y is None else floor_y / img.height)
    return img
