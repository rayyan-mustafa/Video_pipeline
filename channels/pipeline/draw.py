"""
Draw the character, the faces and the furniture in Python, instead of asking a
diffusion model for them.

=============================================================================
 WHY THIS EXISTS
=============================================================================
PLAN.md's rule, proven four times now: the parts that look right in all 160
frames are the parts the model does not touch. The palette lock is code. The
captions are code. The floor band and the line weight are code. All four are
flawless. Everything left to the prompt varies.

The character is the most-drawn object in the channel and the one the pipeline
has spent the most money trying to stabilise -- a LoRA that failed, and a drift
measurement campaign. But a stick figure is a circle and five strokes. It is
the single most drawable object there is, and drawing it in code makes the
things that kept failing free and exact:

    uniform stroke width   `width=LW` is uniform BY CONSTRUCTION. Three rounds
                           of prompt work and five phrasings never achieved it.
    flat fills             a fill is a literal RGB tuple.
    palette                colours are palette entries; the lock is a no-op.
    consistency            100%, not 96%. It is the same function call.
    two figures            two calls. "renders one figure" stops existing.

What this CANNOT do is complex organic subjects. A mosquito coded by hand reads
as a generic bug; a deer is impractical. Those stay model-generated and get
pasted in by compositor.py. The split is geometric -> here, organic -> model.

=============================================================================
 WHY THE WOBBLE IS A BEZIER AND NOT JITTER
=============================================================================
The first version perturbed every point on a line independently. That is white
noise, and it reads as a SHAKY, uncertain line -- the exact "amateur sketch"
quality the channel is trying to get away from.

A real marker stroke deviates SMOOTHLY and at low frequency: the hand travels
in a slight arc, it does not vibrate. So a straight line is drawn as a cubic
bezier whose two control points are pushed perpendicular by a small random
amount, and closed shapes are offset by a sum of two slow harmonics along their
own perimeter. Strokes also overshoot their endpoints a little, which is what a
hand does when it starts and stops. Same principle rough.js uses.

Everything is driven by a SEEDED rng, so a frame redrawn with the same seed is
pixel-identical -- the redraw path in Step 24 depends on that.

=============================================================================
 DEFECTS FOUND BY THE STYLE SHEET, AND WHY THAT TOOL EXISTS
=============================================================================
Every one of these was invisible while tuning scenes two figures at a time,
and obvious the moment all six faces sat side by side (make_style_sheet.py):

    brows drawn OUTSIDE the head   at the brow's outer x the skull's edge is
                                   38.8px above centre and the brow sat at
                                   39.4 -- every face grew two black marks
                                   that read as bad hair.
    the worried face was SMILING   with y growing downward a 20..160 sweep
                                   puts the arc's midpoint BELOW its ends.
                                   That is a U. A frown needs 200..340.
    smug's lid struck THROUGH      a chord and an arc were drawn along the
                                   eye's own outline, doubling into a bar.
    open mouths were black blobs   solid fills gave no internal shape; an
                                   outline with a white interior stays
                                   legible at thumbnail size.

Both of the first two were settled with arithmetic rather than by eye. When a
shape looks wrong, compute where it actually lands before nudging constants.
"""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

from . import config

INK = tuple(config.PALETTE[1])
WHITE = tuple(config.PALETTE[0])

# One stroke width for the whole drawing, as a fraction of frame height.
# Uniform line weight is the single strongest signal separating a deliberate
# style from a doodle -- see the STYLE_PREFIX notes in config.py, which spent
# three rounds trying to ask FLUX for this.
LW_FRAC = 0.0104          # ~7.5px at 720p


def lw(h: int) -> int:
    return max(2, round(h * LW_FRAC))


# =============================================================================
#  PRIMITIVES
# =============================================================================

def _bez(p0, c1, c2, p1, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u*u*u*p0[0] + 3*u*u*t*c1[0] + 3*u*t*t*c2[0] + t*t*t*p1[0],
                    u*u*u*p0[1] + 3*u*u*t*c1[1] + 3*u*t*t*c2[1] + t*t*t*p1[1]))
    return out


def stroke(d, rng, p0, p1, w, col=INK, bow=None, over=0.0):
    """A straight line, drawn by hand.

    The wobble is TWO perpendicular control-point offsets on a cubic bezier --
    not per-point jitter, which reads as shaky rather than hand-drawn.
    """
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    if over:
        x0 -= ux * over; y0 -= uy * over
        x1 += ux * over; y1 += uy * over
        dx, dy = x1 - x0, y1 - y0
    px, py = -dy / L, dx / L
    amt = bow if bow is not None else L * 0.020
    o1, o2 = rng.uniform(-amt, amt), rng.uniform(-amt, amt) * 0.7
    d.line(_bez((x0, y0),
                (x0 + dx*.33 + px*o1, y0 + dy*.33 + py*o1),
                (x0 + dx*.66 + px*o2, y0 + dy*.66 + py*o2),
                (x1, y1)), fill=col, width=w, joint="curve")


def hand_poly(d, rng, pts, w, fill=None, col=INK, amp=1.0, closed=True):
    """Draw a point list with a SMOOTH hand-drawn deviation.

    Each point is pushed along its own outward normal by a sum of two slow
    harmonics of its position around the path. Low frequency is the whole
    point: per-vertex noise gives a crinkled edge, this gives a shape that
    is clearly hand-drawn and still confident.
    """
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    a1, a2 = rng.uniform(0, 6.283), rng.uniform(0, 6.283)
    out = []
    for i, (x, y) in enumerate(pts):
        t = 2 * math.pi * i / n
        off = amp * (math.sin(2*t + a1) + 0.6 * math.sin(3*t + a2))
        vx, vy = x - cx, y - cy
        L = math.hypot(vx, vy) or 1.0
        out.append((x + vx/L * off, y + vy/L * off))
    if fill:
        d.polygon(out, fill=fill)
    if col:
        d.line(out + ([out[0]] if closed else []), fill=col, width=w, joint="curve")
    return out


def blob(d, rng, cx, cy, r, w, fill=None, col=INK, ry=None, wobble=0.030, n=64):
    """An organic circle -- a circle a person drew, not one a compass drew."""
    ry = ry if ry is not None else r
    pts = [(cx + r * math.cos(2*math.pi*i/n), cy + ry * math.sin(2*math.pi*i/n))
           for i in range(n)]
    return hand_poly(d, rng, pts, w, fill, col, amp=r * wobble)


def rbox(d, rng, x0, y0, x1, y1, w, fill=None, col=INK, rad=0.0, amp=1.3):
    """A rectangle with optionally rounded corners, drawn by hand.

    Rounded corners matter for furniture: a hard-cornered box reads as a
    diagram, a softly-rounded one reads as an upholstered object. The
    reference channel's furniture is all soft-cornered.
    """
    rad = min(rad, abs(x1-x0)/2, abs(y1-y0)/2)
    if rad <= 0:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        pts = []
        for i in range(4):
            a, b = corners[i], corners[(i+1) % 4]
            for k in range(8):
                pts.append((a[0] + (b[0]-a[0])*k/8, a[1] + (b[1]-a[1])*k/8))
    else:
        pts = []
        for cx_, cy_, a0 in ((x1-rad, y0+rad, -90), (x1-rad, y1-rad, 0),
                             (x0+rad, y1-rad, 90), (x0+rad, y0+rad, 180)):
            for k in range(9):
                a = math.radians(a0 + 90 * k / 8)
                pts.append((cx_ + rad*math.cos(a), cy_ + rad*math.sin(a)))
    return hand_poly(d, rng, pts, w, fill, col, amp=amp)


def arc_stroke(d, rng, cx, cy, r, a0, a1, w, col=INK, ry=None, amp=0.9):
    """An arc -- used for closed eyes, grins, frowns and eyebrows.

    ANGLES ARE SCREEN ANGLES: y grows downward, so 0..180 sweeps through the
    BOTTOM of the ellipse (a U) and 200..340 through the top (an n). Getting
    this backwards is what made the worried face smile.
    """
    ry = ry if ry is not None else r
    n = 18
    pts = [(cx + r*math.cos(math.radians(a0 + (a1-a0)*i/n)),
            cy + ry*math.sin(math.radians(a0 + (a1-a0)*i/n))) for i in range(n+1)]
    off = rng.uniform(-amp, amp)
    pts = [(x, y + off * math.sin(math.pi * i / n)) for i, (x, y) in enumerate(pts)]
    d.line(pts, fill=col, width=w, joint="curve")


# =============================================================================
#  THE FACE  --  the six expressions look.py already assigns
# =============================================================================
#  These are the SAME six names look.py chooses between from the line's lexical
#  energy score, and each one here is a direct reading of that module's clause.
#  Keeping the names aligned means expression assignment keeps working with no
#  change at all -- the clause stops being sent to a model and starts being
#  drawn.
#
#  WHY THE FACE IS NOT JUST TWO DOTS. The old rendered frames put two dots and a
#  line on a small figure and the face was unreadable at thumbnail size. Real
#  eyebrows are what carry an expression: the eyes barely change between rage
#  and worry, the BROWS do all the work. So every mood here draws brows.
# =============================================================================

# =============================================================================
#  CHARACTER FEATURES  --  flags, not cached images
# =============================================================================
#  A "female doctor stickman" is NOT a new asset to store. If she were a cached
#  PNG she would be locked into one pose and one expression, and every frame she
#  appears in would have to reuse that single picture -- which is exactly the
#  rigidity drawing the figure in code was meant to remove.
#
#  So variety is PARAMETERS. hair / beard / glasses / coat / child compose
#  freely with all six expressions and every pose, at any size, for free. The
#  prop library stores objects; characters are always drawn.
# =============================================================================

def _hair_back(d, rng, cx, cy, hr, w, style):
    """The part of the hair that sits BEHIND the skull. Drawn before the head."""
    if style != "long":
        return
    for s in (-1, 1):
        hand_poly(d, rng, [
            (cx + s*hr*0.60, cy - hr*0.74), (cx + s*hr*1.10, cy - hr*0.12),
            (cx + s*hr*1.04, cy + hr*1.28), (cx + s*hr*0.50, cy + hr*1.20),
            (cx + s*hr*0.72, cy + hr*0.18),
        ], max(2, w-1), fill=INK, amp=1.1)


def _hair_front(d, rng, cx, cy, hr, w, style):
    """The part drawn OVER the skull, after the head circle."""
    if style == "spiky":
        for i in range(7):
            a = math.radians(202 + 22.6*i)
            bx, by = cx + hr*0.97*math.cos(a), cy + hr*0.97*math.sin(a)
            stroke(d, rng, (bx, by),
                   (bx + hr*0.30*math.cos(a), by + hr*0.36*math.sin(a)),
                   max(2, w-1))
    elif style == "short":
        # A SHALLOW cap. The first version dipped to -0.30hr in the middle,
        # covering the top half of the skull -- fine alone, but stacked with a
        # beard it left nothing but a slot for the eyes ("uncle", "old man").
        # Hair and beard have to be able to COMPOSE, so each takes less.
        pts = [(cx + hr*1.02*math.cos(math.radians(a)),
                cy + hr*1.02*math.sin(math.radians(a))) for a in range(205, 336, 7)]
        pts += [(cx - hr*0.56, cy - hr*0.74), (cx, cy - hr*0.60),
                (cx + hr*0.60, cy - hr*0.74)]
        hand_poly(d, rng, pts, max(2, w-1), fill=INK, amp=1.0)
    elif style == "bun":
        blob(d, rng, cx, cy - hr*1.14, hr*0.30, max(2, w-1), fill=INK,
             wobble=0.05, n=24)
        pts = [(cx + hr*1.02*math.cos(math.radians(a)),
                cy + hr*1.02*math.sin(math.radians(a))) for a in range(200, 341, 8)]
        pts += [(cx - hr*0.45, cy - hr*0.55), (cx, cy - hr*0.38),
                (cx + hr*0.45, cy - hr*0.55)]
        hand_poly(d, rng, pts, max(2, w-1), fill=INK, amp=1.0)
    elif style == "long":
        pts = [(cx + hr*1.03*math.cos(math.radians(a)),
                cy + hr*1.03*math.sin(math.radians(a))) for a in range(192, 349, 8)]
        pts += [(cx - hr*0.58, cy - hr*0.46), (cx, cy - hr*0.28),
                (cx + hr*0.66, cy - hr*0.48)]
        hand_poly(d, rng, pts, max(2, w-1), fill=INK, amp=1.1)


def _beard(d, rng, cx, cy, hr, w):
    """A filled jaw shape. Drawn after the head and BEFORE the mouth, so the
    mouth reads as sitting inside the beard rather than being buried by it."""
    n = 26
    outer = [(cx + hr*1.01*math.cos(math.radians(162 - 144*i/n)),
              cy + hr*1.01*math.sin(math.radians(162 - 144*i/n)))
             for i in range(n + 1)]
    # THE TOP EDGE IS SET BY THE EYES. At -0.04hr the beard rose above the
    # eyes' lower rim (+0.16hr) and swallowed the bottom half of both, which is
    # why the bearded and "uncle" figures read as a black head with a slot in
    # it. +0.20hr clears the eye by 0.04hr and still covers the whole jaw.
    inner = [(cx + hr*0.90, cy + hr*0.24), (cx + hr*0.44, cy + hr*0.38),
             (cx,           cy + hr*0.32),
             (cx - hr*0.44, cy + hr*0.38), (cx - hr*0.90, cy + hr*0.24)]
    hand_poly(d, rng, outer + inner, max(2, w-1), fill=INK, amp=1.0)


def _glasses(d, rng, cx, cy, hr, w, ex, ey, er):
    """Round frames over the eyes, plus a bridge and two temples."""
    r = er * 1.34
    for s in (-1, 1):
        blob(d, rng, cx + s*ex, cy - ey, r, max(2, w-1), fill=None,
             wobble=0.035, n=30)
    stroke(d, rng, (cx - ex + r, cy - ey), (cx + ex - r, cy - ey), max(2, w-2))
    for s in (-1, 1):
        stroke(d, rng, (cx + s*(ex + r), cy - ey),
               (cx + s*hr*0.99, cy - ey - hr*0.05), max(2, w-2))


def _coat(d, rng, cx, sh, hip, height, w):
    """A lab coat: a white panel over the torso, wider at the hem.

    Drawn over the spine stroke and UNDER the arms, so the limbs still read as
    limbs. White-filled rather than coloured because the palette's whites are
    what the character is already made of.
    """
    bot = hip + height*0.085
    hand_poly(d, rng, [(cx - height*0.062, sh - height*0.012),
                       (cx + height*0.062, sh - height*0.012),
                       (cx + height*0.115, bot), (cx - height*0.115, bot)],
              w, fill=WHITE, amp=1.3)
    stroke(d, rng, (cx, sh), (cx, bot), max(2, w-2), bow=height*0.006)


def face(d, rng, cx, cy, hr, mood="deadpan", w=None,
         hair=None, beard=False, glasses=False):
    w = w or max(2, round(hr * 0.105))
    bw = max(2, round(hr * 0.075))         # brow stroke
    _hair_back(d, rng, cx, cy, hr, w, hair)
    blob(d, rng, cx, cy, hr, w, fill=WHITE, wobble=0.028)
    if beard:
        _beard(d, rng, cx, cy, hr, w)
    # A BLACK MOUTH ON A BLACK BEARD IS AN INVISIBLE MOUTH. The line-drawn
    # moods (deadpan, worried, smug) vanished into the jaw and the bearded
    # faces read as a solid mask with eye-holes. The open moods already work
    # because their WHITE fill carries the shape on its own.
    mcol = WHITE if beard else INK

    ex, ey = hr * 0.40, hr * 0.10          # eye offset from centre
    er = hr * 0.26
    # BROW BASELINE. It was er*2.1 and that put the brows OUTSIDE the head:
    # at the brow's outer x the circle's edge is only 38.8px above centre while
    # the brow sat at 39.4, so every face grew two black marks on top of its
    # skull that read as bad hair. Verified by arithmetic, not by eye.
    by = cy - ey - er * 1.5                # brow baseline

    def eyes(radius=er, pupil=0.42, lid=False):
        for s in (-1, 1):
            blob(d, rng, cx + s*ex, cy - ey, radius, max(2, w-2),
                 fill=WHITE, wobble=0.05, n=26)
            pr = radius * pupil
            d.ellipse([cx + s*ex - pr, cy - ey - pr,
                       cx + s*ex + pr, cy - ey + pr], fill=INK)
            if lid:
                # HEAVY UPPER LID, for smug. The first version drew a chord and
                # then an arc along the SAME path as the eye's own outline, so
                # the two doubled up and read as a black bar struck through the
                # eye. Instead: paint out the top of the eye entirely, redraw
                # only the lower arc, and rule one straight lid line.
                yl = cy - ey - radius * 0.22
                d.rectangle([cx + s*ex - radius*1.6, cy - ey - radius*1.8,
                             cx + s*ex + radius*1.6, yl], fill=WHITE)
                arc_stroke(d, rng, cx + s*ex, cy - ey, radius, 0, 180,
                           max(2, w-1))
                stroke(d, rng, (cx + s*ex - radius, yl),
                       (cx + s*ex + radius, yl), max(2, w-1), bow=radius*0.06)

    def brows(inner_dy, outer_dy):
        for s in (-1, 1):
            stroke(d, rng, (cx + s*(ex - er*0.92), by + inner_dy),
                   (cx + s*(ex + er*0.92), by + outer_dy), bw, bow=hr*0.03)

    # OPEN MOUTHS ARE OUTLINED, NOT FILLED BLACK.
    #
    # The first version drew rage and delight as solid black rounded boxes. On
    # the style sheet all six faces side by side, those read as a black smear
    # rather than a mouth -- there was no internal shape to tell you what you
    # were looking at. An outline with a white interior is what the reference
    # uses and it stays legible at thumbnail size.
    def open_mouth(w_, h_, dy):
        blob(d, rng, cx, cy + hr*dy, hr*w_, max(2, w-1), fill=WHITE,
             ry=hr*h_, wobble=0.014, n=40)

    if mood == "rage":
        eyes()
        for s in (-1, 1):      # angled DOWN toward the nose -> a hard V
            stroke(d, rng, (cx + s*(ex + er*0.95), by - hr*0.05),
                   (cx + s*(ex - er*0.85), by + hr*0.11), bw, bow=hr*0.03)
        open_mouth(0.26, 0.15, 0.47)
    elif mood == "shock":
        eyes(radius=er*1.18, pupil=0.40)
        brows(-hr*0.05, -hr*0.08)                          # lifted, not a hat
        open_mouth(0.14, 0.17, 0.50)
    elif mood == "smug":
        eyes(lid=True)
        stroke(d, rng, (cx + ex - er*0.95, by - hr*0.07),
               (cx + ex + er*0.95, by - hr*0.13), bw, bow=hr*0.03)   # ONE raised
        stroke(d, rng, (cx - ex - er*0.88, by + hr*0.02),
               (cx - ex + er*0.88, by + hr*0.02), bw, bow=hr*0.02)
        arc_stroke(d, rng, cx + hr*0.05, cy + hr*.36, hr*0.24, 200, 340,
                   max(2, w-1), col=mcol, ry=hr*0.16)
    elif mood == "worried":
        eyes()
        for s in (-1, 1):      # inner ends TILTED UP -> the worry shape
            stroke(d, rng, (cx + s*(ex - er*0.85), by - hr*0.06),
                   (cx + s*(ex + er*0.92), by + hr*0.07), bw, bow=hr*0.03)
        # A SINGLE downward curve, not a zigzag -- the zigzag read as a
        # scribble below full-frame size.
        #
        # THE ANGLES WERE INVERTED and the worried face was SMILING. With y
        # growing downward a 20..160 sweep puts the arc's midpoint BELOW its
        # endpoints, which is a U. A frown needs 200..340.
        arc_stroke(d, rng, cx, cy + hr*.42, hr*0.24, 200, 340,
                   max(2, w-1), col=mcol, ry=hr*0.14)
    elif mood == "delight":
        for s in (-1, 1):      # eyes squeezed shut into upward arcs
            arc_stroke(d, rng, cx + s*ex, cy - ey + er*0.35, er*1.0, 200, 340,
                       bw, ry=er*0.8)
        brows(-hr*0.06, -hr*0.10)      # without these the face reads as empty
        open_mouth(0.30, 0.19, 0.44)
    else:                      # deadpan
        eyes()
        brows(0, -hr*0.03)
        stroke(d, rng, (cx - hr*.22, cy + hr*.46), (cx + hr*.22, cy + hr*.46),
               max(2, w-2), col=mcol, bow=hr*0.04)

    # Hair over the skull and frames over the eyes come LAST, so neither is
    # overdrawn by the features underneath them.
    _hair_front(d, rng, cx, cy, hr, w, hair)
    if glasses:
        _glasses(d, rng, cx, cy, hr, w, ex, ey, er)


# =============================================================================
#  THE FIGURE
# =============================================================================

HEAD_FRAC = 0.150          # head RADIUS as a fraction of total height
CHILD_HEAD_FRAC = 0.196    # a child is mostly head -- that IS the read
CHILD_TORSO = 0.300        # and a shorter body under it (adult 0.36)
#
# MEASURED OFF THE REFERENCE, not guessed. In the reference frames the head is
# roughly a THIRD of the figure's height -- diameter/height ~0.30, so radius
# ~0.15. An earlier pass used 0.115 and the result read as a spider: the head
# is what carries the expression, and the whole style depends on it being the
# dominant shape. The limbs stay thin against it, which is the contrast that
# makes the drawing read as a character rather than a diagram.

ARMS = {
    "stand":  ((-.170, .250), (.170, .250)),
    "point":  ((-.165, .245), (.245, .020)),
    "up":     ((-.165, .240), (.080, -.300)),
    "shrug":  ((-.205, -.055), (.205, -.055)),
    "reach":  ((-.150, .235), (.270, -.080)),
    "walk":   ((-.150, .240), (.185, .185)),
    # SEATED ARMS HANG, they do not splay. At the standing spread (34 deg off
    # vertical) a seated figure's arms shot out sideways over the furniture and
    # were the last thing making it read as standing-behind rather than
    # sitting-in. The reference hangs them ~17 deg off vertical, close to the
    # body, ending near the hip.
    "sit":    ((-.078, .255), (.078, .255)),
}

# SEATED ARM VARIANTS. `sit` above hangs both arms by the hips, which is right
# for a dining chair but wrong for a sofa: in the reference frame the near arm
# is stretched ALONG THE TOP OF THE BACKREST, roughly horizontal, and that one
# stroke is most of what makes the figure read as settled into the sofa rather
# than perched in front of it.
#
# Offsets are from the SHOULDER, in units of figure height, same as ARMS.
# `drape` deliberately runs slightly UPWARD (negative y) because the backrest
# top sits above the seated shoulder line.
SEATED_ARMS = {
    "rest":  ((-.078, .255), (.078, .255)),
    "drape": ((-.078, .255), (.300, -.115)),
    "lap":   ((-.135, .175), (.135, .175)),
}


def figure(d, rng, cx, feet_y, height, pose="stand", mood="deadpan", w=None,
           hair=None, beard=False, glasses=False, coat=False, child=False):
    """Draw a standing/acting figure whose feet land on feet_y.

    Every character variant the channel needs is a combination of these flags:
    a female doctor is hair="long", coat=True; an uncle is beard=True; a kid is
    child=True. None of them is a stored asset, so all of them work in all six
    expressions and every pose.
    """
    w = w or lw(int(height / 0.52))
    hr = height * (CHILD_HEAD_FRAC if child else HEAD_FRAC)
    hcy = feet_y - height + hr
    neck = hcy + hr * 1.02
    hip = neck + height * (CHILD_TORSO if child else 0.36)
    sh = neck + height * 0.05
    o = w * 0.3

    stroke(d, rng, (cx, neck), (cx, hip), w, over=o)
    if coat:
        _coat(d, rng, cx, sh, hip, height, w)
    for ax, ay in ARMS.get(pose, ARMS["stand"]):
        stroke(d, rng, (cx, sh), (cx + height*ax, sh + height*ay), w, over=o)

    def foot(fx, s):
        # A short horizontal foot. The reference figures all have one, and
        # without it the legs read as sticks stopping in mid-air rather than
        # as a person standing ON the floor line.
        stroke(d, rng, (fx, feet_y), (fx + s*height*.038, feet_y), w, over=o)

    if pose == "walk":
        stroke(d, rng, (cx, hip), (cx - height*.150, feet_y), w, over=o)
        knee = (cx + height*.075, hip + height*.16)
        stroke(d, rng, (cx, hip), knee, w, over=o)
        stroke(d, rng, knee, (cx + height*.150, feet_y), w, over=o)
        foot(cx - height*.150, -1)
        foot(cx + height*.150, 1)
    else:
        for s in (-1, 1):
            fx = cx + s * height*.105
            stroke(d, rng, (cx, hip), (fx, feet_y), w, over=o)
            foot(fx, s)

    face(d, rng, cx, hcy, hr, mood, w, hair, beard, glasses)


def figure_seated(d, rng, cx, seat_y, floor_y, height, mood="deadpan", w=None,
                  hair=None, beard=False, glasses=False, coat=False,
                  child=False, arms="rest", back_top=None):
    """A figure SITTING on a surface at seat_y, feet reaching floor_y.

    THE Z-ORDER LESSON, AND IT WAS A CONCEPTUAL ERROR NOT A TUNING ONE.
    An earlier version drew the seated figure BEHIND the furniture and showed
    only two shin stubs below the front edge. Nothing about it read as sitting
    -- it looked like a head parked on top of a box.

    The reference channel draws the whole figure IN FRONT of the seat: the
    thigh runs forward across the cushion, the knee is visible, and the shin
    drops to the floor. The bent leg IS the entire visual evidence of sitting,
    so hiding it removes the only cue there was. Draw the furniture first,
    then the figure complete and on top.

    `arms` picks from SEATED_ARMS. "rest" hangs both by the hips and is right
    for a dining chair. "drape" stretches the near arm along the backrest and
    is what the sofa reference actually does -- see the note on SEATED_ARMS.

    `back_top`, when given, is the y of the backrest top: the draped hand is
    snapped ONTO it instead of floating at whatever the offset produced. The
    arm has to land on the furniture or the pose reads as a shrug, and the
    backrest height varies per prop, so this cannot be a constant.
    """
    w = w or lw(int(height / 0.52))
    hr = height * (CHILD_HEAD_FRAC if child else HEAD_FRAC)
    hip_y = seat_y - height * 0.03
    neck = hip_y - height * (0.285 if child else 0.335)
    hcy = neck - hr * 1.02
    sh = neck + height * 0.05
    o = w * 0.3

    stroke(d, rng, (cx, neck), (cx, hip_y), w, over=o)
    if coat:
        _coat(d, rng, cx, sh, hip_y, height, w)
    for i, (ax, ay) in enumerate(SEATED_ARMS.get(arms, SEATED_ARMS["rest"])):
        hx, hy = cx + height*ax, sh + height*ay
        # Snap the draped hand onto the backrest top. Without this the arm
        # ends in mid-air a few px above or below the cushion, which reads as
        # a raised arm rather than an arm resting on something.
        if arms == "drape" and i == 1 and back_top is not None:
            hy = back_top + height * 0.012
        stroke(d, rng, (cx, sh), (hx, hy), w, over=o)

    # THE LEGS ARE SYMMETRIC, and getting this wrong is what made three earlier
    # passes look like a bundle of sticks rather than a person.
    #
    # The failed version pushed BOTH knees the same direction (a "forward"
    # offset borrowed from the oblique projection). On a front-facing figure
    # that is simply wrong: the viewer sees both legs splayed evenly either
    # side of the body. The reference is unambiguous -- thigh down and slightly
    # OUT, knee, shin straight DOWN, a short foot mark.
    for s in (-1, 1):
        knee = (cx + s * height*.085, hip_y + height*.125)
        foot = (cx + s * height*.105, floor_y)
        stroke(d, rng, (cx + s * height*.025, hip_y), knee, w, over=o)
        stroke(d, rng, knee, foot, w, over=o)
        stroke(d, rng, foot, (foot[0] + s * height*.038, floor_y), w, over=o)

    face(d, rng, cx, hcy, hr, mood, w, hair, beard, glasses)


# =============================================================================
#  FURNITURE IS DRAWN IN 3/4 OBLIQUE, NOT FLAT ON
# =============================================================================
#  A flat front-on rectangle reads as a DIAGRAM. The reference channel draws
#  every piece of furniture in a simple oblique projection -- the top surface
#  recedes back along a fixed offset vector, and the receding faces are filled
#  a shade darker than the top. That one trick is most of what separates their
#  furniture from clip art, and it costs nothing: it is the same polygons with
#  a constant offset added.
#
#  The projection is deliberately NAIVE -- one fixed offset for the whole
#  frame, no vanishing point. A true perspective would be more correct and
#  would look wrong, because the reference is not perspective-correct either.
# =============================================================================

def _depth(width, height):
    """The offset vector that 'going back one unit' adds. Up and to the right.

    MEASURED off the reference frames rather than chosen: the receding edge
    moves about 0.23 of the object's width sideways and about 0.10 of its
    height upward. An earlier guess used 0.16 for the rise, which tipped
    furniture too far and made every table look like it was sliding.
    """
    return width * 0.23, -height * 0.10


def chair(d, rng, x, y, width, height, colour=None, dark=None, w=None):
    """A four-legged chair in 3/4 oblique, sized so a figure can sit on it.

    `y` is the FLOOR line the front legs stand on. Returns the seat surface and
    front-edge coordinates so figure_seated() can be placed onto it.

    Drawn back-to-front: back legs, backrest, seat slab, front legs -- so each
    piece correctly occludes what is behind it.
    """
    colour = colour or tuple(config.PALETTE[4])
    dark = dark or tuple(config.PALETTE[5])
    w = w or lw(int(height / 0.62))
    dx, dy = _depth(width, height)

    # PROPORTIONS ARE MEASURED OFF THE REFERENCE, and the first pass got them
    # badly wrong: the backrest was taller than the seat height and as wide as
    # the seat, which drew a brown wall the figure sat in front of. In the
    # reference the seat is only about as wide as the figure's HEAD, and the
    # backrest is roughly as tall as the seat is off the floor.
    seat_y = y - height * 0.50          # top surface of the seat, front edge
    thick = height * 0.06
    leg_w = width * 0.095
    # BACKREST HEIGHT IS SET BY THE SEATED HEAD, not by the chair's own looks.
    # At 0.50 the backrest top rose ABOVE the seated figure's chin, so the head
    # sat ON the brown panel and you could not tell whether the figure was in
    # front of the chair or behind it. 0.42 leaves about 9px of neck visible
    # between the two, which is what makes the overlap legible. Computed from
    # figure_seated()'s own proportions rather than nudged by eye.
    back_h = height * 0.42
    back_th = width * 0.07              # the backrest is a SLAB, it has depth

    for fx in (0.0, 1.0):               # back legs, furthest away
        lx = x + width * fx - leg_w/2 + dx
        rbox(d, rng, lx, seat_y + dy, lx + leg_w, y + dy, w, dark, rad=leg_w*.25)

    bx0, bx1 = x + dx, x + width + dx
    hand_poly(d, rng, [(bx1, seat_y + dy), (bx1 + back_th, seat_y + dy - back_th*.5),
                       (bx1 + back_th, seat_y + dy - back_h - back_th*.5),
                       (bx1, seat_y + dy - back_h)], w, dark, amp=1.2)
    hand_poly(d, rng, [(bx0, seat_y + dy), (bx1, seat_y + dy),
                       (bx1, seat_y + dy - back_h),
                       (bx0, seat_y + dy - back_h)], w, colour, amp=1.4)

    hand_poly(d, rng, [(x, seat_y), (x + width, seat_y),
                       (bx1, seat_y + dy), (bx0, seat_y + dy)], w, colour, amp=1.4)
    hand_poly(d, rng, [(x, seat_y), (x + width, seat_y),
                       (x + width, seat_y + thick), (x, seat_y + thick)],
              w, dark, amp=1.4)

    for fx in (0.0, 1.0):               # front legs, nearest
        lx = x + width * fx - leg_w/2
        rbox(d, rng, lx, seat_y + thick*.6, lx + leg_w, y, w, colour, rad=leg_w*.25)

    return {"seat_y": seat_y, "front_y": seat_y + thick,
            "cx": x + width*0.5, "floor_y": y}


def table(d, rng, x, y, width, height, colour=None, dark=None, w=None):
    """A table/desk in the same oblique. Same construction as the chair seat."""
    colour = colour or tuple(config.PALETTE[4])
    dark = dark or tuple(config.PALETTE[5])
    w = w or lw(int(height / 0.42))
    dx, dy = _depth(width, height)
    top_y = y - height
    thick = height * 0.11
    leg_w = width * 0.07

    for fx in (0.03, 0.97):
        lx = x + width*fx - leg_w/2 + dx
        rbox(d, rng, lx, top_y + dy, lx + leg_w, y + dy, w, dark, rad=leg_w*.25)
    hand_poly(d, rng, [(x, top_y), (x + width, top_y),
                       (x + width + dx, top_y + dy), (x + dx, top_y + dy)],
              w, colour, amp=1.5)
    hand_poly(d, rng, [(x, top_y), (x + width, top_y),
                       (x + width, top_y + thick), (x, top_y + thick)],
              w, dark, amp=1.5)
    for fx in (0.03, 0.97):
        lx = x + width*fx - leg_w/2
        rbox(d, rng, lx, top_y + thick*.7, lx + leg_w, y, w, colour, rad=leg_w*.25)
    return {"top_y": top_y, "front_y": top_y + thick}


def sofa(d, rng, x, y, width, height, colour, w=None):
    """A soft-cornered upholstered sofa.

    Built as five rounded boxes rather than one outline, because the seams
    between back cushions, seat cushions and arms are what make it read as
    upholstery instead of a crate.

    Drawn BACK TO FRONT so the arms overlap the seat, and returns the seat
    surface / front-edge y values so a seated figure can be placed on it.
    """
    w = w or lw(int(height / 0.30))
    r = height * 0.13
    back_b = y + height * 0.62
    seat_t = y + height * 0.50
    seat_b = y + height * 0.86
    base_b = y + height

    rbox(d, rng, x + width*.085, y, x + width*.915, back_b, w, colour, rad=r)
    for f in (.355, .645):     # back cushion seams
        stroke(d, rng, (x + width*f, y + height*.06),
               (x + width*f, back_b - height*.05), max(2, w-2), bow=height*.012)

    rbox(d, rng, x + width*.085, seat_t, x + width*.915, seat_b, w, colour, rad=r*.8)
    for f in (.355, .645):     # seat cushion seams
        stroke(d, rng, (x + width*f, seat_t + height*.04),
               (x + width*f, seat_b - height*.03), max(2, w-2), bow=height*.010)

    # ARM CORNER RADIUS IS HALVED. At the full body radius the arms were
    # narrower than 2*rad, so they rounded into capsules that read as two
    # teal pills parked beside the sofa rather than as part of it.
    rbox(d, rng, x, y + height*.26, x + width*.135, base_b, w, colour, rad=r*.45)
    rbox(d, rng, x + width*.865, y + height*.26, x + width, base_b, w, colour,
         rad=r*.45)

    for f in (.10, .86):       # short legs, so it sits ON the floor not IN it
        rbox(d, rng, x + width*f, base_b - height*.02,
             x + width*(f+.04), base_b + height*.09, max(2, w-2),
             tuple(config.PALETTE[5]), rad=height*.015)

    return {"seat_y": seat_t, "front_y": seat_b, "back_top": y}


def make_frame(size=None, wall=None, floor=None, floor_y=0.76, seed=42):
    """A background plus a ready draw context and seeded rng."""
    size = size or (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    wall = wall or tuple(config.PALETTE[10])
    floor = floor or tuple(config.PALETTE[5])
    img = Image.new("RGB", size, wall)
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    fy = int(size[1] * floor_y)
    d.rectangle([0, fy, size[0], size[1]], fill=floor)
    stroke(d, rng, (-10, fy), (size[0] + 10, fy), lw(size[1]), bow=size[1]*0.004)
    return img, d, rng, fy
