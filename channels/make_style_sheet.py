"""
Render EVERY face, pose and prop draw.py can produce onto one page.

Why this exists: the drawing engine was being tuned one scene at a time, by
eye, and each round only showed two or three figures. Defects that were
obvious across the whole set took several rounds each to notice, because no
single test frame showed them side by side:

    - eyebrows drawn outside the head, on every one of the six faces
    - the "worried" mouth curving upward, i.e. smiling
    - smug's eyelid drawn as a black bar struck through the eye
    - feet missing from every pose
    - "up" and "reach" almost indistinguishable from "stand"

Every one of those was found in a single glance at this sheet, after surviving
several rounds of scene-by-scene tuning. It is the same lesson the pipeline
already wrote down about renders:

    "Never judge a visual change from prompt text or summary statistics. Draw
     a small batch and look at it."

One page, every variant, one review. Run it after any change to draw.py.

    python make_style_sheet.py [out.png]
"""

from __future__ import annotations

import random
import sys

from PIL import Image, ImageDraw, ImageFont

from pipeline import config
from pipeline import draw as D

MOODS = ["deadpan", "worried", "rage", "shock", "smug", "delight"]
POSES = ["stand", "walk", "point", "up", "shrug", "reach"]

WALL = (255, 255, 255)
FLOOR = tuple(config.PALETTE[2])
LABEL = (90, 90, 90)


def _font(size):
    for name in ("seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(d, x, y, text, font, colour=LABEL):
    d.text((x, y), text, fill=colour, font=font, anchor="ma")


def build() -> Image.Image:
    f_small = _font(20)
    f_head = _font(30)

    CW, CH = 250, 300          # face cell
    PW, PH = 210, 380          # pose cell
    W = max(len(MOODS) * CW, len(POSES) * PW) + 60
    H = 70 + CH + 80 + PH + 80 + 430 + 40

    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    sd = ImageDraw.Draw(sheet)
    sd.text((30, 24), "draw.py style sheet", fill=(30, 30, 30), font=f_head)

    # ---- faces -------------------------------------------------------------
    y0 = 74
    sd.text((30, y0), "EXPRESSIONS  (the six look.py assigns)",
            fill=(30, 30, 30), font=f_small)
    y0 += 30
    for i, mood in enumerate(MOODS):
        cell = Image.new("RGB", (CW, CH), WALL)
        cd = ImageDraw.Draw(cell)
        rng = random.Random(100 + i)
        D.face(cd, rng, CW // 2, CH // 2 - 10, CH * 0.30, mood)
        sheet.paste(cell, (30 + i * CW, y0))
        _label(sd, 30 + i * CW + CW // 2, y0 + CH + 6, mood, f_small)

    # ---- poses -------------------------------------------------------------
    y1 = y0 + CH + 44
    sd.text((30, y1), "POSES  (figure(), deadpan face, feet on the floor line)",
            fill=(30, 30, 30), font=f_small)
    y1 += 30
    for i, pose in enumerate(POSES):
        cell = Image.new("RGB", (PW, PH), WALL)
        cd = ImageDraw.Draw(cell)
        rng = random.Random(200 + i)
        fy = int(PH * 0.86)
        cd.rectangle([0, fy, PW, PH], fill=FLOOR)
        D.stroke(cd, rng, (-5, fy), (PW + 5, fy), D.lw(PH), bow=2.0)
        D.figure(cd, rng, PW // 2, fy, PH * 0.72, pose, "deadpan")
        sheet.paste(cell, (30 + i * PW, y1))
        _label(sd, 30 + i * PW + PW // 2, y1 + PH + 6, pose, f_small)

    # ---- props -------------------------------------------------------------
    y2 = y1 + PH + 44
    sd.text((30, y2), "PROPS  (3/4 oblique, two-tone, with a figure for scale)",
            fill=(30, 30, 30), font=f_small)
    y2 += 30
    PROP_H = 400
    cell = Image.new("RGB", (W - 60, PROP_H), WALL)
    cd = ImageDraw.Draw(cell)
    rng = random.Random(300)
    fy = int(PROP_H * 0.88)
    cd.rectangle([0, fy, W - 60, PROP_H], fill=FLOOR)
    D.stroke(cd, rng, (-5, fy), (W - 55, fy), D.lw(PROP_H), bow=2.5)

    FH = PROP_H * 0.66
    ch = D.chair(cd, rng, 120, fy, FH * 0.28, FH * 0.64)
    D.figure_seated(cd, rng, ch["cx"] - 6, ch["seat_y"], fy, FH, "worried")
    _label(sd, 30 + 175, y2 + PROP_H + 6, "chair + figure_seated", f_small)

    D.table(cd, rng, 380, fy, FH * 0.42, FH * 0.38)
    _label(sd, 30 + 450, y2 + PROP_H + 6, "table", f_small)

    sf = D.sofa(cd, rng, 640, fy - FH * 0.52, FH * 1.05, FH * 0.52,
                tuple(config.PALETTE[7]))
    D.figure_seated(cd, rng, 640 + FH * 0.28, sf["seat_y"], fy, FH, "deadpan")
    D.figure_seated(cd, rng, 640 + FH * 0.72, sf["seat_y"], fy, FH, "smug")
    _label(sd, 30 + 900, y2 + PROP_H + 6, "sofa + two seated", f_small)

    sheet.paste(cell, (30, y2))
    return sheet


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "style_sheet.png"
    img = build()
    img.save(out)
    print(f"wrote {out}  {img.size[0]}x{img.size[1]}")
