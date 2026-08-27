# 100 prompts for the OBJECT LoRA (SDXL)

This trains the model that fills the **prop library**. It is not the abandoned
character LoRA and it is not trying to be — the character is drawn by
`pipeline/draw.py` now, in code, and never comes from a model again.

---

## Read this first — what changed, and why this one should work

The v1 character LoRA failed on 23 images / 500 steps against FLUX. Four things
are different here, and three of them are structural rather than "more effort":

1. **The job got much easier.** This LoRA only has to learn *one object, alone,
   on white, in a flat outlined style*. It does not have to learn a character,
   hold it consistent across 160 frames, or compose a scene. Those were the
   hard parts and they are now code.
2. **The base model has working CFG.** SDXL honours negative prompts. Every
   defect three rounds of FLUX prompt work could not fix — shading on the
   brain, texture, photographic detail — is a negative prompt away. FLUX.1
   -schnell is guidance-distilled and simply cannot do this.
3. **SDXL's weaknesses do not apply.** SDXL is bad at multi-subject scenes,
   attribute binding and spatial relations. A single object on a white square
   involves none of them. This is its easy case.
4. **A review gate stands between the model and the library.** Output does not
   have to be right every time; it has to be right *once per object*, and a
   bad generation is discarded before it can be cached.

### Why an object library at all

Not mainly cost. **Consistency.** Generate a lung fresh every time and video 3's
lung is a different lung from video 40's — the exact drift this project already
paid to eliminate, returning through the props. A cached lung is the same lung
forever. Same argument as the palette lock and the PIL captions: the parts that
look right in all 160 frames are the parts a model touches **once**.

Review cost amortises too. You look at a prop once, not at 160 frames per video.

### The rules

- **Paste `STYLE` verbatim in front of every prompt.** Do not reword it, not
  once. It is the thing being learned.
- **NO PEOPLE. NO STICK FIGURES. In any image, ever.** This is the single most
  important rule on the page. If figures appear in the training set the LoRA
  learns "this style contains a stick figure" and will start adding one to
  every prop you generate afterwards. The character comes from `draw.py`.
- **No background, no floor, no shadow.** Plain white. The scene's background
  is painted in code; a prop that carries its own floor cannot be composited.
- **No text, no labels, no lettering.** Captions are PIL's job.
- **One object per image**, centred, filling most of the frame.
- **Square 1024×1024.** SDXL trains natively at 1024, so unlike the FLUX run
  there is no padding waste at all.
- **Discard and regenerate anything wrong.** 100 *good* images, not 100
  attempts. A bad image teaches a bad style, and here it teaches it to every
  prop you will ever generate.

### STYLE — prefix every prompt with exactly this

```text
flatprop style, bold minimal flat vector illustration, one single object alone,
very thick uniform black outline of even width, flat solid colour fill with no
shading and no gradient and no texture, simple naive 3/4 view, plain pure white
background, no floor, no ground, no shadow, no people, no text, no letters,
no watermark,
```

Then append one line from the list below.

---

## A. Organs and body parts (1–22)

The channel's core subject matter. These are the ones with no substitute — a
lung has to look like a lung.

1. a pair of human lungs, soft pink-red fill
2. a human heart, deep red fill
3. a human brain seen from the side, pale pink fill
4. a human stomach, pink fill
5. a single kidney, dark red fill
6. a human liver, deep red-brown fill
7. a human eyeball with a blue iris
8. a human outer ear, skin tone fill
9. a human nose in profile, skin tone fill
10. a single molar tooth, white fill
11. a human skull facing forward, off-white fill
12. a human spine seen from the side, off-white fill
13. a human rib cage, off-white fill
14. a human hand open palm forward, skin tone fill
15. a human foot in profile, skin tone fill
16. a single long bone, off-white fill
17. a bundle of muscle fibres, dark red fill
18. a single neuron with branching dendrites, pale yellow fill
19. a red blood cell, flat disc shape, red fill
20. a cross section of skin showing three flat layers
21. a single hair follicle in cross section
22. a human tongue, pink fill

## B. Medical and clinical (23–38)

23. an MRI scanner, a large ring machine with a sliding bed, pale teal fill
24. a hospital bed with a headboard and side rail, pale teal fill
25. a stethoscope, dark grey fill
26. a medical syringe, clear barrel with grey plunger
27. a clinical thermometer, white with a grey tip
28. an amber pill bottle with a white cap
29. a single capsule pill, half red half white
30. a rolled bandage, white fill
31. a laboratory microscope, dark grey fill
32. a wheelchair seen from the side, grey and black fill
33. an IV drip stand with a clear bag hanging
34. a clipboard with a blank sheet of paper
35. a flat bathroom scale, white fill
36. a blood pressure cuff with a squeeze bulb
37. a petri dish seen from above, pale green fill
38. a chest x-ray film, dark blue-grey fill

## C. Furniture and room objects (39–52)

Geometric enough that `draw.py` may cover some of these later. They are here so
the LoRA learns the 3/4 oblique on simple boxes, which is what makes everything
else in the set sit correctly.

39. a three-seat sofa in 3/4 view, teal fill
40. a wooden dining chair in 3/4 view, warm brown fill
41. a writing desk in 3/4 view, warm brown fill
42. a single bed in 3/4 view, teal blanket and white pillow
43. a table lamp with a cone shade, mustard fill
44. a closed wooden door with a round handle
45. a window frame with four panes, pale blue glass
46. a bookshelf with three shelves of books
47. a tall standing mirror in a wooden frame
48. a wardrobe with two doors, warm brown fill
49. a round wooden stool with three legs
50. a round wall clock with two hands, white face
51. a flat screen television on a stand, dark grey fill
52. a rectangular floor rug seen in 3/4 view, rust red fill

## D. Everyday objects (53–70)

53. a ceramic mug with a handle, white fill
54. a drinking glass of water, pale blue fill
55. a dinner plate seen in 3/4 view, white fill
56. a metal spoon, grey fill
57. a closed hardback book, dark blue cover
58. a smartphone seen flat on, dark grey body
59. an open laptop computer, silver-grey fill
60. a backpack with two straps, dark green fill
61. a closed cardboard box, light brown fill
62. an old fashioned door key, brass yellow fill
63. a closed umbrella with a curved handle, navy fill
64. a single trainer shoe in profile, white and grey
65. a flat cap hat, dark grey fill
66. a folded bath towel, pale blue fill
67. a rectangular bar of soap, cream fill
68. a toothbrush, white handle with blue bristles
69. a lit candle in a holder, cream wax
70. a plastic water bottle with a blue cap

## E. Food (71–80)

71. a whole red apple with a short stem
72. a lemon cut in half showing the segments
73. a loaf of bread, golden brown crust
74. a bowl of hot soup with steam lines, green bowl
75. an ice cream cone with one white scoop
76. a red chilli pepper with a green stem
77. a brown onion with papery skin
78. a single white egg
79. a wedge of yellow cheese with holes
80. a banana with the skin on

## F. Nature and animals (81–90)

The hardest group and the reason this LoRA exists at all — hand-coding a
mosquito in PIL produced a generic bug blob. Expect to regenerate these most.

81. a broad leafy tree with a brown trunk, green canopy
82. a single green leaf with visible veins
83. a simple flower with five petals, yellow centre
84. a mosquito with long legs and translucent wings, grey-blue body
85. a housefly with translucent wings, dark grey body
86. a sitting dog seen from the side, tan fill
87. a sitting cat seen from the side, grey fill
88. a small bird perched, blue fill
89. a fish seen from the side, orange fill
90. a small mouse seen from the side, grey fill

## G. Symbolic and abstract (91–100)

For the `symbolic-metaphor` shot type, which is ~18-24% of real frames.

91. a thick arrow pointing upward, rust red fill
92. a large question mark, dark blue fill
93. a large exclamation mark, rust red fill
94. a glowing light bulb, yellow fill
95. a gear wheel with eight teeth, grey fill
96. a magnifying glass with a round lens and a handle
97. a warning triangle with a thick black border, yellow fill
98. a simple bar chart with four bars of different heights
99. an empty rounded thought bubble, white fill
100. three curved sound wave lines radiating outward

---

## After you have the 100 images

### Training — SDXL, not FLUX

| setting | value | why |
| --- | --- | --- |
| base | `stabilityai/stable-diffusion-xl-base-1.0` | OpenRAIL++-M, commercial use fine |
| resolution | `1024` | SDXL's native size. No padding waste |
| rank | `16` | drop to 8 if it overfits, raise to 32 if the style is weak |
| learning rate | `1e-4` | UNet |
| text encoder | **off** | you are teaching a look, not new vocabulary. Training it on 100 images is the fastest route to overfitting |
| max steps | `2000–2500` | 100 images × 20-25 passes, the normal range |
| batch / accum | `1` / `4` | fits an L4 comfortably |
| trigger | `flatprop style` | already the first token of every caption above |

**Check `‖B‖` movement between checkpoints before drawing anything.** Under ~10%
over the last few hundred steps means it converged. The v1 run was still moving
71% and that was the tell it had learned noise. This check is cheap and it is
the one that would have saved the previous run.

### ⚠️ Turn SDXL's watermarker off

`StableDiffusionXLPipeline` applies an **invisible watermark by default** when
the `invisible-watermark` package is installed:

```python
pipe = StableDiffusionXLPipeline.from_pretrained(..., add_watermarker=False)
```

Miss this and you ship watermarked props while believing the local-weights path
is clean. This is the one real gotcha in moving off FLUX.

### The negative prompt — this is the whole reason for SDXL

FLUX.1-schnell is guidance-distilled: negative prompts do nothing, which is why
`config.py` records five phrasings failing on shading and line weight. On SDXL
they work. Use this at inference for every prop:

```text
shading, gradient, soft shading, texture, crosshatch, photorealistic, 3d render,
pencil sketch, drop shadow, cast shadow, background, floor, ground, people,
person, stick figure, hands, text, letters, watermark, signature, blurry,
low contrast, thin lines
```

`stick figure` and `person` are in there deliberately — a prop that arrives with
a figure attached cannot be composited.

### Then into the prop store

1. Generate → **review** → admit. Nothing enters the library unreviewed; a bad
   prop cached once appears in forty videos.
2. `props/<key>.png` with alpha, plus `props/manifest.json` carrying the ground
   anchor, nominal height relative to the figure, and `reviewed: true`.
3. `lock_palette()` runs on admission, so every prop is on-palette before it is
   ever pasted.
