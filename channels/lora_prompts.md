# 100 prompts for the character LoRA training set

Generate these with Nano Banana (or whatever tool you verified matches the
reference channel). One image per prompt, **100 images**.

---

## Read this first — it decides whether the LoRA works

The previous LoRA failed on **23 images / 500 steps** and learned noise, not a
character. The three things that were wrong are all fixed by how you use this
list:

1. **Volume.** 100 images, not 23.
2. **Variety.** 100 near-identical poses teaches nothing. The list below varies
   pose, action, prop, framing and expression on purpose.
3. **An identical style block.** The LoRA learns whatever is *constant* across
   the set. If the style wording drifts between images, the constant thing
   becomes mush — which is exactly what happened last time.

### The rules

- **Paste `STYLE` verbatim in front of every single prompt.** Do not reword it,
  not even once. It is the thing being learned.
- **No floor, no ground, no background.** Plain white. `pipeline/finish.py`
  paints the tan floor band in code now, deterministically, on every frame. A
  LoRA that also draws floors wastes capacity on something we overwrite, and
  reintroduces the per-frame inconsistency we just removed.
- **No text, no captions, no lettering.** Captions are drawn by PIL. Same reason.
- **Square output, 1024×1024. NOT 16:9.** This is deliberate, even though the
  pipeline draws 1280×720:
  - training runs at `--resolution 768`, i.e. 768×768, so anything non-square is
    padded or cropped before the model sees it anyway;
  - a 16:9 source padded to square is ~44% blank white bars — capacity spent on
    nothing;
  - a square source gives the character far more pixels, so the LoRA learns the
    **linework** at higher effective resolution, which is the whole problem we
    are trying to fix;
  - it also biases toward a larger figure at inference, which we want — small
    figures are what produced the unreadable dot-eye faces.
  The aspect mismatch does not matter: a style/character LoRA learns what the
  character *looks like*, not where it sits in frame. Composition still comes
  from the shot clauses in `shots.py`.
  (Bonus: Step 21d's prep pads to square rather than centre-cropping — because
  cropping clipped the figure on 4 of the original 23. Generate square and that
  padding becomes a no-op.)
- **Same character every time.** Generate #1 first, keep it, and use it as a
  style/character reference for the rest if your tool supports referencing.
  Consistency across the set is the whole point.
- **Discard bad generations and regenerate.** A wrong-looking figure in the
  training set teaches the wrong figure. Aim for 100 *good* images, not 100
  attempts.

### STYLE — prefix every prompt with exactly this

```text
bold minimal stick figure cartoon, very thick uniform black marker outlines of
even width, a plain bald circle head, one single straight vertical line for the
body, straight line arms and legs, flat solid colour fills with no shading and
no texture, plain pure white background, no floor, no ground, no shadow, no
text, no letters, no watermark,
```

Then append one line from the list below.

---

## A. One figure, body actions (1–30)

Core of the set. These are the `character-moment` shots — 56% of real frames.

1. the figure standing straight facing forward, blank deadpan face
2. the figure touching its own forearm with one hand, deadpan
3. the figure pressing one hand flat on its own chest, worried face
4. the figure pointing at its own head with one finger, smug face
5. the figure touching the back of its own neck, worried face
6. the figure rubbing its own eye, deadpan
7. the figure covering its own mouth with one hand, shocked face
8. the figure holding both hands over its ears, enraged face
9. the figure shrugging with both arms out and palms up, deadpan
10. the figure scratching the top of its head, worried face
11. the figure crossing both arms over its chest, smug face
12. the figure raising one arm straight up, delighted face
13. the figure pointing forward at the viewer, enraged face
14. the figure bending forward slightly holding its own stomach, worried face
15. the figure standing with both arms hanging loose, deadpan
16. the figure tilting its head to one side, worried face
17. the figure holding one arm out to the side, palm forward, deadpan
18. the figure touching its own knee, bending down slightly, deadpan
19. the figure standing on one leg, other leg lifted, delighted face
20. the figure leaning back away from something, shocked face
21. the figure stepping forward with one leg, delighted face
22. the figure sitting cross-legged on nothing, deadpan
23. the figure lying flat on its back, arms at sides, deadpan
24. the figure crouching low with knees bent, worried face
25. the figure jumping with both arms and legs spread, delighted face
26. the figure running to the side, legs mid-stride, shocked face
27. the figure walking slowly, one arm swinging, deadpan
28. the figure turning to look over its own shoulder, worried face
29. the figure standing with hands on hips, smug face
30. the figure stretching both arms above its head, delighted face

## B. One figure, close on the face (31–45)

Teaches the expression range at a size where the eyes and brows are large.
These are what fix the dot-eyes on wide shots.

31. head and shoulders only, two big round white eyes with large black pupils, blank deadpan stare
32. head and shoulders only, eyes squeezed shut into thick happy arcs, wide open grin
33. head and shoulders only, huge round eyes, small round open mouth, shocked
34. head and shoulders only, thick eyebrows down in a hard V, mouth open shouting, enraged
35. head and shoulders only, eyes half closed under heavy lids, one eyebrow raised, smug closed curve mouth
36. head and shoulders only, eyebrows tilted up in the middle, small wavering mouth, worried
37. head and shoulders only, one eye winking, wide grin
38. head and shoulders only, both eyes looking sharply to one side, deadpan
39. head and shoulders only, eyes wide, eyebrows high, mouth a flat line, alarmed
40. head and shoulders only, mouth open wide in a yawn, eyes shut
41. head and shoulders only, cheeks puffed, eyes squeezed shut
42. head and shoulders only, tongue out, eyes crossed
43. head and shoulders only, looking straight up, eyes raised
44. head and shoulders only, looking straight down, eyes lowered, sad mouth
45. head and shoulders only, completely blank face with only two dots and a straight line mouth

## C. One figure with one large symbolic object (46–65)

The `symbolic-metaphor` shots — 24% of real frames. The object must read as a
**flat filled shape**, which is the thing our current renders get wrong.

46. a small figure standing beside a giant closed door, the door a flat solid brown rectangle
47. a small figure standing beside a giant open door, flat solid brown
48. a small figure looking up at a giant flat brown brain shape
49. a small figure beside a giant flat red heart shape
50. a small figure beside a giant flat brown eye shape
51. a small figure beside a giant flat brown ear shape
52. a small figure beside a giant flat brown hand shape
53. a small figure beside a giant flat red exclamation mark
54. a small figure beside a giant flat brown question mark
55. a small figure beside a giant flat brown key
56. a small figure beside a giant flat red padlock
57. a small figure beside a giant flat brown clock face
58. a small figure beside a giant flat brown light bulb
59. a small figure beside a giant flat brown gear wheel
60. a small figure beside a giant flat red arrow pointing up
61. a small figure beside a giant flat brown ladder
62. a small figure beside a giant flat brown mirror
63. a small figure beside a giant flat brown book
64. a small figure beside a giant flat red drop shape
65. a small figure standing inside a giant flat brown circle outline

## D. Two figures (66–85)

The `two-character` shots — 20% of real frames, and the ones that most often
render as a single figure. Every prompt here says **two** explicitly.

66. two figures facing each other, both standing straight, deadpan faces
67. two figures facing each other, one pointing at the other, smug and worried
68. two figures shaking hands, both delighted
69. two figures standing side by side facing forward, both deadpan
70. two figures back to back, both arms crossed, both smug
71. two figures, one handing a small flat red box to the other
72. two figures, one tapping the other on the shoulder
73. two figures walking together in the same direction
74. two figures, one sitting and one standing, talking
75. two figures, one leaning away from the other, shocked
76. two figures with a large flat brown rectangle between them
77. two figures pulling opposite ends of a straight line
78. two figures, one much taller than the other, both deadpan
79. two figures high-fiving, both delighted
80. two figures both pointing at the same spot between them
81. two figures, one covering its eyes, the other watching
82. two figures standing far apart facing each other, empty space between
83. two figures, one holding a flat brown circle, the other reaching for it
84. two figures both shrugging with arms out
85. two figures, one whispering toward the other's ear

## E. One figure with a small prop (86–100)

Teaches the figure holding and interacting with objects without the object
dominating.

86. the figure holding a small flat brown box in both hands
87. the figure holding a flat red ball in one hand
88. the figure holding a flat brown stick out to one side
89. the figure holding a flat brown cup
90. the figure carrying a flat brown bag over one shoulder
91. the figure pushing a flat brown cube along
92. the figure pulling a flat brown rope
93. the figure standing next to a small flat brown table
94. the figure sitting on a flat brown block
95. the figure standing on top of a flat brown block, arms out for balance
96. the figure holding a flat brown rectangle up like a sign, blank and empty
97. the figure reaching up toward a small flat red circle above it
98. the figure looking down at a small flat brown square on the ground
99. the figure holding one flat brown gear in each hand
100. the figure with a flat red circle floating above its head

---

## After you have the 100 images

1. Put them in `lora_data/` — the notebook's Step 21d prep cell already handles
   padding to square and running `lock_palette()` over them, which makes the
   training set look exactly like a finished frame.
2. **Move caption `.txt` files out before training.** v0.39.0's
   `DreamBoothDataset` iterates the directory with no extension filter and a
   caption file crashes it with `UnidentifiedImageError`. Pre-flight 3 in cell
   `O7IVBdeNbv8W` does this — it exists because this bit us before.
3. Training settings, changed from the run that failed:
   - `--max_train_steps` **1500–2500**, not 500. With 100 images that is only
     15–25 passes per image, which is the normal range.
   - `--rank 16` is fine; drop to 8 if it overfits.
   - `--learning_rate 1e-4` is fine at this volume.
4. **Check `||B||` movement between checkpoints before drawing anything.** Cell
   `ATBXCaqKKg7s` prints it. Under ~10% movement over the last few hundred
   steps means it has converged; the failed run was still moving 71%.
5. Set `config.LORA_TRIGGER = "stkmn style"` again (it is `""` now, because the
   old LoRA was abandoned), re-run `rebuild_prompts.py`, and draw the 9-frame
   test batch before committing to a full video.
