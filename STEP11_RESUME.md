# Step 11 — resume here (written 2026-08-26, end of session)

## State when the session stopped

- colab-mcp **connected**; `.mcp.json` timeout raised 30000 -> **600000**
  (needs a Claude Code restart to take effect — that 30s abort was the client
  giving up, NOT the cell failing. Cells kept running server-side every time.)
- Notebook is the real `stickman.ipynb`, 130 cells + 5 new ones appended.
- Runtime was **L4, 23.7 GB, standard RAM**.
- **11a succeeded**: base SDXL resident, `hf_cache` 6.7 GB, 0 incomplete parts.
- **11b was fired and left running.** Last probe returned empty outputs, so
  whether all 10 props landed is UNKNOWN. It is resumable — re-running skips
  finished PNGs.
- 11c never ran. No props have been looked at yet. **Step 11 is not answered.**

## The new cells

| cellId | what |
| --- | --- |
| `e21QSNgVor6B` | 11 probe — runtime / packages / disk |
| `sIC_lYm5o4x5` | 11a — load base SDXL, `add_watermarker=False` |
| `N1DHHd7Fo93V` | 11b — the ten props, resumable |
| `Dn8H4nZxpGuk` | 11c — contact sheet as base64, no `display()` |
| `FE3fSt7npjFb` | state probe — pipe loaded? cache size? props so far? |

## Tomorrow, in order

1. Restart Claude Code so the 600000 timeout loads. Then cells stop aborting.
2. Reconnect L4. `/content` is wiped on disconnect, so 11a re-downloads (~5 min,
   unavoidable) and 11b redraws all ten (~3 min).
3. Run 11a -> 11b -> 11c **in one sitting** — load cost is per-session.
4. Confirm 11a prints `watermarker: None`.
5. Look at the sheet. Consistent + on-style -> generate the other 90, NO
   TRAINING. Inconsistent -> those are a curated seed set, still a win.

## Decisions made this session, flag if wrong

- **L4 over T4 high-RAM.** ~3x the hourly rate but ~3.5x the throughput, so a
  wash at 10 images and cheaper in absolute units at 90. High-RAM buys nothing:
  SDXL fp16 is ~7 GB VRAM, one image at a time.
- **Stripped `flatprop style,`** from the prompts. It is the LoRA trigger; on
  base SDXL it is an untrained token that only dilutes CLIP conditioning and
  would bias the "do we need to train" answer toward yes. Rest of STYLE verbatim.
- Ten chosen one-per-failure-mode, not ten easy ones: lungs(1) brain(3)
  skull(11) MRI(23) microscope(31) chair(40) sofa(39) mosquito(84) bulb(94)
  barchart(98). Brain = the FLUX shading rematch. Barchart = text-leakage trap.

## ⚠️ Unrelated, still outstanding

HF token hardcoded in notebook cells 4, 12, 33, 36 (and in `stickman.ipynb.bak`):
`hf_***ROTATE-ME***`. Cell 13 already does `load_dotenv`
properly. Rotate the token, then switch those four cells to the dotenv path.
