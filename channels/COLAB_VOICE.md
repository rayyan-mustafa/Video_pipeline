# Running the voice stage on a Colab GPU

Only the **voice** stage goes to Colab, and only because `large-v3` wants a GPU.
Planning is arithmetic; muxing is libx264 on CPU. Colab bills wall-clock, so
encoding there spends compute units on work the laptop does identically — the
notebook's own Step 25 says the same thing.

## What travels, and what comes back

```text
   up:  sections/*.txt + voice_recordings/*.wav + pipeline/ (5 modules)
 down:  scenes_timeline.json + words.json + full_vo.wav
```

Those three files are the entire contract with the next stage. The timeline says
when each sentence is spoken, `words.json` gives every word's exact timestamp,
and `full_vo.wav` is the mastered audio the mux reads.

**The mastering has to run in Colab too, not locally first.** The `clip` profile
trims silence, which moves every timestamp — so alignment must see the audio that
actually ships. Master locally and transcribe remotely and you align against
audio that no longer exists.

## 1. Build the bundle (laptop)

```powershell
..\vo_coach\.venv\Scripts\python.exe make_colab_bundle.py --project wierdbiology/chin
```

Writes `wierdbiology\chin\chin_voice_bundle.zip`. It carries the split mode in
`SPLIT.txt` so the Colab timeline uses the same scene boundaries the plan did —
get that wrong and the fingerprint check rejects the result *after* the GPU time
is spent.

## 2. Colab: runtime → GPU (T4 is enough), then run these cells

**Cell 1 — deps.** ~2 min.

```python
!pip install -q faster-whisper soundfile pyloudnorm noisereduce scipy
print("ok")
```

**Cell 2 — upload the zip.**

```python
from google.colab import files
import zipfile, pathlib
up = files.upload()                     # pick chin_voice_bundle.zip
name = next(iter(up))
zipfile.ZipFile(name).extractall("/content/work")
pathlib.Path("/content/work/build").mkdir(exist_ok=True)
print(sorted(p.name for p in pathlib.Path("/content/work").iterdir()))
```

**Cell 3 — check the GPU is actually there.** If this says `cpu`, fix it before
spending 20 minutes: Runtime → Change runtime type → T4 GPU.

```python
import torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

**Cell 4 — master + transcribe + align.** This is the whole stage. ~5–10 min for
10 minutes of audio on a T4 with `large-v3`.

```python
import sys, pathlib
sys.path.insert(0, "/content/work")
from build_video import build_voice

split = pathlib.Path("/content/work/SPLIT.txt").read_text().strip() or "chunk"
build_voice(
    pathlib.Path("/content/work/sections"),
    pathlib.Path("/content/work/voice_recordings"),
    pathlib.Path("/content/work/build"),
    whisper_size="large-v3",
    split=split,
)
```

Watch the **word match %** it prints per section. Below 75% means Whisper could
not follow the script there and those scene boundaries are guesses — re-record
that section rather than shipping it.

**Cell 5 — bring the results back.**

```python
import shutil
from google.colab import files
shutil.make_archive("/content/voice_out", "zip", "/content/work/build")
files.download("/content/voice_out.zip")
```

## 3. Back on the laptop

Unzip into the project's build folder, then carry on:

```powershell
# extract voice_out.zip into wierdbiology\chin\build\
..\vo_coach\.venv\Scripts\python.exe build_video.py --project wierdbiology/chin --stage plan
..\vo_coach\.venv\Scripts\python.exe build_video.py --project wierdbiology/chin --stage mux
```

`--stage plan` re-measures `full_vo.wav` and asserts the timeline, the image
durations and the audio all agree, so a truncated download is caught before any
encoding happens.

## Doing it locally instead

Perfectly viable — `faster-whisper` is already installed in `vo_coach\.venv`:

```powershell
..\vo_coach\.venv\Scripts\python.exe build_video.py --project wierdbiology/chin --stage voice --whisper base
```

`base` on CPU is roughly 8–12 minutes for 10 minutes of audio. `large-v3` on CPU
is about 10x that, which is why the GPU is worth the round trip if you want it.
The device is auto-detected — CUDA with float16 when available, CPU with int8
otherwise, and pre-Volta cards fall back to int8 because float16 is invalid there.

## If colab-mcp will not connect

`open_colab_browser_connection` is an *injected* tool that only exists while the
server is running, and its URL carries a token and port generated inside that
process — it cannot be built from outside. Connect with **`/mcp`** in the
terminal. `.mcp.json` already has `timeout: 600000`, but that value only takes
effect from a Claude Code restart, so restart first if the handshake aborts.

None of the above needs colab-mcp at all — the upload/download cells work in a
plain browser tab.
