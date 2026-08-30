"""The Colab GPU worker: audio in, word timestamps out. Nothing else.

Paste this whole file into ONE Colab cell and run it. It prints a URL; paste
that into the Studio's Voice page (or pass it as --colab-url).

    Runtime -> Change runtime type -> T4 GPU        <- T4, not L4
    then run the cell, then paste the printed URL

WHY T4 AND NOT L4. Colab bills COMPUTE UNITS BY WALL-CLOCK on the runtime, not
by GPU utilisation. Measured on this exact job (598s of audio, large-v3): the
align itself is 95s on a T4, and the session is dominated by pip install and
model load rather than by inference. An L4 is roughly 2.7x the hourly rate and
saves perhaps 30 seconds, so it costs about twice as much for the same work.

    T4   ~0.15 units per video        <- use this
    L4   ~0.32 units per video

WHY THIS FILE KNOWS NOTHING ABOUT THE PIPELINE. The seam is one step wide:
mastering happens on the laptop, alignment happens on the laptop, and only the
transcribe call comes here. So there is no scene logic, no fingerprint, no
mastering chain to keep in step with `pipeline/voice.py` - and therefore nothing
that can silently drift out of date with the repo. If you change how audio is
mastered, this file does not care.

NEVER RUN THE RENDER HERE. ffmpeg is pure CPU and a GPU runtime bills it at GPU
rates. That was worth ~2x the entire monthly bill when it was measured.
"""


# =============================================================================
#  STUDIO VOICE WORKER - transcribe only
# =============================================================================
import subprocess, sys, os, threading, time, json, re, shutil

subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "faster-whisper>=1.1", "fastapi", "uvicorn[standard]",
                "python-multipart"], check=True)

# cloudflared quick tunnel: no account, no login, new hostname each run.
if not shutil.which("cloudflared"):
    subprocess.run(
        "wget -q -O /usr/local/bin/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared",
        shell=True, check=True)

import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, UploadFile
import uvicorn

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
if GPU is None:
    print("!! NO GPU. Runtime -> Change runtime type -> T4 GPU, then re-run.")
elif "L4" in GPU or "A100" in GPU:
    print(f"!! You are on a {GPU}. A T4 does this job for about half the units.")

# THE MODEL IS LOADED ONCE, HERE. Loading it per request would add ~15s to
# every section and is the whole reason this is a long-lived worker rather
# than a script you re-run.
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
print(f"loading {MODEL} on {GPU or 'cpu'} ...")
_t = time.time()
_m = WhisperModel(MODEL, device="cuda" if GPU else "cpu",
                  compute_type="float16" if GPU else "int8")
print(f"loaded in {time.time()-_t:.0f}s")

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "gpu": GPU or "cpu", "model": MODEL}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), model: str = Form(MODEL)):
    # The laptop sends audio that is ALREADY MASTERED, so this must not touch
    # it - resampling or normalising here would move every timestamp away from
    # the audio the render actually uses.
    path = f"/content/_in_{int(time.time()*1000)}.wav"
    with open(path, "wb") as fh:
        fh.write(await audio.read())
    t0 = time.time()
    segs, _ = _m.transcribe(path, word_timestamps=True)
    words = [{"word": w.word, "start": float(w.start), "end": float(w.end)}
             for s in segs for w in (s.words or [])]
    os.remove(path)
    print(f"  {audio.filename}: {len(words)} words in {time.time()-t0:.1f}s")
    return {"words": words, "model": MODEL, "seconds": round(time.time()-t0, 2)}

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8899, log_level="warning"),
    daemon=True).start()
time.sleep(3)

proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8899", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

url = None
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        break

print("\n" + "=" * 62)
print("  WORKER URL - paste this into the Studio Voice page:")
print(f"\n     {url}\n")
print("  Keep this cell running. The URL dies when the runtime does,")
print("  and a quick tunnel gets a NEW one every session.")
print("=" * 62)



