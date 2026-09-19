# Video Pipeline — Weird Biology

An end-to-end pipeline for producing short-form explainer videos: script
generation and fact-checking, voice recording and coaching, pacing-aware
timing/motion planning, rendering, and YouTube packaging/upload — driven
either from the CLI or from a local React/FastAPI Studio UI.

Pictures and motion clips are generated externally (Nano Banana / Google
Flow); this pipeline's job is to write the script, time the narration,
place and cut the assets, render, and publish.

## Architecture

```text
                         React (Vite, :5173)
                                |  /api proxy
                                v
                         FastAPI (:8770)  --imports-->  channels/pipeline/*
                                |
                                +--> channels/build_video.py (render)
                                +--> channels/youtube_upload.py (publish)

  channels/           CLI-driven script + scene pipeline (CPU only, no torch)
  channels/pipeline/  shared modules: script, pacing, motion, assetsort,
                       render jobs, packaging, YouTube metadata
  studio/backend/      FastAPI app wrapping the pipeline for the UI
  studio/frontend/     React + TypeScript + Vite UI (Script, Voice, Plan,
                       Images, Motion, Timeline, Package, Jobs)
  vo_coach/            standalone voice recording/coaching tool (own venv)
```

`channels/pipeline/` deliberately never imports `torch`, `diffusers` or
`faster-whisper` — that GPU work stays in a Colab notebook (Whisper
alignment) and a Colab render worker, kept on separate runtimes to control
cost. See [`STUDIO_HANDOFF.md`](STUDIO_HANDOFF.md) and
[`channels/README.md`](channels/README.md) for the full design history.

## Prerequisites

- **Python 3.12** (pinned — some voice-stage dependencies ship compiled
  wheels that lag newer Python releases)
- **Node.js 18+** and npm
- An [OpenRouter](https://openrouter.ai) API key (routes script/scene
  generation through Claude and other models)
- A Hugging Face token (`HF_TOKEN`), if using local voice/alignment tooling
- *Optional:* a Google Cloud OAuth "Desktop app" client, for YouTube upload

## Setup

### 1. Backend — Studio + pipeline

```powershell
cd F:\wierd_bio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> The project has historically run against a venv at `vo_coach\.venv`
> (Python 3.12.10) — that venv also satisfies everything in
> `requirements.txt`. Either a fresh `.venv` built from `requirements.txt`
> or the existing `vo_coach\.venv` will work; just be consistent about
> which one you launch `uvicorn` with.

### 2. Frontend — Studio UI

```powershell
cd F:\wierd_bio\studio\frontend
npm install
```

### 3. Environment variables

Create a `.env` file in the repo root (gitignored — never commit it):

```ini
OPENROUTER_API_KEY=sk-or-v1-...
HF_TOKEN=hf_...
```

Other environment variables read by the pipeline, set only if you need
non-default behaviour:

| variable | purpose |
| --- | --- |
| `VIDEO_PIPELINE_WORKDIR` | shared working folder (e.g. a synced Google Drive path) so the local half and the Colab notebook read/write the same files |
| `WHISPER_MODEL` | override the default Whisper model size for alignment |
| `STUDIO_ALLOW_GPU` | opt-in flag for GPU-only code paths in the Studio backend |
| `VO_COACH_TAKES` | override where `vo_coach` writes recorded takes |
| `HTTPS_PROXY` | route YouTube API calls through a proxy, if configured |

YouTube upload credentials are **not** environment variables — they live in
the gitignored `channels/.youtube/` directory (OAuth client + one refresh
token per channel), created by running `youtube_upload.py auth`. See
[`channels/youtube_upload.py`](channels/youtube_upload.py) for the full
flow.

## Running the Studio UI

Two processes, run in separate terminals. The Vite dev server proxies
`/api` to the backend, so no CORS configuration is needed.

```powershell
# 1. Backend
cd F:\wierd_bio
.\.venv\Scripts\python.exe -m uvicorn studio.backend.app:app --host 127.0.0.1 --port 8770

# 2. Frontend
cd F:\wierd_bio\studio\frontend
npm run dev
```

Then open **http://localhost:5173**.

To stop the backend on Windows (`pkill` is unreliable here):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*uvicorn*studio.backend*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Running the CLI pipeline

For driving script generation and scene planning directly, without the UI:

```powershell
cd F:\wierd_bio\channels
.venv\Scripts\python.exe run_script.py --preflight   # verify key + model slugs, free
.venv\Scripts\python.exe run_script.py --list         # the topic bank
.venv\Scripts\python.exe run_script.py --topic <topic>
.venv\Scripts\python.exe run_scenes.py                # plan scenes from the recorded voice
```

See [`channels/README.md`](channels/README.md) for the full CLI reference,
cost breakdowns, and the file contract between pipeline stages.

## YouTube publishing

Metadata (title, description, tags, thumbnail) is generated by
`youtube_packager.py` and reviewed in the Studio's **Package** page. Upload
via:

```powershell
.venv\Scripts\python.exe channels\youtube_upload.py auth   --channel <channel>
.venv\Scripts\python.exe channels\youtube_upload.py upload --project <channel>/<topic> \
    --video "<file>.mp4" --privacy private
```

One upload per project is enforced (pass `--force` to override) to avoid
duplicate videos against YouTube's daily quota.

## Project layout

```text
channels/            CLI pipeline (script, evidence, fact-check, pacing, packaging)
channels/pipeline/   shared modules imported by both the CLI and the Studio backend
studio/backend/      FastAPI app (projects, jobs, rendering, YouTube, settings)
studio/frontend/     React + TypeScript + Vite UI
vo_coach/            voice recording/coaching tool (separate Python 3.12 venv)
```

## Notes

- `.env`, `channels/.youtube/`, and other credential paths are gitignored —
  never commit API keys or OAuth tokens.
- GPU-dependent work (image generation, Whisper alignment) runs externally
  or in Colab; the local pipeline is CPU-only by design so it stays
  installable on a laptop.
