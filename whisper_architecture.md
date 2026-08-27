# Hybrid Voiceover Sync & Custom Asset Architecture

This document outlines the architectural pivot for the video generator: moving from AI-generated audio/visuals to a Hybrid model utilizing human voiceovers (with local Whisper auto-syncing) and manual asset uploads.

## 1. Core Architecture Pivot

- **Previous Model:** The app generated isolated AI TTS audio chunks for each scene and concatenated them. Images were generated via API.
- **New Model:** The user uploads a single, continuous Human Voiceover (`.mp3`/`.wav`) and custom visual assets (e.g., local images). The system maps the script to the audio using local AI transcription, with a manual drag-and-drop timeline for visual overrides.

## 2. Local Whisper Implementation (Zero API Cost)

To avoid paying for the OpenAI Whisper API, the transcription engine will be run locally in the browser using WebAssembly.

### Technology Stack: `transformers.js`

- **Library:** `@xenova/transformers`
- **Model:** `Xenova/whisper-tiny.en` (or `base.en`). These are highly compressed models (under 100MB) that download directly into the user's browser cache.
- **Execution:** Runs entirely client-side using WebGL/WebGPU. No server costs, no API keys required.

### Workflow

1. User drops their voiceover file into the browser.
2. `transformers.js` decodes the audio and runs the Whisper model locally.
3. It returns a JSON object containing `chunks` with start and end timestamps for phrases/words.
4. The application logic fuzzy-matches these transcription chunks against the original generated script to assign precise `startTime` and `endTime` values to each Scene object in the state.

## 3. The Interactive Timeline Editor (Wavesurfer.js)

The core UI will transition to a non-linear editor (NLE) style timeline.

### Components

1. **Audio Waveform:** Use `wavesurfer.js` to render the uploaded audio file visually, allowing the user to see the peaks (speech) and valleys (silences).
2. **Scene Regions:** Use the `wavesurfer.js` Regions plugin (or a custom React overlay) to draw colored blocks on top of the waveform, representing each scene based on the Whisper timestamps.
3. **Manual Override:**
   - The edges of these region blocks must be draggable.
   - If Whisper incorrectly aligns a scene, the user can click and drag the boundary to visually align it with the correct audio spike in the waveform.
   - Dragging updates the React state for that scene's `startTime` or `endTime`.

## 4. Custom Asset Pipeline (Blob URLs)

Bypassing the AI image generation APIs entirely.

### Workflow

1. The Timeline Editor acts as a Dropzone.
2. The user drags a local file (e.g., a "nano banana" image) directly onto a Scene Region block in the timeline.
3. The browser generates a temporary local URL using `URL.createObjectURL(file)`.
4. This local Blob URL is saved to the Scene's `assetUrl` state.
5. **No server uploads required:** The images remain in local browser memory, ensuring instant loading and zero hosting costs.

## 5. Rendering Pipeline

The final video compiler will read the updated React state:

- It plays the master audio file from `0.00s`.
- It reads the timeline state. At exactly `scene.startTime`, it renders the local Blob URL image onto the canvas/video frame.
- It holds that image until `scene.endTime`, then switches to the next scene's Blob URL.

## Summary for Implementation Hand-off

To build this, the following major components need to be implemented:

1. **Whisper Worker:** A Web Worker implementing `@xenova/transformers` for non-blocking local transcription.
2. **Timeline Component:** A React wrapper around `wavesurfer.js` with the Regions plugin enabled for draggable boundaries.
3. **Asset Manager:** A drag-and-drop handler that maps local file drops to specific Scene IDs using Blob URLs.
