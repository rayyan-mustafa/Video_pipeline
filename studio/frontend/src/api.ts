// One place that knows how to talk to the backend.
//
// Every call goes through `req()`, so a FastAPI error - which always arrives as
// {"detail": "..."} - becomes a thrown Error carrying the message the backend
// actually wrote. Page code can then just catch and show it, rather than each
// page inventing its own way to notice a 409.

export type Problem = { message: string; status: number };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = body.detail.map((d: any) => d.msg).join("; ");
    } catch {
      /* not JSON: keep the status text */
    }
    const err = new Error(detail) as Error & Problem;
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// ---------------------------------------------------------------- types

export type ProjectStatus = {
  channel: string; topic: string; id: string; dir: string;
  has_script: boolean; script_words: number;
  sections: number; recordings: number; images: number;
  has_plan: boolean; plan_split: string | null;
  plan_scenes: number; plan_images: number; fingerprint: string | null;
  has_timeline: boolean; timeline_scenes: number;
  has_words: boolean; has_voice: boolean; has_render_plan: boolean;
  has_overrides: boolean; has_video: boolean; video_mb: number;
};

export type Topic = {
  topic: string; channel: string; id: string; title: string;
  stake: string; audience: string; sources: number; exists: boolean;
};

export type Pacing = {
  n: number; total: number; avg: number; body_avg: number; median: number;
  min: number; max: number; accents: number; accent_share: number;
  over_hard: number; flashes: number;
  target_low: number; target_high: number; hard_max: number;
  want_images: number; rating: string; note: string;
};

export type Job = {
  id: string; kind: string; project: string; status: string;
  returncode: number | null;
  started: number; finished: number | null; elapsed: number;
  lines: string[];
};

export type VoiceStatus = {
  sections: { section: number; words: number; files: string[]; text: string }[];
  n_sections: number;
  files: { name: string; path: string; canonical: boolean }[];
  n_files: number;
  missing_sections: number[];
  ready: boolean;
  can_revert: boolean;
  models: string[];
  default_model: string;
  voice_profiles: string[];
  colab: { ok: boolean; url: string; gpu?: string; model?: string;
           error?: string; hint?: string };
  proposal: {
    status: "ok" | "ambiguous" | "impossible" | "already";
    confidence: string; needs_rename: boolean;
    spread: number; runner_up: number; note: string;
    assignments: {
      section: number; files: string[]; targets: string[];
      seconds: number; words: number; wpm: number;
    }[];
    unmatched: string[];
  };
  results: null | {
    scenes: number; words: number; has_voice: boolean;
    voice_path: string | null; runtime: number;
    pacing?: Pacing;
    motion_dose?: {
      runtime: number; clip_s: number; clips_low: number; clips_high: number;
      seconds_low: number; seconds_high: number; window: number[];
    };
  };
};

// `move` is the Ken Burns move pipeline/motion.py will actually render for
// this slot - zoom and framing at both ends of the hold. The preview animates
// it; nothing here decides it.
export type Move = {
  name: string; static: boolean;
  z0: number; z1: number;
  px0: number; px1: number;
  py0: number; py1: number;
};

export type Slot = {
  index: number; path: string; abs_path: string; has_file: boolean;
  section: number; order: number; scene_indices: number[];
  start: number; end: number; duration: number;
  kind: "image" | "video"; is_motion: boolean; fits_motion: boolean;
  clip_path?: string; clip_abs_path?: string;
  move: Move;
};

export type Word = { section: number; w: string; s: number; e: number };

export type Overrides = {
  version: number; fingerprint: string | null;
  cuts: Record<string, number>;
  assets: Record<string, string>;
  motion: Record<string, string>;
};

export type TimelineData = {
  ready: boolean; why: string;
  slots: Slot[]; warnings: string[]; words: Word[];
  overrides: Overrides;
  available_images: string[];
  // Files in images/ whose name matches no slot. Offered in the replace
  // picker under their own heading - an override resolves by name, so they
  // work; they are just never chosen automatically by the plan.
  unconventional_images: string[];
  available_clips: string[];
  fingerprint: string | null;
};

// The YouTube package. Shaped by channels/youtube_packager.py's package.json -
// the copyable twin of metadata.txt / thumbnail_prompt.txt, which the UI reads
// instead of scraping those two files. See studio/backend/packager.py.
export type PackageData = {
  version: number; topic: string; base_title: string; stake: string;
  generated_at: string; model: string;
  core_fact: string; hook: string;
  best_title: string; best_title_reason: string;
  titles: { text: string; cap_word: string; best: boolean }[];
  description: string;
  description_parts: { hook: string; tease: string; cta: string };
  keywords: { broad: string[]; niche: string[]; specific: string[] };
  tags: string; tag_list: string[];
  thumbnail: {
    character_choice: string; overlay_text: string;
    prompt: string; full_prompt: string;
  };
  sources: { title: string; url: string }[];
};

export type PackageFile = {
  name: string; abs_path: string; exists: boolean; bytes: number;
};

export type PackageStatus = {
  can_run: boolean; why: string;
  has_package: boolean; stale: boolean; has_hook: boolean;
  package: PackageData | null;
  checks: null | {
    title_chars: number; title_max: number; title_ok: boolean;
    desc_chars: number; desc_max: number; desc_ok: boolean;
    tags_chars: number; tags_max: number; tags_ok: boolean;
    n_tags: number;
  };
  thumbnail_image: string | null;
  files: {
    metadata: PackageFile;
    thumbnail_prompt: PackageFile;
    package_json: PackageFile;
  };
};

// The YouTube upload: channel connection, candidate videos, the thumbnail on
// disk, and build/upload.json once it has gone. See studio/backend/youtube.py.
export type YoutubeIdentity = {
  id: string; title: string; custom_url: string; checked_at: string;
};

export type YoutubeRecord = {
  video_id: string; url: string; studio_url: string;
  channel: YoutubeIdentity; privacy: string; publish_at: string | null;
  title: string; video_file: string; thumbnail_file: string | null;
  thumbnail_set: boolean; thumbnail_error: string | null; uploaded_at: string;
};

export type YoutubeStatus = {
  auth: { has_client_secrets: boolean; connected: boolean; identity: YoutubeIdentity | null };
  videos: { name: string; abs_path: string; mb: number; mtime: number; canonical: boolean }[];
  thumbnail: null | { name: string; abs_path: string; bytes: number; mtime: number; ok: boolean };
  record: YoutubeRecord | null;
  blockers: string[];
  can_upload: boolean;
  privacy_choices: string[];
  categories: Record<string, string>;
  proxy: string;
};

export type YoutubeUpload = {
  video: string; title?: string; privacy: string; publish_at?: string;
  category: string; synthetic: boolean; force?: boolean;
};

// A patch section's key -> new value, or -> null to remove that override.
export type TimelinePatch = {
  cuts?: Record<string, number | null>;
  assets?: Record<string, string | null>;
  motion?: Record<string, string | null>;
};

// ---------------------------------------------------------------- calls

export const api = {
  health: () => req<{ ok: boolean; problems: string[]; repo: string }>("/api/health"),

  projects: () => req<ProjectStatus[]>("/api/projects"),
  topics: () => req<Topic[]>("/api/topics"),
  project: (id: string) => req<ProjectStatus>(`/api/projects/${id}`),
  createProject: (channel: string, topic: string) =>
    req<ProjectStatus>(`/api/projects?channel=${encodeURIComponent(channel)}&topic=${encodeURIComponent(topic)}`,
      { method: "POST" }),

  script: (id: string) => req<any>(`/api/projects/${id}/script`),
  saveScript: (id: string, text: string) =>
    req<any>(`/api/projects/${id}/script`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),
  generateScript: (id: string) =>
    req<Job>(`/api/projects/${id}/script/generate`, { method: "POST" }),
  importGeneratedScript: (id: string) =>
    req<{ text: string }>(`/api/projects/${id}/script/generate/import`, { method: "POST" }),

  plan: (id: string) => req<any>(`/api/projects/${id}/plan`),
  runPlan: (id: string, body: unknown) => req<Job>(`/api/projects/${id}/plan`, json(body)),
  generatePrompts: (id: string) => req<Job>(`/api/projects/${id}/prompts`, { method: "POST" }),


  images: (id: string) => req<any>(`/api/projects/${id}/images`),
  imagePacing: (id: string) => req<any>(`/api/projects/${id}/images/pacing`),

  voice: (id: string) => req<VoiceStatus>(`/api/projects/${id}/voice`),
  applyNaming: (id: string, force = false) =>
    req<any>(`/api/projects/${id}/voice/naming`, json({ force })),
  revertNaming: (id: string) =>
    req<any>(`/api/projects/${id}/voice/naming`, { method: "DELETE" }),
  runVoice: (id: string, body: unknown) =>
    req<Job>(`/api/projects/${id}/voice/run`, json(body)),

  upload: (path: string, files: FileList | File[]) => {
    const fd = new FormData();
    for (const f of Array.from(files)) fd.append("files", f);
    return req<any>(path, { method: "POST", body: fd });
  },

  settings: () => req<any>("/api/settings"),
  saveSettings: (body: unknown) =>
    req<any>("/api/settings", { method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) }),
  checkColab: (url?: string) => req<any>("/api/settings/colab/check", json({ url })),
  // The RENDER worker is a different runtime from the voice one - CPU, not a
  // T4 - so it has its own URL and its own health check. See settings.py.
  checkRenderWorker: (url?: string) =>
    req<any>("/api/settings/render-worker/check", json({ url })),

  motion: (id: string) => req<any>(`/api/projects/${id}/motion`),
  saveMotion: (id: string, slots?: { index: number }[]) =>
    req<any>(`/api/projects/${id}/motion`, json(slots ? { slots } : {})),

  timeline: (id: string) => req<TimelineData>(`/api/projects/${id}/timeline`),
  saveTimeline: (id: string, patch: TimelinePatch) =>
    req<TimelineData>(`/api/projects/${id}/timeline`, json(patch)),
  revertTimeline: (id: string) =>
    req<TimelineData>(`/api/projects/${id}/timeline`, { method: "DELETE" }),

  renderStatus: (id: string) => req<any>(`/api/projects/${id}/render`),
  runRender: (id: string, body: { motion?: string; only_section?: number; target?: string }) =>
    req<Job>(`/api/projects/${id}/render`, json(body)),

  packageStatus: (id: string) => req<PackageStatus>(`/api/projects/${id}/package`),
  runPackage: (id: string) =>
    req<Job>(`/api/projects/${id}/package`, { method: "POST" }),

  youtube: (id: string) => req<YoutubeStatus>(`/api/projects/${id}/youtube`),
  youtubeAuth: (channel: string) =>
    req<Job>(`/api/youtube/${channel}/auth`, { method: "POST" }),
  youtubeUpload: (id: string, body: YoutubeUpload) =>
    req<Job>(`/api/projects/${id}/youtube/upload`, json(body)),
  youtubeRetryThumbnail: (id: string) =>
    req<Job>(`/api/projects/${id}/youtube/thumbnail/retry`, { method: "POST" }),

  jobs: (project?: string) =>
    req<Job[]>(`/api/jobs${project ? `?project=${encodeURIComponent(project)}` : ""}`),
  job: (jobId: string) => req<Job>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => req<any>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
};

// Absolute disk path -> a URL the browser can put in <img>/<audio>/<video>.
export const fileUrl = (absPath: string) =>
  `/api/file?path=${encodeURIComponent(absPath)}`;
