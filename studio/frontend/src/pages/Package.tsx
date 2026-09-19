// The upload form, filled in.
//
// Every other page in this app exists to change a video. This one changes
// nothing - it hands over four strings (title, description, tags, thumbnail
// prompt) and two .txt files. So the whole page is built around one gesture:
// click, paste, next field. Nothing here is editable, because an edit made in
// this box would not survive a re-run and would silently diverge from the .txt
// the operator archives next to the video.
//
// The values come from build/package.json, written by youtube_packager.py
// alongside the two human-readable files. See studio/backend/packager.py for
// why the UI reads the JSON rather than parsing metadata.txt.
//
// The same values can also be sent straight to YouTube from the upload card at
// the bottom - see components/YoutubeUpload.tsx.

import { useEffect, useState } from "react";
import { api, fileUrl, type PackageStatus, type YoutubeStatus } from "../api";
import type { PageProps } from "../App";
import { Banner, Drop, ErrorBox, JobLog, Pill } from "../components/bits";
import YoutubeUpload from "../components/YoutubeUpload";

/** Copy-to-clipboard with a receipt, and a fallback for non-secure contexts. */
function useCopy() {
  const [done, setDone] = useState<string | null>(null);

  async function copy(key: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // navigator.clipboard is undefined outside a secure context - which this
      // app is on http://localhost, but not if it is ever opened over a LAN IP.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;opacity:0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } finally { ta.remove(); }
    }
    setDone(key);
    setTimeout(() => setDone((k) => (k === key ? null : k)), 1400);
  }

  return { done, copy };
}

type FieldProps = {
  k: string; label: string; value: string;
  note?: React.ReactNode; rows?: number;
  copied: string | null;
  onCopy: (k: string, v: string) => void;
};

function CopyField({ k, label, value, note, rows = 1, copied, onCopy }: FieldProps) {
  const empty = !value.trim();
  return (
    <div style={{ marginTop: 14 }}>
      <div className="spread" style={{ marginBottom: 5 }}>
        <strong style={{ fontSize: 13 }}>{label}</strong>
        <div className="row" style={{ gap: 8 }}>
          {note}
          <button disabled={empty} onClick={() => onCopy(k, value)}>
            {copied === k ? "copied ✓" : "Copy"}
          </button>
        </div>
      </div>
      <div className="copybox" style={{ maxHeight: rows > 1 ? rows * 20 + 18 : undefined }}>
        {empty ? <span className="faint">— nothing here; re-run the packager</span> : value}
      </div>
    </div>
  );
}

/** A character budget YouTube enforces by truncating rather than complaining. */
function Count({ n, max, unit = "chars" }: { n: number; max: number; unit?: string }) {
  return (
    <Pill kind={n <= max ? "" : "bad"}>
      {n}/{max} {unit}
    </Pill>
  );
}

function Download({ file }: { file: { name: string; abs_path: string; exists: boolean; bytes: number } }) {
  if (!file.exists) return <span className="faint mono">{file.name} — not written yet</span>;
  return (
    <a className="dl" href={fileUrl(file.abs_path)} download={file.name}>
      ⭳ {file.name}
      <small>{(file.bytes / 1024).toFixed(1)} kB</small>
    </a>
  );
}

export default function Package({ id }: PageProps) {
  const [data, setData] = useState<PackageStatus | null>(null);
  const [yt, setYt] = useState<YoutubeStatus | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [thumbBusy, setThumbBusy] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  // The thumbnail prompt is shipped with the locked style block prepended,
  // because that is what the image generator needs in one paste. The toggle is
  // for the case where the style is already in the generator's system prompt.
  const [withStyle, setWithStyle] = useState(true);
  const { done, copy } = useCopy();

  const loadYt = () => api.youtube(id).then(setYt).catch(setErr);
  const load = () => { loadYt(); return api.packageStatus(id).then(setData).catch(setErr); };
  useEffect(() => { setData(null); setYt(null); setJobId(null); setErr(null); load(); }, [id]);

  async function uploadThumb(files: FileList) {
    setThumbBusy(true); setErr(null);
    try { setYt(await api.upload(`/api/projects/${id}/youtube/thumbnail`, files)); }
    catch (e) { setErr(e); } finally { setThumbBusy(false); }
  }

  async function run() {
    setBusy(true); setErr(null);
    try {
      const job = await api.runPackage(id);
      setJobId(job.id);
    } catch (e) { setErr(e); } finally { setBusy(false); }
  }

  if (!data) return <div className="faint">loading…</div>;

  const p = data.package;
  const c = data.checks;
  const topic = id.split("/")[1];
  const thumbPrompt = p
    ? (withStyle ? p.thumbnail.full_prompt : p.thumbnail.prompt)
    : "";

  return (
    <>
      <div className="spread">
        <h2>Package</h2>
        {p
          ? <Pill kind="ok">packaged · {new Date(p.generated_at).toLocaleString()}</Pill>
          : <Pill>not packaged</Pill>}
      </div>
      <div className="faint">
        YouTube metadata and the thumbnail image prompt. Nothing here enters a render —
        it is what you paste into the upload form and into the image generator.
      </div>
      <ErrorBox err={err} />

      {/* ------------------------------------------------------------ run */}
      <div className="card" style={{ marginTop: 10 }}>
        <div className="row">
          <code className="mono">python youtube_packager.py --topic {topic}</code>
          <div style={{ flex: 1 }} />
          <button className="primary" disabled={busy || !data.can_run} onClick={run}>
            {p ? "Re-run packager" : "Run packager"}
          </button>
        </div>
        <div className="faint" style={{ marginTop: 8 }}>
          Two model calls: titles and description grounded in the real cold open, then a
          thumbnail prompt with the character chosen for this topic.
        </div>

        {!data.can_run && <Banner kind="bad">{data.why}</Banner>}
        {data.can_run && !data.has_hook && (
          <Banner kind="warn">
            No script or <code>sections/section_01.txt</code> yet. The packager will fall back
            to the source excerpts, and the titles will be weaker for it — they earn clicks by
            mirroring words the viewer hears in the first five seconds.
          </Banner>
        )}
        {data.stale && (
          <Banner kind="warn">
            This project was packaged before the copyable <code>package.json</code> existed —
            the .txt files below are still readable, but the copy fields need one re-run.
          </Banner>
        )}

        {jobId && (
          <div style={{ marginTop: 12 }}>
            <JobLog jobId={jobId} onDone={() => load()} />
          </div>
        )}
      </div>

      {/* ---------------------------------------------------------- files */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="spread"><strong>Files</strong><span className="faint">build/</span></div>
        <div className="row" style={{ marginTop: 8 }}>
          <Download file={data.files.metadata} />
          <Download file={data.files.thumbnail_prompt} />
          <Download file={data.files.package_json} />
        </div>
      </div>

      {!p ? (
        <div className="card faint" style={{ marginTop: 12 }}>
          Nothing packaged yet — run the packager above.
        </div>
      ) : (
        <>
          {/* ------------------------------------------------- copy fields */}
          <div className="card" style={{ marginTop: 12 }}>
            <div className="spread">
              <strong>Paste into YouTube</strong>
              <span className="faint">{p.model}</span>
            </div>

            <CopyField
              k="title" label="Title" value={p.best_title} copied={done} onCopy={copy}
              note={c && <Count n={c.title_chars} max={c.title_max} />}
            />
            <div className="faint" style={{ marginTop: 4 }}>
              Chosen over the other two: {p.best_title_reason}
            </div>

            <CopyField
              k="desc" label="Description" value={p.description} rows={12}
              copied={done} onCopy={copy}
              note={c && <Count n={c.desc_chars} max={c.desc_max} />}
            />

            <CopyField
              k="tags" label="Tags (comma separated)" value={p.tags} rows={3}
              copied={done} onCopy={copy}
              note={c && (
                <>
                  <Pill>{c.n_tags} tags</Pill>
                  <Count n={c.tags_chars} max={c.tags_max} />
                </>
              )}
            />
            {c && !c.tags_ok && (
              <Banner kind="warn">
                Over YouTube's {c.tags_max}-character tag budget — the box will cut the tail off
                silently. Drop the last few specific tags before pasting.
              </Banner>
            )}
          </div>

          {/* -------------------------------------------------- thumbnail */}
          <div className="card" style={{ marginTop: 12 }}>
            <div className="spread">
              <strong>Thumbnail</strong>
              <label className="row faint" style={{ gap: 6 }}>
                <input type="checkbox" checked={withStyle}
                       onChange={(e) => setWithStyle(e.target.checked)} />
                include the locked style block
              </label>
            </div>

            <div className="row" style={{ gap: 22, marginTop: 10, alignItems: "flex-start" }}>
              {/* The upload card's view of the thumbnail carries an mtime, so a
                  replaced file gets a new URL - same path, new bytes, and the
                  browser would otherwise keep showing the old one. */}
              {yt?.thumbnail ? (
                <img className="thumb-shot" alt="thumbnail"
                     src={`${fileUrl(yt.thumbnail.abs_path)}&v=${yt.thumbnail.mtime}`} />
              ) : data.thumbnail_image && (
                <img className="thumb-shot" src={fileUrl(data.thumbnail_image)} alt="thumbnail" />
              )}
              <div style={{ flex: 1, minWidth: 240 }}>
                <div className="faint">overlay text</div>
                <div style={{ fontSize: 19, fontFamily: "var(--mono)" }}>
                  {p.thumbnail.overlay_text || "—"}
                </div>
                <div className="faint" style={{ marginTop: 8 }}>character</div>
                <div className="muted">{p.thumbnail.character_choice || "—"}</div>
              </div>
            </div>

            <CopyField
              k="thumb" label="Thumbnail prompt" value={thumbPrompt} rows={14}
              copied={done} onCopy={copy}
              note={<Pill>{thumbPrompt.length} chars</Pill>}
            />
            <div style={{ marginTop: 12 }}>
              <Drop label={yt?.thumbnail
                      ? `Drop a new thumbnail to replace ${yt.thumbnail.name} — the old one is kept as thumbnail_prev`
                      : "Drop the generated thumbnail here, or click to choose — JPG or PNG, under 2 MB"}
                    accept="image/jpeg,image/png" busy={thumbBusy} onFiles={uploadThumb} />
            </div>
            <div className="faint" style={{ marginTop: 6 }}>
              1280×720, 16:9. Saved into the project as <code>thumbnail.jpeg</code> /{" "}
              <code>.png</code> and sent with the upload below.
            </div>
          </div>

          {yt && <YoutubeUpload id={id} pkg={p} yt={yt} reload={loadYt} />}

          {/* ---------------------------------------------- other titles */}
          <div className="card" style={{ marginTop: 12 }}>
            <strong>Other title options</strong>
            <div className="faint" style={{ marginTop: 2 }}>
              All three mirror the hook; the starred one is the model's pick.
            </div>
            {p.titles.map((t, i) => (
              <div className="row alt-title" key={i}>
                <span style={{ width: 14 }}>{t.best ? "★" : ""}</span>
                <span style={{ flex: 1 }}>{t.text}</span>
                <span className="faint mono">{t.cap_word}</span>
                <span className="faint">{t.text.length}</span>
                <button onClick={() => copy(`t${i}`, t.text)}>
                  {done === `t${i}` ? "copied ✓" : "Copy"}
                </button>
              </div>
            ))}
          </div>

          {/* ------------------------------------------------- reference */}
          <div className="card" style={{ marginTop: 12 }}>
            <strong>Reference</strong>
            <div className="faint" style={{ marginTop: 8 }}>core fact</div>
            <div className="muted">{p.core_fact || "—"}</div>

            <div className="faint" style={{ marginTop: 10 }}>keywords</div>
            {(["broad", "niche", "specific"] as const).map((g) => (
              <div className="row" key={g} style={{ gap: 8, marginTop: 4 }}>
                <span className="faint" style={{ width: 60 }}>{g}</span>
                <span className="mono">{p.keywords[g].join(", ") || "—"}</span>
              </div>
            ))}

            <div className="faint" style={{ marginTop: 10 }}>sources (already in the description)</div>
            <ul className="srcs">
              {p.sources.map((s, i) => (
                <li key={i}>
                  <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
                </li>
              ))}
              {p.sources.length === 0 && <li className="faint">—</li>}
            </ul>
          </div>
        </>
      )}
    </>
  );
}
