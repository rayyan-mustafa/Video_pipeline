// The last click: package + video + thumbnail, sent to YouTube.
//
// The metadata is build/package.json as-is - the same strings the Package page
// shows above this card - so there is nothing to type here, only the choices
// the package cannot make for you: which title, which video file, privacy and
// schedule. The upload is channels/youtube_upload.py run as a job; the log
// below is its stdout, including the channel it is about to upload to, so a
// token connected to the wrong Brand Account is visible before any bytes go.

import { useEffect, useState } from "react";
import { api, fileUrl, type PackageData, type YoutubeStatus } from "../api";
import { Banner, Drop, ErrorBox, JobLog, Pill } from "./bits";

type Props = { id: string; pkg: PackageData; yt: YoutubeStatus; reload: () => void };

/** datetime-local (the viewer's clock) -> the UTC instant YouTube wants. */
function toUtcIso(local: string): string {
  return new Date(local).toISOString().replace(/\.\d{3}Z$/, "Z");
}

export default function YoutubeUpload({ id, pkg, yt, reload }: Props) {
  const channel = id.split("/")[0];
  const [video, setVideo] = useState(yt.videos[0]?.name ?? "");
  const [title, setTitle] = useState(pkg.best_title);
  const [privacy, setPrivacy] = useState("private");
  const [when, setWhen] = useState("");
  const [category, setCategory] = useState("27");
  const [synthetic, setSynthetic] = useState(false);
  const [proxy, setProxy] = useState(yt.proxy);
  const [jobId, setJobId] = useState<string | null>(null);
  const [authJob, setAuthJob] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  // A new render or a rename changes the list; never keep a choice that is gone.
  useEffect(() => {
    if (!yt.videos.some((v) => v.name === video)) setVideo(yt.videos[0]?.name ?? "");
  }, [yt.videos]);
  useEffect(() => setTitle(pkg.best_title), [pkg.best_title]);

  // An upload is minutes long; a page reload mid-way should find it again
  // rather than offer a second upload over the first.
  useEffect(() => {
    api.jobs(id).then((js) => {
      const live = js.find((j) => j.kind.startsWith("youtube-") && j.status === "running");
      if (live) setJobId(live.id);
    }).catch(() => {});
  }, [id]);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true); setErr(null);
    try { await fn(); } catch (e) { setErr(e); } finally { setBusy(false); }
  }

  const who = yt.auth.identity;
  const picked = yt.videos.find((v) => v.name === video);
  const scheduled = when !== "";
  const whenBad = scheduled && new Date(when).getTime() < Date.now() + 15 * 60_000;

  function upload(force = false) {
    const target = who ? who.title : "the connected channel";
    const how = scheduled ? `private, going public ${new Date(when).toLocaleString()}` : privacy;
    const msg = `${force ? "Upload a SECOND copy" : "Upload"} "${video}" to ${target} as ${how}?\n\nTitle: ${title}`;
    if (!window.confirm(msg)) return;
    act(async () => {
      const job = await api.youtubeUpload(id, {
        video, title, category, synthetic, force,
        privacy: scheduled ? "private" : privacy,
        publish_at: scheduled ? toUtcIso(when) : undefined,
      });
      setJobId(job.id);
    });
  }

  const rec = yt.record;
  const titles = pkg.titles.length ? pkg.titles : [{ text: pkg.best_title, cap_word: "", best: true }];

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="spread">
        <strong>Upload to YouTube</strong>
        {rec
          ? <Pill kind="ok">uploaded · {rec.privacy}</Pill>
          : who
            ? <Pill kind="ok">→ {who.title}</Pill>
            : <Pill>not connected</Pill>}
      </div>
      <ErrorBox err={err} />

      {/* ------------------------------------------------ connection */}
      {!yt.auth.has_client_secrets && (
        <div style={{ marginTop: 10 }}>
          <div className="faint" style={{ marginBottom: 8 }}>
            One-time setup. In Google Cloud Console: enable <b>YouTube Data API v3</b> →
            Credentials → Create OAuth client ID → <b>Desktop app</b> → download the JSON and
            drop it here. It is stored in <code>channels/.youtube/</code> (gitignored).
          </div>
          <Drop label="Drop the OAuth client JSON here, or click to choose" accept=".json,application/json"
                busy={busy}
                onFiles={(f) => act(async () => { await api.upload("/api/youtube/client-secrets", f); reload(); })} />
        </div>
      )}

      {yt.auth.has_client_secrets && (
        <div className="row" style={{ marginTop: 10, gap: 10 }}>
          {who
            ? <span className="muted">Uploads go to <b>{who.title}</b>{" "}
                <span className="faint mono">{who.custom_url || who.id}</span></span>
            : <span className="faint">
                {yt.auth.connected ? "Token saved, channel not identified yet." : "No channel connected."}
              </span>}
          <div style={{ flex: 1 }} />
          <button className={yt.auth.connected ? "" : "primary"} disabled={busy || !!authJob}
                  onClick={() => act(async () => setAuthJob((await api.youtubeAuth(channel)).id))}>
            {yt.auth.connected ? "Reconnect" : "Connect YouTube"}
          </button>
        </div>
      )}
      {authJob && (
        <div style={{ marginTop: 10 }}>
          <div className="faint" style={{ marginBottom: 6 }}>
            A Google tab opened in your browser. Pick the account — or Brand Account — that
            <b> owns the channel</b>, then Allow.
          </div>
          <JobLog jobId={authJob} onDone={() => { setAuthJob(null); reload(); }} />
        </div>
      )}

      {/* ------------------------------------------------ already uploaded */}
      {rec && (
        <Banner kind={rec.thumbnail_set || !rec.thumbnail_file ? "ok" : "warn"}>
          <div>
            Uploaded to <b>{rec.channel?.title}</b> on {new Date(rec.uploaded_at).toLocaleString()} as{" "}
            <b>{rec.privacy}</b>{rec.publish_at && <> — public at {new Date(rec.publish_at).toLocaleString()}</>}.
          </div>
          <div className="row" style={{ gap: 12, marginTop: 6 }}>
            <a href={rec.url} target="_blank" rel="noreferrer">{rec.url}</a>
            <a href={rec.studio_url} target="_blank" rel="noreferrer">open in YouTube Studio</a>
          </div>
          {rec.thumbnail_file && !rec.thumbnail_set && (
            <div style={{ marginTop: 6 }}>
              Thumbnail did not apply: {rec.thumbnail_error}
              <div style={{ marginTop: 6 }}>
                <button disabled={busy || !!jobId}
                        onClick={() => act(async () => setJobId((await api.youtubeRetryThumbnail(id)).id))}>
                  Retry thumbnail
                </button>
              </div>
            </div>
          )}
        </Banner>
      )}

      {/* ------------------------------------------------ the form */}
      <div className="yt-form">
        <span className="faint">video</span>
        <div>
          {yt.videos.length === 0
            ? <span className="faint">no .mp4 in the project folder — render first</span>
            : (
              <select value={video} onChange={(e) => setVideo(e.target.value)} style={{ maxWidth: "100%" }}>
                {yt.videos.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name} — {v.mb} MB{v.canonical ? " (render output)" : ""}
                  </option>
                ))}
              </select>
            )}
          {picked && (
            <video className="yt-video" style={{ marginTop: 8 }} controls preload="metadata"
                   src={`${fileUrl(picked.abs_path)}&v=${picked.mtime}`} />
          )}
        </div>

        <span className="faint">title</span>
        <div className="yt-titles">
          {titles.map((t, i) => (
            <label key={i}>
              <input type="radio" name="yt-title" checked={title === t.text}
                     onChange={() => setTitle(t.text)} />
              <span>{t.text}</span>
              {t.best && <span className="faint">★ pick</span>}
              <span className="faint mono">{t.text.length}</span>
            </label>
          ))}
        </div>

        <span className="faint">visibility</span>
        <div className="row" style={{ gap: 10 }}>
          <select value={scheduled ? "private" : privacy} disabled={scheduled}
                  onChange={(e) => setPrivacy(e.target.value)}>
            {yt.privacy_choices.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          {privacy === "public" && !scheduled && <Pill kind="warn">goes live immediately</Pill>}
        </div>

        <span className="faint">schedule</span>
        <div className="row" style={{ gap: 10 }}>
          <input type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
          {scheduled && <button onClick={() => setWhen("")}>clear</button>}
          <span className="faint">
            {scheduled ? "uploads private, YouTube flips it public at this time" : "optional — empty uploads now"}
          </span>
        </div>

        <span className="faint">category</span>
        <select value={category} onChange={(e) => setCategory(e.target.value)} style={{ width: "fit-content" }}>
          {Object.entries(yt.categories).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>

        <span className="faint">disclosure</span>
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} />
          <span>Altered or synthetic content</span>
          <span className="faint">— for realistic AI people/events; 2D cartoon art doesn't need it</span>
        </label>
      </div>

      {/* ------------------------------------------------ what goes */}
      <div className="yt-checks">
        <Pill kind={who ? "ok" : "bad"}>channel</Pill>
        <Pill kind={picked ? "ok" : "bad"}>video{picked ? ` · ${picked.mb} MB` : ""}</Pill>
        <Pill kind="ok">title · {title.length}/100</Pill>
        <Pill kind="ok">description · {pkg.description.length} chars</Pill>
        <Pill kind="ok">{pkg.tag_list.length} tags</Pill>
        <Pill kind={yt.thumbnail ? (yt.thumbnail.ok ? "ok" : "bad") : "warn"}>
          {yt.thumbnail ? `thumbnail · ${(yt.thumbnail.bytes / 1024).toFixed(0)} kB` : "no thumbnail"}
        </Pill>
      </div>
      {!yt.thumbnail && (
        <Banner kind="warn">
          No thumbnail yet — drop one in the Thumbnail card above, or YouTube will pick a frame.
        </Banner>
      )}
      {whenBad && <Banner kind="warn">Schedule at least 15 minutes ahead — YouTube rejects a publish time in the past.</Banner>}
      {yt.blockers.length > 0 && <Banner kind="bad">Before uploading: {yt.blockers.join(" · ")}</Banner>}

      <div className="row" style={{ marginTop: 12, gap: 10 }}>
        {!rec ? (
          <button className="primary" disabled={busy || !!jobId || !yt.can_upload || !video || whenBad}
                  onClick={() => upload(false)}>
            Upload to YouTube
          </button>
        ) : (
          <button className="danger" disabled={busy || !!jobId || !yt.can_upload || !video || whenBad}
                  onClick={() => upload(true)}>
            Upload again (second copy)
          </button>
        )}
        <span className="faint">
          ~1,600 of your 10,000 daily API quota units per upload.
        </span>
      </div>

      {jobId && (
        <div style={{ marginTop: 12 }}>
          <JobLog jobId={jobId} onDone={() => { setJobId(null); reload(); }} />
        </div>
      )}

      {/* ------------------------------------------------ proxy */}
      <details style={{ marginTop: 14 }}>
        <summary className="faint" style={{ cursor: "pointer" }}>
          API proxy {yt.proxy ? <Pill kind="ok">on</Pill> : <Pill>direct</Pill>}
        </summary>
        <div className="faint" style={{ margin: "8px 0" }}>
          Send the connect + upload calls through the same proxy you manage the channel from,
          so they come from the same IP as your YouTube Studio sessions. Empty = direct.
        </div>
        <div className="row" style={{ gap: 8 }}>
          <input type="password" placeholder="http://user:pass@host:port" value={proxy}
                 onChange={(e) => setProxy(e.target.value)} style={{ flex: 1, minWidth: 220 }} />
          <button disabled={busy || proxy === yt.proxy}
                  onClick={() => act(async () => { await api.saveSettings({ youtube_proxy: proxy.trim() }); reload(); })}>
            Save
          </button>
        </div>
      </details>
    </div>
  );
}
