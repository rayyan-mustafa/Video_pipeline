// Motion slots: which cuts get an external Flow clip.
//
// The page has one job beyond picking slots, and it is the honest one: showing
// when the video CANNOT take its full dose. A 4s clip fits a hold of 3.5-4.0s
// and nothing else — a 15s hold is where motion is most wanted and exactly where
// it cannot go, because covering it would mean slowing or freezing the clip.
// Both are banned, so the planner reports the shortfall instead of bending.
//
// The fix for a shortfall is upstream: re-plan at the picture count the pacing
// verdict asks for, which pulls the long holds down into the window.

import { useEffect, useState } from "react";
import { api, fileUrl } from "../api";
import type { PageProps } from "../App";
import { Banner, Drop, ErrorBox, Pill, Stat } from "../components/bits";

export default function Motion({ id, refresh }: PageProps) {
  const [st, setSt] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const load = () => api.motion(id).then(setSt).catch(setErr);
  useEffect(() => { setSt(null); load(); }, [id]);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true); setErr(null);
    try { await fn(); await load(); refresh(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }

  if (!st) return <div className="faint">loading…</div>;

  if (st.blocked) {
    return (
      <>
        <h2>Motion</h2>
        <Banner kind="warn">{st.why}</Banner>
        <div className="faint">
          Motion goes on the slots that would otherwise sit at the 4s ceiling, and which slots those
          are is only knowable from measured Whisper holds — not from word-count estimates.
        </div>
      </>
    );
  }

  const p = st.proposal;
  const slots: any[] = st.slots ?? [];

  return (
    <>
      <div className="spread">
        <h2>Motion</h2>
        <Pill kind={p.short_by ? "warn" : "ok"}>
          {slots.length} slot(s) · {st.ready}/{slots.length} clips in
        </Pill>
      </div>
      <div className="faint">
        Clips come from Google Flow, external. This page places them — it never generates one.
      </div>
      <ErrorBox err={err} />

      <div className="card" style={{ marginTop: 10 }}>
        <div className="row" style={{ gap: 22 }}>
          <Stat label="candidates" value={String(p.candidates)}
                note={`holds in ${p.window[0]}–${p.window[1]}s`} />
          <Stat label="selected" value={String(p.selected)} />
          <Stat label="dose wants" value={`${p.want[0]}–${p.want[1]}`}
                note={`10–15% of ${(p.runtime / 60).toFixed(1)} min`} />
          <Stat label="clips in" value={`${st.ready}/${slots.length}`}
                note={st.waiting ? `${st.waiting} waiting on Flow` : "all here"} />
        </div>
      </div>

      {/* The shortfall is the finding, not an error. Say what causes it and
          what actually fixes it, rather than just colouring a number red. */}
      {p.short_by > 0 && (
        <Banner kind="warn">
          <strong>This video can only take {p.selected} clip(s), not {p.want[0]}–{p.want[1]}.</strong>
          <div style={{ marginTop: 6 }}>{p.note}</div>
        </Banner>
      )}
      {p.short_by === 0 && p.selected > 0 && (
        <Banner kind="ok">{p.note}</Banner>
      )}

      <div className="row" style={{ marginTop: 10 }}>
        <button className="primary" disabled={busy}
                onClick={() => act(() => api.saveMotion(id))}>
          {st.saved ? "Re-propose from measured holds" : "Save this selection"}
        </button>
        {st.saved && <span className="faint">saved to build/motion_slots.json — hand-editable</span>}
      </div>

      <h3>Slots</h3>
      {slots.length === 0 ? (
        <div className="card faint">No slot can carry a clip at this picture count.</div>
      ) : (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th className="num">slot</th><th className="num">hold</th><th className="num">section</th>
                <th>seed picture</th><th>flow clip</th><th></th>
              </tr>
            </thead>
            <tbody>
              {slots.map((s) => (
                <tr key={s.index}>
                  <td className="num mono">{s.index}</td>
                  <td className="num">{s.hold.toFixed(2)}s</td>
                  <td className="num">{s.section}</td>
                  <td className="faint mono">{s.filename}</td>
                  <td className="mono">
                    {s.clip ?? <span className="faint">drop {String(s.index).padStart(3, "0")}.mp4</span>}
                  </td>
                  <td>{s.have ? <Pill kind="ok">ready</Pill> : <Pill>waiting</Pill>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {/* Each motion slot still needs its seed picture - Flow is driven
              image-to-video, so the picture count does not drop. */}
          <div className="faint" style={{ marginTop: 8 }}>
            A motion slot <strong>replaces</strong> a still, it never adds runtime — and it still needs
            its seed picture, because Flow is driven image-to-video.
          </div>
        </div>
      )}

      <h3>Clips</h3>
      <Drop label={`Drop Flow clips here — name them <slot>.mp4 (e.g. 068.mp4) to auto-assign`}
            accept="video/*" busy={busy}
            onFiles={(f) => act(() => api.upload(`/api/projects/${id}/motion/clips`, f))} />

      {slots.filter((s) => s.have).length > 0 && (
        <div className="thumbs" style={{ marginTop: 12 }}>
          {slots.filter((s) => s.have).map((s) => (
            <div className="thumb" key={s.index}>
              <video src={fileUrl(s.clip_path)} controls muted
                     style={{ width: "100%", aspectRatio: "16/9", background: "#000" }} />
              <div className="cap">slot {s.index} · {s.hold.toFixed(2)}s · {s.clip}</div>
            </div>
          ))}
        </div>
      )}

      {st.clips_on_disk?.length > 0 && (
        <div className="faint mono" style={{ marginTop: 10 }}>
          in motion/: {st.clips_on_disk.join(", ")}
        </div>
      )}
    </>
  );
}
