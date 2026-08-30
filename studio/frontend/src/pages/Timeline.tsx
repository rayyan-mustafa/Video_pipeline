// The timeline editor (Phase 5): roll edits, replace, reorder, and the motion
// toggle - the one piece of the pipeline that was still entirely manual, per
// STUDIO_HANDOFF.md. Everything here edits build/overrides.json through the
// backend, which recomputes render_plan.json through the SAME
// imageplan.plan_from_binding + motionplan.attach_kinds the CLI's --stage
// plan uses - so what this page shows is never a guess at what the real
// build would do, it IS what the real build will do.
//
// THREE INTERACTIONS, THREE MECHANISMS:
//   roll edit   drag the thin handle between two slots - a pointer drag,
//               snapped to the nearest real Whisper word boundary on release.
//   replace     click a slot, pick a different file from the ones already on
//               disk - a plain select, not a drag target.
//   reorder     drag one slot onto another - native HTML5 drag/drop, which
//               swaps their assigned files (two replaces, per the spec).
//
// A slot can also be toggled between still and motion, but ONLY among slots
// whose hold already fits pacing's [3.5, 4.0] window - the backend refuses
// (with a warning) anything outside it, so the picker here just doesn't
// offer the option rather than letting you hit a wall after saving.

import { useEffect, useMemo, useRef, useState } from "react";
import { api, fileUrl, type Slot, type TimelineData } from "../api";
import type { PageProps } from "../App";
import { Banner, ErrorBox, JobLog, Pill, Stat } from "../components/bits";

const basename = (p: string) => p.replace(/\\/g, "/").split("/").pop() || p;
const mmss = (t: number) => {
  const s = Math.max(0, Math.round(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

export default function Timeline({ id, refresh }: PageProps) {
  const [data, setData] = useState<TimelineData | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [pxPerSec, setPxPerSec] = useState(24);

  const load = () => api.timeline(id).then(setData).catch(setErr);
  useEffect(() => { setData(null); setSelected(null); setErr(null); load(); }, [id]);

  async function patch(p: Parameters<typeof api.saveTimeline>[1]) {
    setBusy(true); setErr(null);
    try { const d = await api.saveTimeline(id, p); setData(d); refresh(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }

  if (!data) return <div className="faint">loading…</div>;

  if (!data.ready) {
    return (
      <>
        <h2>Timeline</h2>
        <Banner kind="warn">{data.why}</Banner>
      </>
    );
  }

  const total = data.slots.length
    ? data.slots[data.slots.length - 1].end
    : 0;
  const sel = selected != null ? data.slots.find((s) => s.index === selected) ?? null : null;

  return (
    <>
      <div className="spread">
        <h2>Timeline</h2>
        <div className="row">
          <button disabled={pxPerSec <= 6} onClick={() => setPxPerSec((z) => Math.max(6, z - 6))}>−</button>
          <span className="faint mono">{pxPerSec}px/s</span>
          <button onClick={() => setPxPerSec((z) => Math.min(80, z + 6))}>+</button>
        </div>
      </div>
      <div className="faint">
        {data.slots.length} slot(s) over {mmss(total)} — drag a boundary to roll a cut, click a
        slot to replace it or toggle motion, drag one slot onto another to swap them.
      </div>
      <ErrorBox err={err} />

      {data.warnings.length > 0 && (
        <Banner kind="warn">
          {data.warnings.map((w, i) => <div key={i}>{w}</div>)}
        </Banner>
      )}

      <div style={{ marginTop: 10 }}>
        <TimelineTrack data={data} pxPerSec={pxPerSec} selected={selected}
                       onSelect={setSelected} onRollEdit={(idx, t) => patch({ cuts: { [idx]: t } })}
                       onSwap={(a, b) => {
                         const A = data.slots[a], B = data.slots[b];
                         patch({ assets: { [a]: basename(B.path), [b]: basename(A.path) } });
                       }} busy={busy} />
      </div>

      {sel && (
        <SlotDetail slot={sel} data={data} busy={busy} onPatch={patch}
                   onClose={() => setSelected(null)} />
      )}

      <div className="row" style={{ marginTop: 14 }}>
        <button className="danger" disabled={busy}
                onClick={() => confirm(
                  "Discard every roll edit, replacement and motion toggle on this timeline?",
                ) && (async () => {
                  setBusy(true); setErr(null);
                  try { setData(await api.revertTimeline(id)); refresh(); }
                  catch (e) { setErr(e); } finally { setBusy(false); }
                })()}>
          Revert all edits
        </button>
        <span className="faint">writes build/render_plan.json fresh from script_plan.json, no overrides</span>
      </div>

      <RenderPanel id={id} refresh={refresh} />
    </>
  );
}

/** The scrollable strip of slots, the ruler above it, and the roll-edit
 *  handles between them. Absolute-positioned by time * pxPerSec rather than
 *  flex, so a handle's drag delta converts to a time delta with no layout
 *  reflow to fight. */
function TimelineTrack({ data, pxPerSec, selected, onSelect, onRollEdit, onSwap, busy }: {
  data: TimelineData; pxPerSec: number; selected: number | null;
  onSelect: (i: number) => void;
  onRollEdit: (boundaryIndex: number, t: number) => void;
  onSwap: (a: number, b: number) => void;
  busy: boolean;
}) {
  const { slots, words } = data;
  const total = slots.length ? slots[slots.length - 1].end : 0;
  const width = Math.max(200, total * pxPerSec);

  // Local drag preview so the handle follows the pointer smoothly; the real
  // save (and the snap) only happens on release.
  const [drag, setDrag] = useState<{ boundary: number; t: number } | null>(null);
  const dragInfo = useRef<{ boundary: number; startX: number; startT: number } | null>(null);

  function snap(t: number): number {
    if (!words.length) return Math.round(t * 1000) / 1000;
    let best = t, bestDist = Infinity;
    for (const w of words) {
      for (const cand of [w.s, w.e]) {
        const d = Math.abs(cand - t);
        if (d < bestDist) { bestDist = d; best = cand; }
      }
    }
    return Math.round(best * 1000) / 1000;
  }

  // `boundary` is the index of the slot whose START moves - i.e. dragging the
  // handle on slot i's right edge moves the boundary belonging to slot i+1.
  function down(boundary: number, e: React.PointerEvent) {
    // Stop this from ALSO being read as the start of the slot's own native
    // drag (used for reorder) - without both of these, grabbing the edge
    // pixel reliably fires dragstart on the slot instead of a pointer drag on
    // the handle, because the handle sits inside a draggable ancestor.
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    const startT = slots[boundary].start;
    dragInfo.current = { boundary, startX: e.clientX, startT };
    setDrag({ boundary, t: startT });
  }
  function move(e: React.PointerEvent) {
    if (!dragInfo.current) return;
    const { boundary, startX, startT } = dragInfo.current;
    const t = startT + (e.clientX - startX) / pxPerSec;
    setDrag({ boundary, t: Math.max(0, Math.min(total, t)) });
  }
  function up(e: React.PointerEvent) {
    if (!dragInfo.current) return;
    e.stopPropagation();
    const { boundary, startT } = dragInfo.current;
    const t = snap(drag ? drag.t : startT);
    dragInfo.current = null;
    setDrag(null);
    // A plain click (no real movement) should not fire a save - it would just
    // re-save the same cut the plan already has.
    if (Math.abs(t - startT) > 0.005) onRollEdit(boundary, t);
  }

  const ticks = [];
  const step = pxPerSec >= 40 ? 5 : pxPerSec >= 15 ? 15 : 30;
  for (let t = 0; t <= total; t += step) ticks.push(t);

  return (
    <div className="tl-scroll">
      <div style={{ width }}>
        <div className="tl-ruler">
          {ticks.map((t) => (
            <span key={t} style={{ left: t * pxPerSec }}>{mmss(t)}</span>
          ))}
        </div>
        <div className="tl-track">
          {slots.map((s, i) => {
            // A boundary being dragged moves TWO slots' edges at once - the
            // one being dragged (its right edge) and its neighbour (whose
            // left edge is the exact same point) - both read from the same
            // `drag.t` so they always meet with no gap or overlap mid-drag.
            const draggingMe = drag?.boundary === i + 1;
            const effStart = drag?.boundary === i ? drag!.t : s.start;
            const effEnd = draggingMe ? drag!.t : s.end;
            const left = effStart * pxPerSec;
            const w = Math.max(2, (effEnd - effStart) * pxPerSec);
            const cls = ["tl-slot",
              s.index === selected ? "selected" : "",
              s.is_motion ? "motion" : "",
              !s.has_file ? "missing" : ""].filter(Boolean).join(" ");
            const hasHandle = i < slots.length - 1;
            return (
              <div key={s.index} className={cls}
                   style={{ left, width: w }}
                   title={`${basename(s.path)} · ${s.duration.toFixed(2)}s`}
                   draggable={!busy}
                   onDragStart={(e) => e.dataTransfer.setData("text/plain", String(s.index))}
                   onDragOver={(e) => e.preventDefault()}
                   onDrop={(e) => {
                     e.preventDefault();
                     const from = Number(e.dataTransfer.getData("text/plain"));
                     if (!Number.isNaN(from) && from !== s.index) onSwap(from, s.index);
                   }}
                   onClick={() => onSelect(s.index)}>
                {s.has_file && w > 14 && (
                  s.is_motion
                    ? <div style={{ width: "100%", height: "100%", display: "flex",
                                    alignItems: "center", justifyContent: "center", fontSize: 16 }}>▶</div>
                    : <img src={fileUrl(s.abs_path)} loading="lazy" alt="" />
                )}
                <span className="tl-idx">{s.index}</span>
                {s.is_motion && <span className="tl-mo">mo</span>}

                {/* The resize handle: THIS slot's own right edge, not a
                    separate element floating between two slots - so there is
                    never any ambiguity about which slot's screentime a drag
                    here changes. draggable=false plus stopping the pointer
                    event here is what keeps the parent's reorder-drag from
                    hijacking a grab that starts right on the edge pixel. */}
                {hasHandle && (
                  <div className={`tl-handle ${draggingMe ? "active" : ""}`}
                       draggable={false}
                       onPointerDown={(e) => down(i + 1, e)}
                       onPointerMove={move}
                       onPointerUp={up}
                       onClick={(e) => e.stopPropagation()} />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** Everything about one selected slot: replace its file, and - only when its
 *  hold actually fits a 4s clip - toggle it to motion. */
function SlotDetail({ slot, data, busy, onPatch, onClose }: {
  slot: Slot; data: TimelineData; busy: boolean;
  onPatch: (p: Parameters<typeof api.saveTimeline>[1]) => void;
  onClose: () => void;
}) {
  const currentAsset = data.overrides.assets[String(slot.index)] ?? basename(slot.path);
  const currentMotion = data.overrides.motion[String(slot.index)]
    ?? (slot.is_motion && slot.clip_path ? basename(slot.clip_path) : "off");

  return (
    <div className="card tl-detail">
      <div className="spread">
        <strong>Slot {slot.index}</strong>
        <button onClick={onClose}>close</button>
      </div>
      <div className="row" style={{ marginTop: 6, gap: 22 }}>
        <Stat label="section" value={String(slot.section)} />
        <Stat label="start" value={`${slot.start.toFixed(2)}s`} />
        <Stat label="hold" value={`${slot.duration.toFixed(2)}s`} />
        <Stat label="file" value={basename(slot.path)} />
      </div>
      {!slot.has_file && <Banner kind="warn">This file is not on disk yet.</Banner>}

      <div className="row" style={{ marginTop: 10 }}>
        <label className="faint">replace with</label>
        <select disabled={busy} value={currentAsset}
                onChange={(e) => onPatch({ assets: { [slot.index]: e.target.value } })}>
          <option value={basename(slot.path)}>{basename(slot.path)} (current)</option>
          {data.available_images
            .filter((n) => n !== basename(slot.path))
            .map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        {data.overrides.assets[String(slot.index)] && (
          <button disabled={busy}
                  onClick={() => onPatch({ assets: { [slot.index]: null } })}>
            revert to planned
          </button>
        )}
      </div>
      {data.available_images.length === 0 && (
        <div className="faint" style={{ marginTop: 6 }}>
          No image picker available for this project - see the warning above the timeline.
        </div>
      )}

      <div className="row" style={{ marginTop: 10 }}>
        <label className="faint">motion</label>
        {slot.fits_motion ? (
          <>
            <select disabled={busy} value={currentMotion}
                    onChange={(e) => onPatch({ motion: { [slot.index]: e.target.value } })}>
              <option value="off">still</option>
              {data.available_clips.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            {data.overrides.motion[String(slot.index)] && (
              <button disabled={busy}
                      onClick={() => onPatch({ motion: { [slot.index]: null } })}>
                revert to planned
              </button>
            )}
          </>
        ) : (
          <span className="faint">
            {slot.duration.toFixed(2)}s hold — outside the 3.5–4.0s window a clip can cover
          </span>
        )}
      </div>
    </div>
  );
}

/** The local half of Phase 6: render a real video and preview it here. */
function RenderPanel({ id, refresh }: { id: string; refresh: () => void }) {
  const [st, setSt] = useState<any>(null);
  const [motion, setMotion] = useState("kenburns");
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  const load = () => api.renderStatus(id).then((s) => {
    setSt(s);
    setMotion((m) => s.motion_choices.includes(m) ? m : s.motion_choices[0]);
  }).catch(setErr);
  useEffect(() => { load(); }, [id]);

  if (!st) return null;

  return (
    <>
      <h3 style={{ marginTop: 22 }}>Render &amp; preview</h3>
      <div className="card">
        <div className="row">
          <label className="faint">motion</label>
          <select value={motion} onChange={(e) => setMotion(e.target.value)}>
            {st.motion_choices.map((m: string) => <option key={m} value={m}>{m}</option>)}
          </select>
          <div style={{ flex: 1 }} />
          <button className="primary" disabled={!st.ready || !!jobId}
                  onClick={async () => {
                    setErr(null);
                    try { const job = await api.runRender(id, { motion }); setJobId(job.id); }
                    catch (e) { setErr(e); }
                  }}>
            {st.ready ? "Render" : "needs a plan + voice first"}
          </button>
        </div>
        <ErrorBox err={err} />

        {jobId && (
          <div style={{ marginTop: 10 }}>
            <JobLog jobId={jobId} onDone={(status) => { if (status === "done") { load(); refresh(); } setJobId(null); }} />
          </div>
        )}

        {st.has_video && (
          <div style={{ marginTop: 12 }}>
            <div className="spread">
              <span className="faint">{st.video_mb} MB</span>
              <Pill kind="ok">rendered</Pill>
            </div>
            {/* key AND a cache-busting query param, both keyed on mtime - the
                output path never changes across a re-render, so without this
                the browser (and the <video> element itself) has no signal
                that the bytes under that same URL are now different. */}
            <video key={st.video_mtime} controls
                   src={`${fileUrl(st.video_path)}&v=${st.video_mtime}`}
                   style={{ width: "100%", marginTop: 6, background: "#000" }} />
          </div>
        )}
      </div>
    </>
  );
}
