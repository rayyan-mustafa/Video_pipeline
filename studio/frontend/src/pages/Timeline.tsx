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

  // Playhead time, shared between the preview player (which drives it via the
  // audio element's own clock) and the track (which draws it and lets a click
  // on the ruler set it) - the whole point of one shared number is that
  // "what plays" and "where the line is" can never drift apart.
  const [time, setTime] = useState(0);
  // The render's motion provider, held HERE rather than inside the render
  // panel: the preview has to show the same choice the button will send, or
  // the page would be previewing one build and rendering another.
  const [motionMode, setMotionMode] = useState("kenburns");
  const audioRef = useRef<HTMLAudioElement>(null);
  const seek = (t: number) => {
    if (audioRef.current) audioRef.current.currentTime = t;
    setTime(t);
  };

  const load = () => api.timeline(id).then(setData).catch(setErr);
  useEffect(() => { setData(null); setSelected(null); setErr(null); setTime(0); load(); }, [id]);

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

      <PreviewPlayer id={id} data={data} time={time} onTime={setTime} audioRef={audioRef}
                     motionMode={motionMode} />

      <div style={{ marginTop: 10 }}>
        <TimelineTrack data={data} pxPerSec={pxPerSec} selected={selected} time={time}
                       onSelect={setSelected} onSeek={seek}
                       onRollEdit={(idx, t) => patch({ cuts: { [idx]: t } })}
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

      <RenderPanel id={id} refresh={refresh} motion={motionMode} setMotion={setMotionMode} />
    </>
  );
}

/** The scrollable strip of slots, the ruler above it, and the roll-edit
 *  handles between them. Absolute-positioned by time * pxPerSec rather than
 *  flex, so a handle's drag delta converts to a time delta with no layout
 *  reflow to fight. */
function TimelineTrack({ data, pxPerSec, selected, time, onSelect, onSeek, onRollEdit, onSwap, busy }: {
  data: TimelineData; pxPerSec: number; selected: number | null; time: number;
  onSelect: (i: number) => void;
  onSeek: (t: number) => void;
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
      <div style={{ width, position: "relative" }}>
        <div className="tl-ruler"
             onClick={(e) => {
               const rect = e.currentTarget.getBoundingClientRect();
               onSeek(Math.max(0, Math.min(total, (e.clientX - rect.left) / pxPerSec)));
             }}>
          {ticks.map((t) => (
            <span key={t} style={{ left: t * pxPerSec }}>{mmss(t)}</span>
          ))}
        </div>
        {/* The playhead - one line spanning ruler and track, sharing the same
            time the preview player's <audio> element is actually at, so
            "where the line is" and "what's playing" never disagree. */}
        <div className="tl-playhead" style={{ left: time * pxPerSec }} />
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

/** The pre-render preview: the mastered voiceover, playing against whichever
 *  slot its own timestamp falls in - a still, or a real motion clip synced to
 *  the same clock. This is what answers "does the cut work" WITHOUT waiting
 *  on ffmpeg to mux something to find out; the actual Render button below
 *  still exists for the real file, this is the fast loop before that.
 *
 *  A motion slot's clip plays muted (the voiceover is the only audio track)
 *  and is kept in step with the audio by setting its `currentTime` to
 *  `time - slot.start` whenever they drift more than a couple of frames -
 *  never every tick, or the constant micro-seeks would themselves cause the
 *  stutter this is trying to avoid. */
/** The CSS transform that frames a slot at one end of its move.
 *
 *  This is the browser's translation of what `motion.filter_chain` hands
 *  ffmpeg, and it lands on the same picture for the same reason: zoompan's
 *  `x = (iw - iw/zoom) * px` puts the crop window's left edge `px` of the way
 *  through the margin the zoom opened up, and `translate(-px*(z-1)*100%)`
 *  against `transform-origin: 0 0` slides the scaled element by exactly that
 *  margin fraction. px = 0 shows the left edge, 1 the right, 0.5 centred -
 *  same three anchors, same numbers, no second opinion about the framing.
 *
 *  It is applied to the WHOLE frame box rather than to the <img>, because
 *  ffmpeg pads to the frame FIRST and zoompans the padded result - so on a
 *  picture that is not 16:9, the white bars are inside the move, and the
 *  preview has to zoom them too or it would be flattering the render. */
const framing = (z: number, px: number, py: number) =>
  `translate(${(-px * (z - 1) * 100).toFixed(4)}%, ${(-py * (z - 1) * 100).toFixed(4)}%) scale(${z.toFixed(5)})`;

function PreviewPlayer({ id, data, time, onTime, audioRef, motionMode }: {
  id: string; data: TimelineData; time: number; onTime: (t: number) => void;
  audioRef: React.RefObject<HTMLAudioElement>;
  motionMode: string;
}) {
  const [voicePath, setVoicePath] = useState<string | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    api.renderStatus(id).then((s) => { if (live) setVoicePath(s.voice_path ?? null); })
                        .catch(() => { if (live) setVoicePath(null); });
    return () => { live = false; };
  }, [id]);

  const total = data.slots.length ? data.slots[data.slots.length - 1].end : 0;
  let active: Slot | null = data.slots[0] ?? null;
  for (const s of data.slots) { if (s.start <= time) active = s; else break; }

  const clipRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<Animation | null>(null);

  // ONLY the `kenburns` provider moves a still. `still` and `none` hold the
  // frame, and a motion slot is covered by its own Flow clip whatever the
  // provider is - so those three cases must preview as static, or the page
  // would be promising a move the render is not going to make.
  const mv = active?.move;
  const moves = motionMode === "kenburns" && !!active && !active.is_motion
                && !!mv && !mv.static;

  // One Animation object per slot, driven by the audio's clock rather than by
  // React: re-rendering 133 slots per frame to move one picture would cost
  // more than the move is worth, and the browser can interpolate a transform
  // off the main thread once it has been told the two ends and the duration.
  useEffect(() => {
    animRef.current?.cancel();
    animRef.current = null;
    const el = stageRef.current;
    if (!el || !active) return;
    if (!moves || !mv) { el.style.transform = "none"; return; }
    animRef.current = el.animate(
      [{ transform: framing(mv.z0, mv.px0, mv.py0) },
       { transform: framing(mv.z1, mv.px1, mv.py1) }],
      { duration: Math.max(1, active.duration * 1000), fill: "both", easing: "linear" },
    );
    animRef.current.pause();
    // `voicePath` is a dependency because the whole component renders `null`
    // until the render status arrives - so on first load these effects run
    // BEFORE the stage element exists, find no node, and would never run
    // again on their own: the slot has not changed, so nothing else in this
    // list has changed either. Without it the very first slot silently never
    // moves, and every slot you navigate to afterwards does.
  }, [active?.index, moves, mv?.name, mv?.z0, mv?.z1, voicePath]);

  // Scrub the move to wherever the playhead is - including while paused, so a
  // seek shows the frame as it will actually be composed at that instant and
  // not the start of the move.
  useEffect(() => {
    const anim = animRef.current;
    if (!anim || !active) return;
    anim.currentTime = Math.max(0, Math.min(active.duration, time - active.start)) * 1000;
  }, [time, active?.index]);

  useEffect(() => {
    const v = clipRef.current;
    if (!v || !active?.is_motion || !active.clip_abs_path) return;
    const local = Math.max(0, time - active.start);
    if (Math.abs(v.currentTime - local) > 0.12) v.currentTime = local;
  }, [time, active]);

  // Keep the clip's own play/pause - and the Ken Burns move's - in lockstep
  // with the voiceover, since neither carries audio and both are only ever a
  // silent visual layer over it.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onPlay = () => { clipRef.current?.play().catch(() => {}); animRef.current?.play(); };
    const onPause = () => { clipRef.current?.pause(); animRef.current?.pause(); };
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    if (!audio.paused) onPlay(); else onPause();
    return () => { audio.removeEventListener("play", onPlay); audio.removeEventListener("pause", onPause); };
  }, [active?.index, moves, voicePath]);

  if (voicePath === undefined) return null; // still loading render status
  if (voicePath === null) {
    return (
      <Banner kind="warn">
        No mastered voiceover yet — run the Voice stage first to preview the
        cut here. The strip below still shows the plan.
      </Banner>
    );
  }

  return (
    <div className="card tl-preview">
      <div className="tl-preview-frame">
        <div className="tl-preview-stage" ref={stageRef}>
          {active?.has_file ? (
            active.is_motion && active.clip_abs_path
              ? <video ref={clipRef} src={fileUrl(active.clip_abs_path)} muted playsInline />
              : <img src={fileUrl(active.abs_path)} alt="" />
          ) : (
            <div className="faint" style={{ display: "grid", placeItems: "center", height: "100%" }}>
              slot {active?.index} has no file on disk
            </div>
          )}
        </div>
        <span className="tl-preview-badge">
          {mmss(time)} / {mmss(total)} · slot {active?.index}
          {active?.is_motion ? " · motion clip"
            : moves ? ` · ${mv!.name.replace(/_/g, " ")}`
            : motionMode === "kenburns" ? " · static (accent)" : ""}
        </span>
      </div>
      <audio ref={audioRef} controls style={{ width: "100%", marginTop: 8 }}
             src={fileUrl(voicePath)}
             onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
             onSeeked={(e) => onTime(e.currentTarget.currentTime)} />
      <div className="faint" style={{ marginTop: 6, fontSize: 12 }}>
        Preview approximates the <strong>{motionMode}</strong> render: the move is the one
        pipeline/motion.py will use, interpolated by the browser rather than by ffmpeg's
        zoompan, and without the supersampling that keeps the real lines from shimmering.
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
          {/* Files whose name matches no slot - a one-off drop-in, or the
              generator's own descriptive title. Grouped rather than hidden:
              an override is resolved by name, so these render fine; they are
              only excluded from what the PLAN picks on its own. */}
          {data.unconventional_images.length > 0 && (
            <optgroup label="not named for a slot">
              {data.unconventional_images
                .filter((n) => n !== basename(slot.path))
                .map((n) => <option key={n} value={n}>{n}</option>)}
            </optgroup>
          )}
        </select>
        {data.overrides.assets[String(slot.index)] && (
          <button disabled={busy}
                  onClick={() => onPatch({ assets: { [slot.index]: null } })}>
            revert to planned
          </button>
        )}
      </div>
      {data.available_images.length === 0 && data.unconventional_images.length === 0 && (
        <div className="faint" style={{ marginTop: 6 }}>
          {data.warnings.some((w) => w.startsWith("image picker unavailable"))
            ? "No image picker available for this project - see the warning above the timeline."
            : "No unused pictures to swap in - every conventionally-named file "
              + "in images/ is already assigned to a slot. Drop a new file into "
              + "images/ to make it available here."}
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

/** The render worker's URL and whether it is alive.
 *
 *  Deliberately NOT health-checked on mount: a dead cloudflared tunnel takes
 *  20s to time out and this panel renders on every visit to the page. The
 *  operator presses check when they have just started the notebook and
 *  actually want to know. */
function RenderWorkerBox({ initialUrl, last }: {
  initialUrl: string;
  last: { ok: boolean; at: string; detail: string } | null;
}) {
  const [url, setUrl] = useState(initialUrl);
  // Seeded from the LAST RECORDED CHECK so the box says something the moment
  // it opens. Probing on mount instead would cost a 20s timeout every time
  // the runtime has gone away, which is most of the time.
  const [state, setState] = useState<any>(
    last ? { ok: last.ok, error: last.ok ? "" : last.detail, stale: true } : null);
  const [busy, setBusy] = useState(false);

  async function saveAndCheck() {
    setBusy(true);
    try {
      await api.saveSettings({ colab_render_url: url.trim() });
      setState({ ...(await api.checkRenderWorker(url.trim())), stale: false });
    } catch (e) {
      setState({ ok: false, error: String(e) });
    } finally { setBusy(false); }
  }

  return (
    <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--line)" }}>
      <div className="row">
        <label className="faint">worker</label>
        <input style={{ flex: 1, minWidth: 260 }} value={url} placeholder="https://….trycloudflare.com"
               onChange={(e) => setUrl(e.target.value)} />
        <button disabled={busy || !url.trim()} onClick={saveAndCheck}>
          {busy ? "checking…" : "save & check"}
        </button>
        {state && (
          <Pill kind={state.ok ? "ok" : ""}>
            {state.ok ? (state.cpus ? `${state.cpus} cores` : "reachable")
                      : "not connected"}
          </Pill>
        )}
      </div>
      {state?.stale && last && (
        <div className="faint" style={{ marginTop: 6, fontSize: 12 }}>
          last checked {new Date(last.at).toLocaleString()} — {last.detail}
        </div>
      )}
      {state && !state.ok && state.error && (
        <div className="faint" style={{ marginTop: 6, color: "var(--bad)" }}>
          {state.error}{state.hint ? ` — ${state.hint}` : ""}
        </div>
      )}
      <div className="faint" style={{ marginTop: 6, fontSize: 12 }}>
        Run <code>studio/colab_render.py</code> in one Colab cell on a <strong>CPU</strong> runtime
        and paste the URL it prints. Not a T4 — rendering is pure ffmpeg, and a GPU
        runtime bills by wall-clock at GPU rates for work the GPU never touches
        (the worker refuses to start on one). The whole project is uploaded as a
        bundle, so the first render pays for the upload.
      </div>
    </div>
  );
}

/** Render a real video - here or on a CPU runtime - and preview it. */
function RenderPanel({ id, refresh, motion, setMotion }: {
  id: string; refresh: () => void;
  motion: string; setMotion: (m: string) => void;
}) {
  const [st, setSt] = useState<any>(null);
  const [target, setTarget] = useState("local");
  const [jobId, setJobId] = useState<string | null>(null);
  // Tracked apart from `jobId` because the log stays mounted after the job
  // ends - so "is one running" can no longer be read as "is a log showing".
  const [jobStatus, setJobStatus] = useState<string>("");
  const [err, setErr] = useState<unknown>(null);

  const load = () => api.renderStatus(id).then((s) => {
    setSt(s);
    if (s.render_target) setTarget(s.render_target);
    // The choice lives in the parent now (the preview reads it too), so fall
    // back to the first offered provider only when the held one is not on
    // this backend's list at all.
    if (!s.motion_choices.includes(motion)) setMotion(s.motion_choices[0]);
  }).catch(setErr);

  useEffect(() => {
    setJobId(null); setJobStatus(""); setSt(null);
    load();
    // SHOW THE LAST RENDER'S LOG ON ARRIVAL, not just during the run that
    // produced it. A render that failed while the operator was on another
    // page - or before a reload - otherwise left no trace anywhere in the UI,
    // and "why is there still no video" had no answer short of the terminal.
    api.jobs(id).then((all) => {
      const mine = all.filter((j) => j.kind === "mux" || j.kind === "mux-colab");
      const last = mine.sort((a, b) => b.started - a.started)[0];
      if (last) { setJobId(last.id); setJobStatus(last.status); }
    }).catch(() => {});
  }, [id]);

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
          <label className="faint" style={{ marginLeft: 14 }}>run on</label>
          <select value={target} onChange={(e) => {
                    setTarget(e.target.value);
                    // Persisted so the choice survives a reload - the backend
                    // falls back to this when a render arrives with no target.
                    api.saveSettings({ render_target: e.target.value }).catch(() => {});
                  }}>
            {(st.targets ?? ["local"]).map((t: string) => (
              <option key={t} value={t}>{t === "local" ? "this laptop" : "Colab CPU"}</option>
            ))}
          </select>
          <div style={{ flex: 1 }} />
          <button className="primary"
                  disabled={!st.ready || jobStatus === "running"}
                  onClick={async () => {
                    setErr(null);
                    try {
                      const job = await api.runRender(id, { motion, target });
                      setJobId(job.id); setJobStatus("running");
                    } catch (e) { setErr(e); }
                  }}>
            {jobStatus === "running" ? "rendering…"
              : st.ready ? "Render" : "needs a plan + voice first"}
          </button>
        </div>

        {target === "colab" && (
          <RenderWorkerBox initialUrl={st.render_worker_url ?? ""}
                           last={st.render_worker_last ?? null} />
        )}

        <ErrorBox err={err} />

        {jobId && (
          <div style={{ marginTop: 10 }}>
            {/* THE LOG STAYS UP AFTER THE JOB ENDS. It used to clear `jobId`
                here, which unmounted this the instant the render finished -
                so a FAILURE flashed once and vanished, and the only way to
                find out why a render had not appeared was the terminal. A
                finished job's stream replays its backlog (see
                jobs.registry.stream), so leaving it mounted costs nothing
                and a reload still shows it. */}
            <JobLog jobId={jobId}
                    onDone={(status) => {
                      setJobStatus(status);
                      if (status === "done") { load(); refresh(); }
                    }} />
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
