// The script, and what it implies about pacing before anything is recorded.
//
// The one thing this page must not let you do quietly: edit the script after
// planning. The image binding is fingerprinted against the exact sentence texts,
// so a changed script invalidates every picture's placement. The pipeline
// catches it and refuses - but by then you have lost the plan, so the warning
// belongs here, next to the textarea, before the edit.

import { useEffect, useState } from "react";
import { api } from "../api";
import type { PageProps } from "../App";
import { Banner, ErrorBox, JobLog, Pill, Stat } from "../components/bits";

export default function Script({ id, proj, refresh }: PageProps) {
  const [data, setData] = useState<any>(null);
  const [text, setText] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [genJobId, setGenJobId] = useState<string | null>(null);
  const [genRunning, setGenRunning] = useState(false);

  const load = () =>
    api.script(id).then((d) => { setData(d); setText(d.text); }).catch(setErr);

  useEffect(() => { setData(null); setGenJobId(null); setGenRunning(false); load(); }, [id]);

  const dirty = data && text !== data.text;

  async function save() {
    setBusy(true); setErr(null);
    try { await api.saveScript(id, text); await load(); refresh(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }

  async function generate() {
    if (text.trim() && !confirm(
      "Generate a new script from the AI pipeline for this project's topic?\n\n" +
      "This fills the textarea below when it finishes - it does not save over the " +
      "current script until you click Save.",
    )) return;
    setErr(null); setGenRunning(true);
    try { const job = await api.generateScript(id); setGenJobId(job.id); }
    catch (e) { setErr(e); setGenRunning(false); }
  }

  async function onGenerateDone(status: string) {
    setGenRunning(false);
    if (status !== "done") return;
    setErr(null);
    try { const r = await api.importGeneratedScript(id); setText(r.text); }
    catch (e) { setErr(e); }
  }

  if (!data) return <div className="faint">loading…</div>;

  const pac = data.pacing;

  return (
    <>
      <div className="spread">
        <h2>Script</h2>
        <Pill kind={data.bracket_ok ? "ok" : "bad"}>
          {data.bracket_ok ? "brackets ok" : `${data.brackets.length} stray bracket(s)`}
        </Pill>
      </div>
      <ErrorBox err={err} />

      <div className="card" style={{ marginTop: 10 }}>
        <div className="row" style={{ gap: 22 }}>
          <Stat label="words" value={String(data.words)} />
          <Stat label="estimated" value={`${data.est_minutes} min`} note={`at ${data.wpm} wpm`} />
          <Stat label="sections" value={String(data.sections.length)} />
          <Stat label="wants" value={`~${pac.options.find((o: any) => o.seconds_per_image === 3)?.images ?? "?"} images`}
            note="at the 3.0s band midpoint" />
        </div>
      </div>

      {!data.bracket_ok && (
        <Banner kind="bad">
          Stray brackets would be read aloud: {data.brackets.slice(0, 5).join(" · ")}
        </Banner>
      )}

      {proj?.has_plan && dirty && (
        <Banner kind="bad">
          <strong>This project is already planned.</strong> Saving a changed script invalidates the
          fingerprint that binds all {proj.plan_images} pictures to their sentences — the plan will
          have to be re-run and the binding re-checked.
        </Banner>
      )}

      <h3>How many pictures the runtime implies</h3>
      <div className="card">
        <table>
          <thead>
            <tr><th className="num">seconds/image</th><th className="num">images</th><th>in the band</th></tr>
          </thead>
          <tbody>
            {pac.options.map((o: any) => (
              <tr key={o.seconds_per_image}>
                <td className="num">{o.seconds_per_image.toFixed(1)}</td>
                <td className="num">{o.images}</td>
                <td>{o.in_band ? <Pill kind="ok">yes</Pill> : <Pill>outside {pac.target_low}–{pac.target_high}s</Pill>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Split modes</h3>
      <div className="card">
        <div className="row" style={{ gap: 18 }}>
          {Object.entries(data.split_counts as Record<string, number>).map(([mode, n]) => (
            <Stat key={mode} label={mode} value={String(n)}
              note={pac.recommended_split === mode ? "recommended" : "scenes"} />
          ))}
        </div>
        {/* Scene count is not the cap on cut speed - words are. More pictures
            than scenes is word-cuts, which imageplan supports natively. */}
        <div className="faint" style={{ marginTop: 10 }}>
          More pictures than scenes is not a problem: cuts then land inside sentences at exact
          Whisper word timestamps rather than on sentence edges.
        </div>
      </div>

      <h3>Sections</h3>
      <div className="card">
        <table>
          <thead>
            <tr><th className="num">#</th><th className="num">words</th><th className="num">est.</th><th>opens</th></tr>
          </thead>
          <tbody>
            {data.sections.map((s: any) => (
              <tr key={s.n}>
                <td className="num">{s.n}</td>
                <td className="num">{s.words}</td>
                <td className="num">{s.est_seconds}s</td>
                <td className="faint">{s.preview}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="spread" style={{ marginTop: 18 }}>
        <h3 style={{ margin: 0 }}>The words</h3>
        <button disabled={genRunning} onClick={generate} title={proj?.topic}>
          {genRunning ? "generating…" : "Generate with AI"}
        </button>
      </div>
      <div className="faint" style={{ marginBottom: 8 }}>
        Runs the full pipeline (grounding → hook tournament → retention → humanizer →
        re-grounding) against this project's topic and drops the result in below —
        review it and click Save when you're happy, same as pasting text by hand.
      </div>
      {genJobId && (
        <div style={{ marginBottom: 12 }}>
          <JobLog jobId={genJobId} onDone={onGenerateDone} />
        </div>
      )}
      <textarea rows={20} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" disabled={!dirty || busy} onClick={save}>
          {busy ? "saving…" : "Save script"}
        </button>
        {dirty && <button disabled={busy} onClick={() => setText(data.text)}>Discard changes</button>}
        {dirty && <span className="faint">unsaved</span>}
      </div>
    </>
  );
}
