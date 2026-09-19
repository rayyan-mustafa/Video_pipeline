// The voice stage: takes in, naming resolved, Whisper alignment out.
//
// The page is deliberately in two halves, because they fail for different
// reasons and the operator needs to know which one they are looking at.
//
//   NAMING   which file is which section. Inferred by measurement, shown with
//            its evidence and its margin, applied only on confirmation.
//   ALIGN    the expensive part. Cannot start until naming resolves, so the
//            button is disabled rather than failing four minutes in.
//
// The results panel reports ANCHORED separately from word-match %, because they
// are different things and conflating them sends people off to re-record a take
// that was fine. See the note in voicestage.results().

import { useEffect, useState } from "react";
import { api, fileUrl, type VoiceStatus } from "../api";
import type { PageProps } from "../App";
import { Banner, Drop, ErrorBox, JobLog, PacingCard, Pill, Stat } from "../components/bits";

export default function Voice({ id, refresh }: PageProps) {
  const [st, setSt] = useState<VoiceStatus | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [model, setModel] = useState<string>("");
  const [profile, setProfile] = useState<string>("natural");
  const [useColab, setUseColab] = useState(false);

  const load = () =>
    api.voice(id).then((s) => {
      setSt(s);
      setModel((m) => m || s.default_model);
      // Default to the GPU when one is reachable: it is both faster and
      // cheaper than a local large-v3 run, and it is what ships.
      setUseColab((u) => u || s.colab.ok);
    }).catch(setErr);

  useEffect(() => {
    setSt(null); setJobId(null);
    load();
    // SHOW THE LAST ALIGNMENT'S LOG ON ARRIVAL. Same reasoning as the render
    // panel: a run that failed while the operator was on another page, or
    // before a reload, otherwise left nothing behind to read.
    api.jobs(id).then((all) => {
      const last = all.filter((j) => j.kind === "voice")
                      .sort((a, b) => b.started - a.started)[0];
      if (last) setJobId(last.id);
    }).catch(() => {});
  }, [id]);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true); setErr(null);
    try { await fn(); await load(); refresh(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }

  if (!st) return <div className="faint">loading…</div>;

  const p = st.proposal;
  const canApply = p.status === "ok" && p.needs_rename;
  const canForce = p.status === "ambiguous";

  return (
    <>
      <div className="spread">
        <h2>Voice</h2>
        <Pill kind={st.ready ? "ok" : "warn"}>
          {st.ready ? "ready to align"
            : st.n_sections === 0 ? "no sections yet"
            : `${st.missing_sections.length} section(s) unresolved`}
        </Pill>
      </div>
      <div className="faint">
        {st.n_files} take(s) on disk for {st.n_sections} section(s).
      </div>
      <ErrorBox err={err} />

      {/* ---------------------------------------------------------- naming */}
      <h3>1 · Which take is which section</h3>

      {p.status === "already" && (
        <Banner kind="ok">Every section is already named <code>section_NN</code> — nothing to resolve.</Banner>
      )}
      {p.status === "impossible" && (
        st.n_sections === 0
          ? <Banner kind="warn">
              No sections yet — write the script and split it first. Section word counts are what
              identify which take is which.
            </Banner>
          : <Banner kind="bad">{p.note}</Banner>
      )}
      {p.status === "ambiguous" && (
        <Banner kind="warn">
          <strong>Too close to call.</strong> {p.note}
        </Banner>
      )}
      {p.status === "ok" && p.needs_rename && (
        <Banner kind="ok">
          Confident: {p.note}
        </Banner>
      )}

      {p.assignments.length > 0 && (
        <div className="card" style={{ marginTop: 10 }}>
          <table>
            <thead>
              <tr>
                <th>section</th><th>take(s)</th><th></th><th>will be named</th>
                <th className="num">words</th><th className="num">length</th><th className="num">wpm</th>
              </tr>
            </thead>
            <tbody>
              {p.assignments.map((a) => (
                <tr key={a.section}>
                  <td className="mono">{String(a.section).padStart(2, "0")}</td>
                  <td className="mono">{a.files.join(" + ") || <span className="faint">none</span>}</td>
                  <td className="faint">→</td>
                  <td className="mono">{a.targets.join(" + ")}</td>
                  <td className="num">{a.words}</td>
                  <td className="num">{a.seconds ? `${a.seconds.toFixed(1)}s` : "—"}</td>
                  <td className="num">{a.wpm ? a.wpm.toFixed(0) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {/* The margin IS the evidence. A mapping that only just wins is not
              one to trust to arithmetic, and hiding that would be dishonest. */}
          {p.spread > 0 && (
            <div className="faint" style={{ marginTop: 8 }}>
              Reader holds a {p.spread.toFixed(0)} wpm spread on this mapping
              {p.runner_up > 0 && <> versus {p.runner_up.toFixed(0)} on the next-best grouping</>}.
              A consistent pace across sections is what identifies the right one.
            </div>
          )}
        </div>
      )}

      <div className="row" style={{ marginTop: 10 }}>
        <button className="primary" disabled={!canApply || busy}
                onClick={() => act(() => api.applyNaming(id))}>
          Apply this naming
        </button>
        {canForce && (
          <button disabled={busy}
                  onClick={() => confirm(
                    "The arithmetic could not choose confidently. Apply it anyway?\n\n" +
                    "Listen to the takes first — a wrong mapping wastes an alignment pass.",
                  ) && act(() => api.applyNaming(id, true))}>
            Apply anyway (I checked by ear)
          </button>
        )}
        {st.can_revert && (
          <button className="danger" disabled={busy}
                  onClick={() => act(() => api.revertNaming(id))}>
            Undo rename
          </button>
        )}
      </div>

      <div style={{ marginTop: 12 }}>
        <Drop label="Drop takes here, or click to choose — named anything, resolved above"
              accept="audio/*" busy={busy}
              onFiles={(f) => act(() => api.upload(`/api/projects/${id}/voice/recordings`, f))} />
      </div>

      {st.files.length > 0 && (
        <div className="faint mono" style={{ marginTop: 8 }}>
          on disk: {st.files.map((f) => f.name).join(", ")}
        </div>
      )}

      {/* ----------------------------------------------------------- align */}
      <h3>2 · Align with Whisper</h3>
      {!st.ready && (
        <Banner kind="warn">
          {st.n_sections === 0
            ? <>There are no sections to align yet.</>
            : <>Resolve the naming above first — section(s) {st.missing_sections.join(", ")} have no
               audio, and the alignment would refuse to start.</>}
        </Banner>
      )}
      {/* THE GPU WORKER. Only the transcribe step moves - mastering and
          alignment stay local either way, so switching this changes which
          Whisper weights produced the timestamps and nothing else. */}
      <ColabBox colab={st.colab} onChanged={load} />

      <div className="card" style={{ marginTop: 10 }}>
        <div className="row">
          <label className="faint">model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {st.models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>

          <label className="faint" style={{ marginLeft: 8 }}>voice</label>
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {st.voice_profiles.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>

          <label className="row" style={{ gap: 6, marginLeft: 8 }}>
            <input type="checkbox" checked={useColab} disabled={!st.colab.ok}
                   onChange={(e) => setUseColab(e.target.checked)} />
            <span className="faint">run on Colab GPU</span>
          </label>

          <div style={{ flex: 1 }} />
          <button className="primary" disabled={!st.ready || busy}
                  onClick={() => act(async () => {
                    const job = await api.runVoice(id, {
                      whisper: model, voice_profile: profile, use_colab: useColab,
                    });
                    setJobId(job.id);
                  })}>
            Run alignment
          </button>
        </div>

        <div className="faint" style={{ marginTop: 8 }}>
          {profile === "natural" && <>
            <strong>natural</strong> — pauses trimmed, level set to −14 LUFS, and nothing else.
            No denoise, EQ, de-esser, compressor or high-pass.
          </>}
          {profile === "clip" && <>
            <strong>clip</strong> — the full mastering chain including the silence trim.
          </>}
          {profile === "raw" && <>
            <strong>raw</strong> — level only. Every pause kept, so the video runs longer.
          </>}
        </div>

        {/* large-v3 is what ships but is ~10x base on CPU - which is the entire
            reason the GPU worker exists. Say so before, not after. */}
        {model === "large-v3" && !useColab && (
          <Banner kind="warn">
            <strong>large-v3 on this laptop is roughly 10× base</strong> — tens of minutes for a full
            video. On the Colab worker it is about 95 seconds. Tick “run on Colab GPU” above.
          </Banner>
        )}
        {useColab && (
          <div className="faint" style={{ marginTop: 6 }}>
            Only the transcribe step goes to the GPU. Mastering and alignment stay on this machine,
            so the result is identical apart from which weights read the audio.
          </div>
        )}
        <Banner kind="warn">
          Changing the model or the voice profile moves every timestamp, so the slots are re-cut and
          the shot sheet has to be regenerated. Decide both <em>before</em> generating pictures.
        </Banner>
      </div>

      {jobId && (
        <div style={{ marginTop: 12 }}>
          <JobLog jobId={jobId} onDone={() => { load(); refresh(); }} />
        </div>
      )}

      {/* --------------------------------------------------------- results */}
      {st.results && (
        <>
          <h3>3 · What the alignment produced</h3>
          <div className="card">
            <div className="row" style={{ gap: 22 }}>
              <Stat label="scenes" value={String(st.results.scenes)} />
              <Stat label="word timestamps" value={String(st.results.words)}
                    note="ground truth for every cut" />
              <Stat label="runtime" value={`${(st.results.runtime / 60).toFixed(2)} min`} />
            </div>
            {st.results.voice_path && (
              <audio controls src={fileUrl(st.results.voice_path)}
                     style={{ width: "100%", marginTop: 12 }} />
            )}
          </div>

          {st.results.pacing && (
            <div style={{ marginTop: 12 }}>
              <PacingCard p={st.results.pacing} title="Measured pacing" />
            </div>
          )}

          {/* Motion dose is only meaningful once holds are MEASURED - which
              slots sit at the ceiling is not knowable from word-count estimates.
              That is why this panel appears here and nowhere earlier. */}
          {st.results.motion_dose && (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="spread">
                <strong>Motion budget</strong>
                <span className="faint">
                  {st.results.motion_dose.clips_low}–{st.results.motion_dose.clips_high} clips
                </span>
              </div>
              <div className="faint" style={{ marginTop: 6 }}>
                At 10–15% of {(st.results.motion_dose.runtime / 60).toFixed(1)} min this video wants{" "}
                {st.results.motion_dose.clips_low}–{st.results.motion_dose.clips_high} external clips of{" "}
                {st.results.motion_dose.clip_s}s ({st.results.motion_dose.seconds_low}–
                {st.results.motion_dose.seconds_high}s of screen time). They go on the slots that would
                otherwise sit at the {st.results.motion_dose.window[1]}s ceiling, never on an accent.
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}


/** The GPU worker: paste a URL, see whether it answers.
 *
 *  A cloudflared quick tunnel gets a NEW hostname every time the notebook runs,
 *  so "the saved URL is dead" is the normal state rather than an error - the box
 *  says so plainly instead of hiding it behind a failed job four minutes later.
 */
function ColabBox({ colab, onChanged }: {
  colab: VoiceStatus["colab"]; onChanged: () => void;
}) {
  const [url, setUrl] = useState(colab.url || "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  async function save() {
    setBusy(true); setErr(null);
    try { await api.saveSettings({ colab_url: url.trim() }); onChanged(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }

  return (
    <div className="card" style={{ marginTop: 10 }}>
      <div className="spread">
        <strong>Colab GPU worker</strong>
        <Pill kind={colab.ok ? "ok" : ""}>
          {colab.ok ? `${colab.gpu} · ${colab.model}` : "not connected"}
        </Pill>
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <input type="text" value={url} placeholder="https://....trycloudflare.com"
               style={{ flex: 1, minWidth: 260 }}
               onChange={(e) => setUrl(e.target.value)} />
        <button disabled={busy} onClick={save}>Save &amp; check</button>
      </div>
      <ErrorBox err={err} />
      {!colab.ok && colab.error && (
        <div className="faint" style={{ marginTop: 8 }}>
          {colab.error}{colab.hint ? ` — ${colab.hint}` : ""}
        </div>
      )}
      <div className="faint" style={{ marginTop: 8 }}>
        Run <code>studio/colab_worker.py</code> in one Colab cell on a <strong>T4</strong> and paste
        the URL it prints. T4 costs about half what an L4 does for this job, because the session is
        dominated by setup rather than by GPU time.
      </div>
    </div>
  );
}
