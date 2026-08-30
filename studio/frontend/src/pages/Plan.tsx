// The plan binds each picture to the sentences it illustrates.
//
// `match` is the default and the only mode that cannot be re-derived from the
// script alone - its scenes come from the image folder, so they have to travel
// inside script_plan.json. The other modes split the script by a rule and are
// reproducible from the text.

import { useEffect, useState } from "react";
import { api, fileUrl } from "../api";
import type { PageProps } from "../App";
import { Banner, ErrorBox, JobLog, PacingCard, Pill, Stat } from "../components/bits";

const SPLITS = [
  ["match", "one scene per picture already on disk (1:1 binding)"],
  ["chunk", "aligner chunks — the coarsest split"],
  ["sentence", "one scene per sentence"],
  ["dash", "sentences, further split on dashes — the finest"],
] as const;

export default function Plan({ id, proj, refresh }: PageProps) {
  const [plan, setPlan] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [promptsJobId, setPromptsJobId] = useState<string | null>(null);
  const [split, setSplit] = useState("match");
  const [spi, setSpi] = useState(3);

  const load = () =>
    api.plan(id).then(setPlan).catch((e) => {
      if ((e as any).status === 404) setPlan(null); else setErr(e);
    });

  useEffect(() => { setPlan(null); setJobId(null); setPromptsJobId(null); load(); }, [id]);

  async function run() {
    setBusy(true); setErr(null);
    try {
      const job = await api.runPlan(id, { split, seconds_per_image: spi });
      setJobId(job.id);
    } catch (e) { setErr(e); } finally { setBusy(false); }
  }

  async function runPrompts() {
    setBusy(true); setErr(null);
    try {
      const job = await api.generatePrompts(id);
      setPromptsJobId(job.id);
    } catch (e) { setErr(e); } finally { setBusy(false); }
  }

  const images: any[] = plan?.images ?? [];

  return (
    <>
      <div className="spread">
        <h2>Plan</h2>
        {plan && <Pill kind="ok">{plan.split} · {plan.scenes?.length ?? 0} scenes</Pill>}
      </div>
      <div className="faint">Binds each picture to the sentences it illustrates, and fingerprints them.</div>
      <ErrorBox err={err} />

      <div className="card" style={{ marginTop: 10 }}>
        <div className="row">
          <label className="faint">split</label>
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            {SPLITS.map(([v]) => <option key={v} value={v}>{v}</option>)}
          </select>
          <label className="faint">seconds/image</label>
          <input type="number" step={0.5} min={0.5} max={15} value={spi} style={{ width: 74 }}
                 onChange={(e) => setSpi(Number(e.target.value))} />
          <div style={{ flex: 1 }} />
          <button className="primary" disabled={busy} onClick={run}>
            {plan ? "Re-run plan" : "Run plan"}
          </button>
        </div>
        <div className="faint" style={{ marginTop: 8 }}>
          {SPLITS.find(([v]) => v === split)?.[1]}
        </div>
        {split === "match" && (proj?.images ?? 0) === 0 && (
          <Banner kind="warn">
            <code>match</code> takes its scene count from the images on disk, and there are none yet.
            Upload pictures first, or pick another split.
          </Banner>
        )}
      </div>

      {jobId && (
        <div style={{ marginTop: 12 }}>
          <JobLog jobId={jobId} onDone={() => { load(); refresh(); }} />
        </div>
      )}

      {!plan ? (
        <div className="card faint" style={{ marginTop: 12 }}>No plan yet — run one above.</div>
      ) : (
        <>
          <div className="card" style={{ marginTop: 12 }}>
            <div className="row" style={{ gap: 22 }}>
              <Stat label="scenes" value={String(plan.scenes?.length ?? 0)} />
              <Stat label="images" value={String(images.length)} />
              <Stat label="split" value={plan.split} />
              <Stat label="fingerprint" value={String(plan.fingerprint ?? "—").slice(0, 12)}
                    note="binds pictures to sentence texts" />
              <div style={{ flex: 1 }} />
              <button disabled={busy} onClick={runPrompts}>
                Generate Prompts (Beat Sheet)
              </button>
            </div>
            
            {promptsJobId && (
              <div style={{ marginTop: 16, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
                <h4>Generating Prompts</h4>
                <div className="faint" style={{ marginBottom: 8 }}>
                  Generating beat sheet and prompts for each scene using OpenRouter.
                  You can view the final <code>shot_sheet.txt</code> in the project directory when done.
                </div>
                <JobLog jobId={promptsJobId} onDone={() => { /* nothing to reload */ }} />
              </div>
            )}
          </div>

          {plan.pacing && <div style={{ marginTop: 12 }}><PacingCard p={plan.pacing} title="Planned pacing (estimated)" /></div>}
          {plan.pacing && (
            <div className="faint" style={{ marginTop: 6 }}>
              Estimated from word counts. The real numbers come from the Voice stage — only measured
              Whisper holds decide which slots actually sit at the ceiling.
            </div>
          )}

          <h3>Bound pictures</h3>
          <div className="thumbs">
            {images.slice(0, 60).map((im: any, i: number) => (
              <div className="thumb" key={i}>
                {im.abs_path && <img src={fileUrl(im.abs_path)} loading="lazy" alt="" />}
                <div className="cap">{i + 1}. {String(im.path ?? "").split(/[\\/]/).pop()}</div>
              </div>
            ))}
          </div>
          {images.length > 60 && <div className="faint" style={{ marginTop: 8 }}>+{images.length - 60} more</div>}
        </>
      )}
    </>
  );
}
