// The landing page: every project that exists, and every topic that could be one.
//
// WHY THE TOPIC BANK IS HERE AND NOT A TEXT BOX. `run_script.py` accepts exactly
// 15 topic keys, each with curated source URLs - a free-text topic has nowhere to
// get its evidence from. Listing the bank makes that a feature rather than a
// wall you hit later: the fourteen unstarted topics are visible from the first
// screen, one click from existing.

import { useEffect, useState } from "react";
import { api, type ProjectStatus, type Topic } from "../api";
import { go } from "../App";
import { ErrorBox, Pill } from "../components/bits";

export default function Projects() {
  const [projects, setProjects] = useState<ProjectStatus[] | null>(null);
  const [topics, setTopics] = useState<Topic[] | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = () =>
    Promise.all([api.projects(), api.topics()])
      .then(([p, t]) => { setProjects(p); setTopics(t); })
      .catch(setErr);

  useEffect(() => { load(); }, []);

  async function create(t: Topic) {
    setBusy(t.id); setErr(null);
    try {
      await api.createProject(t.channel, t.topic);
      await load();
      go(t.id, "script");
    } catch (e) { setErr(e); } finally { setBusy(null); }
  }

  return (
    <>
      <h2>Projects</h2>
      <div className="faint">One folder per video. Pick one to work on, or start a topic.</div>
      <ErrorBox err={err} />

      <h3>In progress</h3>
      {projects === null ? <div className="faint">loading…</div>
        : projects.length === 0 ? <div className="card faint">No projects yet — start one below.</div>
        : (
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(310px,1fr))" }}>
            {projects.map((p) => <ProjectCard key={p.id} p={p} />)}
          </div>
        )}

      <h3>Topic bank {topics && <span className="faint">— {topics.filter((t) => !t.exists).length} not started</span>}</h3>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(310px,1fr))" }}>
        {topics?.map((t) => (
          <div key={t.id} className="card" style={{ opacity: t.exists ? 0.55 : 1 }}>
            <div className="spread">
              <strong>{t.title}</strong>
              {t.exists ? <Pill kind="ok">started</Pill> : <Pill>{t.sources} sources</Pill>}
            </div>
            <div className="faint mono" style={{ margin: "2px 0 8px" }}>{t.topic}</div>
            {t.stake && <div className="faint" style={{ marginBottom: 10 }}>Stake: {t.stake}</div>}
            {t.exists
              ? <button onClick={() => go(t.id, "script")}>Open</button>
              : <button className="primary" disabled={busy === t.id} onClick={() => create(t)}>
                  {busy === t.id ? "creating…" : "Start this topic"}
                </button>}
          </div>
        ))}
      </div>
    </>
  );
}

/** A project's real state, as flags rather than a guessed "current step".
 *  The backend deliberately reports facts; which stage is next is a judgement,
 *  and showing the facts lets the operator make it. */
function ProjectCard({ p }: { p: ProjectStatus }) {
  const steps: [string, boolean, string][] = [
    ["script", p.has_script, `${p.script_words} words`],
    ["sections", p.sections > 0, `${p.sections}`],
    ["images", p.images > 0, `${p.images}`],
    ["plan", p.has_plan, p.has_plan ? `${p.plan_scenes} scenes` : "—"],
    ["voice", p.has_render_plan, p.has_render_plan ? "aligned" : `${p.recordings} takes`],
    ["video", p.has_video, p.has_video ? `${p.video_mb} MB` : "—"],
  ];
  return (
    <div className="card">
      <div className="spread">
        <strong><a href={`#/${p.id}/script`}>{p.topic}</a></strong>
        <span className="faint mono">{p.channel}</span>
      </div>
      <table style={{ marginTop: 8 }}>
        <tbody>
          {steps.map(([name, done, note]) => (
            <tr key={name}>
              <td style={{ padding: "3px 0", border: 0, width: 22 }}>
                <span style={{ color: done ? "var(--ok)" : "var(--faint)" }}>{done ? "●" : "○"}</span>
              </td>
              <td style={{ padding: "3px 0", border: 0 }} className={done ? "" : "faint"}>{name}</td>
              <td style={{ padding: "3px 0", border: 0 }} className="faint mono num">{note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
