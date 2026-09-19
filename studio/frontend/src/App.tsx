// The shell: which project is selected, which stage page is showing.
//
// ROUTING IS A HASH AND ~30 LINES, not a router dependency. The whole app is
// `#/<channel>/<topic>/<page>`, there are no nested routes and no loaders, and a
// local tool that one person runs does not need a router's code-splitting or its
// upgrade surface. Back/forward and a pasteable URL still work.

import { useCallback, useEffect, useState } from "react";
import { api, type ProjectStatus } from "./api";
import Projects from "./pages/Projects";
import Script from "./pages/Script";
import Plan from "./pages/Plan";
import Images from "./pages/Images";
import Voice from "./pages/Voice";
import Motion from "./pages/Motion";
import Timeline from "./pages/Timeline";
import Package from "./pages/Package";
import Jobs from "./pages/Jobs";

const PAGES = ["script", "plan", "images", "voice", "motion", "timeline",
               "package", "jobs"] as const;
type Page = (typeof PAGES)[number];

type Route = { id: string | null; page: Page };

function parse(): Route {
  const raw = location.hash.replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean);
  if (parts.length >= 3 && PAGES.includes(parts[2] as Page))
    return { id: `${parts[0]}/${parts[1]}`, page: parts[2] as Page };
  if (parts.length === 2) return { id: `${parts[0]}/${parts[1]}`, page: "script" };
  return { id: null, page: "script" };
}

export const go = (id: string | null, page: Page = "script") => {
  location.hash = id ? `#/${id}/${page}` : "#/";
};

export default function App() {
  const [route, setRoute] = useState<Route>(parse);
  const [proj, setProj] = useState<ProjectStatus | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; problems: string[] } | null>(null);

  useEffect(() => {
    const on = () => setRoute(parse());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);

  useEffect(() => { api.health().then(setHealth).catch(() => setHealth(null)); }, []);

  // Every stage page changes something the sidebar reports (a script gets
  // written, takes get renamed), so they call this rather than each keeping a
  // private copy of the project's state that would then go stale.
  const refresh = useCallback(() => {
    if (!route.id) { setProj(null); return; }
    api.project(route.id).then(setProj).catch(() => setProj(null));
  }, [route.id]);

  useEffect(() => { refresh(); }, [refresh]);

  const id = route.id;
  const P = { script: Script, plan: Plan, images: Images, voice: Voice,
              motion: Motion, timeline: Timeline, package: Package,
              jobs: Jobs }[route.page];

  return (
    <div className="shell">
      <aside className="side">
        <h1>Video Studio</h1>
        <div className="sub">{health ? (health.ok ? "backend ok" : "backend has problems") : "backend unreachable"}</div>

        <a className={`nav ${!id ? "on" : ""}`} href="#/">Projects</a>

        {id && (
          <>
            <div className="faint" style={{ margin: "14px 8px 4px" }}>{id}</div>
            <Nav id={id} page="script" now={route.page} label="Script"
                 note={proj ? `${proj.script_words} words · ${proj.sections} sections` : ""} />
            <Nav id={id} page="plan" now={route.page} label="Plan"
                 note={proj?.has_plan ? `${proj.plan_scenes} scenes · ${proj.plan_split}` : "not planned"} />
            <Nav id={id} page="images" now={route.page} label="Images"
                 note={proj ? `${proj.images} on disk` : ""} />
            <Nav id={id} page="voice" now={route.page} label="Voice"
                 note={proj?.has_render_plan ? "aligned" : proj ? `${proj.recordings} takes` : ""} />
            <Nav id={id} page="motion" now={route.page} label="Motion"
                 note={proj?.has_render_plan ? "flow clips" : "needs measured holds"} />
            <Nav id={id} page="timeline" now={route.page} label="Timeline"
                 note={proj?.has_render_plan ? "edit cuts & render" : "needs a plan first"} />
            <Nav id={id} page="package" now={route.page} label="Package"
                 note="title · tags · thumbnail prompt" />
            <Nav id={id} page="jobs" now={route.page} label="Jobs" note="running & finished" />
          </>
        )}

        <div style={{ flex: 1 }} />
        {health && !health.ok && (
          <div className="faint" style={{ color: "var(--bad)" }}>
            {health.problems.join("; ")}
          </div>
        )}
      </aside>

      <main className="main">
        {id ? <P id={id} proj={proj} refresh={refresh} /> : <Projects />}
      </main>
    </div>
  );
}

function Nav({ id, page, now, label, note }: {
  id: string; page: Page; now: Page; label: string; note?: string;
}) {
  return (
    <a className={`nav ${now === page ? "on" : ""}`} href={`#/${id}/${page}`}>
      {label}
      {note && <small>{note}</small>}
    </a>
  );
}

export type PageProps = {
  id: string;
  proj: ProjectStatus | null;
  refresh: () => void;
};
