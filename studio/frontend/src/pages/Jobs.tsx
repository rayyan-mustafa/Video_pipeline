// Every pipeline run for this project, live and finished.
//
// The long stages are subprocesses (they are argparse + print + SystemExit, so
// calling them in-process would take the server down), and jobs.py streams their
// stdout over SSE. This page is the window onto that: pick a run, watch it, or
// read back what a finished one printed.

import { useEffect, useState } from "react";
import { api, type Job } from "../api";
import type { PageProps } from "../App";
import { ErrorBox, JobLog, Pill } from "../components/bits";

const KIND_STATUS: Record<string, string> = {
  done: "ok", running: "", failed: "bad", cancelled: "warn",
};

export default function Jobs({ id, refresh }: PageProps) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  const load = () => api.jobs(id).then((j) => {
    setJobs(j);
    setSel((s) => s ?? j.find((x) => x.status === "running")?.id ?? j[0]?.id ?? null);
  }).catch(setErr);

  useEffect(() => { setJobs(null); setSel(null); load(); }, [id]);

  // Poll only the list, not the log - the log has its own SSE stream, and a
  // second source of truth for the same lines would fight it.
  useEffect(() => {
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [id]);

  const selected = jobs?.find((j) => j.id === sel);

  return (
    <>
      <h2>Jobs</h2>
      <div className="faint">Pipeline runs for {id}.</div>
      <ErrorBox err={err} />

      {jobs === null ? <div className="faint">loading…</div>
        : jobs.length === 0 ? <div className="card faint" style={{ marginTop: 10 }}>Nothing has been run yet.</div>
        : (
          <div className="card" style={{ marginTop: 10 }}>
            <table>
              <thead>
                <tr><th>kind</th><th>status</th><th>started</th><th className="num">took</th><th></th></tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} style={{ cursor: "pointer", background: j.id === sel ? "var(--panel-2)" : undefined }}
                      onClick={() => setSel(j.id)}>
                    <td className="mono">{j.kind}</td>
                    <td><Pill kind={KIND_STATUS[j.status] ?? ""}>{j.status}</Pill></td>
                    <td className="faint">{new Date(j.started * 1000).toLocaleTimeString()}</td>
                    <td className="num faint">{j.elapsed}s</td>
                    <td>
                      {j.status === "running" && (
                        <button className="danger"
                                onClick={(e) => { e.stopPropagation(); api.cancelJob(j.id).then(load).catch(setErr); }}>
                          cancel
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      {selected && (
        <div style={{ marginTop: 14 }}>
          <h3>{selected.kind} · {selected.status}</h3>
          {/* A finished job has no live stream left, so its kept backlog is
              rendered directly rather than opening an EventSource that would
              immediately close. */}
          {selected.status === "running"
            ? <JobLog jobId={selected.id} onDone={() => { load(); refresh(); }} />
            : <div className="log">{(selected.lines ?? []).join("\n") || "no output kept"}</div>}
        </div>
      )}
    </>
  );
}
