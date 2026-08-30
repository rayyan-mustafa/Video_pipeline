// Small shared pieces. Kept together because each is a dozen lines and a file
// per component would be more navigation than code.

import { useEffect, useRef, useState } from "react";
import type { Pacing } from "../api";

export function Pill({ kind = "", children }: { kind?: string; children: React.ReactNode }) {
  return <span className={`pill ${kind}`}>{children}</span>;
}

export function Banner({ kind, children }: { kind: "ok" | "warn" | "bad"; children: React.ReactNode }) {
  return <div className={`banner ${kind}`}>{children}</div>;
}

/** A stage's readiness, shown the same way everywhere so it reads at a glance. */
export function Ready({ ok, yes, no }: { ok: boolean; yes: string; no: string }) {
  return <Pill kind={ok ? "ok" : ""}>{ok ? yes : no}</Pill>;
}

export function Spinner({ label }: { label: string }) {
  return <span className="faint">{label}…</span>;
}

/** Errors are shown, never swallowed - the backend writes good messages. */
export function ErrorBox({ err }: { err: unknown }) {
  if (!err) return null;
  return <Banner kind="bad">{err instanceof Error ? err.message : String(err)}</Banner>;
}

// ---------------------------------------------------------------- pacing

const RATING_KIND: Record<string, string> = {
  "on target": "ok", "too slow": "bad", "too fast": "warn", empty: "",
};

/** The pacing verdict, rendered identically on every page that shows one.
 *
 *  Shows the BODY average rather than the raw mean, because that is what
 *  pacing.verdict() judges - accents deliberately pull the mean down and a
 *  correctly paced video would otherwise look "too fast" here.
 */
export function PacingCard({ p, title = "Pacing" }: { p?: Pacing | null; title?: string }) {
  if (!p || !p.n) return null;
  const kind = RATING_KIND[p.rating] ?? "";
  const accent = p.accents, over = p.over_hard, flash = p.flashes;
  const body = p.n - accent - over - flash;
  const pct = (v: number) => `${(v / p.n) * 100}%`;

  return (
    <div className="card">
      <div className="spread">
        <strong>{title}</strong>
        <Pill kind={kind}>{p.rating}</Pill>
      </div>
      <div className="row" style={{ margin: "10px 0 6px", gap: 18 }}>
        <Stat label="body average" value={`${p.body_avg}s`} note={`band ${p.target_low}–${p.target_high}s`} />
        <Stat label="images" value={String(p.n)} note={p.want_images !== p.n ? `wants ~${p.want_images}` : "as planned"} />
        <Stat label="longest" value={`${p.max}s`} note={`ceiling ${p.hard_max}s`} />
        <Stat label="runtime" value={`${(p.total / 60).toFixed(1)} min`} />
      </div>
      <div className="bar" title="body / accents / over ceiling / flashes">
        {body > 0 && <i style={{ width: pct(body), background: "var(--ok)" }} />}
        {accent > 0 && <i style={{ width: pct(accent), background: "var(--accent)" }} />}
        {over > 0 && <i style={{ width: pct(over), background: "var(--bad)" }} />}
        {flash > 0 && <i style={{ width: pct(flash), background: "var(--warn)" }} />}
      </div>
      <div className="faint" style={{ marginTop: 8 }}>{p.note}</div>
    </div>
  );
}

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div>
      <div className="faint">{label}</div>
      <div style={{ fontSize: 19, fontFamily: "var(--mono)" }}>{value}</div>
      {note && <div className="faint">{note}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- upload

export function Drop({ label, onFiles, accept, busy }: {
  label: string; accept?: string; busy?: boolean;
  onFiles: (f: FileList) => void;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  return (
    <div
      className={`drop ${over ? "over" : ""}`}
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault(); setOver(false);
        if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
      }}
    >
      {busy ? "uploading…" : label}
      <input
        ref={input} type="file" multiple accept={accept} hidden
        onChange={(e) => { if (e.target.files?.length) onFiles(e.target.files); e.target.value = ""; }}
      />
    </div>
  );
}

// ---------------------------------------------------------------- job log

/** Streams one job's output over SSE and follows it to completion.
 *
 *  The backend keeps a backlog of lines per job, so a page that mounts partway
 *  through a run still shows everything from the start rather than joining
 *  midway - which matters because these runs are minutes long and the operator
 *  will navigate away and back.
 */
export function JobLog({ jobId, onDone }: { jobId: string; onDone?: (status: string) => void }) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState("running");
  const box = useRef<HTMLDivElement>(null);
  const done = useRef(onDone);
  done.current = onDone;

  useEffect(() => {
    setLines([]); setStatus("running");
    const es = new EventSource(`/api/jobs/${jobId}/stream`);
    es.onmessage = (e) => setLines((l) => [...l, e.data]);
    es.addEventListener("end", (e) => {
      const s = (e as MessageEvent).data as string;
      setStatus(s); es.close(); done.current?.(s);
    });
    // A dropped connection would otherwise retry forever against a finished job.
    es.onerror = () => { es.close(); setStatus((s) => (s === "running" ? "disconnected" : s)); };
    return () => es.close();
  }, [jobId]);

  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <div>
      <div className="spread" style={{ marginBottom: 6 }}>
        <span className="faint mono">job {jobId.slice(0, 8)}</span>
        <Pill kind={status === "done" ? "ok" : status === "running" ? "" : "bad"}>{status}</Pill>
      </div>
      <div className="log" ref={box}>{lines.join("\n") || "waiting for output…"}</div>
    </div>
  );
}
