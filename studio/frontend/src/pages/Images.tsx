// The pictures, which are produced OUTSIDE this pipeline and dropped in.
//
// Nothing here generates a pixel. Nano Banana makes the images, this page
// receives them, and the only judgement it offers is the one that matters:
// whether there are enough of them for the pacing band.
//
// `_2`-suffixed files are regenerations of the SAME slot, not extra pictures.
// scan_images keeps the lowest variant and reports the rest as alternates, which
// is why "105 used, 122 on disk" is a correct pair of numbers rather than a bug.

import { useEffect, useState } from "react";
import { api, fileUrl } from "../api";
import type { PageProps } from "../App";
import { Banner, Drop, ErrorBox, PacingCard, Pill, Stat } from "../components/bits";

export default function Images({ id, refresh }: PageProps) {
  const [data, setData] = useState<any>(null);
  const [pac, setPac] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [section, setSection] = useState<string>("all");

  const load = () => Promise.all([
    api.images(id).then(setData),
    api.imagePacing(id).then(setPac).catch(() => setPac(null)),
  ]).catch(setErr);

  useEffect(() => { setData(null); setPac(null); load(); }, [id]);

  async function upload(files: FileList) {
    setBusy(true); setErr(null);
    try { await api.upload(`/api/projects/${id}/images`, files); await load(); refresh(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }

  if (!data) return <div className="faint">loading…</div>;
  if (!data.ok) return <><h2>Images</h2><Banner kind="bad">{data.error}</Banner></>;

  const shown = section === "all"
    ? data.images
    : data.images.filter((i: any) => String(i.section) === section);

  return (
    <>
      <div className="spread">
        <h2>Images</h2>
        <Pill>{data.count} used · {data.on_disk} on disk</Pill>
      </div>
      <div className="faint">
        Generated externally (Nano Banana) and dropped in — this pipeline places them, it never makes them.
      </div>
      <ErrorBox err={err} />

      <div className="card" style={{ marginTop: 10 }}>
        <div className="row" style={{ gap: 22 }}>
          <Stat label="used" value={String(data.count)} />
          <Stat label="on disk" value={String(data.on_disk)} />
          <Stat label="alternates" value={String(data.alternates.length)} note="`_2` regenerations, not extras" />
          <Stat label="sections" value={String(Object.keys(data.by_section).length)} />
        </div>
      </div>

      {data.notes?.length > 0 && (
        <div className="faint" style={{ marginTop: 8 }}>{data.notes.join(" · ")}</div>
      )}

      <div style={{ marginTop: 12 }}>
        <Drop label="Drop pictures here, or click to choose" accept="image/*" busy={busy} onFiles={upload} />
      </div>

      {pac?.verdict && (
        <div style={{ marginTop: 12 }}>
          <PacingCard p={pac.verdict} title="Pacing at this picture count (estimated)" />
        </div>
      )}

      <h3>
        By section
        <select value={section} onChange={(e) => setSection(e.target.value)}
                style={{ marginLeft: 10, fontSize: 12 }}>
          <option value="all">all</option>
          {Object.keys(data.by_section).map((s) => (
            <option key={s} value={s}>section {s} ({data.by_section[s]})</option>
          ))}
        </select>
      </h3>

      <div className="thumbs">
        {shown.map((im: any) => (
          <div className="thumb" key={im.name}>
            <img src={fileUrl(im.abs_path)} loading="lazy" alt={im.name} />
            <div className="cap">s{im.section} · #{im.order}{im.variant > 1 ? ` · v${im.variant}` : ""}</div>
          </div>
        ))}
      </div>
      {shown.length === 0 && <div className="card faint">No pictures yet.</div>}

      {data.alternates.length > 0 && (
        <>
          <h3>Alternates not used ({data.alternates.length})</h3>
          <div className="faint mono">{data.alternates.join(", ")}</div>
        </>
      )}
    </>
  );
}
