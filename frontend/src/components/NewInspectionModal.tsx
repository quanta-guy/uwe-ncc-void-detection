/**
 * Import modal.
 *
 * Calibration is mandatory and material is a selection, never a guess. Both rules come
 * straight from the spec's product guardrails: without a trusted µm/pixel there is no
 * physical measurement, only pixel counts, and apparent fibre radius cannot be used to
 * infer material because it is dominated by microscope magnification.
 *
 * Import is live: the selected files are posted to the local backend, which runs the
 * active ensemble over every one and writes real measurements. Inference is five
 * forward passes plus scipy hulls per field, so the request returns a job id and the
 * modal polls it - a synchronous wait would read as a hang rather than as work.
 */

import { useEffect, useRef, useState } from "react";
import { FolderOpen, Images, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useData } from "../app";
import { jobStatus, startImport } from "../api";
import type { ImportJob } from "../api";

export function NewInspectionModal({ onClose }: { onClose: () => void }) {
  const { fx } = useData();
  const navigate = useNavigate();
  const next = `INS-2026-${String(43 + fx.inspections.length).padStart(3, "0")}`;

  const [inspectionId, setInspectionId] = useState(next);
  const [sampleId, setSampleId] = useState("");
  const [folder, setFolder] = useState("");
  const [material, setMaterial] = useState("C/PEEK");
  const [calibration, setCalibration] = useState("0.57");
  const [scan, setScan] = useState<{ images: number; dupes: string[]; skipped: number } | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<ImportJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dirRef = useRef<HTMLInputElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Poll while the batch runs. Stops on its own once the job leaves `running`, so
  // there is no interval left behind after the modal closes.
  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(async () => {
      try {
        const s = await jobStatus(jobId);
        setJob(s);
        if (s.state !== "running") {
          clearInterval(t);
          if (s.state === "error") setError(s.error ?? "inference failed");
        }
      } catch (e) {
        clearInterval(t);
        setError((e as Error).message);
      }
    }, 700);
    return () => clearInterval(t);
  }, [jobId]);

  /**
   * Real selection, from either picker. A browser cannot hand back an absolute path,
   * but it hands back the file list - which is what has to be validated anyway:
   * supported extensions, duplicate stems, empty selection.
   *
   * Two pickers rather than one, because `webkitdirectory` takes a whole folder and
   * offers no way to import a subset. Re-running a handful of fields after a recut is
   * an ordinary thing to want, and a folder-only picker forbids it.
   */
  const onFolder = (files: FileList | null) => {
    if (!files || files.length === 0) {
      setScan({ images: 0, dupes: [], skipped: 0 });
      return;
    }
    const ok = /\.(png|jpe?g|tiff?)$/i;
    const all = Array.from(files);
    const images = all.filter((f) => ok.test(f.name));
    const stems = images.map((f) => f.name.replace(/\.[^.]+$/, ""));
    const seen = new Set<string>();
    const dupes = [...new Set(stems.filter((st) => seen.size === seen.add(st).size))];
    const rel = (all[0] as File & { webkitRelativePath?: string }).webkitRelativePath ?? "";
    setFolder(rel.split("/")[0]
      || (images.length === 1 ? images[0].name : `${images.length} selected images`));
    setFiles(images);
    setScan({ images: images.length, dupes, skipped: all.length - images.length });
  };

  const cal = Number(calibration);
  const calBad = !Number.isFinite(cal) || cal <= 0;
  const running = job?.state === "running" || (jobId !== null && job === null);
  const ready = sampleId.trim() !== "" && files.length > 0 && !calBad
    && scan !== null && scan.dupes.length === 0 && !running;

  const submit = async () => {
    setError(null);
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    form.append("inspectionId", inspectionId);
    form.append("sampleId", sampleId);
    form.append("material", material);
    form.append("umPerPixel", calibration);
    try {
      const { jobId: id, total } = await startImport(form);
      setJob({ state: "running", done: 0, total, current: "", inspectionId });
      setJobId(id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="backdrop" role="dialog" aria-modal aria-label="New inspection"
         onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <h2>New inspection</h2>
        <p className="sub">
          Images are copied into an immutable originals folder; the source is never
          modified. Batch inference starts automatically once the manifest is written.
        </p>

        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <div>
            <label className="field" htmlFor="ins">Inspection ID</label>
            <input id="ins" type="text" value={inspectionId}
                   onChange={(e) => setInspectionId(e.target.value)} />
          </div>
          <div>
            <label className="field" htmlFor="smp">Sample ID</label>
            <input id="smp" type="text" value={sampleId} placeholder="CPEEK-043"
                   onChange={(e) => setSampleId(e.target.value)} />
          </div>
          <div style={{ gridColumn: "1 / -1" }}>
            <label className="field" htmlFor="fld">Image folder</label>
            <div className="row">
              <input id="fld" type="text" value={folder}
                     placeholder="Choose a folder of microscopy images"
                     onChange={(e) => setFolder(e.target.value)} />
              <input ref={dirRef} type="file" multiple style={{ display: "none" }}
                     aria-label="Choose folder"
                     {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
                     onChange={(e) => onFolder(e.target.files)} />
              <input ref={fileRef} type="file" multiple style={{ display: "none" }}
                     aria-label="Choose images"
                     accept=".png,.jpg,.jpeg,.tif,.tiff"
                     onChange={(e) => onFolder(e.target.files)} />
              <button className="btn" onClick={() => dirRef.current?.click()}>
                <FolderOpen size={16} aria-hidden /> Folder
              </button>
              <button className="btn" onClick={() => fileRef.current?.click()}>
                <Images size={16} aria-hidden /> Images
              </button>
            </div>
            {scan && (
              <div className={`note ${scan.images === 0 || scan.dupes.length ? "bad" : ""}`}
                   style={{ marginTop: 10 }}>
                {scan.images === 0
                  ? "No supported images found. Allowed: .png, .jpg, .jpeg, .tif, .tiff."
                  : <><strong>{scan.images} image{scan.images === 1 ? "" : "s"}</strong> ready to import.</>}
                {scan.skipped > 0 && <> {scan.skipped} unsupported file
                  {scan.skipped === 1 ? "" : "s"} ignored.</>}
                {scan.dupes.length > 0 && (
                  <div style={{ marginTop: 6 }}>
                    Duplicate stems would collide and are rejected:{" "}
                    <span className="mono">{scan.dupes.slice(0, 4).join(", ")}</span>
                    {scan.dupes.length > 4 && ` +${scan.dupes.length - 4} more`}
                  </div>
                )}
              </div>
            )}
          </div>
          <div>
            <label className="field" htmlFor="mat">Material</label>
            <select id="mat" value={material} onChange={(e) => setMaterial(e.target.value)}>
              <option>C/PEEK</option>
              <option>C/LM-PEAK</option>
              <option>Other</option>
            </select>
          </div>
          <div>
            <label className="field" htmlFor="cal">Calibration (µm/pixel)</label>
            <input id="cal" type="number" step="0.01" value={calibration}
                   onChange={(e) => setCalibration(e.target.value)} />
          </div>
          <div style={{ gridColumn: "1 / -1" }}>
            <label className="field" htmlFor="prof">Measurement profile</label>
            <select id="prof" defaultValue={fx.profile.id}>
              <option value={fx.profile.id}>
                {fx.profile.name} — limit {fx.profile.clusterSeverityLimitUm} µm
              </option>
            </select>
          </div>
        </div>

        {calBad && (
          <div className="note bad" style={{ marginTop: 14 }}>
            Calibration is required. Without a trusted µm/pixel the sample is
            <strong> Uncalibrated</strong>: pixel measurements only, and no physical
            pass/fail disposition.
          </div>
        )}

        <div className="note warn" style={{ marginTop: 14 }}>
          Material is an inspection input, not a model output. It is recorded as
          operator-entered and never inferred from fibre radius, which is dominated by
          microscope magnification rather than the material.
        </div>

        {error && (
          <div className="note bad" style={{ marginTop: 12 }}>
            <strong>Import failed.</strong>
            <div className="mono" style={{ marginTop: 6 }}>{error}</div>
          </div>
        )}

        {job && job.state === "running" && (
          <div className="note" style={{ marginTop: 12 }}>
            <div className="row" style={{ gap: 8, alignItems: "center" }}>
              <Loader2 size={16} className="spin" aria-hidden />
              <strong>Running inference — field {job.done} of {job.total}</strong>
            </div>
            <div className="progress" style={{ marginTop: 10 }}
                 role="progressbar" aria-valuenow={job.done} aria-valuemin={0}
                 aria-valuemax={job.total}>
              <span style={{ width: `${job.total ? (100 * job.done) / job.total : 0}%` }} />
            </div>
            {job.current && (
              <div className="mono" style={{ marginTop: 8, fontSize: 12 }}>{job.current}</div>
            )}
          </div>
        )}

        {job?.state === "done" && (
          <div className="note good" style={{ marginTop: 12 }}>
            <strong>{job.total} field{job.total === 1 ? "" : "s"} analysed.</strong>{" "}
            Preliminary disposition{" "}
            <strong>{job.inspection?.preliminary.disposition}</strong> — worst-cluster
            severity {job.inspection?.kpis.worstClusterSeverityUm} µm against a{" "}
            {fx.profile.clusterSeverityLimitUm} µm limit.
          </div>
        )}

        <div className="row" style={{ justifyContent: "flex-end", marginTop: 20 }}>
          <button className="btn" disabled={running}
                  onClick={() => (job?.state === "done" ? navigate(0) : onClose())}>
            {job?.state === "done" ? "Close" : "Cancel"}
          </button>
          {job?.state === "done" ? (
            // Fixtures load once at boot, so the new inspection needs a reload to
            // appear. A router push would land on a route the data does not have yet.
            <button className="btn primary"
                    onClick={() => window.location.assign(`/inspections/${inspectionId}`)}>
              Open sample analysis
            </button>
          ) : (
            <button className="btn primary" disabled={!ready} onClick={submit}>
              {running ? "Analysing…" : "Import and run inference"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
