/**
 * Import modal.
 *
 * Calibration is mandatory and material is a selection, never a guess. Both rules come
 * straight from the spec's product guardrails: without a trusted µm/pixel there is no
 * physical measurement, only pixel counts, and apparent fibre radius cannot be used to
 * infer material because it is dominated by microscope magnification.
 *
 * In this prototype build the folder-import and inference pipeline is not wired, so
 * the modal validates and then says so plainly rather than pretending to import.
 */

import { useState } from "react";
import { FolderOpen } from "lucide-react";
import { useData } from "../app";

export function NewInspectionModal({ onClose }: { onClose: () => void }) {
  const { fx } = useData();
  const next = `INS-2026-${String(43 + fx.inspections.length).padStart(3, "0")}`;

  const [inspectionId, setInspectionId] = useState(next);
  const [sampleId, setSampleId] = useState("");
  const [folder, setFolder] = useState("");
  const [material, setMaterial] = useState("C/PEEK");
  const [calibration, setCalibration] = useState("0.57");
  const [submitted, setSubmitted] = useState(false);

  const cal = Number(calibration);
  const calBad = !Number.isFinite(cal) || cal <= 0;
  const ready = sampleId.trim() !== "" && folder.trim() !== "" && !calBad;

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
                     placeholder="data\Data sets\Test data set\Images"
                     onChange={(e) => setFolder(e.target.value)} />
              <button className="btn" onClick={() => setFolder("data\\Data sets\\Test data set\\Images")}>
                <FolderOpen size={16} aria-hidden /> Browse
              </button>
            </div>
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

        {submitted && (
          <div className="note" style={{ marginTop: 12 }}>
            Validated. Folder import and live batch inference are not wired in this
            prototype build — the two Test-set samples are preloaded from the model
            output in <span className="mono">frontend/public/data/fixtures.json</span>.
          </div>
        )}

        <div className="row" style={{ justifyContent: "flex-end", marginTop: 20 }}>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={!ready} onClick={() => setSubmitted(true)}>
            Import and run inference
          </button>
        </div>
      </div>
    </div>
  );
}
