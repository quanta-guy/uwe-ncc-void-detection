/**
 * The correction flow. Edit and upload controls appear only after a reviewer has said
 * the mask is wrong - that ordering is in the spec, and it matters: offering an editor
 * before a judgement invites fiddling with masks that were already correct.
 *
 * An error reason is required. Without it the reviewed dataset records that something
 * was wrong but not what, which is useless for deciding whether the model has a
 * systematic problem or the input did.
 *
 * The upload path validates before it commits, and a correction is always a NEW
 * version - the original prediction is never overwritten. That is what makes the
 * inspection record reproducible.
 */

import { useRef, useState } from "react";
import { AlertTriangle, Pencil, Upload } from "lucide-react";
import { asset } from "../api";
import type { Field, Inspection } from "../types";

const REASONS: Array<[string, string]> = [
  ["missed_void", "Missed void"],
  ["false_void", "False void"],
  ["boundary_error", "Boundary error"],
  ["class_error", "Fibre/matrix classification error"],
  ["input_quality", "Input-quality problem"],
  ["other", "Other"],
];

interface Props {
  field: Field;
  inspection: Inspection;
  onClose: () => void;
  onCorrected: (reason: string) => void;
}

/** Client-side mask validation, mirroring the server rules in the spec. */
async function validateMask(file: File, expect: number): Promise<string[]> {
  const errors: string[] = [];
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = () => rej(new Error("unreadable"));
      i.src = url;
    });
    if (img.width !== expect || img.height !== expect) {
      errors.push(`Dimensions ${img.width}×${img.height} do not match the field (${expect}×${expect}).`);
    }
    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;
    c.getContext("2d")!.drawImage(img, 0, 0);
    const d = c.getContext("2d")!.getImageData(0, 0, img.width, img.height).data;
    const seen = new Set<number>();
    for (let i = 0; i < d.length; i += 4) seen.add(d[i]);
    const bad = [...seen].filter((v) => v > 2);
    if (bad.length) {
      errors.push(
        `Class values must be exactly 0, 1 and 2. Found ${bad.length} other value(s), ` +
        `e.g. ${bad.slice(0, 4).join(", ")}. A palette-rendered mask will fail this check — ` +
        `upload the raw class mask.`,
      );
    }
  } catch {
    errors.push("File could not be read as an image.");
  } finally {
    URL.revokeObjectURL(url);
  }
  return errors;
}

export function CorrectionModal({ field, inspection, onClose, onCorrected }: Props) {
  const [reason, setReason] = useState("");
  const [other, setOther] = useState("");
  const [errors, setErrors] = useState<string[] | null>(null);
  const [uploaded, setUploaded] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const ready = reason !== "" && (reason !== "other" || other.trim() !== "");
  const label = reason === "other" ? other.trim() : REASONS.find(([k]) => k === reason)?.[1] ?? "";

  const onFile = async (f: File) => {
    const errs = await validateMask(f, 256);
    setErrors(errs);
    setUploaded(errs.length === 0 ? f.name : null);
  };

  return (
    <div className="backdrop" role="dialog" aria-modal aria-label="Correct the prediction"
         onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal wide">
        <div className="spread" style={{ marginBottom: 6 }}>
          <h2 style={{ margin: 0 }}>Correct the prediction</h2>
          <button className="btn sm" onClick={onClose}>Close</button>
        </div>
        <p className="sub" style={{ marginBottom: 18 }}>
          Field {field.id} · {inspection.sampleId}. The correction enters the reviewed
          dataset after approval. The original prediction is retained.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 22 }}>
          <div style={{ position: "relative", alignSelf: "start" }}>
            <img src={asset("originals", field.stem)} alt={`Field ${field.id} original`}
                 style={{ width: "100%", borderRadius: 6, display: "block" }} />
            <img src={asset("void", field.stem)} alt="" aria-hidden
                 style={{ position: "absolute", inset: 0, width: "100%", opacity: 0.7 }} />
            <div className="provisional" style={{ marginTop: 8 }}>
              Current prediction · {field.voidCount} void
              {field.voidCount === 1 ? "" : "s"} · severity{" "}
              {field.clusterSeverityUm.toFixed(2)} µm
            </div>
          </div>

          <div>
            <h3>What is wrong?</h3>
            <div style={{ display: "grid", gap: 6, marginBottom: 14 }}>
              {REASONS.map(([k, text]) => (
                <label key={k} className="row" style={{ cursor: "pointer" }}>
                  <input type="radio" name="reason" value={k} checked={reason === k}
                         style={{ width: 16, minHeight: 16 }}
                         onChange={() => setReason(k)} />
                  <span style={{ fontSize: 14 }}>{text}</span>
                </label>
              ))}
              {reason === "other" && (
                <input type="text" value={other} placeholder="Describe the problem"
                       aria-label="Other reason" onChange={(e) => setOther(e.target.value)} />
              )}
            </div>

            <input ref={fileRef} type="file" accept="image/png" style={{ display: "none" }}
                   onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />

            <div className="row" style={{ marginBottom: 14 }}>
              <button className="btn primary" disabled={!ready}
                      onClick={() => fileRef.current?.click()}>
                <Upload size={16} aria-hidden /> Upload corrected mask
              </button>
              <button className="btn" disabled={!ready} title="Brush editor is not in this prototype build">
                <Pencil size={16} aria-hidden /> Edit mask
              </button>
            </div>

            {!ready && (
              <div className="note">Select an error reason to enable the correction actions.</div>
            )}

            {errors && errors.length > 0 && (
              <div className="note bad">
                <strong style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <AlertTriangle size={15} aria-hidden /> Mask rejected — nothing was written
                </strong>
                <ul style={{ margin: "8px 0 0 18px", padding: 0 }}>
                  {errors.map((e) => <li key={e} style={{ marginBottom: 4 }}>{e}</li>)}
                </ul>
              </div>
            )}

            {uploaded && (
              <div className="note" style={{ borderLeftColor: "var(--success)" }}>
                <strong>{uploaded}</strong> passed validation: dimensions match and all
                values are 0, 1 or 2. It will be written as version v001 alongside the
                prediction, and the field measurements recalculated from it.
              </div>
            )}

            <div className="row" style={{ marginTop: 18, justifyContent: "flex-end" }}>
              <button className="btn" onClick={onClose}>Cancel</button>
              <button className="btn primary" disabled={!ready}
                      onClick={() => onCorrected(label)}>
                Save correction
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
