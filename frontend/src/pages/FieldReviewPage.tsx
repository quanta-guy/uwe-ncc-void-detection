/**
 * Field review - the core loop, and the screen the product lives or dies on.
 *
 * A reviewer validates one prediction and records a trustworthy accepted mask. The
 * queue is ordered by measured risk rather than filename, which is the central product
 * claim: the most consequential field is seen first instead of found by scrolling.
 *
 * Everything on screen for the active field is a real measurement from evaluation.py,
 * including the reason it was flagged. A colour alone would tell a reviewer nothing
 * about why they are looking at this image.
 */

import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Ban, Check, ChevronLeft, ChevronRight, Pencil } from "lucide-react";
import { useData } from "../app";
import { appendReview, asset, bandOf, fieldStatus, priorityOrder } from "../api";
import type { Field, Inspection } from "../types";
import { BandChip, Card, MRow, StatusChip } from "../components/common";
import { ImageViewer, NEUTRAL, type Display, type Layers, type ViewMode } from "../components/ImageViewer";
import { SeverityFormula } from "../components/SeverityOverlay";
import { CorrectionModal } from "../components/CorrectionModal";

export function FieldReviewPage() {
  const { fx, status } = useData();
  const params = useParams();
  const nav = useNavigate();

  const real = fx.inspections.filter((i) => i.fields.length > 0);
  const inspection: Inspection | undefined =
    real.find((i) => i.inspectionId === params.inspectionId) ?? real[0];

  const queue = useMemo(() => (inspection ? priorityOrder(inspection.fields) : []), [inspection]);
  const activeId = params.fieldId ?? queue[0]?.id;
  const field = queue.find((f) => f.id === activeId) ?? queue[0];

  const [mode, setMode] = useState<ViewMode>("side");
  const [opacity, setOpacity] = useState(0.65);
  const [display, setDisplay] = useState<Display>(NEUTRAL);
  const [layers, setLayers] = useState<Layers>({
    fibre: true, void: true, controlling: true, evidence: true,
  });
  const [note, setNote] = useState("");
  const [correcting, setCorrecting] = useState(false);

  const idx = queue.findIndex((f) => f.id === field?.id);
  const go = (n: number) => {
    const t = queue[Math.max(0, Math.min(queue.length - 1, idx + n))];
    if (t && inspection) nav(`/inspections/${inspection.inspectionId}/fields/${t.id}`);
  };

  const record = (decision: "accepted" | "unsuitable", errorReason?: string) => {
    if (!inspection || !field) return;
    appendReview({
      inspectionId: inspection.inspectionId, fieldId: field.id, decision,
      errorReason, note: note || undefined, reviewer: "Demo reviewer",
    });
    setNote("");
    if (idx < queue.length - 1) go(1);
  };

  // Shortcuts are disabled while typing, or a note would be impossible to write.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || correcting) return;
      const k = e.key.toLowerCase();
      if (k === "a") record("accepted");
      else if (k === "w") setCorrecting(true);
      else if (k === "u") record("unsuitable");
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === " ") {
        e.preventDefault();
        setMode((m) => (m === "overlay" ? "side" : "overlay"));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => { setDisplay(NEUTRAL); setNote(""); }, [field?.stem]);

  if (!inspection || !field) {
    return <div className="page"><div className="empty">No fields to review.</div></div>;
  }

  const band = bandOf(field);
  const st = fieldStatus(status, inspection.inspectionId, field.id);
  const critical = queue.filter((f) => bandOf(f) === "critical");
  const review = queue.filter((f) => bandOf(f) === "review");
  const routine = queue.filter((f) => bandOf(f) === "routine");
  const over = field.distanceToLimitUm;

  const QueueGroup = ({ title, items, tone }: { title: string; items: Field[]; tone: string }) => (
    items.length === 0 ? null : (
      <>
        <div className="qhead">
          <span><span className={`dot ${tone}`} style={{ marginRight: 7 }} aria-hidden />{title}</span>
          <span>{items.length}</span>
        </div>
        {items.map((f) => (
          <div key={f.id} className={`qitem ${f.id === field.id ? "on" : ""}`}
               onClick={() => nav(`/inspections/${inspection.inspectionId}/fields/${f.id}`)}
               role="button" tabIndex={0}
               onKeyDown={(e) => e.key === "Enter" &&
                 nav(`/inspections/${inspection.inspectionId}/fields/${f.id}`)}>
            <img src={asset("thumbs", f.stem)} alt="" aria-hidden />
            <div className="meta" style={{ minWidth: 0 }}>
              <div className="name">Field {f.id}</div>
              <div>{f.clusterSeverityUm > 0
                ? `Cluster ${f.clusterSeverityUm.toFixed(2)} µm`
                : "No void detected"}</div>
              <StatusChip s={fieldStatus(status, inspection.inspectionId, f.id)} />
            </div>
          </div>
        ))}
      </>
    )
  );

  return (
    <div className="page">
      <div className="crumbs">
        <Link to="/review">Review queue</Link> /{" "}
        <Link to={`/inspections/${inspection.inspectionId}`}>{inspection.sampleId}</Link> /{" "}
        Field {field.id}
      </div>

      <div className="spread" style={{ marginBottom: 18 }}>
        <div>
          <h1 style={{ display: "flex", alignItems: "center", gap: 14 }}>
            Field {field.id} <BandChip band={band} />
          </h1>
          <div className="sub" style={{ margin: 0 }}>
            priority {idx + 1} of {queue.length} · {inspection.material} · {field.stem}
          </div>
        </div>
        <div className="row">
          <button className="btn" onClick={() => go(-1)} disabled={idx === 0}>
            <ChevronLeft size={16} aria-hidden /> Previous
          </button>
          <button className="btn primary" onClick={() => record("accepted")}>
            Save &amp; next <ChevronRight size={16} aria-hidden />
          </button>
        </div>
      </div>

      <div className="review-layout"
           style={{ display: "grid", gridTemplateColumns: "260px 1fr 320px", gap: 16 }}>
        <Card style={{ padding: 8 }}>
          <div className="qhead" style={{ paddingTop: 6 }}>
            <span>Priority queue</span><span>{queue.length}</span>
          </div>
          <div className="queue">
            <QueueGroup title="Critical" items={critical} tone="critical" />
            <QueueGroup title="Review" items={review} tone="review" />
            <QueueGroup title="Routine" items={routine} tone="routine" />
          </div>
        </Card>

        <div className="grid" style={{ alignContent: "start" }}>
          <Card>
            <div className="spread" style={{ marginBottom: 14 }}>
              <strong style={{ fontSize: 16 }}>Is the segmentation correct?</strong>
              <div className="row">
                <button className="btn success" onClick={() => record("accepted")}>
                  <Check size={16} aria-hidden /> Accept mask <kbd className="sr-only">A</kbd>
                </button>
                <button className="btn danger" onClick={() => setCorrecting(true)}>
                  <Pencil size={16} aria-hidden /> Wrong — correct mask
                </button>
                <button className="btn" onClick={() => record("unsuitable")}>
                  <Ban size={16} aria-hidden /> Mark unsuitable
                </button>
              </div>
            </div>

            <div className="spread" style={{ marginBottom: 14 }}>
              <div className="tabs">
                {([["side", "Side by side"], ["swipe", "Swipe"], ["overlay", "Overlay"]] as const)
                  .map(([m, label]) => (
                    <button key={m} className={`tab ${mode === m ? "on" : ""}`}
                            onClick={() => setMode(m)}>{label}</button>
                  ))}
              </div>
              <div className="row" style={{ minWidth: 240 }}>
                <span style={{ fontSize: 13, color: "var(--muted)" }}>Mask</span>
                <strong style={{ minWidth: 42 }}>{Math.round(opacity * 100)}%</strong>
                <input type="range" min={0} max={100} value={opacity * 100}
                       aria-label="Mask opacity"
                       onChange={(e) => setOpacity(Number(e.target.value) / 100)} />
              </div>
            </div>

            <ImageViewer field={field} mode={mode} maskOpacity={opacity} display={display}
                         layers={layers} evidence={field.severityEvidence} />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginTop: 16 }}>
              <div>
                {([["hue", "Hue", 180], ["brightness", "Brightness", 60],
                   ["contrast", "Contrast", 60]] as const).map(([k, label, max]) => (
                  <div key={k} className="row" style={{ marginBottom: 8 }}>
                    <span style={{ width: 82, fontSize: 13, color: "var(--muted)" }}>{label}</span>
                    <input type="range" min={-max} max={max} value={display[k]}
                           aria-label={`${label} (display only)`}
                           onChange={(e) => setDisplay({ ...display, [k]: Number(e.target.value) })} />
                    <span style={{ width: 34, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                      {display[k]}
                    </span>
                  </div>
                ))}
                <div className="provisional">Display only — never alters the mask or measurements</div>
              </div>
              <div>
                <h3>Show classes</h3>
                {([["fibre", "Fibre"], ["void", "Void"],
                   ["controlling", "Controlling cluster"],
                   ["evidence", "Severity evidence (L / A / D)"]] as const).map(([k, label]) => (
                  <label key={k} className="row" style={{ marginBottom: 7, cursor: "pointer" }}>
                    <input type="checkbox" checked={layers[k]} style={{ width: 16, minHeight: 16 }}
                           onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} />
                    <span style={{ fontSize: 14 }}>{label}</span>
                  </label>
                ))}
              </div>
            </div>

            <SeverityFormula evidence={field.severityEvidence} />
          </Card>

          <Card title="Review note">
            <textarea rows={2} value={note} placeholder="Add a note (optional)"
                      aria-label="Review note" onChange={(e) => setNote(e.target.value)} />
            <div className="provisional" style={{ marginTop: 8 }}>
              {inspection.model.id} · original prediction retained · reviewer decision audited
            </div>
          </Card>
        </div>

        <Card title="Field measurements" right={<StatusChip s={st} />}>
          <div className="mlist">
            <MRow k="Void areal fraction" v={`${field.voidArealFractionPct.toFixed(3)}%`} />
            <MRow k="Void count" v={field.voidCount} />
            <MRow k="Largest void Feret" v={`${field.largestFeretUm.toFixed(2)} µm`} />
            <MRow k="Cluster severity" v={`${field.clusterSeverityUm.toFixed(2)} µm`}
                  tone={field.verdict === "FAIL" ? "bad" : undefined} />
            <MRow k="Distance to limit" v={`${over >= 0 ? "+" : ""}${over.toFixed(2)} µm`}
                  tone={over >= 0 ? "bad" : "good"}
                  title="Positive means the field is over the profile limit" />
            <MRow k="Model disagreement"
                  v={field.modelDisagreement > 0.15 ? "High" : field.modelDisagreement > 0.05 ? "Moderate" : "Low"}
                  tone={field.modelDisagreement > 0.15 ? "bad" : undefined}
                  title={`Mean inter-model spread ${field.modelDisagreement.toFixed(4)}`} />
            <MRow k="Calibration" v={`${field.umPerPixel} µm/px`} />
          </div>

          <div className={`note ${band === "critical" ? "bad" : band === "review" ? "warn" : ""}`}
               style={{ marginTop: 14 }}>
            <strong>Why this is in the queue</strong>
            <div style={{ marginTop: 4 }}>{field.triageReason}</div>
          </div>

          {field.severityEvidence.voids.length > 0 && (
            <>
              <h3 style={{ marginTop: 18 }}>Controlling defect</h3>
              <div className="mlist">
                {field.severityEvidence.voids.map((v) => (
                  <MRow key={v.label} k={`${v.label} · ${v.areaLabel}`}
                        v={`${v.lengthUm.toFixed(1)} µm · ${v.areaUm2.toFixed(0)} µm²`} />
                ))}
                {field.severityEvidence.gaps.map((g) => (
                  <MRow key={g.label} k={`${g.label} gap`} v={`${g.gapUm.toFixed(1)} µm`}
                        title={`Under ${field.severityEvidence.merge_distance_um} µm, so these voids merge`} />
                ))}
              </div>
            </>
          )}
        </Card>
      </div>

      {correcting && (
        <CorrectionModal
          field={field} inspection={inspection}
          onClose={() => setCorrecting(false)}
          onCorrected={(reason) => {
            setCorrecting(false);
            appendReview({
              inspectionId: inspection.inspectionId, fieldId: field.id,
              decision: "corrected", errorReason: reason,
              note: note || undefined, reviewer: "Demo reviewer",
            });
            setNote("");
            if (idx < queue.length - 1) go(1);
          }}
        />
      )}
    </div>
  );
}
