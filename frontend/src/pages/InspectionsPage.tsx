/**
 * Inspection workspace - find work and understand operational state.
 *
 * Operational KPIs only. The spec is explicit that the full material KPI set does not
 * belong here: this screen answers "what should I open", not "is this part good".
 *
 * Real inspections carry measurements; seeded rows exist only to show what a populated
 * workspace looks like and are visibly marked, so no number here can be mistaken for a
 * measurement that was never taken.
 */

import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Boxes, CheckCircle2, FolderTree, LoaderCircle, Search, Table2, UserCheck, X,
} from "lucide-react";
import { useData } from "../app";
import { reviewedCount, seededRows } from "../api";
import type { Inspection } from "../types";
import { Card, DispositionChip, MRow } from "../components/common";
import { NewInspectionModal } from "../components/NewInspectionModal";

type Filter = "all" | "processing" | "ready" | "complete";

export function InspectionsPage() {
  const { fx, status } = useData();
  const nav = useNavigate();
  const [filter, setFilter] = useState<Filter>("all");
  const [view, setView] = useState<"folder" | "table">("table");
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);

  const rows = useMemo(() => [...fx.inspections, ...seededRows()], [fx]);

  const stateOf = (i: Inspection) => {
    if (i.state === "processing") return "processing";
    const done = i.fields.length ? reviewedCount(i, status) : i.reviewedCount;
    if (done >= i.fieldCount && i.fieldCount > 0) return "complete";
    return "ready";
  };

  const shown = rows.filter((i) => {
    if (filter !== "all" && stateOf(i) !== filter) return false;
    const t = q.trim().toLowerCase();
    return !t || `${i.inspectionId} ${i.sampleId} ${i.material}`.toLowerCase().includes(t);
  });

  const sel = rows.find((i) => i.inspectionId === selected);
  const ops = {
    active: rows.filter((i) => stateOf(i) !== "complete").length,
    processing: rows.filter((i) => stateOf(i) === "processing")
      .reduce((n, i) => n + i.fieldCount, 0),
    ready: rows.filter((i) => stateOf(i) === "ready").length,
    complete: rows.filter((i) => stateOf(i) === "complete").length,
  };

  return (
    <div className="page">
      <div className="spread" style={{ marginBottom: 18 }}>
        <h1>Inspection workspace</h1>
        <button className="btn primary" onClick={() => setImporting(true)}>New inspection</button>
      </div>

      <div className="spread" style={{ marginBottom: 16 }}>
        <div className="tabs">
          {(["all", "processing", "ready", "complete"] as const).map((f) => (
            <button key={f} className={`tab ${filter === f ? "on" : ""}`} onClick={() => setFilter(f)}>
              {{ all: "All", processing: "Processing", ready: "Ready for review", complete: "Completed" }[f]}
            </button>
          ))}
        </div>
        <div className="row">
          <div className="search" style={{ width: 320 }}>
            <Search size={16} aria-hidden />
            <input type="text" value={q} placeholder="Search inspections…"
                   aria-label="Search inspections" onChange={(e) => setQ(e.target.value)} />
          </div>
          <div className="tabs">
            <button className={`tab ${view === "folder" ? "on" : ""}`} onClick={() => setView("folder")}>
              <FolderTree size={16} aria-hidden /> Folder
            </button>
            <button className={`tab ${view === "table" ? "on" : ""}`} onClick={() => setView("table")}>
              <Table2 size={16} aria-hidden /> Table
            </button>
          </div>
        </div>
      </div>

      <Card style={{ marginBottom: 16 }}>
        <div className="kpis">
          <div className="kpi row" style={{ gap: 14 }}>
            <Boxes size={22} color="var(--muted)" aria-hidden />
            <div><div className="label">Active inspections</div><div className="value">{ops.active}</div></div>
          </div>
          <div className="kpi row" style={{ gap: 14 }}>
            <LoaderCircle size={22} color="var(--muted)" aria-hidden />
            <div><div className="label">Images processing</div><div className="value">{ops.processing}</div></div>
          </div>
          <div className="kpi row" style={{ gap: 14 }}>
            <UserCheck size={22} color="var(--muted)" aria-hidden />
            <div><div className="label">Ready for review</div><div className="value">{ops.ready}</div></div>
          </div>
          <div className="kpi row" style={{ gap: 14 }}>
            <CheckCircle2 size={22} color="var(--muted)" aria-hidden />
            <div><div className="label">Completed</div><div className="value">{ops.complete}</div></div>
          </div>
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: sel ? "1fr 380px" : "1fr", gap: 16 }}>
        <Card style={{ padding: view === "table" ? "8px 0 0" : undefined }}>
          {view === "table" ? (
            <table>
              <thead>
                <tr>
                  <th>Inspection</th><th>Sample</th><th>Material</th>
                  <th className="num">Fields</th><th>Review</th>
                  <th>Preliminary result</th><th>Model</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((i) => {
                  const done = i.fields.length ? reviewedCount(i, status) : i.reviewedCount;
                  const st = stateOf(i);
                  return (
                    <tr key={i.inspectionId}
                        className={`clickable ${selected === i.inspectionId ? "selected" : ""}`}
                        onClick={() => setSelected(i.inspectionId)}>
                      <td><strong>{i.inspectionId}</strong></td>
                      <td>{i.sampleId}</td>
                      <td>{i.material}</td>
                      <td className="num">{i.fieldCount}</td>
                      <td className="num">{done} / {i.fieldCount}</td>
                      <td>
                        {st === "processing"
                          ? <span className="chip info">Processing</span>
                          : <DispositionChip v={i.preliminary.disposition} size="sm" />}
                        {i.seeded && <span className="chip muted" style={{ marginLeft: 6 }}>seeded</span>}
                      </td>
                      <td className="mono">{i.model.id.split(" ")[0]}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <div style={{ padding: 6 }}>
              {shown.map((i) => (
                <div key={i.inspectionId} style={{ marginBottom: 14 }}>
                  <div className="row" style={{ cursor: "pointer", fontWeight: 600 }}
                       onClick={() => setSelected(i.inspectionId)}>
                    <FolderTree size={16} color="var(--muted)" aria-hidden />
                    {i.inspectionId} · {i.sampleId}
                    <DispositionChip v={i.preliminary.disposition} size="sm" />
                  </div>
                  <div style={{ paddingLeft: 26, marginTop: 6, fontSize: 13, color: "var(--muted)" }}>
                    {i.fields.length
                      ? <>cross-section · {i.fieldCount} fields ·{" "}
                          <a href={`/inspections/${i.inspectionId}`}
                             onClick={(e) => { e.preventDefault(); nav(`/inspections/${i.inspectionId}`); }}>
                            open sample analysis
                          </a></>
                      : <>seeded row — no field data in the prototype dataset</>}
                  </div>
                </div>
              ))}
            </div>
          )}
          {shown.length === 0 && <div className="empty">No inspections match.</div>}
        </Card>

        {sel && (
          <Card title={sel.sampleId}
                right={<button className="btn sm" onClick={() => setSelected(null)}
                               aria-label="Close details"><X size={15} /></button>}>
            <div className="mlist">
              <MRow k="Sample type" v={sel.sampleType} />
              <MRow k="Material" v={sel.material} />
              <MRow k="Batch" v={sel.batch} />
              <MRow k="Calibration" v={`${sel.umPerPixel} µm/px`} />
              <MRow k="Profile" v={sel.measurementProfile} />
              <MRow k="Production model" v={sel.model.id} />
            </div>

            <div className={`note ${sel.preliminary.disposition === "FAIL" ? "bad" : ""}`}
                 style={{ marginTop: 14 }}>
              <div className="row" style={{ gap: 8 }}>
                <DispositionChip v={sel.preliminary.disposition} />
                <strong>Preliminary</strong>
              </div>
              {sel.fields.length > 0 && (
                <div style={{ fontSize: 24, fontWeight: 700, margin: "8px 0 2px",
                              color: sel.preliminary.disposition === "FAIL" ? "var(--danger)" : "var(--success)" }}>
                  {sel.kpis.worstClusterSeverityUm.toFixed(2)} µm
                </div>
              )}
              <div style={{ marginTop: 4 }}>{sel.preliminary.reason}</div>
              <div className="provisional" style={{ marginTop: 6 }}>
                Model-derived · human review pending
              </div>
            </div>

            {sel.userEntered.length > 0 && (
              <div className="note warn" style={{ marginTop: 12 }}>
                Operator-entered, not inferred from the image:{" "}
                <strong>{sel.userEntered.join(", ")}</strong>. Material is an inspection
                input; apparent fibre radius depends on microscope scale and must not be
                used to infer it.
              </div>
            )}

            <button className="btn primary" style={{ width: "100%", marginTop: 16, justifyContent: "center" }}
                    disabled={sel.fields.length === 0}
                    onClick={() => nav(`/inspections/${sel.inspectionId}`)}>
              Open sample analysis
            </button>
          </Card>
        )}
      </div>

      {importing && <NewInspectionModal onClose={() => setImporting(false)} />}
    </div>
  );
}
