/**
 * Cross-section quality report - the complete evidence package for one sample.
 *
 * This is the only inspection screen carrying the full KPI set, and the only one that
 * exports. The controlling evidence panel shows the severity annotation over the real
 * field, so the disposition can be checked rather than taken on trust.
 *
 * CSV and JSON export from the fixture data directly, so what is exported is exactly
 * what is displayed.
 */

import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CheckCircle2, Download, FileJson, Table2 } from "lucide-react";
import { useData } from "../app";
import { asset, bandOf, fieldStatus, reviewedCount } from "../api";
import { BandChip, Card, Kpi, MRow, ProvisionalTag, StatusChip } from "../components/common";
import { SeverityOverlay, SeverityFormula } from "../components/SeverityOverlay";
import { ReportChat } from "../components/ReportChat";

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function CrossSectionReportPage() {
  const { fx, status, reviews } = useData();
  const { inspectionId } = useParams();
  const insp = fx.inspections.find((i) => i.inspectionId === inspectionId);

  const csv = useMemo(() => {
    if (!insp) return "";
    const head = [
      "field", "stem", "um_per_pixel", "void_areal_fraction_pct", "void_count",
      "largest_feret_um", "cluster_severity_um", "distance_to_limit_um", "verdict",
      "triage_action", "triage_reason", "review_status",
    ];
    const rows = insp.fields.map((f) => [
      f.id, f.stem, f.umPerPixel, f.voidArealFractionPct, f.voidCount,
      f.largestFeretUm, f.clusterSeverityUm, f.distanceToLimitUm, f.verdict,
      f.triageAction, `"${f.triageReason}"`,
      fieldStatus(status, insp.inspectionId, f.id),
    ]);
    return [head, ...rows].map((r) => r.join(",")).join("\n");
  }, [insp, status]);

  if (!insp || insp.fields.length === 0) {
    return (
      <div className="page">
        <div className="empty">
          No report for this inspection. <Link to="/reports">Back to reports</Link>
        </div>
      </div>
    );
  }

  const k = insp.kpis;
  const done = reviewedCount(insp, status);
  const fail = insp.preliminary.disposition === "FAIL";
  const controlling = insp.fields.find((f) => f.id === insp.controllingFieldId)!;
  const exceptions = insp.fields
    .filter((f) => f.triageAction !== "ACCEPT")
    .sort((a, b) => b.clusterSeverityUm - a.clusterSeverityUm);

  const evidence = {
    inspectionId: insp.inspectionId, sampleId: insp.sampleId, material: insp.material,
    materialSource: "operator-entered", calibrationUmPerPixel: insp.umPerPixel,
    measurementProfile: fx.profile, model: fx.model,
    fieldsReviewed: done, fieldsTotal: insp.fieldCount,
    kpis: k, controllingFieldId: insp.controllingFieldId,
    fields: insp.fields.map((f) => ({
      id: f.id, stem: f.stem, severityUm: f.clusterSeverityUm, verdict: f.verdict,
      reviewStatus: fieldStatus(status, insp.inspectionId, f.id),
    })),
    reviewEvents: reviews.filter((r) => r.inspectionId === insp.inspectionId),
    generatedAt: new Date().toISOString(),
  };

  return (
    <div className="page">
      <div className="crumbs">
        <Link to="/reports">Reports</Link> / {insp.inspectionId} / {insp.sampleId}
      </div>

      <div className="spread" style={{ marginBottom: 18, alignItems: "flex-start" }}>
        <div>
          <h1>Cross-section quality report</h1>
          <div className="sub" style={{ margin: 0 }}>
            {insp.material} sample · {insp.fieldCount} microscopy fields · analysed area{" "}
            {k.analysedAreaMm2.toFixed(4)} mm²
          </div>
        </div>
        <div className="row">
          <button className="btn" onClick={() => window.print()}>
            <Download size={16} aria-hidden /> Print / PDF
          </button>
          <button className="btn" onClick={() => download(`${insp.sampleId}-measurements.csv`, csv, "text/csv")}>
            <Table2 size={16} aria-hidden /> Export CSV
          </button>
          <button className="btn"
                  onClick={() => download(`${insp.sampleId}-evidence.json`,
                                          JSON.stringify(evidence, null, 2), "application/json")}>
            <FileJson size={16} aria-hidden /> Evidence JSON
          </button>
          <button className="btn primary" disabled={done < insp.fieldCount}
                  title={done < insp.fieldCount
                    ? `Review all ${insp.fieldCount} fields before approval`
                    : "Approve this report"}>
            <CheckCircle2 size={16} aria-hidden /> Approve report
          </button>
        </div>
      </div>

      <Card style={{ marginBottom: 16, borderColor: fail ? "var(--danger)" : "var(--border)" }}>
        <div className="spread">
          <div className="row" style={{ gap: 34 }}>
            <div>
              <div className="label" style={{ fontSize: 13, color: "var(--muted)" }}>DISPOSITION</div>
              <div style={{ fontSize: 40, fontWeight: 700, letterSpacing: "-0.03em",
                            color: fail ? "var(--danger)" : "var(--success)" }}>
                {insp.preliminary.disposition}
              </div>
            </div>
            <div style={{ maxWidth: 380 }}>
              <div className="label" style={{ fontSize: 13, color: "var(--muted)" }}>REASON</div>
              <div style={{ marginTop: 6 }}>{insp.preliminary.reason}</div>
            </div>
            <div>
              <div className="label" style={{ fontSize: 13, color: "var(--muted)" }}>STATUS</div>
              <div style={{ marginTop: 8 }}>
                <span className="chip review">
                  {done < insp.fieldCount ? "Awaiting review" : "Awaiting approval"}
                </span>
              </div>
            </div>
          </div>
          <ProvisionalTag reviewed={done} total={insp.fieldCount} />
        </div>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div className="kpis">
          <Kpi label="Void areal fraction" value={k.voidArealFractionPct.toFixed(3)} unit="%" />
          <Kpi label="Worst-cluster severity" value={k.worstClusterSeverityUm.toFixed(2)}
               unit="µm" tone={fail ? "bad" : "normal"} />
          <Kpi label="Fields over limit"
               value={`${k.fieldsOverLimit} / ${k.fieldsTotal}`}
               unit={` (${((100 * k.fieldsOverLimit) / k.fieldsTotal).toFixed(1)}%)`}
               tone={k.fieldsOverLimit ? "bad" : "normal"} />
          <Kpi label="Largest void Feret" value={k.largestFeretUm.toFixed(2)} unit="µm" />
        </div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr 1fr", alignItems: "start" }}>
        <div className="grid">
          <Card title="Spatial evidence">
            <div className="riskmap" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
              {insp.fields.map((f) => (
                <div key={f.id} className={`riskcell ${bandOf(f)}`} title={`Field ${f.id}`}>
                  <img src={asset("thumbs", f.stem)} alt="" aria-hidden />
                  <span className="no">{f.id}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card title="Cluster severity by field">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={insp.fields.map((f) => ({
                field: f.id, severity: Number(f.clusterSeverityUm.toFixed(2)),
                fill: f.clusterSeverityUm >= fx.profile.clusterSeverityLimitUm ? "#d51f26" : "#00858b",
              }))} margin={{ top: 6, right: 10, bottom: 2, left: -20 }}>
                <CartesianGrid stroke="#eef2f4" vertical={false} />
                <XAxis dataKey="field" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [`${v} µm`, "Severity"]} />
                <ReferenceLine y={fx.profile.clusterSeverityLimitUm} stroke="#d51f26" strokeDasharray="4 3" />
                <Bar dataKey="severity" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <Card title={`Controlling evidence — field ${controlling.id}`}>
          <div className="viewer">
            <div className="cap">Original with severity annotation</div>
            <div className="stack">
              <img src={asset("originals", controlling.stem)} alt={`Field ${controlling.id}`} />
              <img src={asset("void", controlling.stem)} alt="" aria-hidden style={{ opacity: 0.6 }} />
              <img src={asset("controlling", controlling.stem)} alt="" aria-hidden />
              <SeverityOverlay evidence={controlling.severityEvidence} size={256} />
            </div>
          </div>
          <SeverityFormula evidence={controlling.severityEvidence} />
          <div className="mlist" style={{ marginTop: 12 }}>
            <MRow k="Field void fraction" v={`${controlling.voidArealFractionPct.toFixed(3)}%`} />
            <MRow k="Void count" v={controlling.voidCount} />
            <MRow k="Cluster severity" v={`${controlling.clusterSeverityUm.toFixed(2)} µm`} tone="bad" />
            <MRow k="Distance to limit"
                  v={`${controlling.distanceToLimitUm >= 0 ? "+" : ""}${controlling.distanceToLimitUm.toFixed(2)} µm`}
                  tone={controlling.distanceToLimitUm >= 0 ? "bad" : "good"} />
          </div>
        </Card>

        <div className="grid">
          <Card title="Composition and morphology">
            <div className="mlist">
              <MRow k="Fibre areal fraction" v={`${k.fibreArealFractionPct.toFixed(3)}%`} />
              <MRow k="Matrix areal fraction" v={`${k.matrixArealFractionPct.toFixed(3)}%`} />
              <MRow k="Void density" v={`${k.voidDensityPerMm2.toFixed(1)} /mm²`} />
              <MRow k="Void count" v={k.voidCount} />
              <MRow k="Feret p50" v={`${k.feretP50Um.toFixed(2)} µm`} />
              <MRow k="Feret p95" v={`${k.feretP95Um.toFixed(2)} µm`} />
              <MRow k="Maximum Feret" v={`${k.largestFeretUm.toFixed(2)} µm`} />
            </div>
          </Card>

          <Card title="Traceability and assurance">
            <div className="mlist">
              <MRow k="Inspection" v={insp.inspectionId} />
              <MRow k="Production model" v={<span className="mono">{fx.model.id}</span>} />
              <MRow k="Model SHA-256" v={<span className="mono">{fx.model.sha256}…</span>} />
              <MRow k="Operating point"
                    v={<span className="mono">t={fx.model.threshold} min={fx.model.minSize}</span>} />
              <MRow k="Measurement profile" v={fx.profile.name} />
              <MRow k="Calibration" v={`${insp.umPerPixel} µm/px`} />
              <MRow k="Fields reviewed" v={`${done} / ${insp.fieldCount}`} />
              <MRow k="Prediction retained" v="Yes" tone="good" />
            </div>
            <div className="note" style={{ marginTop: 12 }}>
              Model metrics are deliberately absent from this report. Void Dice and
              recall describe the model, not the material, and belong in Model
              improvement.
            </div>
          </Card>
        </div>
      </div>

      <Card title="Exceptions and review decisions" style={{ marginTop: 16, padding: "8px 0 0" }}>
        <table>
          <thead>
            <tr>
              <th>Field</th><th>Issue</th><th>Band</th>
              <th className="num">Severity</th><th>Why</th><th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {exceptions.map((f) => (
              <tr key={f.id}>
                <td><strong>Field {f.id}</strong></td>
                <td>{f.triageAction === "REJECT" ? "Over profile limit" : "Needs review"}</td>
                <td><BandChip band={bandOf(f)} /></td>
                <td className="num">{f.clusterSeverityUm.toFixed(2)} µm</td>
                <td>{f.triageReason}</td>
                <td><StatusChip s={fieldStatus(status, insp.inspectionId, f.id)} /></td>
              </tr>
            ))}
            {exceptions.length === 0 && (
              <tr><td colSpan={6} style={{ color: "var(--muted)" }}>No exceptions raised.</td></tr>
            )}
          </tbody>
        </table>
      </Card>

      <ReportChat inspection={insp} fx={fx} status={status} />
    </div>
  );
}
