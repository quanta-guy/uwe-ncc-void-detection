/**
 * Sample analysis - understand the cross-section and decide where review starts.
 *
 * The risk map is built from real thumbnails rather than coloured blocks, so a
 * reviewer recognises the microscopy before reading the number. The priority table
 * carries evaluation.py's own reason string for each flagged field: a colour says
 * "look here", a reason says "look here because the models disagree on the extent",
 * and only the second is actionable. It sits in its own row rather than beneath a tall
 * risk map, because it is the action the reviewer takes next.
 *
 * The void-size distribution is a histogram rather than the reference screen's dot
 * plot. Eight overlapping dots on a shared axis cannot be read; the shape of the
 * distribution is the thing worth seeing.
 */

import { Link, useNavigate, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useData } from "../app";
import { asset, bandOf, fieldStatus, priorityOrder, reviewedCount } from "../api";
import { BandChip, Card, Kpi, MRow, ProvisionalTag, StatusChip } from "../components/common";

export function SampleAnalysisPage() {
  const { fx, status } = useData();
  const { inspectionId } = useParams();
  const nav = useNavigate();

  const insp = fx.inspections.find((i) => i.inspectionId === inspectionId);
  if (!insp || insp.fields.length === 0) {
    return (
      <div className="page">
        <div className="empty">
          This inspection has no field data in the prototype dataset.{" "}
          <Link to="/inspections">Back to workspace</Link>
        </div>
      </div>
    );
  }

  const k = insp.kpis;
  const done = reviewedCount(insp, status);
  const fail = insp.preliminary.disposition === "FAIL";
  const queue = priorityOrder(insp.fields).filter((f) => bandOf(f) !== "routine");
  const open = (id: string) => nav(`/inspections/${insp.inspectionId}/fields/${id}`);

  const ferets = insp.fields.flatMap((f) => f.feretsUm);
  const step = Math.ceil(Math.max(35, ...ferets) / 8);
  const bins = Array.from({ length: 8 }, (_, i) => ({
    range: `${i * step}-${(i + 1) * step}`,
    lo: i * step,
    count: ferets.filter((v) => v >= i * step && v < (i + 1) * step).length,
  }));

  const severityByField = insp.fields.map((f) => ({
    field: f.id,
    severity: Number(f.clusterSeverityUm.toFixed(2)),
    fill: f.clusterSeverityUm >= fx.profile.clusterSeverityLimitUm ? "#d51f26" : "#00858b",
  }));

  return (
    <div className="page">
      <div className="crumbs">
        <Link to="/inspections">Inspections</Link> / {insp.inspectionId} / {insp.sampleId}
      </div>

      <div className="spread" style={{ marginBottom: 18 }}>
        <div>
          <h1>Sample analysis</h1>
          <div className="sub" style={{ margin: 0 }}>
            {insp.material} cross-section &middot; {insp.fieldCount} microscopy fields
            &middot; {insp.umPerPixel} &micro;m/px &middot; {insp.model.id}
          </div>
        </div>
        <div className="row">
          <button className="btn primary" onClick={() => open(priorityOrder(insp.fields)[0].id)}>
            Start priority review
          </button>
          <button className="btn" onClick={() => open(insp.fields[0].id)}>
            Sequential review
          </button>
        </div>
      </div>

      <Card style={{ marginBottom: 16 }}>
        <div className="spread" style={{ marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Preliminary quality result</h2>
          <ProvisionalTag reviewed={done} total={insp.fieldCount} />
        </div>
        <div className="kpis">
          <div className="kpi" style={{ flex: "0 0 300px" }}>
            <div style={{ fontSize: 46, fontWeight: 700, letterSpacing: "-0.03em",
                          color: fail ? "var(--danger)" : "var(--success)" }}>
              {insp.preliminary.disposition}
            </div>
            <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 4 }}>
              {insp.preliminary.reason}
            </div>
          </div>
          <Kpi label="Void areal fraction" value={k.voidArealFractionPct.toFixed(3)} unit="%" />
          <Kpi label="Worst-cluster severity" value={k.worstClusterSeverityUm.toFixed(2)}
               unit="&micro;m" tone={fail ? "bad" : "normal"} />
          <Kpi label="Fields over limit" value={`${k.fieldsOverLimit} / ${k.fieldsTotal}`}
               tone={k.fieldsOverLimit ? "bad" : "normal"} />
          <Kpi label="Largest void Feret" value={k.largestFeretUm.toFixed(2)} unit="&micro;m" />
        </div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1.35fr 1fr", alignItems: "start" }}>
        <Card title="Field risk map"
              right={<span className="provisional">Reviewed {done} / {insp.fieldCount}</span>}>
          <div className="row" style={{ marginBottom: 12, gap: 18 }}>
            {(["critical", "review", "routine"] as const).map((b) => (
              <span key={b} className="row" style={{ gap: 6, fontSize: 13, color: "var(--muted)" }}>
                <span className={`dot ${b}`} aria-hidden />
                {b[0].toUpperCase() + b.slice(1)}
              </span>
            ))}
          </div>
          <div className="riskmap">
            {insp.fields.map((f) => (
              <div key={f.id} className={`riskcell ${bandOf(f)}`} onClick={() => open(f.id)}
                   role="button" tabIndex={0}
                   aria-label={`Field ${f.id}, ${bandOf(f)}, severity ${f.clusterSeverityUm.toFixed(1)} microns`}
                   onKeyDown={(e) => e.key === "Enter" && open(f.id)}>
                <img src={asset("thumbs", f.stem)} alt="" aria-hidden />
                <span className="no">{f.id}</span>
                {fieldStatus(status, insp.inspectionId, f.id) !== "pending" && (
                  <span className="flag">
                    <StatusChip s={fieldStatus(status, insp.inspectionId, f.id)} />
                  </span>
                )}
              </div>
            ))}
          </div>
        </Card>

        <div className="grid">
          <Card title="Sample status">
            <div className="mlist">
              <MRow k="Preliminary disposition" v={insp.preliminary.disposition}
                    tone={fail ? "bad" : "good"} />
              <MRow k="Batch inference" v={`${insp.fieldCount} / ${insp.fieldCount}`} />
              <MRow k="Human review" v={`${done} / ${insp.fieldCount}`} />
              <MRow k="Measurement profile" v={fx.profile.name} />
              <MRow k="Calibration" v={`${insp.umPerPixel} µm/px`} />
              <MRow k="Analysed area" v={`${k.analysedAreaMm2.toFixed(4)} mm²`} />
            </div>
            {!fx.profile.approvedForProductionUse && (
              <div className="note warn" style={{ marginTop: 12 }}>{fx.profile.note}</div>
            )}
          </Card>

          <Card title="Composition and porosity"
                right={<span className="provisional">Estimated from prediction mask</span>}>
            <div className="kpis">
              <Kpi label="Fibre" value={k.fibreArealFractionPct.toFixed(3)} unit="%" small />
              <Kpi label="Matrix" value={k.matrixArealFractionPct.toFixed(3)} unit="%" small />
              <Kpi label="Void count" value={k.voidCount} small />
              <Kpi label="Void density" value={k.voidDensityPerMm2.toFixed(1)}
                   unit="/mm&sup2;" small />
            </div>
            <div className="note" style={{ marginTop: 12 }}>
              Two-dimensional area fraction approximates volume fraction only under
              valid stereological assumptions. Not a certified fibre volume fraction.
            </div>
          </Card>

          <Card title="Void-size distribution"
                right={<span className="provisional">{ferets.length} detected voids</span>}>
            <div className="kpis" style={{ marginBottom: 10 }}>
              <Kpi label="Feret p50" value={k.feretP50Um.toFixed(2)} unit="&micro;m" small />
              <Kpi label="Feret p95" value={k.feretP95Um.toFixed(2)} unit="&micro;m" small />
              <Kpi label="Maximum" value={k.largestFeretUm.toFixed(2)} unit="&micro;m" small />
            </div>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={bins} margin={{ top: 4, right: 8, bottom: 4, left: -22 }}>
                <CartesianGrid stroke="#eef2f4" vertical={false} />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} interval={0} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [v, "voids"]}
                         labelFormatter={(l) => `Feret ${l} microns`} />
                <ReferenceLine
                  x={bins.find((b) => k.feretP50Um >= b.lo && k.feretP50Um < b.lo + step)?.range}
                  stroke="#00858b" strokeDasharray="3 3"
                  label={{ value: "p50", fontSize: 10, fill: "#00858b" }} />
                <Bar dataKey="count" fill="#00858b" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "1.35fr 1fr",
                                     alignItems: "start", marginTop: 16 }}>
        <Card title="Review priority"
              right={<span className="provisional">Ordered by measured risk, not filename</span>}>
          {queue.length === 0 ? (
            <div className="empty" style={{ padding: 24 }}>No fields flagged for review.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Field</th><th>Reason</th><th>Band</th>
                  <th className="num">Severity</th><th>Review</th>
                </tr>
              </thead>
              <tbody>
                {queue.map((f) => (
                  <tr key={f.id} className="clickable" onClick={() => open(f.id)}>
                    <td><strong>{f.id}</strong></td>
                    <td>{f.triageReason}</td>
                    <td><BandChip band={bandOf(f)} /></td>
                    <td className="num">{f.clusterSeverityUm.toFixed(2)} &micro;m</td>
                    <td><StatusChip s={fieldStatus(status, insp.inspectionId, f.id)} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <Card title="Cluster severity by field">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={severityByField} margin={{ top: 6, right: 12, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="#eef2f4" vertical={false} />
              <XAxis dataKey="field" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => [`${v} microns`, "Cluster severity"]} />
              <ReferenceLine y={fx.profile.clusterSeverityLimitUm} stroke="#d51f26"
                             strokeDasharray="4 3"
                             label={{ value: `limit ${fx.profile.clusterSeverityLimitUm}`,
                                      position: "insideTopRight", fill: "#d51f26", fontSize: 11 }} />
              <Bar dataKey="severity" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
