/**
 * Reports dashboard - operational value across inspections.
 *
 * Correction rate and accepted-first-time are the numbers that decide whether the
 * model is earning its place in the workflow, so they are computed from real review
 * events rather than shown as decoration.
 */

import { useNavigate } from "react-router-dom";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { useData } from "../app";
import { fieldStatus, reviewedCount } from "../api";
import { Card, DispositionChip, Kpi } from "../components/common";

export function ReportsPage() {
  const { fx, status, reviews } = useData();
  const nav = useNavigate();
  const real = fx.inspections.filter((i) => i.fields.length > 0);

  const totalFields = real.reduce((n, i) => n + i.fieldCount, 0);
  const reviewed = real.reduce((n, i) => n + reviewedCount(i, status), 0);
  const corrected = reviews.filter((r) => r.decision === "corrected").length;
  const accepted = reviews.filter((r) => r.decision === "accepted").length;
  const correctionRate = reviewed ? (100 * corrected) / reviewed : 0;
  const firstTime = reviewed ? (100 * accepted) / reviewed : 0;

  const bands = real.flatMap((i) => i.fields).reduce(
    (a, f) => {
      const k = f.triageAction === "REJECT" ? "Fail" : f.triageAction === "REVIEW" ? "Review" : "Pass";
      a[k] = (a[k] ?? 0) + 1;
      return a;
    }, {} as Record<string, number>);
  const donut = [
    { name: "Pass", value: bands.Pass ?? 0, fill: "#258a38" },
    { name: "Review", value: bands.Review ?? 0, fill: "#d99000" },
    { name: "Fail", value: bands.Fail ?? 0, fill: "#d51f26" },
  ];

  return (
    <div className="page">
      <h1>Reports</h1>
      <p className="sub">
        Aggregate view across inspections in the prototype dataset. Click a sample for
        its full evidence package.
      </p>

      <Card style={{ marginBottom: 16 }}>
        <div className="kpis">
          <Kpi label="Samples in dataset" value={real.length} />
          <Kpi label="Fields reviewed" value={`${reviewed} / ${totalFields}`} />
          <Kpi label="Correction rate" value={correctionRate.toFixed(0)} unit="%" />
          <Kpi label="Accepted first time" value={firstTime.toFixed(0)} unit="%" />
          <Kpi label="Corrected masks" value={corrected} />
        </div>
        {reviewed === 0 && (
          <div className="note" style={{ marginTop: 12 }}>
            No review decisions recorded yet. Correction and acceptance rates populate
            as fields are reviewed in the review queue.
          </div>
        )}
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1.6fr", alignItems: "start" }}>
        <Card title="Field disposition"
              right={<span className="provisional">Model-derived</span>}>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <Pie data={donut} dataKey="value" nameKey="name" innerRadius={54} outerRadius={86}
                   paddingAngle={2} isAnimationActive={false}
                   label={(e: { name: string; value: number }) => `${e.name} ${e.value}`}>
                {donut.map((d) => <Cell key={d.name} fill={d.fill} />)}
              </Pie>
              <Tooltip /><Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Samples" style={{ padding: "8px 0 0" }}>
          <table>
            <thead>
              <tr>
                <th>Sample</th><th>Material</th><th className="num">Fields</th>
                <th className="num">Over limit</th><th className="num">Worst severity</th>
                <th>Preliminary</th><th className="num">Reviewed</th>
              </tr>
            </thead>
            <tbody>
              {real.map((i) => (
                <tr key={i.inspectionId} className="clickable"
                    onClick={() => nav(`/reports/${i.inspectionId}`)}>
                  <td><strong>{i.sampleId}</strong></td>
                  <td>{i.material}</td>
                  <td className="num">{i.fieldCount}</td>
                  <td className="num">{i.kpis.fieldsOverLimit}</td>
                  <td className="num">{i.kpis.worstClusterSeverityUm.toFixed(2)} µm</td>
                  <td><DispositionChip v={i.preliminary.disposition} size="sm" /></td>
                  <td className="num">
                    {reviewedCount(i, status)} / {i.fieldCount}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      {reviews.length > 0 && (
        <Card title="Recent review decisions" style={{ marginTop: 16, padding: "8px 0 0" }}>
          <table>
            <thead>
              <tr><th>When</th><th>Sample</th><th>Field</th><th>Decision</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {[...reviews].reverse().slice(0, 12).map((r, n) => {
                const insp = real.find((i) => i.inspectionId === r.inspectionId);
                return (
                  <tr key={n}>
                    <td className="mono">{new Date(r.at).toLocaleTimeString()}</td>
                    <td>{insp?.sampleId ?? r.inspectionId}</td>
                    <td>{r.fieldId}</td>
                    <td>{r.decision}</td>
                    <td>{r.errorReason ?? r.note ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="provisional" style={{ padding: "10px 12px" }}>
            {fieldStatus(status, real[0]?.inspectionId ?? "", "01") !== "pending"
              ? "Decisions are appended, never overwritten."
              : ""}
          </div>
        </Card>
      )}
    </div>
  );
}
