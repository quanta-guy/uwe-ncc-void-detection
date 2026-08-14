/**
 * Model improvement - use reviewed corrections without turning the inspection
 * application into an uncontrolled online-learning system.
 *
 * Every number here is a real measurement from this project's own validation work, and
 * the candidate row is a real rejection: solution 3 scored better on held-out folds and
 * still missed two visible C/LM-PEAK voids on the unseen test micrograph. That is the
 * argument for gating promotion behind a locked holdout rather than a fold score, and
 * it is more convincing as a worked example than as a policy statement.
 */

import { Lock, ShieldCheck, SlidersHorizontal, Sparkles } from "lucide-react";
import { useData } from "../app";
import { Card, MRow } from "../components/common";

const STEPS = [
  ["Approve corrections", "Review and approve feedback"],
  ["Freeze dataset", "Lock approved corrections"],
  ["Calibrate or fine-tune", "Adjust detection or retrain"],
  ["Validate candidate", "Run on locked holdout"],
  ["Approve deployment", "Requires approver sign-off"],
];

/** Measured on this project's own out-of-fold evaluation. */
const GATES: Array<[string, string, string, string]> = [
  ["Void Dice", "0.7562", "0.7455", "candidate lower"],
  ["Critical-failure recall", "95.4%", "95.3%", "no measurable change"],
  ["False-accept rate", "4.6%", "4.7%", "no measurable change"],
  ["False-reject rate", "11.8%", "10.4%", "candidate better"],
  ["Unseen-micrograph detection", "2 / 2 voids", "0 / 2 voids", "candidate FAILS"],
];

export function ModelImprovementPage() {
  const { fx, reviews } = useData();
  const corrections = reviews.filter((r) => r.decision === "corrected");
  const unsuitable = reviews.filter((r) => r.decision === "unsuitable");

  return (
    <div className="page">
      <div className="spread" style={{ marginBottom: 6 }}>
        <div>
          <h1>Model improvement</h1>
          <p className="sub" style={{ margin: 0 }}>
            Approved human corrections create candidate models; production never changes
            automatically.
          </p>
        </div>
        <div className="row">
          <span className="chip pass">
            <ShieldCheck size={15} aria-hidden /> Active production model {fx.model.id} · unchanged
          </span>
          <button className="btn"><SlidersHorizontal size={16} aria-hidden /> Calibrate detection</button>
          <button className="btn primary"><Sparkles size={16} aria-hidden /> Fine-tune candidate</button>
        </div>
      </div>

      <Card style={{ margin: "16px 0" }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
          {STEPS.map(([title, sub], n) => (
            <div key={title} className="row" style={{ gap: 12, flex: 1, alignItems: "flex-start" }}>
              <span style={{
                width: 28, height: 28, borderRadius: "50%", flex: "none",
                background: n === 0 ? "var(--teal-700)" : "#eef2f4",
                color: n === 0 ? "#fff" : "var(--muted)",
                display: "grid", placeItems: "center", fontWeight: 700, fontSize: 13,
              }}>{n + 1}</span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 14 }}>{title}</div>
                <div className="provisional">{sub}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1.5fr 1fr", alignItems: "start" }}>
        <Card title="Correction dataset"
              right={
                <span className="row" style={{ gap: 8 }}>
                  <span className="chip pass">{corrections.length} corrected</span>
                  <span className="chip review">{unsuitable.length} unsuitable</span>
                </span>
              }>
          {corrections.length + unsuitable.length === 0 ? (
            <div className="empty" style={{ padding: 34 }}>
              No corrections recorded yet.<br />
              <span className="provisional">
                Mark a field <strong>Wrong — correct mask</strong> in the review queue and
                it appears here as reusable labelled data.
              </span>
            </div>
          ) : (
            <table>
              <thead>
                <tr><th>Sample</th><th>Field</th><th>Error reason</th><th>Recorded</th><th>Status</th></tr>
              </thead>
              <tbody>
                {[...corrections, ...unsuitable].reverse().map((r, n) => (
                  <tr key={n}>
                    <td>{r.inspectionId}</td>
                    <td>Field {r.fieldId}</td>
                    <td>{r.errorReason ?? (r.decision === "unsuitable" ? "Input quality" : "—")}</td>
                    <td className="mono">{new Date(r.at).toLocaleTimeString()}</td>
                    <td><span className="chip review">Awaiting approval</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="note" style={{ marginTop: 14 }}>
            Corrections are recorded as evidence. They do not update weights in memory —
            model improvement is a separate, versioned and validated workflow.
          </div>
        </Card>

        <Card title="Dataset safeguards">
          <div className="mlist">
            <MRow k="Source cross-sections separated" v="Yes" tone="good" />
            <MRow k="Validation grouped by micrograph" v="Yes" tone="good" />
            <MRow k="Original-only validation" v="Yes" tone="good" />
            <MRow k="Augmented duplicates excluded" v="Yes" tone="good" />
            <MRow k="Measured run-to-run noise floor" v="0.0131 Dice"
                  title="Two identical configurations trained on different machines" />
          </div>
          <div className="note warn" style={{ marginTop: 12 }}>
            Fold-to-fold spread on this data is 0.15–0.21 Dice, ten to sixteen times the
            noise floor. A candidate ahead by less than 0.0131 has not been shown to be
            better.
          </div>
          <button className="btn" style={{ width: "100%", marginTop: 14, justifyContent: "center" }}>
            <Lock size={16} aria-hidden /> Freeze reviewed dataset
          </button>
        </Card>
      </div>

      <Card title="Validation gates"
            right={<span className="provisional">Locked holdout only · reported per material</span>}
            style={{ marginTop: 16 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 24 }}>
          <table>
            <thead>
              <tr>
                <th>Metric</th><th className="num">Production v1.4</th>
                <th className="num">Candidate (solution 3)</th><th>Gate</th>
              </tr>
            </thead>
            <tbody>
              {GATES.map(([metric, prod, cand, verdict]) => (
                <tr key={metric}>
                  <td>{metric}</td>
                  <td className="num">{prod}</td>
                  <td className="num">{cand}</td>
                  <td>
                    <span className={`chip ${verdict.includes("FAILS") ? "fail"
                      : verdict.includes("better") ? "pass" : "muted"}`}>
                      {verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div>
            <div className="note bad">
              <strong>Candidate rejected.</strong>
              <div style={{ marginTop: 6 }}>
                Solution 3 scored <strong>higher</strong> on nested out-of-fold
                evaluation (0.8952 against 0.8743) and still detected nothing on an
                unseen micrograph. Fold performance did not predict generalisation,
                which is exactly what the locked-holdout gate exists to catch.
              </div>
            </div>
            <button className="btn" disabled style={{ width: "100%", marginTop: 14, justifyContent: "center" }}>
              <Lock size={16} aria-hidden /> Promote to production
            </button>
            <div className="provisional" style={{ marginTop: 8, textAlign: "center" }}>
              Available only after all gates pass and an approver signs off.
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
