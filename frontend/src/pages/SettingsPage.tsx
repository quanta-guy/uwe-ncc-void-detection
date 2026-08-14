/**
 * Settings.
 *
 * Selecting a checkpoint does not activate it. Validation must first confirm the file
 * loads and returns a three-class mask at the expected dimensions - a model that fails
 * that check would otherwise become active and silently produce nonsense for a whole
 * inspection.
 *
 * The severity limit lives here because it is a business rule, not a model property.
 * The challenge handout says 60 um and the NCC deck says 25 um; evaluation.py uses 25.
 * Making it an editable profile is the honest resolution.
 */

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, FolderOpen, Loader2, ShieldCheck } from "lucide-react";
import { useData } from "../app";
import {
  activateModel, clearReviews, listModels, restoreInspections, validateModel,
} from "../api";
import type { ModelGroup, ModelRecord, ValidationResult } from "../api";
import { Card, MRow } from "../components/common";

/**
 * Specimen-level confusion at the 25 µm line, for models that were actually given an
 * out-of-fold run. FN is the cell that matters - a missed failing specimen ships a
 * bad part - which is why it is tinted, not TP.
 */
function ConfusionMatrix({ r }: { r: ModelRecord }) {
  if (r.tp === undefined) {
    return (
      <div className="note warn" style={{ marginTop: 12 }}>
        No out-of-fold confusion was measured for this model — held-out split Dice{" "}
        <strong>{r.valDice?.toFixed(4)}</strong> only. Showing a matrix here would be
        inventing one.
      </div>
    );
  }
  const cell = (v: number | undefined, tone?: "good" | "bad") => (
    <td style={{
      padding: "8px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums",
      fontWeight: 600,
      background: tone === "bad" ? "#fdeaea" : tone === "good" ? "#e8f5ea" : undefined,
    }}>{v?.toLocaleString()}</td>
  );
  const th = { padding: "8px 14px", fontWeight: 600, color: "var(--muted)", fontSize: 12.5 };
  return (
    <div style={{ marginTop: 12 }}>
      <div className="row" style={{ gap: 16, flexWrap: "wrap", marginBottom: 8 }}>
        <span className="chip pass">Final score {r.final?.toFixed(4)}</span>
        <span className="chip">Dice_void {r.diceVoid?.toFixed(4)}</span>
        <span className="chip">F2 {r.f2?.toFixed(4)}</span>
      </div>
      <table style={{ borderCollapse: "collapse", border: "1px solid var(--border)" }}
             aria-label="Specimen-level confusion matrix">
        <thead>
          <tr>
            <th style={th}></th>
            <th style={th}>Actual FAIL</th>
            <th style={th}>Actual PASS</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{ borderTop: "1px solid var(--border)" }}>
            <th style={{ ...th, textAlign: "left" }}>Predicted FAIL</th>
            {cell(r.tp, "good")}
            {cell(r.fp)}
          </tr>
          <tr style={{ borderTop: "1px solid var(--border)" }}>
            <th style={{ ...th, textAlign: "left" }}>Predicted PASS</th>
            {cell(r.fn, "bad")}
            {cell(r.tn)}
          </tr>
        </tbody>
      </table>
      <div className="note" style={{ marginTop: 10 }}>
        {r.protocol} · {r.operatingPoint}. Pass/fail is per specimen at the 25 µm
        severity line, scored by NCC's evaluation code. The {r.fn} missed failures are
        the cost the F2 metric weighs 4× a false alarm.
      </div>
    </div>
  );
}

export function SettingsPage() {
  const { fx, reviews, hidden } = useData();
  const [models, setModels] = useState<ModelGroup[]>([]);
  const [backendErr, setBackendErr] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [activeId, setActiveId] = useState("");
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [limit, setLimit] = useState(String(fx.profile.clusterSeverityLimitUm));

  // Checkpoints are discovered on disk, not listed in the source. A new run appears
  // here without a code change, which is the point of making weights selectable.
  useEffect(() => {
    listModels()
      .then(({ models: m, active }) => {
        setModels(m);
        setActiveId(active);
        setSelected(active || m[0]?.id || "");
      })
      .catch((e: Error) => setBackendErr(e.message));
  }, []);

  const runValidation = async () => {
    setBusy(true);
    setResult(null);
    try {
      const r = await validateModel(selected);
      setResult(r);
      // Loading is not activating. Only a checkpoint that passed the gate becomes
      // the one new inspections run on.
      if (r.ok) {
        await activateModel(selected);
        setActiveId(selected);
      }
    } catch (e) {
      setResult({ ok: false, id: selected, detail: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <h1>Settings</h1>
      <p className="sub">
        Local prototype configuration. All state is filesystem and browser storage - no
        database is created.
      </p>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "start" }}>
        <Card title="Active model">
          {backendErr ? (
            <div className="note bad">
              <strong>Inference backend is not running.</strong> Checkpoints cannot be
              listed or validated, and new inspections cannot be imported.
              <div className="mono" style={{ marginTop: 8 }}>
                python frontend/server/app.py
              </div>
            </div>
          ) : (
            <>
              <label className="field" htmlFor="mdl">
                Checkpoint — {models.length} discovered on disk
              </label>
              <select id="mdl" value={selected}
                      onChange={(e) => { setSelected(e.target.value); setResult(null); }}>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label} · {m.sizeMb} MB{m.id === activeId ? "  (active)" : ""}
                  </option>
                ))}
              </select>

              <div className="row" style={{ marginTop: 14 }}>
                <button className="btn primary" onClick={runValidation} disabled={busy || !selected}>
                  {busy
                    ? <><Loader2 size={16} className="spin" aria-hidden /> Loading checkpoint…</>
                    : <><ShieldCheck size={16} aria-hidden /> Validate and activate</>}
                </button>
                {result && (
                  <span className={`chip validate-result ${result.ok ? "pass" : "fail"}`}>
                    {result.ok
                      ? <CheckCircle2 size={14} aria-hidden />
                      : <AlertTriangle size={14} aria-hidden />}
                    {result.ok
                      ? `${result.classes} classes · ${result.dims} · ${result.device} · ${result.elapsedMs} ms`
                      : "Rejected"}
                  </span>
                )}
              </div>

              {result && (
                <div className={`note ${result.ok ? "good" : "bad"}`} style={{ marginTop: 12 }}>
                  {result.detail}
                  {result.ok && result.disagreementAvailable === false && (
                    <> Single checkpoint: model disagreement will be zero and no field
                      will ever be queued for review on disagreement.</>
                  )}
                </div>
              )}

              {selected !== activeId && (
                <div className="note warn" style={{ marginTop: 14 }}>
                  Selecting a file does not activate it. The checkpoint must load and
                  return a three-class mask at the expected dimensions before an
                  inspection is allowed to run on it.
                </div>
              )}

              {(() => {
                const rec = models.find((m) => m.id === selected)?.record;
                return rec ? (
                  <>
                    <ConfusionMatrix r={rec} />
                    <div className="note" style={{ marginTop: 10 }}>{rec.note}</div>
                  </>
                ) : null;
              })()}
            </>
          )}

          <div className="mlist" style={{ marginTop: 14 }}>
            <MRow k="Active for new imports" v={<span className="mono">{activeId || "—"}</span>} />
            <MRow k="Fixture model ID" v={<span className="mono">{fx.model.id}</span>} />
            <MRow k="SHA-256" v={<span className="mono">{fx.model.sha256}...</span>} />
            <MRow k="Threshold" v={fx.model.threshold} />
            <MRow k="Minimum object size" v={`${fx.model.minSize} px`} />
            <MRow k="Checkpoints" v={fx.model.checkpoints.length} />
          </div>

          <div className="note" style={{ marginTop: 12 }}>
            Model disagreement needs at least two checkpoints. With a single model the
            confidence signal is identically zero and fields are never queued for
            review on disagreement.
          </div>
        </Card>

        <div className="grid">
          <Card title="Measurement profile">
            <div className="mlist">
              <MRow k="Profile" v={fx.profile.name} />
              <MRow k="Merge distance" v={`${fx.profile.mergeDistanceUm} um`} />
              <MRow k="Area factor" v={fx.profile.areaFactor} />
              <MRow k="Approved for production" v={fx.profile.approvedForProductionUse ? "Yes" : "No"}
                    tone={fx.profile.approvedForProductionUse ? "good" : "bad"} />
            </div>
            <label className="field" htmlFor="lim" style={{ marginTop: 14 }}>
              Cluster severity limit (um)
            </label>
            <input id="lim" type="number" step="1" value={limit}
                   onChange={(e) => setLimit(e.target.value)} />
            <div className="note warn" style={{ marginTop: 12 }}>
              The challenge handout states 60 um and the NCC partner deck states 25 um.
              The scoring harness uses 25. It is exposed here as a business rule rather
              than hard-coded, because a pass/fail line belongs to an approved
              measurement profile, not to a model.
            </div>
          </Card>

          <Card title="Data and feedback">
            <div className="mlist">
              <MRow k="Prototype data folder" v={<span className="mono">frontend/public/data</span>} />
              <MRow k="Fixtures generated" v={new Date(fx.generatedAt).toLocaleString()} />
              <MRow k="Review events recorded" v={reviews.length} />
              <MRow k="Inspections hidden" v={hidden.length}
                    title="Removed from the workspace but not destroyed" />
              <MRow k="Keep original predictions" v="Always on" tone="good"
                    title="Corrections are new versions; predictions are never overwritten" />
            </div>
            <div className="row" style={{ marginTop: 14 }}>
              <button className="btn"><FolderOpen size={16} aria-hidden /> Export corrections</button>
              <button className="btn" onClick={() => {
                if (confirm("Clear all recorded review decisions in this browser?")) clearReviews();
              }}>
                Reset review history
              </button>
              <button className="btn" disabled={hidden.length === 0}
                      onClick={restoreInspections}>
                Restore {hidden.length} hidden inspection{hidden.length === 1 ? "" : "s"}
              </button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
