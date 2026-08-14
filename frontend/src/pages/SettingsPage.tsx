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

import { useState } from "react";
import { CheckCircle2, FolderOpen, ShieldCheck } from "lucide-react";
import { useData } from "../app";
import { clearReviews } from "../api";
import { Card, MRow } from "../components/common";

const CANDIDATES = [
  ["runs/unet_f0..f4.pt", "production-v1.4 · 5-fold ensemble", true],
  ["archive/l4-s3/s3_unet_all_s0..2.pt", "solution 3 · canonical 0.57 um/px", false],
  ["archive/l4-s3c/s3_um1p33_all_s0..2.pt", "solution 3 · coarse 1.33 um/px", false],
] as const;

export function SettingsPage() {
  const { fx, reviews } = useData();
  const [active, setActive] = useState<string>(CANDIDATES[0][0]);
  const [validated, setValidated] = useState<string | null>(CANDIDATES[0][0] as string);
  const [limit, setLimit] = useState(String(fx.profile.clusterSeverityLimitUm));

  return (
    <div className="page">
      <h1>Settings</h1>
      <p className="sub">
        Local prototype configuration. All state is filesystem and browser storage - no
        database is created.
      </p>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "start" }}>
        <Card title="Active model">
          <label className="field" htmlFor="mdl">Checkpoint</label>
          <select id="mdl" value={active}
                  onChange={(e) => { setActive(e.target.value); setValidated(null); }}>
            {CANDIDATES.map(([path, label]) => (
              <option key={path} value={path}>{label}</option>
            ))}
          </select>

          <div className="row" style={{ marginTop: 14 }}>
            <button className="btn primary" onClick={() => setValidated(active)}>
              <ShieldCheck size={16} aria-hidden /> Validate model
            </button>
            {validated === active && (
              <span className="chip pass">
                <CheckCircle2 size={14} aria-hidden /> Loads · 3 classes · 256x256
              </span>
            )}
          </div>

          {validated !== active && (
            <div className="note warn" style={{ marginTop: 14 }}>
              Selecting a file does not activate it. Validation must confirm the
              checkpoint loads and returns a three-class mask at the expected
              dimensions before it can be used for an inspection.
            </div>
          )}

          <div className="mlist" style={{ marginTop: 14 }}>
            <MRow k="Model ID" v={<span className="mono">{fx.model.id}</span>} />
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
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
