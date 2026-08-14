/**
 * Small shared pieces. Per the spec, a component lives here only once at least two
 * screens use it - there is no design-system package for a prototype.
 */

import { AlertCircle, CheckCircle2, CircleDot, HelpCircle } from "lucide-react";
import type { Band } from "../api";
import { BAND_LABEL } from "../api";
import type { ReviewStatus, Verdict } from "../types";

/** Status is never colour alone: icon plus text, always. */
export function DispositionChip({ v, size = "md" }: { v: Verdict | "REVIEW"; size?: "sm" | "md" }) {
  const map = {
    PASS: { cls: "pass", Icon: CheckCircle2, text: "Pass" },
    FAIL: { cls: "fail", Icon: AlertCircle, text: "FAIL" },
    REVIEW: { cls: "review", Icon: HelpCircle, text: "Review" },
  }[v];
  return (
    <span className={`chip ${map.cls}`} style={size === "sm" ? { fontSize: 11.5 } : undefined}>
      <map.Icon size={14} aria-hidden /> {map.text}
    </span>
  );
}

export function BandChip({ band }: { band: Band }) {
  return (
    <span className={`chip ${band}`}>
      <span className={`dot ${band}`} aria-hidden /> {BAND_LABEL[band]}
    </span>
  );
}

export function StatusChip({ s }: { s: ReviewStatus }) {
  const map: Record<ReviewStatus, { cls: string; text: string }> = {
    pending: { cls: "muted", text: "Pending" },
    accepted: { cls: "pass", text: "Accepted" },
    corrected: { cls: "info", text: "Corrected" },
    unsuitable: { cls: "review", text: "Unsuitable" },
  };
  return <span className={`chip ${map[s].cls}`}>{map[s].text}</span>;
}

/**
 * The spec requires every model-derived KPI to be labelled until the fields behind it
 * have been reviewed. Showing a preliminary number as if it were an inspection result
 * is the single most misleading thing this UI could do.
 */
export function ProvisionalTag({ reviewed, total }: { reviewed: number; total: number }) {
  if (reviewed >= total && total > 0) {
    return (
      <span className="chip pass">
        <CheckCircle2 size={14} aria-hidden /> Recalculated from accepted masks
      </span>
    );
  }
  return (
    <span className="provisional">
      <CircleDot size={12} style={{ verticalAlign: -2, marginRight: 4 }} aria-hidden />
      Model-derived · human review pending ({reviewed}/{total} reviewed)
    </span>
  );
}

export function Kpi({
  label, value, unit, tone, small,
}: {
  label: string; value: string | number; unit?: string;
  tone?: "bad" | "normal"; small?: boolean;
}) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className={`value ${small ? "sm" : ""} ${tone === "bad" ? "bad" : ""}`}>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
  );
}

export function Card({
  title, right, children, style,
}: {
  title?: string; right?: React.ReactNode;
  children: React.ReactNode; style?: React.CSSProperties;
}) {
  return (
    <div className="card" style={style}>
      {title && (
        <div className="card-title">
          <span>{title}</span>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function MRow({
  k, v, tone, title,
}: { k: string; v: React.ReactNode; tone?: "bad" | "good"; title?: string }) {
  return (
    <div className="mrow" title={title}>
      <span className="k">{k}</span>
      <span className={`v ${tone ?? ""}`}>{v}</span>
    </div>
  );
}
