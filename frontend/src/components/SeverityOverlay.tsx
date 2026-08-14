/**
 * The NCC deck's severity annotation, drawn on a real prediction.
 *
 * L arrows span each void's longest internal axis; A labels sit on the void body; D
 * arrows join voids whose gap fell under 40 um and therefore merged them into one
 * defect. Every coordinate comes from `severity_geometry.py`, which locates the points
 * behind numbers evaluation.py already scored - so the picture cannot disagree with
 * the number it explains.
 *
 * Rendered as SVG in the mask's own pixel coordinates with a viewBox, so it scales
 * with the image at any zoom without recomputing anything.
 */

import type { SeverityEvidence } from "../types";

const ARROW_L = "#ffffff";
const ARROW_D = "#ffc700";

interface Props {
  evidence: SeverityEvidence;
  size: number;      // mask is square; this is its pixel width
  showLengths?: boolean;
  showGaps?: boolean;
}

export function SeverityOverlay({ evidence, size, showLengths = true, showGaps = true }: Props) {
  if (!evidence.voids.length) return null;

  // Keep strokes and text visually constant regardless of the image's pixel size.
  const s = size / 256;
  const stroke = 1.4 * s;
  const font = 11 * s;

  // A void near an edge would otherwise have its label drawn outside the image and
  // silently clipped - the reader loses the measurement rather than the decoration.
  const pad = font * 1.1;
  const clampY = (y: number) => Math.max(pad, Math.min(size - pad * 0.4, y));
  const clampX = (x: number) => Math.max(pad * 2.6, Math.min(size - pad * 2.6, x));

  return (
    <svg viewBox={`0 0 ${size} ${size}`} role="img"
         aria-label="Severity measurement annotation">
      <defs>
        {[["al", ARROW_L], ["ad", ARROW_D]].map(([id, colour]) => (
          <marker key={id} id={id} viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill={colour} />
          </marker>
        ))}
        <filter id="lbl" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="0" stdDeviation={1.6 * s} floodColor="#000" floodOpacity="0.95" />
        </filter>
      </defs>

      {/* D arrows first, so the L arrows and labels sit above them. */}
      {showGaps && evidence.gaps.map((g) => {
        const [y0, x0] = g.d0;
        const [y1, x1] = g.d1;
        return (
          <g key={g.label}>
            <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={ARROW_D} strokeWidth={stroke}
                  markerStart="url(#ad)" markerEnd="url(#ad)" />
            <text x={clampX((x0 + x1) / 2 + 5 * s)} y={clampY((y0 + y1) / 2 - 3 * s)} fill={ARROW_D}
                  fontSize={font} fontWeight="700" filter="url(#lbl)">
              {g.label} {g.gapUm.toFixed(1)} µm
            </text>
          </g>
        );
      })}

      {showLengths && evidence.voids.map((v) => {
        const [y0, x0] = v.l0;
        const [y1, x1] = v.l1;
        const [cy, cx] = v.centroid;
        return (
          <g key={v.label}>
            <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={ARROW_L} strokeWidth={stroke}
                  markerStart="url(#al)" markerEnd="url(#al)" />
            <text x={clampX((x0 + x1) / 2)} y={clampY((y0 + y1) / 2 - 5 * s)} fill={ARROW_L}
                  fontSize={font} fontWeight="700" textAnchor="middle" filter="url(#lbl)">
              {v.label} {v.lengthUm.toFixed(1)} µm
            </text>
            <text x={clampX(cx)} y={clampY(cy + 4 * s)} fill={ARROW_L} fontSize={font} fontWeight="700"
                  textAnchor="middle" filter="url(#lbl)">
              {v.areaLabel}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/**
 * The arithmetic in words, so a reviewer can see the severity being formed rather
 * than having to trust it. Mirrors evaluation.py: lengths summed, then one square
 * root of the summed area, scaled by 0.5.
 */
export function SeverityFormula({ evidence }: { evidence: SeverityEvidence }) {
  const e = evidence;
  if (!e.voids.length) return null;
  const fail = e.verdict === "FAIL";
  return (
    <div className={`note ${fail ? "bad" : ""}`} style={{ marginTop: 12 }}>
      <div className="mono" style={{ color: "var(--text)", marginBottom: 6 }}>
        Σ L = {e.length_term_um.toFixed(2)} µm &nbsp;+&nbsp; 0.5·√(Σ A) ={" "}
        {e.area_term_um.toFixed(2)} µm &nbsp;=&nbsp;{" "}
        <strong style={{ color: fail ? "var(--danger)" : "var(--success)" }}>
          {e.severity_um.toFixed(2)} µm
        </strong>
      </div>
      <div>
        {e.voids.length} void{e.voids.length > 1 ? "s" : ""} in the controlling defect
        {e.gaps.length > 0
          ? `, merged because ${e.gaps.length === 1 ? "a gap of" : "gaps of"} ${e.gaps
              .map((g) => `${g.gapUm.toFixed(1)}`)
              .join(", ")} µm ${e.gaps.length === 1 ? "is" : "are"} under the ${e.merge_distance_um} µm limit`
          : ""}
        . Profile limit {e.limit_um} µm — <strong>{e.verdict}</strong>.
      </div>
    </div>
  );
}
