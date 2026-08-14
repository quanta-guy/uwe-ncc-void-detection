/**
 * Mask editor. Brush, eraser, class selector, brush size, undo, reset, save.
 *
 * Plain canvas rather than Konva: the spec suggested Konva, but a 256x256 three-class
 * mask is a Uint8Array and a paint loop. Adding a scene-graph library to draw circles
 * into an array would be more dependency than the job needs.
 *
 * The class mask is reconstructed from the two RGBA render layers already shipped
 * (fibre and void), so no extra asset is needed: any pixel opaque in the void layer is
 * class 2, opaque in the fibre layer is class 1, everything else is matrix.
 *
 * On save it writes a PNG whose red channel carries the class values 0/1/2 - the same
 * format the upload validator accepts - and never touches the prediction. A correction
 * is a new version, always.
 *
 * Void pixel count and areal fraction are recomputed here because they are exact pixel
 * counts that cannot drift. Severity, Feret and clustering are NOT: those are
 * evaluation.py's geometry and recomputing them in TypeScript is exactly how a UI ends
 * up disagreeing with the score. They are marked pending instead.
 */

import { useEffect, useRef, useState } from "react";
import { Brush, Eraser, RotateCcw, Save, Undo2 } from "lucide-react";
import { asset } from "../api";
import type { Field } from "../types";

const N = 256;
const VIEW = 460;
const CLASS_RGB: Record<number, [number, number, number]> = {
  0: [95, 111, 125],   // matrix - neutral
  1: [0, 133, 139],    // fibre  - teal
  2: [213, 31, 38],    // void   - red
};
const CLASS_NAME = ["Matrix", "Fibre", "Void"];

/** Load an image and return its pixel data at mask resolution. */
async function pixels(src: string): Promise<Uint8ClampedArray> {
  const img = await new Promise<HTMLImageElement>((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = () => rej(new Error(`could not load ${src}`));
    i.src = src;
  });
  const c = document.createElement("canvas");
  c.width = N;
  c.height = N;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, N, N);
  return ctx.getImageData(0, 0, N, N).data;
}

interface Props {
  field: Field;
  onCancel: () => void;
  onSave: (result: { voidPx: number; voidPct: number; changedPx: number; png: string }) => void;
}

export function MaskEditor({ field, onCancel, onSave }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<Uint8Array | null>(null);
  const originalRef = useRef<Uint8Array | null>(null);
  const imgDataRef = useRef<Uint8ClampedArray | null>(null);
  const undoRef = useRef<Uint8Array[]>([]);
  const painting = useRef(false);

  const [tool, setTool] = useState<"brush" | "eraser">("brush");
  const [cls, setCls] = useState(2);
  const [size, setSize] = useState(6);
  const [opacity, setOpacity] = useState(0.55);
  const [ready, setReady] = useState(false);
  const [stats, setStats] = useState({ voidPx: 0, changed: 0 });

  // ---- load and reconstruct the class mask from the shipped render layers ----
  useEffect(() => {
    let live = true;
    (async () => {
      const [orig, fibre, voidL] = await Promise.all([
        pixels(asset("originals", field.stem)),
        pixels(asset("fibre", field.stem)),
        pixels(asset("void", field.stem)),
      ]);
      if (!live) return;
      const m = new Uint8Array(N * N);
      for (let i = 0; i < N * N; i++) {
        m[i] = voidL[i * 4 + 3] > 127 ? 2 : fibre[i * 4 + 3] > 127 ? 1 : 0;
      }
      imgDataRef.current = orig;
      maskRef.current = m;
      originalRef.current = Uint8Array.from(m);
      undoRef.current = [];
      setStats({ voidPx: m.reduce((n, v) => n + (v === 2 ? 1 : 0), 0), changed: 0 });
      setReady(true);
    })().catch(() => setReady(false));
    return () => { live = false; };
  }, [field.stem]);

  // ---- render ----
  const draw = () => {
    const c = canvasRef.current;
    const img = imgDataRef.current;
    const m = maskRef.current;
    if (!c || !img || !m) return;
    const ctx = c.getContext("2d")!;
    const out = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const [r, g, b] = CLASS_RGB[m[i]];
      const a = m[i] === 0 ? 0 : opacity;   // matrix stays as bare microscopy
      out.data[i * 4] = img[i * 4] * (1 - a) + r * a;
      out.data[i * 4 + 1] = img[i * 4 + 1] * (1 - a) + g * a;
      out.data[i * 4 + 2] = img[i * 4 + 2] * (1 - a) + b * a;
      out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  };
  useEffect(draw);

  // ---- painting ----
  const paintAt = (e: React.MouseEvent) => {
    const c = canvasRef.current;
    const m = maskRef.current;
    if (!c || !m) return;
    const r = c.getBoundingClientRect();
    const cx = Math.round(((e.clientX - r.left) / r.width) * N);
    const cy = Math.round(((e.clientY - r.top) / r.height) * N);
    const value = tool === "eraser" ? 0 : cls;
    const rad = size / 2;

    for (let y = Math.max(0, cy - size); y <= Math.min(N - 1, cy + size); y++) {
      for (let x = Math.max(0, cx - size); x <= Math.min(N - 1, cx + size); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= rad * rad) m[y * N + x] = value;
      }
    }
    draw();
  };

  const pushUndo = () => {
    const m = maskRef.current;
    if (!m) return;
    undoRef.current.push(Uint8Array.from(m));
    if (undoRef.current.length > 40) undoRef.current.shift();
  };

  const recount = () => {
    const m = maskRef.current;
    const o = originalRef.current;
    if (!m || !o) return;
    let voidPx = 0;
    let changed = 0;
    for (let i = 0; i < m.length; i++) {
      if (m[i] === 2) voidPx++;
      if (m[i] !== o[i]) changed++;
    }
    setStats({ voidPx, changed });
  };

  const undo = () => {
    const prev = undoRef.current.pop();
    if (prev && maskRef.current) {
      maskRef.current.set(prev);
      draw();
      recount();
    }
  };

  const reset = () => {
    if (maskRef.current && originalRef.current) {
      pushUndo();
      maskRef.current.set(originalRef.current);
      draw();
      recount();
    }
  };

  /** PNG whose red channel is the class value - the format the validator accepts. */
  const toPng = () => {
    const m = maskRef.current!;
    const c = document.createElement("canvas");
    c.width = N;
    c.height = N;
    const ctx = c.getContext("2d")!;
    const out = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      out.data[i * 4] = m[i];
      out.data[i * 4 + 1] = m[i];
      out.data[i * 4 + 2] = m[i];
      out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    return c.toDataURL("image/png");
  };

  const voidPct = (100 * stats.voidPx) / (N * N);
  const delta = voidPct - field.voidArealFractionPct;

  return (
    <div className="backdrop" role="dialog" aria-modal aria-label="Mask editor">
      <div className="modal wide" style={{ maxWidth: 1000 }}>
        <div className="spread" style={{ marginBottom: 4 }}>
          <h2 style={{ margin: 0 }}>Edit mask — field {field.id}</h2>
          <button className="btn sm" onClick={onCancel}>Close</button>
        </div>
        <p className="sub" style={{ marginBottom: 16 }}>
          Starts from a copy of the prediction. The original prediction is never
          overwritten; saving writes a new correction version.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: `${VIEW}px 1fr`, gap: 22 }}>
          <div>
            <canvas ref={canvasRef} width={N} height={N}
                    style={{ width: VIEW, height: VIEW, borderRadius: 6,
                             imageRendering: "pixelated", cursor: "crosshair",
                             background: "#0d1418", touchAction: "none" }}
                    onMouseDown={(e) => { pushUndo(); painting.current = true; paintAt(e); }}
                    onMouseMove={(e) => painting.current && paintAt(e)}
                    onMouseUp={() => { painting.current = false; recount(); }}
                    onMouseLeave={() => { if (painting.current) { painting.current = false; recount(); } }} />
            {!ready && <div className="provisional" style={{ marginTop: 8 }}>Loading mask…</div>}
          </div>

          <div>
            <h3>Tool</h3>
            <div className="tabs" style={{ marginBottom: 14 }}>
              <button className={`tab ${tool === "brush" ? "on" : ""}`} onClick={() => setTool("brush")}>
                <Brush size={15} aria-hidden /> Brush
              </button>
              <button className={`tab ${tool === "eraser" ? "on" : ""}`} onClick={() => setTool("eraser")}>
                <Eraser size={15} aria-hidden /> Eraser
              </button>
            </div>

            <h3>Paint class</h3>
            <div className="tabs" style={{ marginBottom: 14 }}>
              {[0, 1, 2].map((v) => (
                <button key={v} className={`tab ${cls === v && tool === "brush" ? "on" : ""}`}
                        onClick={() => { setCls(v); setTool("brush"); }}>
                  <span className="dot" aria-hidden
                        style={{ background: `rgb(${CLASS_RGB[v].join(",")})` }} />
                  {CLASS_NAME[v]}
                </button>
              ))}
            </div>

            <label className="field" htmlFor="bs">Brush size — {size} px</label>
            <input id="bs" type="range" min={1} max={30} value={size}
                   onChange={(e) => setSize(Number(e.target.value))} />

            <label className="field" htmlFor="op" style={{ marginTop: 12 }}>
              Overlay opacity — {Math.round(opacity * 100)}%
            </label>
            <input id="op" type="range" min={10} max={100} value={opacity * 100}
                   onChange={(e) => setOpacity(Number(e.target.value) / 100)} />

            <div className="row" style={{ marginTop: 16 }}>
              <button className="btn" onClick={undo} disabled={!undoRef.current.length}>
                <Undo2 size={15} aria-hidden /> Undo
              </button>
              <button className="btn" onClick={reset}>
                <RotateCcw size={15} aria-hidden /> Reset to prediction
              </button>
            </div>

            <div className="mlist" style={{ marginTop: 18 }}>
              <div className="mrow">
                <span className="k">Void pixels</span>
                <span className="v">{stats.voidPx} <span className="provisional">
                  (was {field.voidPx})</span></span>
              </div>
              <div className="mrow">
                <span className="k">Void areal fraction</span>
                <span className={`v ${Math.abs(delta) > 0.001 ? "bad" : ""}`}>
                  {voidPct.toFixed(3)}%{" "}
                  {Math.abs(delta) > 0.0005 && (
                    <span className="provisional">
                      ({delta > 0 ? "+" : ""}{delta.toFixed(3)})
                    </span>
                  )}
                </span>
              </div>
              <div className="mrow">
                <span className="k">Pixels changed</span>
                <span className="v">{stats.changed}</span>
              </div>
            </div>

            <div className="note" style={{ marginTop: 14 }}>
              Void pixel count and areal fraction update live — they are exact pixel
              counts. <strong>Cluster severity, Feret diameters and 40 µm clustering
              are not recalculated here</strong>: that geometry belongs to
              evaluation.py, and deriving it in the browser is how a UI ends up
              disagreeing with the score. It recalculates when the correction is
              committed to the measurement service.
            </div>

            <div className="row" style={{ justifyContent: "flex-end", marginTop: 18 }}>
              <button className="btn" onClick={onCancel}>Cancel</button>
              <button className="btn primary" disabled={!ready || stats.changed === 0}
                      onClick={() => onSave({
                        voidPx: stats.voidPx, voidPct, changedPx: stats.changed, png: toPng(),
                      })}>
                <Save size={16} aria-hidden /> Save correction
              </button>
            </div>
            {stats.changed === 0 && ready && (
              <div className="provisional" style={{ textAlign: "right", marginTop: 6 }}>
                Paint a change to enable saving.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
