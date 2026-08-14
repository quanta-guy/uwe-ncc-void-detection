/**
 * Original and prediction, with synchronised pan/zoom and three comparison modes.
 *
 * Class layers are separate RGBA PNGs (fibre, void, controlling outline) so toggling
 * and fading a class is a CSS opacity change rather than per-pixel work in JavaScript.
 *
 * Hue, brightness and contrast are CSS filters on the original only. The spec is
 * explicit that these are display-only: they must never touch the source file, the
 * model input, the mask or any measurement. Keeping them in CSS makes that
 * structurally true rather than a promise.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { asset } from "../api";
import type { Field, SeverityEvidence } from "../types";
import { SeverityOverlay } from "./SeverityOverlay";

export type ViewMode = "side" | "swipe" | "overlay";

export interface Display {
  hue: number;
  brightness: number;
  contrast: number;
}
export const NEUTRAL: Display = { hue: 0, brightness: 0, contrast: 0 };

export interface Layers {
  fibre: boolean;
  void: boolean;
  controlling: boolean;
  evidence: boolean;
}

interface Props {
  field: Field;
  mode: ViewMode;
  maskOpacity: number;
  display: Display;
  layers: Layers;
  evidence: SeverityEvidence;
}

const SIZE = 256;

/** A scale bar sized to a round number of microns, per the reference screens. */
function ScaleBar({ umPerPixel }: { umPerPixel: number }) {
  const um = 50;
  const pct = ((um / umPerPixel) / SIZE) * 100;
  return (
    <div className="scalebar" style={{ width: `${pct}%`, minWidth: 54 }}>
      {um} µm
      <div className="bar" />
    </div>
  );
}

export function ImageViewer({ field, mode, maskOpacity, display, layers, evidence }: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [swipe, setSwipe] = useState(50);
  const drag = useRef<{ x: number; y: number } | null>(null);

  // Reset when the field changes, or the reviewer inherits the previous view.
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSwipe(50);
  }, [field.stem]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") return;
      if (e.key === "+" || e.key === "=") setZoom((z) => Math.min(6, z * 1.25));
      if (e.key === "-") setZoom((z) => Math.max(1, z / 1.25));
      if (e.key === "0") { setZoom(1); setPan({ x: 0, y: 0 }); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY }; };
  const onUp = () => { drag.current = null; };
  const onMove = useCallback((e: React.MouseEvent) => {
    if (!drag.current || zoom === 1) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
  }, [zoom]);

  const tf = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    transformOrigin: "center center",
  };
  const filter = `hue-rotate(${display.hue}deg) brightness(${1 + display.brightness / 100}) contrast(${1 + display.contrast / 100})`;

  const Original = (
    <img src={asset("originals", field.stem)} alt={`Field ${field.id} original microscopy`}
         style={{ ...tf, filter }} draggable={false} />
  );

  const MaskLayers = (
    <>
      {layers.fibre && (
        <img src={asset("fibre", field.stem)} alt="" aria-hidden
             style={{ ...tf, opacity: maskOpacity }} draggable={false} />
      )}
      {layers.void && (
        <img src={asset("void", field.stem)} alt="" aria-hidden
             style={{ ...tf, opacity: maskOpacity }} draggable={false} />
      )}
      {layers.controlling && (
        <img src={asset("controlling", field.stem)} alt="" aria-hidden
             style={{ ...tf }} draggable={false} />
      )}
      {layers.evidence && (
        <div style={{ ...tf, position: "absolute", inset: 0 }}>
          <SeverityOverlay evidence={evidence} size={SIZE} />
        </div>
      )}
    </>
  );

  const stackProps = {
    className: "stack",
    onMouseDown: onDown,
    onMouseUp: onUp,
    onMouseLeave: onUp,
    onMouseMove: onMove,
    style: { cursor: zoom > 1 ? ("grab" as const) : ("default" as const) },
  };

  if (mode === "side") {
    return (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div className="viewer">
          <div className="cap">Original</div>
          <div {...stackProps}>{Original}</div>
          <ScaleBar umPerPixel={field.umPerPixel} />
        </div>
        <div className="viewer">
          <div className="cap">Prediction</div>
          <div {...stackProps}>{Original}{MaskLayers}</div>
          <ScaleBar umPerPixel={field.umPerPixel} />
        </div>
      </div>
    );
  }

  if (mode === "overlay") {
    return (
      <div className="viewer" style={{ maxWidth: 640, margin: "0 auto" }}>
        <div className="cap">Overlay — original with prediction</div>
        <div {...stackProps}>{Original}{MaskLayers}</div>
        <ScaleBar umPerPixel={field.umPerPixel} />
      </div>
    );
  }

  return (
    <div className="viewer" style={{ maxWidth: 640, margin: "0 auto" }}
         onMouseMove={(e) => {
           if (!drag.current) {
             const r = e.currentTarget.getBoundingClientRect();
             setSwipe(Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)));
           }
           onMove(e);
         }}>
      <div className="cap">Swipe — drag across to compare</div>
      <div {...stackProps}>
        {Original}
        <div style={{ position: "absolute", inset: 0, clipPath: `inset(0 0 0 ${swipe}%)` }}>
          {MaskLayers}
        </div>
      </div>
      <div className="swipe-handle" style={{ left: `${swipe}%` }} />
      <ScaleBar umPerPixel={field.umPerPixel} />
    </div>
  );
}
