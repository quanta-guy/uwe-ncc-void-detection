/**
 * Data access and the review store.
 *
 * Fixtures are read from a static JSON built by `tools/build_fixtures.py`, so the app
 * runs with no backend. Review decisions live in localStorage for the prototype; the
 * FastAPI layer writes them to `reviews.jsonl` when it is running. Either way the
 * frontend never edits prediction artifacts - corrections are new versions, and the
 * original prediction is retained. That rule is not negotiable in the spec.
 */

import type { Field, Fixtures, Inspection, ReviewEvent, ReviewStatus } from "./types";

const DATA = "/data";
const STORE = "cip.reviews.v1";

export const asset = (kind: "originals" | "fibre" | "void" | "controlling" | "thumbs", stem: string) =>
  `${DATA}/${kind}/${stem}.${kind === "originals" || kind === "thumbs" ? "jpg" : "png"}`;

export async function loadFixtures(): Promise<Fixtures> {
  const res = await fetch(`${DATA}/fixtures.json`);
  if (!res.ok) {
    throw new Error(
      `fixtures.json missing (${res.status}). Run: python frontend/tools/build_fixtures.py`,
    );
  }
  const fx = (await res.json()) as Fixtures;

  // Inspections the user imported live are written to a separate file, never merged
  // into fixtures.json - that stays a reproducible build artifact. Missing file just
  // means nothing has been imported yet, which is not an error.
  try {
    const extra = await fetch(`${DATA}/imported.json`, { cache: "no-store" });
    if (extra.ok) {
      const { inspections } = (await extra.json()) as { inspections: Inspection[] };
      const ids = new Set(inspections.map((i) => i.inspectionId));
      fx.inspections = [...inspections, ...fx.inspections.filter((i) => !ids.has(i.inspectionId))];
    }
  } catch {
    /* backend not running: the two fixture samples still open */
  }
  return fx;
}

// ---------- live backend ----------
/**
 * Everything below needs `python frontend/server/app.py`. The app is deliberately
 * usable without it - fixtures render read-only - so each call reports the backend
 * being absent as a plain message rather than throwing the UI into an error state.
 */

export interface ModelGroup {
  id: string;
  label: string;
  count: number;
  paths: string[];
  sizeMb: number;
}

export interface ValidationResult {
  ok: boolean;
  id: string;
  detail: string;
  classes?: number;
  dims?: string;
  members?: number;
  device?: string;
  elapsedMs?: number;
  disagreementAvailable?: boolean;
}

export interface ImportJob {
  state: "running" | "done" | "error";
  done: number;
  total: number;
  current: string;
  inspectionId: string;
  error?: string;
  inspection?: Inspection;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new Error("Inference backend is not running. Start it with: python frontend/server/app.py");
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const listModels = () =>
  api<{ models: ModelGroup[]; active: string; threshold: number; minSize: number }>("/models");

export const validateModel = (id: string) =>
  api<ValidationResult>("/models/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });

export const activateModel = (id: string) =>
  api<{ active: string }>("/models/activate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });

export function startImport(form: FormData) {
  return api<{ jobId: string; total: number; model: string; device: string }>(
    "/inspections",
    { method: "POST", body: form },
  );
}

export const jobStatus = (jobId: string) => api<ImportJob>(`/jobs/${jobId}`);

export const deleteImported = (id: string) =>
  api<{ removed: string }>(`/inspections/${id}`, { method: "DELETE" });

// ---------- review store ----------

export function readReviews(): ReviewEvent[] {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "[]") as ReviewEvent[];
  } catch {
    return [];
  }
}

export function appendReview(e: Omit<ReviewEvent, "at">): ReviewEvent[] {
  const events = [...readReviews(), { ...e, at: new Date().toISOString() }];
  localStorage.setItem(STORE, JSON.stringify(events));
  window.dispatchEvent(new Event("cip:reviews"));
  return events;
}

export function clearReviews() {
  localStorage.removeItem(STORE);
  window.dispatchEvent(new Event("cip:reviews"));
}

/** Latest decision per field. Later events supersede earlier ones. */
export function statusMap(events: ReviewEvent[]): Record<string, ReviewEvent> {
  const out: Record<string, ReviewEvent> = {};
  for (const e of events) out[`${e.inspectionId}/${e.fieldId}`] = e;
  return out;
}

export const fieldStatus = (
  map: Record<string, ReviewEvent>,
  inspectionId: string,
  fieldId: string,
): ReviewStatus => map[`${inspectionId}/${fieldId}`]?.decision ?? "pending";

// ---------- derived views ----------

/** Risk band. Drives colour, but is always shown with text alongside it. */
export type Band = "critical" | "review" | "routine";

export const bandOf = (f: Field): Band =>
  f.triageAction === "REJECT" ? "critical" : f.triageAction === "REVIEW" ? "review" : "routine";

export const BAND_LABEL: Record<Band, string> = {
  critical: "Critical",
  review: "Review",
  routine: "Routine",
};

/**
 * Queue order: worst first. Critical before review before routine, then by severity.
 * This is the product claim - reviewers see the most consequential field first
 * instead of scrolling a folder.
 */
export function priorityOrder(fields: Field[]): Field[] {
  const rank: Record<Band, number> = { critical: 0, review: 1, routine: 2 };
  return [...fields].sort(
    (a, b) => rank[bandOf(a)] - rank[bandOf(b)] || b.clusterSeverityUm - a.clusterSeverityUm,
  );
}

export const reviewedCount = (i: Inspection, map: Record<string, ReviewEvent>) =>
  i.fields.filter((f) => fieldStatus(map, i.inspectionId, f.id) !== "pending").length;

/**
 * Seeded workspace rows. The two real inspections are not enough to show what an
 * operational workspace looks like, and the spec asks for that view. These carry no
 * field data and are marked `seeded` so no screen can mistake them for measurements.
 */
export function seededRows(): Inspection[] {
  const spec: Array<[string, string, string, number, number, number, string]> = [
    ["INS-2026-040", "CPEEK-040", "C/PEEK", 48, 48, 12, "PASS"],
    ["INS-2026-039", "CPEEK-039", "C/PEEK", 40, 40, 8, "PASS"],
    ["INS-2026-038", "LMPEAK-017", "C/LM-PEAK", 60, 60, 0, "PASS"],
    ["INS-2026-037", "LMPEAK-016", "C/LM-PEAK", 36, 36, 4, "PASS"],
    ["INS-2026-036", "CPEEK-036", "C/PEEK", 24, 24, 0, "PASS"],
    ["INS-2026-035", "CPEEK-035", "C/PEEK", 28, 20, 0, "PROCESSING"],
    ["INS-2026-034", "LMPEAK-015", "C/LM-PEAK", 52, 52, 6, "PASS"],
    ["INS-2026-033", "CPEEK-033", "C/PEEK", 44, 44, 10, "PASS"],
    ["INS-2026-032", "LMPEAK-014", "C/LM-PEAK", 64, 64, 0, "PASS"],
    ["INS-2026-031", "CPEEK-031", "C/PEEK", 32, 32, 32, "FAIL"],
  ];
  return spec.map(([inspectionId, sampleId, material, total, done, reviewed, disp]) => ({
    inspectionId, sampleId, material, batch: "B-24xx", micrograph: "-",
    sampleType: "polished cross-section", umPerPixel: 0.57,
    userEntered: ["material", "sampleId", "batch"],
    measurementProfile: "challenge-severity-v1",
    model: { id: "production-v1.4", sha256: "-", threshold: 0.5, minSize: 4 },
    state: done < total ? "processing" : reviewed >= total ? "complete" : "ready_for_review",
    reviewedCount: reviewed, fieldCount: total,
    preliminary: {
      disposition: disp === "FAIL" ? "FAIL" : "PASS",
      reason: disp === "PROCESSING" ? "Batch inference in progress" : "Seeded workspace row",
      modelDerived: true,
    },
    kpis: {
      voidArealFractionPct: 0, fibreArealFractionPct: 0, matrixArealFractionPct: 0,
      worstClusterSeverityUm: 0, fieldsOverLimit: 0, fieldsTotal: total,
      largestFeretUm: 0, voidCount: 0, voidDensityPerMm2: 0,
      feretP50Um: 0, feretP95Um: 0, analysedAreaMm2: 0,
    },
    controllingFieldId: "-", fields: [], seeded: true,
    processed: done,
  } as Inspection & { processed: number }));
}

// ---------- local removal ----------
/**
 * Removing an inspection from the workspace.
 *
 * Fixtures are a read-only build artifact, so "delete" hides the inspection locally
 * rather than destroying measurements - which is the honest behaviour for a prototype
 * and is reversible from Settings. A real deletion would archive the inspection folder
 * through the API, never unlink it.
 */
const HIDDEN = "cip.hidden.v1";

export function readHidden(): string[] {
  try {
    return JSON.parse(localStorage.getItem(HIDDEN) ?? "[]") as string[];
  } catch {
    return [];
  }
}

export function hideInspection(id: string) {
  const next = [...new Set([...readHidden(), id])];
  localStorage.setItem(HIDDEN, JSON.stringify(next));
  window.dispatchEvent(new Event("cip:reviews"));
}

export function restoreInspections() {
  localStorage.removeItem(HIDDEN);
  window.dispatchEvent(new Event("cip:reviews"));
}
