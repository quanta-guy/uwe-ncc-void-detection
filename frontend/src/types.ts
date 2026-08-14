/**
 * Shapes written by `frontend/tools/build_fixtures.py`.
 *
 * Every physical number here was produced by evaluation.py through Python. Nothing in
 * TypeScript derives a measurement - it only renders one. Keeping that boundary is
 * what stops the UI and the score drifting apart.
 */

export type TriageAction = "ACCEPT" | "REVIEW" | "REJECT";
export type ReviewStatus = "pending" | "accepted" | "corrected" | "unsuitable";
export type Verdict = "PASS" | "FAIL";

/** One void inside the cluster that set the severity, with its L arrow and A label. */
export interface EvidenceVoid {
  index: number;
  label: string;      // L1, L2, ...
  areaLabel: string;  // A1, A2, ...
  lengthUm: number;
  areaUm2: number;
  areaPx: number;
  l0: [number, number]; // (row, col) - one end of the max Feret chord
  l1: [number, number];
  centroid: [number, number];
}

/** A gap that fell under the merge distance, and therefore joined two voids. */
export interface EvidenceGap {
  label: string;  // D12, D13, ...
  gapUm: number;
  d0: [number, number];
  d1: [number, number];
}

export interface SeverityEvidence {
  severity_um: number;
  limit_um: number;
  verdict: Verdict;
  length_term_um: number;   // sum of L
  area_term_um: number;     // 0.5 * sqrt(sum of A)
  merge_distance_um: number;
  voids: EvidenceVoid[];
  gaps: EvidenceGap[];
}

export interface Field {
  id: string;      // "01".."16"
  stem: string;
  umPerPixel: number;
  voidArealFractionPct: number;
  fibreArealFractionPct: number;
  matrixArealFractionPct: number;
  voidCount: number;
  voidAreaUm2: number;
  sampledAreaMm2: number;
  largestFeretUm: number;
  feretsUm: number[];
  clusterSeverityUm: number;
  distanceToLimitUm: number;
  verdict: Verdict;
  modelDisagreement: number;
  triageAction: TriageAction;
  triageReason: string;
  severityEvidence: SeverityEvidence;
  voidPx: number;
}

export interface SampleKpis {
  voidArealFractionPct: number;
  fibreArealFractionPct: number;
  matrixArealFractionPct: number;
  worstClusterSeverityUm: number;
  fieldsOverLimit: number;
  fieldsTotal: number;
  largestFeretUm: number;
  voidCount: number;
  voidDensityPerMm2: number;
  feretP50Um: number;
  feretP95Um: number;
  analysedAreaMm2: number;
}

export interface Inspection {
  inspectionId: string;
  sampleId: string;
  material: string;
  batch: string;
  micrograph: string;
  sampleType: string;
  umPerPixel: number;
  /** Fields with no source in metadata.csv - operator input, never model output. */
  userEntered: string[];
  measurementProfile: string;
  model: { id: string; sha256: string; threshold: number; minSize: number };
  state: string;
  reviewedCount: number;
  fieldCount: number;
  preliminary: { disposition: Verdict; reason: string; modelDerived: boolean };
  kpis: SampleKpis;
  controllingFieldId: string;
  fields: Field[];
  /** Present only on seeded workspace rows, which carry no field data. */
  seeded?: boolean;
}

export interface Profile {
  id: string;
  name: string;
  clusterSeverityLimitUm: number;
  mergeDistanceUm: number;
  areaFactor: number;
  approvedForProductionUse: boolean;
  note: string;
}

export interface Fixtures {
  generatedAt: string;
  profile: Profile;
  model: {
    id: string; sha256: string; threshold: number; minSize: number; checkpoints: string[];
  };
  inspections: Inspection[];
}

/** A reviewer decision. Kept separately from predictions, which are immutable. */
export interface ReviewEvent {
  at: string;
  inspectionId: string;
  fieldId: string;
  decision: Exclude<ReviewStatus, "pending">;
  errorReason?: string;
  note?: string;
  reviewer: string;
}
