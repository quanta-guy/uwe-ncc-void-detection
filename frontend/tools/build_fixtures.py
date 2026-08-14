"""Turn real model output into everything the frontend renders.

    python frontend/tools/build_fixtures.py

Runs the 5-fold ensemble over the 32-image Test set and writes measurements, triage
and render layers into `frontend/public/data/`. Every number the UI shows comes from
here, so nothing on screen is invented.

Why a build step rather than a live API: the severity geometry is CPU-bound scipy
(convex hulls, KD-trees) and the ensemble is five forward passes per image. Doing it
once produces a dataset the UI can open instantly, which is what a demo needs. The
FastAPI layer still recomputes on demand when a reviewer uploads a corrected mask -
that path has to be live, because the whole point is showing the KPIs move.

Render layers, not a single mask image: fibre and void ship as separate RGBA overlays
so the frontend can toggle and fade each class independently without per-pixel work in
JavaScript. Matrix is transparent - it is the background.

Sample and material identity (`sampleId`, `material`) exist nowhere in metadata.csv.
They are assigned here and flagged `userEntered` in the JSON, because the spec is
explicit that material is an input, never a model conclusion.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from data import VOID_CLASS, index_test, load_image  # noqa: E402
from evaluation import MERGE_DISTANCE_UM, SEVERITY_THRESHOLD, _VoidRegions  # noqa: E402
from predict import load_nets, to_mask, void_prob  # noqa: E402
from report import confidence, triage  # noqa: E402
from severity_geometry import controlling_geometry, feret_endpoints  # noqa: E402

OUT = REPO / "frontend" / "public" / "data"
THRESHOLD, MIN_SIZE = 0.5, 4          # verified OOF optimum; see plan
THUMB = 132

#: Two Test micrographs -> two inspections. Material is operator-entered, per spec.
SAMPLES = {
    "2-6-1_mid": {
        "inspectionId": "INS-2026-041", "sampleId": "CPEEK-041",
        "material": "C/PEEK", "batch": "B-2408",
    },
    "17-5-2_zoom_mid": {
        "inspectionId": "INS-2026-042", "sampleId": "LMPEAK-018",
        "material": "C/LM-PEAK", "batch": "B-2411",
    },
}
PALETTE = {"fibre": (0, 133, 139), "void": (213, 31, 38)}


def micrograph_of(stem):
    """`2-6-1_mid_768_1536` -> `2-6-1_mid`. Test stems are {micrograph}_{cut}_{offset}."""
    return stem.rsplit("_", 2)[0]


def layer(mask, cls, rgb):
    """One class as an RGBA overlay, transparent everywhere else."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    sel = mask == cls
    out[sel, :3] = rgb
    out[sel, 3] = 255
    return Image.fromarray(out, "RGBA")


def outline(mask, indices, regions):
    """Yellow outline of the controlling cluster, as its own RGBA layer."""
    from scipy.ndimage import binary_dilation

    h, w = mask.shape
    sel = np.zeros((h, w), bool)
    for i in indices:
        sel[tuple(regions.coords[i].T)] = True
    edge = binary_dilation(sel, np.ones((3, 3), bool)) & ~sel
    out = np.zeros((h, w, 4), np.uint8)
    out[edge] = (255, 199, 0, 255)
    return Image.fromarray(out, "RGBA")


def measure(mask, um, spread, img_shape):
    """Every per-field number the UI shows, from evaluation.py's own primitives."""
    px2 = um * um
    total = mask.size
    void_px = int((mask == VOID_CLASS).sum())
    fibre_px = int((mask == 1).sum())

    regions = _VoidRegions(mask)
    ferets = sorted((regions.diameter(i) * um for i in range(regions.n)), reverse=True)
    geom = controlling_geometry(mask, um)
    conf = confidence(spread, mask == VOID_CLASS)
    action, reason = triage(geom["severity_um"], conf, void_px > 0)

    return {
        "voidArealFractionPct": round(100 * void_px / total, 3),
        "fibreArealFractionPct": round(100 * fibre_px / total, 3),
        "matrixArealFractionPct": round(100 * (total - void_px - fibre_px) / total, 3),
        "voidCount": int(regions.n),
        "voidAreaUm2": round(void_px * px2, 2),
        "sampledAreaMm2": round(total * px2 / 1e6, 4),
        "largestFeretUm": round(ferets[0], 2) if ferets else 0.0,
        "feretsUm": [round(f, 2) for f in ferets],
        "clusterSeverityUm": geom["severity_um"],
        "distanceToLimitUm": round(geom["severity_um"] - SEVERITY_THRESHOLD, 2),
        "verdict": geom["verdict"],
        "modelDisagreement": round(conf, 4),
        "triageAction": action,
        "triageReason": reason,
        "severityEvidence": geom,
        "voidPx": void_px,
    }


def percentile(values, q):
    return round(float(np.percentile(values, q)), 2) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--min-size", type=int, default=MIN_SIZE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(Path(args.runs).glob("unet_f*.pt"))
    if not ckpts:
        sys.exit(f"no unet_f*.pt in {args.runs}")
    # report.confidence needs >=2 models or disagreement is silently always zero,
    # and the REVIEW-on-disagreement rule would never fire.
    if len(ckpts) < 2:
        print("WARNING: fewer than 2 checkpoints - model disagreement will be 0")
    nets = load_nets([str(c) for c in ckpts], device)
    model_id = f"production-v1.4 ({len(ckpts)}-fold ensemble)"
    model_sha = hashlib.sha256(b"".join(c.read_bytes()[:65536] for c in ckpts)).hexdigest()

    out = Path(args.out)
    for sub in ("originals", "fibre", "void", "controlling", "thumbs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    df = index_test()
    if args.limit:
        df = df.head(args.limit)

    # Lexicographic stem order gives the field numbering the spec assumes:
    # 2-6-1_mid_512_0 -> field 03 and 2-6-1_mid_768_1536 -> field 12.
    df = df.sort_values("stem").reset_index(drop=True)

    per_sample = {}
    print(f"\n{'field':>5}  {'stem':34} {'sev':>7} {'voids':>6} {'action':>7}  reason")
    for _, row in df.iterrows():
        stem, um = row["stem"], row["um_per_pixel"]
        mg = micrograph_of(stem)
        if mg not in SAMPLES:
            continue
        img = load_image(row["image"])

        per_model = [void_prob([n], img, device)[0] for n in nets]
        p_mean = np.mean(per_model, axis=0)
        spread = np.std(per_model, axis=0) if len(nets) > 1 else np.zeros_like(p_mean)
        _, base = void_prob(nets, img, device)
        mask = to_mask(p_mean, base, args.threshold, args.min_size)

        m = measure(mask, um, spread, img.shape)
        bucket = per_sample.setdefault(mg, [])
        field_no = f"{len(bucket) + 1:02d}"
        bucket.append({"id": field_no, "stem": stem, "umPerPixel": um, **m})

        shutil.copy(row["image"], out / "originals" / f"{stem}.jpg")
        layer(mask, 1, PALETTE["fibre"]).save(out / "fibre" / f"{stem}.png")
        layer(mask, VOID_CLASS, PALETTE["void"]).save(out / "void" / f"{stem}.png")
        regions = _VoidRegions(mask)
        outline(mask, [v["index"] for v in m["severityEvidence"]["voids"]], regions).save(
            out / "controlling" / f"{stem}.png")

        # Thumbnail is the original with the void tinted, so the risk map reads as
        # microscopy rather than as an abstract colour block.
        thumb = img.astype(np.float32).copy()
        sel = mask == VOID_CLASS
        thumb[sel] = 0.35 * thumb[sel] + 0.65 * np.array(PALETTE["void"], np.float32)
        Image.fromarray(thumb.astype(np.uint8)).resize((THUMB, THUMB), Image.LANCZOS).save(
            out / "thumbs" / f"{stem}.jpg", quality=88)

        print(f"{field_no:>5}  {stem:34} {m['clusterSeverityUm']:7.2f} "
              f"{m['voidCount']:6d} {m['triageAction']:>7}  {m['triageReason']}")

    inspections = []
    for mg, fields in per_sample.items():
        meta = SAMPLES[mg]
        um = fields[0]["umPerPixel"]
        all_ferets = [f for fl in fields for f in fl["feretsUm"]]
        area_mm2 = sum(f["sampledAreaMm2"] for f in fields)
        void_count = sum(f["voidCount"] for f in fields)
        over = [f for f in fields if f["clusterSeverityUm"] >= SEVERITY_THRESHOLD]
        controlling = max(fields, key=lambda f: f["clusterSeverityUm"])

        def mean_of(key):
            return round(float(np.mean([f[key] for f in fields])), 3)

        inspections.append({
            **meta, "micrograph": mg, "sampleType": "polished cross-section",
            "umPerPixel": um, "userEntered": ["material", "sampleId", "batch"],
            "measurementProfile": "challenge-severity-v1",
            "model": {"id": model_id, "sha256": model_sha[:16],
                      "threshold": args.threshold, "minSize": args.min_size},
            "state": "ready_for_review", "reviewedCount": 0,
            "fieldCount": len(fields),
            "preliminary": {
                "disposition": "FAIL" if over else "PASS",
                "reason": (f"Worst-cluster severity exceeds the {SEVERITY_THRESHOLD} µm "
                           f"profile limit" if over else
                           f"All fields below the {SEVERITY_THRESHOLD} µm profile limit"),
                "modelDerived": True,
            },
            "kpis": {
                "voidArealFractionPct": mean_of("voidArealFractionPct"),
                "fibreArealFractionPct": mean_of("fibreArealFractionPct"),
                "matrixArealFractionPct": mean_of("matrixArealFractionPct"),
                "worstClusterSeverityUm": controlling["clusterSeverityUm"],
                "fieldsOverLimit": len(over), "fieldsTotal": len(fields),
                "largestFeretUm": round(max((f["largestFeretUm"] for f in fields), default=0), 2),
                "voidCount": void_count,
                "voidDensityPerMm2": round(void_count / area_mm2, 1) if area_mm2 else 0.0,
                "feretP50Um": percentile(all_ferets, 50),
                "feretP95Um": percentile(all_ferets, 95),
                "analysedAreaMm2": round(area_mm2, 4),
            },
            "controllingFieldId": controlling["id"],
            "fields": fields,
        })

    payload = {
        "generatedAt": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": {"id": "challenge-severity-v1", "name": "Challenge cluster severity v1",
                    "clusterSeverityLimitUm": SEVERITY_THRESHOLD,
                    "mergeDistanceUm": MERGE_DISTANCE_UM, "areaFactor": 0.5,
                    "approvedForProductionUse": False,
                    "note": "Prototype challenge rule; replace with an approved business profile."},
        "model": {"id": model_id, "sha256": model_sha[:16],
                  "threshold": args.threshold, "minSize": args.min_size,
                  "checkpoints": [c.name for c in ckpts]},
        "inspections": inspections,
    }
    (out / "fixtures.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'=' * 78}")
    for i in inspections:
        k = i["kpis"]
        print(f"{i['sampleId']:12} {i['material']:10} {i['fieldCount']:2d} fields  "
              f"{i['preliminary']['disposition']:4}  worst {k['worstClusterSeverityUm']:6.2f} µm  "
              f"over limit {k['fieldsOverLimit']}/{k['fieldsTotal']}  "
              f"void {k['voidArealFractionPct']}%")
    print(f"\n-> {out / 'fixtures.json'}")


if __name__ == "__main__":
    main()
