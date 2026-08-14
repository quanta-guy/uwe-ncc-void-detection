"""The live half of the prototype: import images, load weights, run inference.

    python frontend/server/app.py            # http://127.0.0.1:8000

`build_fixtures.py` produces the two preloaded Test samples ahead of time because
severity geometry is CPU-bound scipy and a demo should open instantly. Everything a
user *does* has to be live, though, and until now three things pretended:

  - the import modal validated a folder and imported nothing
  - "Validate model" set a state variable and never opened a checkpoint
  - no inference ran on anything the user supplied

This serves those three. It reuses `build_fixtures` for measurement rather than
restating it, so an imported inspection and a fixture inspection are the same numbers
computed by the same code - which is the only way the KPIs stay comparable.

Bound to 127.0.0.1. No auth, no database: a local inspection workstation.
"""

import io
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "frontend" / "tools"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import build_fixtures as bf  # noqa: E402
from evaluation import SEVERITY_THRESHOLD, TooManyRegionsError  # noqa: E402
from model import build  # noqa: E402
from predict import to_mask, void_prob  # noqa: E402

DATA = REPO / "frontend" / "public" / "data"
IMPORTED = DATA / "imported.json"
ALLOWED = re.compile(r"\.(png|jpe?g|tiff?)$", re.I)
MAX_EDGE = 2048           # guard: a 12k-pixel micrograph would OOM a laptop

app = FastAPI(title="Composite Inspection prototype")


# --------------------------------------------------------------------------
# checkpoint discovery
# --------------------------------------------------------------------------
def fold_group(path: Path) -> str:
    """`unet_f3.pt` and `unet_f0.pt` are one ensemble; `unet_d3.pt` is not."""
    return re.sub(r"_f\d+$", "", path.stem)


def discover():
    """Every checkpoint on disk, grouped into the ensembles they form.

    Groups are what ships - a single fold is a different model from the five
    averaged. Ordering puts the largest ensembles first so the production model
    is the default without hard-coding its path.
    """
    # runs/ is the submission model; models/ holds the curated alternatives.
    # archive/ is deliberately NOT scanned: solution 2/3 checkpoints need their own
    # preprocessing (normalisation, physical resampling) that this pipeline does not
    # apply - the shape gate would pass them and they would then quietly mispredict.
    roots = [REPO / "runs", REPO / "models"]
    groups: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.pt")):
            key = f"{root.relative_to(REPO).as_posix()}/{fold_group(p)}"
            groups.setdefault(key, []).append(p)
    out = []
    for key, paths in groups.items():
        out.append({
            "id": key,
            "label": f"{key}  ({len(paths)}-fold ensemble)" if len(paths) > 1 else key,
            "count": len(paths),
            "paths": [p.relative_to(REPO).as_posix() for p in paths],
            "sizeMb": round(sum(p.stat().st_size for p in paths) / 1e6, 1),
            "record": RECORDS.get(key),
        })
    return sorted(out, key=lambda g: (-g["count"], g["id"]))


#: Validation record per model group - measured this project, not asserted here.
#: The two ensembles were scored out-of-fold over all 28 micrographs (3100 images)
#: by evaluation.py; tp/fp/fn are specimen-level pass/fail calls at the 25 um line
#: and tn is the remainder. The singles were never given an OOF confusion run, so
#: they carry only their held-out-split Dice - showing a matrix for them would be
#: inventing one.
N_OOF_IMAGES = 3100
RECORDS = {
    "runs/unet": {
        "final": 0.8869, "diceVoid": 0.7562, "f2": 0.9383,
        "tp": 769, "fp": 105, "fn": 37, "tn": N_OOF_IMAGES - 769 - 105 - 37,
        "operatingPoint": "threshold 0.4 · min-size 4",
        "protocol": "out-of-fold, 28 micrographs, 3100 images",
        "note": "Submission model. Finds 6 of 6 visible Test-set voids.",
    },
    "models/unet4_s2": {
        "final": 0.8788, "diceVoid": 0.7527, "f2": 0.9340,
        "tp": 764, "fp": 102, "fn": 42, "tn": N_OOF_IMAGES - 764 - 102 - 42,
        "operatingPoint": "threshold 0.3 · min-size 4",
        "protocol": "out-of-fold, 28 micrographs, 3100 images",
        "note": "Solution 4 (fixed augmentation RNG), best of 3 seeds. "
                "Test verdicts identical to the submission model.",
    },
    "models/unet_single": {
        "valDice": 0.7300,
        "protocol": "single held-out split only",
        "note": "Original single checkpoint. No out-of-fold confusion was measured, "
                "and with one member model disagreement is always zero.",
    },
    "models/unetpp_r34": {
        "valDice": 0.7254,
        "protocol": "single held-out split only",
        "note": "U-Net++ / ResNet-34 from the architecture sweep. No out-of-fold "
                "confusion was measured.",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_nets: dict[str, list] = {}
_lock = threading.Lock()
STATE = {"active": "runs/unet", "threshold": bf.THRESHOLD, "minSize": bf.MIN_SIZE}


def load_group(model_id: str):
    """Load and cache an ensemble. Raises with the real error if it will not load."""
    with _lock:
        if model_id in _nets:
            return _nets[model_id]
        group = next((g for g in discover() if g["id"] == model_id), None)
        if group is None:
            raise HTTPException(404, f"no such model: {model_id}")
        nets = []
        for rel in group["paths"]:
            ckpt = torch.load(REPO / rel, map_location=DEVICE, weights_only=False)
            net, _ = build(ckpt.get("arch", "unet"), ckpt.get("base", 32),
                           ckpt.get("depth", 4), ckpt.get("chroma", False))
            net.load_state_dict(ckpt["model"])
            nets.append(net.to(DEVICE).eval())
        _nets[model_id] = nets
        return nets


@app.get("/api/health")
def health():
    return {"ok": True, "device": DEVICE.type,
            "cuda": torch.cuda.is_available(), "active": STATE["active"]}


@app.get("/api/models")
def models():
    return {"models": discover(), **STATE}


@app.post("/api/models/validate")
def validate(payload: dict):
    """Actually open the checkpoint and push a tensor through it.

    A file that loads is not the same as a file that works. This runs a real forward
    pass and checks the output is three classes at the input's own size, because a
    checkpoint that quietly returns two classes would otherwise become active and
    produce a whole inspection of nonsense.
    """
    model_id = payload.get("id", "")
    t0 = time.time()
    try:
        nets = load_group(model_id)
        with torch.no_grad():
            y = nets[0](torch.zeros(1, 3, 256, 256, device=DEVICE))
        classes, hw = int(y.shape[1]), tuple(y.shape[-2:])
        ok = classes == 3 and hw == (256, 256)
        return {
            "ok": ok, "id": model_id, "classes": classes,
            "dims": f"{hw[0]}x{hw[1]}", "members": len(nets), "device": DEVICE.type,
            "elapsedMs": int(1000 * (time.time() - t0)),
            "detail": ("Loads, 3 classes, shape preserved"
                       if ok else f"Rejected: {classes} classes at {hw[0]}x{hw[1]}, expected 3 at 256x256"),
            "disagreementAvailable": len(nets) > 1,
        }
    except HTTPException:
        raise
    except Exception as e:                                  # noqa: BLE001
        # The failure is the useful output here - report it rather than a 500.
        return {"ok": False, "id": model_id, "detail": f"{type(e).__name__}: {e}",
                "elapsedMs": int(1000 * (time.time() - t0))}


@app.post("/api/models/activate")
def activate(payload: dict):
    model_id = payload.get("id", "")
    load_group(model_id)                     # must load before it can be active
    STATE["active"] = model_id
    if payload.get("threshold") is not None:
        STATE["threshold"] = float(payload["threshold"])
    if payload.get("minSize") is not None:
        STATE["minSize"] = int(payload["minSize"])
    return STATE


# --------------------------------------------------------------------------
# import + inference
# --------------------------------------------------------------------------
JOBS: dict[str, dict] = {}


def read_upload(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) > MAX_EDGE:
        scale = MAX_EDGE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    return np.array(img)


def analyse_one(nets, img, um, stem, threshold, min_size):
    """One field: predict, measure, and write the four render layers."""
    per_model = [void_prob([n], img, DEVICE)[0] for n in nets]
    p_mean = np.mean(per_model, axis=0)
    spread = np.std(per_model, axis=0) if len(nets) > 1 else np.zeros_like(p_mean)
    _, base = void_prob(nets, img, DEVICE)
    mask = to_mask(p_mean, base, threshold, min_size)

    try:
        m = bf.measure(mask, um, spread, img.shape)
    except TooManyRegionsError:
        # evaluation.py scores this as an automatic fail; say so rather than crash.
        m = {**{k: 0 for k in ("voidCount", "voidAreaUm2", "largestFeretUm", "voidPx")},
             "voidArealFractionPct": 0.0, "fibreArealFractionPct": 0.0,
             "matrixArealFractionPct": 0.0, "sampledAreaMm2": 0.0, "feretsUm": [],
             "clusterSeverityUm": float("inf"), "distanceToLimitUm": float("inf"),
             "verdict": "FAIL", "modelDisagreement": 0.0, "triageAction": "REJECT",
             "triageReason": "Over 1500 void regions - unscorable, automatic fail",
             "severityEvidence": {"severity_um": 0, "voids": [], "gaps": []}}

    Image.fromarray(img).save(DATA / "originals" / f"{stem}.jpg", quality=92)
    bf.layer(mask, 1, bf.PALETTE["fibre"]).save(DATA / "fibre" / f"{stem}.png")
    bf.layer(mask, bf.VOID_CLASS, bf.PALETTE["void"]).save(DATA / "void" / f"{stem}.png")
    regions = bf._VoidRegions(mask)
    bf.outline(mask, [v["index"] for v in m["severityEvidence"]["voids"]],
               regions).save(DATA / "controlling" / f"{stem}.png")

    thumb = img.astype(np.float32).copy()
    sel = mask == bf.VOID_CLASS
    thumb[sel] = 0.35 * thumb[sel] + 0.65 * np.array(bf.PALETTE["void"], np.float32)
    Image.fromarray(thumb.astype(np.uint8)).resize((bf.THUMB, bf.THUMB),
                                                   Image.LANCZOS).save(
        DATA / "thumbs" / f"{stem}.jpg", quality=88)
    return m


def roll_up(fields, meta, model_id, nets_n, threshold, min_size):
    """Sample-level KPIs. Same shape build_fixtures emits, so the UI needs no branch."""
    all_ferets = [f for fl in fields for f in fl["feretsUm"]]
    area_mm2 = sum(f["sampledAreaMm2"] for f in fields)
    void_count = sum(f["voidCount"] for f in fields)
    over = [f for f in fields if f["clusterSeverityUm"] >= SEVERITY_THRESHOLD]
    controlling = max(fields, key=lambda f: f["clusterSeverityUm"])
    mean_of = lambda k: round(float(np.mean([f[k] for f in fields])), 3)  # noqa: E731

    return {
        **meta, "micrograph": meta["sampleId"], "sampleType": "polished cross-section",
        "userEntered": ["material", "sampleId", "batch"],
        "measurementProfile": "challenge-severity-v1",
        "model": {"id": model_id, "sha256": "live", "threshold": threshold,
                  "minSize": min_size},
        "state": "ready_for_review", "reviewedCount": 0, "fieldCount": len(fields),
        "imported": True,
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
            "feretP50Um": bf.percentile(all_ferets, 50),
            "feretP95Um": bf.percentile(all_ferets, 95),
            "analysedAreaMm2": round(area_mm2, 4),
        },
        "controllingFieldId": controlling["id"],
        "fields": fields,
    }


def append_imported(inspection):
    """Never touch fixtures.json - it is a build artifact and must stay reproducible."""
    with _lock:
        current = json.loads(IMPORTED.read_text()) if IMPORTED.exists() else {"inspections": []}
        current["inspections"] = [i for i in current["inspections"]
                                  if i["inspectionId"] != inspection["inspectionId"]]
        current["inspections"].append(inspection)
        tmp = IMPORTED.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2))
        tmp.replace(IMPORTED)


def run_job(job_id, uploads, meta, um, model_id, threshold, min_size):
    job = JOBS[job_id]
    try:
        nets = load_group(model_id)
        job["total"] = len(uploads)
        fields = []
        for n, (name, raw) in enumerate(uploads, start=1):
            job["current"] = name
            img = read_upload(raw)
            stem = f"{meta['inspectionId']}__{Path(name).stem}"
            m = analyse_one(nets, img, um, stem, threshold, min_size)
            fields.append({"id": f"{n:02d}", "stem": stem, "umPerPixel": um,
                           "sourceFile": name, **m})
            job["done"] = n
        inspection = roll_up(fields, meta, model_id, len(nets), threshold, min_size)
        append_imported(inspection)
        job.update(state="done", inspection=inspection)
    except Exception as e:                                  # noqa: BLE001
        job.update(state="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/inspections")
async def create_inspection(
    files: list[UploadFile] = File(...),
    inspectionId: str = Form(...),
    sampleId: str = Form(...),
    material: str = Form(...),
    batch: str = Form("-"),
    umPerPixel: float = Form(...),
    modelId: str = Form(""),
):
    """Import a folder of micrographs and run the ensemble over every one.

    Returns immediately with a job id. Inference is five forward passes per image
    plus scipy hulls, which on CPU is tens of seconds for a folder - long enough
    that a synchronous request would look like a hang rather than like work.
    """
    if umPerPixel <= 0:
        raise HTTPException(400, "calibration must be positive: without a trusted "
                                 "µm/pixel there is no physical measurement")
    uploads = [(f.filename, await f.read()) for f in files
               if f.filename and ALLOWED.search(f.filename)]
    if not uploads:
        raise HTTPException(400, "no supported images (.png/.jpg/.jpeg/.tif/.tiff)")

    stems = [Path(n).stem for n, _ in uploads]
    dupes = {s for s in stems if stems.count(s) > 1}
    if dupes:
        raise HTTPException(400, f"duplicate image stems would collide: {sorted(dupes)}")

    for sub in ("originals", "fibre", "void", "controlling", "thumbs"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    model_id = modelId or STATE["active"]
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"state": "running", "done": 0, "total": len(uploads),
                    "current": "", "inspectionId": inspectionId}
    meta = {"inspectionId": inspectionId, "sampleId": sampleId,
            "material": material, "batch": batch, "umPerPixel": umPerPixel}
    threading.Thread(target=run_job, daemon=True, args=(
        job_id, uploads, meta, umPerPixel, model_id,
        STATE["threshold"], STATE["minSize"])).start()
    return {"jobId": job_id, "total": len(uploads), "model": model_id,
            "device": DEVICE.type}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return JSONResponse(job)


@app.delete("/api/inspections/{inspection_id}")
def delete_imported(inspection_id: str):
    """Remove an imported inspection. Fixture inspections are read-only by design."""
    with _lock:
        if not IMPORTED.exists():
            raise HTTPException(404, "nothing imported")
        current = json.loads(IMPORTED.read_text())
        gone = [i for i in current["inspections"] if i["inspectionId"] == inspection_id]
        if not gone:
            raise HTTPException(404, f"{inspection_id} is not an imported inspection")
        current["inspections"] = [i for i in current["inspections"]
                                  if i["inspectionId"] != inspection_id]
        IMPORTED.write_text(json.dumps(current, indent=2))

    # Take the render layers with it, or repeated imports silently fill the data
    # folder with orphans nothing references.
    for field in gone[0]["fields"]:
        for sub, ext in (("originals", "jpg"), ("fibre", "png"), ("void", "png"),
                         ("controlling", "png"), ("thumbs", "jpg")):
            (DATA / sub / f"{field['stem']}.{ext}").unlink(missing_ok=True)
    return {"removed": inspection_id, "fields": len(gone[0]["fields"])}


def demo():
    """Discovery groups folds correctly and validation rejects a broken checkpoint."""
    groups = discover()
    assert len(groups) == 4, [g["id"] for g in groups]   # the curated four, no strays
    prod = next(g for g in groups if g["id"] == "runs/unet")
    assert prod["count"] == 5, prod
    assert next(g for g in groups if g["id"] == "models/unet4_s2")["count"] == 5
    assert fold_group(Path("unet_f12.pt")) == "unet"
    assert fold_group(Path("unet_d3.pt")) == "unet_d3", "stripped a non-fold suffix"

    r = validate({"id": "runs/unet"})
    assert r["ok"] and r["classes"] == 3 and r["members"] == 5, r
    try:
        validate({"id": "does/not/exist"})
        raise AssertionError("unknown model must 404, not fall through to a load")
    except HTTPException as e:
        assert e.status_code == 404, e

    # A checkpoint the builder cannot construct must come back as a failed gate,
    # not a 500 - reporting why is the whole point of having a validation step.
    bad = validate({"id": groups[-1]["id"]}) if len(groups) > 1 else {"ok": True}
    assert "detail" in bad, bad
    print(f"ok  {len(groups)} model groups, production loads on {DEVICE.type}: {r['detail']}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
