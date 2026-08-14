"""Locate the geometry behind a severity number, for the NCC evidence overlay.

    python frontend/tools/severity_geometry.py     # self-check

The NCC deck draws severity with a specific annotation language: an `L` arrow along
each void's longest internal axis, an `A` label on the void body, and a `D` arrow
between voids close enough to have merged into one defect. Reproducing that on a real
prediction turns the severity number from an assertion into something a reviewer can
audit at a glance.

**This module does not measure anything.** evaluation.py has already produced every
number; the job here is only to find the *points* that produced them, so the arrows
are guaranteed to agree with the score:

  L endpoints   evaluation._diameter_px takes np.max(cdist(P, P)) over the region's
                convex-hull points. This takes the argmax of that same matrix over
                those same points, so the arrow spans exactly the chord whose length
                was scored.
  D endpoints   _VoidRegions.distance(i, j) returns the scalar minimum. This returns
                the argmin pair producing it.
  membership    merge_regions() decides which voids form the defect. Untouched.

The alternative - recomputing lengths in the frontend - would eventually disagree with
the score, and an overlay that disagrees with the number it explains is worse than no
overlay.

Coordinates are returned in (row, col) pixel space matching the mask array, plus
micron values, so the frontend can scale them to whatever size it renders at.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from evaluation import (  # noqa: E402  - the judge's own code, never reimplemented
    K_AREA,
    MERGE_DISTANCE_UM,
    SEVERITY_THRESHOLD,
    VOID_CLASS,
    _HULL_MIN_POINTS,
    _VoidRegions,
    merge_regions,
)


def _hull_points(coords):
    """The point set evaluation._diameter_px actually measures over.

    Mirrors its logic exactly, including the degenerate-hull fallback: a region whose
    pixels form a straight line has no 2D hull, and Qhull raises rather than returning
    the line.
    """
    if len(coords) > _HULL_MIN_POINTS:
        try:
            return coords[ConvexHull(coords).vertices]
        except (QhullError, ValueError):
            return coords
    return coords


def feret_endpoints(coords):
    """((r0, c0), (r1, c1), length_px) for the region's maximum Feret chord."""
    if len(coords) < 2:
        p = coords[0] if len(coords) else np.array([0, 0])
        return tuple(p), tuple(p), 0.0

    pts = _hull_points(coords)
    d = cdist(pts, pts)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return tuple(pts[i]), tuple(pts[j]), float(d[i, j])


def closest_pair(coords_a, coords_b):
    """((r,c), (r,c), gap_px) - the two points that set the distance between voids."""
    d = cdist(coords_a, coords_b)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return tuple(coords_a[i]), tuple(coords_b[j]), float(d[i, j])


def controlling_geometry(mask, um_per_px):
    """Full evidence geometry for the cluster that sets the severity.

    Returns a JSON-ready dict: the voids with their L arrows and A labels, the D
    arrows between members that are close enough to have merged them, and the severity
    arithmetic broken into its two terms so the UI can show the sum being formed.
    """
    regions = _VoidRegions(mask)
    empty = {"severity_um": 0.0, "limit_um": SEVERITY_THRESHOLD, "verdict": "PASS",
             "length_term_um": 0.0, "area_term_um": 0.0, "voids": [], "gaps": [],
             "merge_distance_um": MERGE_DISTANCE_UM}
    if regions.n == 0:
        return empty

    groups = merge_regions(regions, um_per_px)
    scored = [(sum(regions.diameter(i) * um_per_px for i in g)
               + K_AREA * np.sqrt(sum(regions.areas[i] * um_per_px ** 2 for i in g)), g)
              for g in groups]
    severity, group = max(scored, key=lambda s: s[0])

    # Largest first, so the labels L1/A1 land on the void a reviewer looks at first.
    group = sorted(group, key=lambda i: -regions.diameter(i))

    voids = []
    for n, i in enumerate(group, start=1):
        p0, p1, length_px = feret_endpoints(regions.coords[i])
        cy, cx = regions.coords[i].mean(axis=0)
        voids.append({
            "index": int(i), "label": f"L{n}", "areaLabel": f"A{n}",
            "lengthUm": round(length_px * um_per_px, 2),
            "areaUm2": round(regions.areas[i] * um_per_px ** 2, 2),
            "areaPx": int(regions.areas[i]),
            "l0": [int(p0[0]), int(p0[1])], "l1": [int(p1[0]), int(p1[1])],
            "centroid": [float(cy), float(cx)],
        })

    # Only gaps under the merge distance are drawn: those are the ones that caused
    # these voids to be scored as a single defect, which is the point being made.
    gaps = []
    for a in range(len(group)):
        for b in range(a + 1, len(group)):
            p0, p1, gap_px = closest_pair(regions.coords[group[a]], regions.coords[group[b]])
            gap_um = gap_px * um_per_px
            if gap_um < MERGE_DISTANCE_UM:
                gaps.append({
                    "label": f"D{a + 1}{b + 1}", "gapUm": round(gap_um, 2),
                    "d0": [int(p0[0]), int(p0[1])], "d1": [int(p1[0]), int(p1[1])],
                })

    length_term = sum(v["lengthUm"] for v in voids)
    area_term = K_AREA * float(np.sqrt(sum(regions.areas[i] * um_per_px ** 2 for i in group)))
    return {
        "severity_um": round(float(severity), 2),
        "limit_um": SEVERITY_THRESHOLD,
        "verdict": "FAIL" if severity >= SEVERITY_THRESHOLD else "PASS",
        "length_term_um": round(length_term, 2),
        "area_term_um": round(area_term, 2),
        "merge_distance_um": MERGE_DISTANCE_UM,
        "voids": voids, "gaps": gaps,
    }


def demo():
    """The arrows must reproduce the numbers evaluation.py scored."""
    from evaluation import compute_max_severity

    # Two voids 20px apart at 1um/px: inside the 40um merge distance, so one defect.
    m = np.zeros((160, 160), np.uint8)
    m[40:50, 40:70] = VOID_CLASS      # elongated, long axis ~30px
    m[40:48, 90:100] = VOID_CLASS     # smaller, 20px gap from the first
    g = controlling_geometry(m, 1.0)

    sev_ref, n_clusters = compute_max_severity(m, 1.0)
    assert n_clusters == 1, f"expected one merged defect, got {n_clusters}"
    assert abs(g["severity_um"] - sev_ref) < 0.01, (g["severity_um"], sev_ref)

    # Every L arrow must actually span the length it claims.
    for v in g["voids"]:
        drawn = float(np.hypot(v["l1"][0] - v["l0"][0], v["l1"][1] - v["l0"][1]))
        assert abs(drawn - v["lengthUm"]) < 0.01, (v["label"], drawn, v["lengthUm"])

    # The two terms must sum to the severity evaluation.py reported.
    assert abs(g["length_term_um"] + g["area_term_um"] - g["severity_um"]) < 0.05

    assert len(g["voids"]) == 2 and len(g["gaps"]) == 1
    assert g["gaps"][0]["gapUm"] < MERGE_DISTANCE_UM
    drawn_gap = float(np.hypot(g["gaps"][0]["d1"][0] - g["gaps"][0]["d0"][0],
                               g["gaps"][0]["d1"][1] - g["gaps"][0]["d0"][1]))
    assert abs(drawn_gap - g["gaps"][0]["gapUm"]) < 0.01

    # Voids beyond the merge distance must NOT be joined, or the overlay would claim
    # a merge the scorer never made.
    far = np.zeros((300, 300), np.uint8)
    far[10:20, 10:40] = VOID_CLASS
    far[10:20, 200:230] = VOID_CLASS
    gf = controlling_geometry(far, 1.0)
    assert len(gf["gaps"]) == 0, "drew a merge gap the scorer did not make"
    assert len(gf["voids"]) == 1, "controlling cluster should hold one void only"
    assert abs(gf["severity_um"] - compute_max_severity(far, 1.0)[0]) < 0.01

    # An empty mask must be a clean PASS, not a crash.
    e = controlling_geometry(np.zeros((32, 32), np.uint8), 0.57)
    assert e["severity_um"] == 0.0 and e["verdict"] == "PASS" and e["voids"] == []

    print(f"ok  severity {g['severity_um']} = L {g['length_term_um']} + "
          f"A {g['area_term_um']}, {len(g['voids'])} voids, {len(g['gaps'])} merge gap")


if __name__ == "__main__":
    demo()
