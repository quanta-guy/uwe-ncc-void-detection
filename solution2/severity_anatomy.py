"""What actually drives the severity score, measured on the ground truth.

    python solution2/severity_anatomy.py

severity = SUM(diameter_i) + 0.5 * sqrt(SUM(area_i)) over one cluster, where
clusters are single-linkage groups of void regions closer than 40um, and an
image scores the maximum over its clusters. Fail at >= 25.

Two things follow from that formula which decide what a model should optimise,
and neither is visible in the Dice score:

1. The length term is a SUM over regions, so it grows linearly with the NUMBER
   of regions in a cluster. The area term is damped by a square root and
   multiplied by 0.5. Severity is therefore mostly a detection-and-count
   property, not a boundary-precision property.

2. For a single circular void of diameter d, area = 0.785 d^2, so
   severity = d + 0.5*sqrt(0.785 d^2) = 1.443 d. Failing alone needs
   d >= 17.3um. Most single voids in this data are smaller than that, so most
   failures must come from clustering.

This script tests both claims against every ground-truth mask rather than
leaving them as algebra. Everything is imported from evaluation.py - the
competition's own file - so nothing here is a reimplementation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data import VOID_CLASS, index_training, load_mask  # noqa: E402
from evaluation import (K_AREA, MERGE_DISTANCE_UM, SEVERITY_THRESHOLD,  # noqa: E402
                        _VoidRegions, group_severity, merge_regions)


def main():
    df = index_training(0)
    print(f"scanning {len(df)} ground-truth masks\n")
    print(f"formula: sum(diameter) + {K_AREA} * sqrt(sum(area))   "
          f"cluster < {MERGE_DISTANCE_UM}um   fail >= {SEVERITY_THRESHOLD}\n")

    rows = []
    for _, r in df.iterrows():
        gt = load_mask(r["mask"])
        if not (gt == VOID_CLASS).any():
            continue
        um = r["um_per_pixel"]
        regions = _VoidRegions(gt)
        groups = merge_regions(regions, um)

        sev = [group_severity(g, regions, um) for g in groups]
        worst = int(np.argmax(sev))
        g = groups[worst]

        length = sum(regions.diameter(i) * um for i in g)
        area = sum(regions.areas[i] * um * um for i in g)
        rows.append({
            "stem": r["stem"], "um": um, "severity": sev[worst],
            "n_regions_image": regions.n, "n_clusters": len(groups),
            "n_in_worst_cluster": len(g),
            "length_um": length, "area_term": K_AREA * np.sqrt(area),
            "biggest_diameter_um": max(regions.diameter(i) * um for i in g),
            "fail": sev[worst] >= SEVERITY_THRESHOLD,
        })

    d = pd.DataFrame(rows)
    fails = d[d.fail]
    print(f"{len(d)} void-containing masks, {len(fails)} failing "
          f"({len(fails) / len(d):.1%})\n")

    # Claim 1: severity is dominated by the length term, not the area term.
    share = d["length_um"] / d["severity"]
    print("share of severity contributed by the LENGTH term")
    print(f"  median {share.median():.1%}   p10 {share.quantile(.1):.1%}   "
          f"p90 {share.quantile(.9):.1%}")
    print(f"  -> the sqrt(area) term contributes the remainder, "
          f"median {1 - share.median():.1%}\n")

    # Claim 2: failures are driven by clustering, not by single large voids.
    print("regions in the worst cluster, for FAILING images")
    vc = fails["n_in_worst_cluster"].value_counts().sort_index()
    for n, c in vc.head(8).items():
        print(f"  {n:3d} region(s)  {c:5d} images  {c / len(fails):6.1%}")
    multi = (fails["n_in_worst_cluster"] > 1).mean()
    print(f"  -> {multi:.1%} of failures come from clusters of 2+ regions\n")

    # Could the worst cluster's biggest single void have failed by itself?
    alone = 1.443 * fails["biggest_diameter_um"]
    solo = (alone >= SEVERITY_THRESHOLD).mean()
    print(f"failing images whose LARGEST single void would fail on its own: {solo:.1%}")
    print(f"  (a lone circular void needs diameter >= "
          f"{SEVERITY_THRESHOLD / 1.443:.1f}um to reach severity {SEVERITY_THRESHOLD})\n")

    # What a model loses per missed region, on average, in the worst cluster.
    per_region = fails["length_um"] / fails["n_in_worst_cluster"]
    print(f"mean diameter per region inside a failing cluster: "
          f"{per_region.median():.1f}um")
    print(f"  every missed member of a cluster removes about that much severity,")
    print(f"  and the fail line sits at {SEVERITY_THRESHOLD} - so on a cluster scoring "
          f"{fails['severity'].median():.0f} (median),")
    print(f"  missing {max(1, int((fails['severity'].median() - SEVERITY_THRESHOLD) / max(per_region.median(), 1e-9)) + 1)}"
          f" region(s) can flip it to a pass\n")

    print("severity distribution of failing images")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q * 100):<3d} {fails['severity'].quantile(q):8.1f}")
    near = ((fails["severity"] >= SEVERITY_THRESHOLD) &
            (fails["severity"] < SEVERITY_THRESHOLD * 1.4)).mean()
    print(f"  -> {near:.1%} of failures sit within 40% of the fail line, "
          f"where a single missed region decides the verdict")

    out = REPO / "solution2" / "results" / "severity_anatomy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nper-image table -> {out}")


if __name__ == "__main__":
    main()
