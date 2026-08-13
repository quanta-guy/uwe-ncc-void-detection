"""Post-process a predicted label map using fibre geometry and a KNN vote.

Two ideas, both classical, both aimed at the same weakness: the network decides
each pixel from local texture, and the one thing it has no way to represent is
that a fibre is a CIRCLE of a known size.

**Circles.** metadata.csv gives fibre_radius_px per image, so the Hough
transform can be given a tight radius band instead of searching blind. That
turns a weak generic detector into a strong one. The payoff is not a better
fibre mask - the score never reads the fibre class - it is that voids do not
occur inside fibres. Porosity forms in the resin between them. So a predicted
void sitting inside a confidently detected fibre circle is a false positive,
and removing it raises precision without touching any true void.

**KNN.** The network's own confidence separates pixels it is sure about from
pixels it is not. The sure ones are free labelled training data, in this exact
image, under this exact illumination. A k-nearest-neighbour vote in colour and
texture space relabels the unsure pixels from them. This is per-image
adaptation with no training and no extra parameters, and unlike the circle step
it can move the void class in both directions.

scipy's cKDTree does the KNN - sklearn is not installed, and a KD-tree over a
few thousand points is all a KNN is.

Neither step is assumed to help. evaluate.py --refine runs the same threshold
sweep with it on, so the effect is measured on held-out data rather than
argued.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data import N_CLASSES, VOID_CLASS  # noqa: E402

FIBRE_CLASS = 1


def fibre_circles(img, radius_px, tol=0.4, dp=1.2, param2=22):
    """Detect fibre cross-sections as circles. Returns (N, 3) of x, y, r.

    Canny runs inside cv2.HoughCircles (param1 is its upper threshold), on a
    median-blurred grey image - fibres are smooth discs and the blur stops
    every speck of sensor noise voting in the accumulator.

    The radius band comes from the image's own metadata, which is the whole
    reason this is worth doing: an unconstrained Hough search on a texture this
    dense returns mostly nonsense, while +/-40% around a known radius is a
    strong prior that costs nothing.
    """
    if not radius_px or radius_px <= 1:
        return np.empty((0, 3), np.float32)

    grey = cv2.medianBlur(img.mean(axis=2).astype(np.uint8), 5)
    lo = max(2, int(radius_px * (1 - tol)))
    hi = max(lo + 1, int(radius_px * (1 + tol)))

    found = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=dp,
        minDist=max(3, int(radius_px * 1.2)),   # fibres touch but do not overlap
        param1=120,      # Canny upper threshold
        param2=param2,   # accumulator threshold; lower finds more, and more junk
        minRadius=lo, maxRadius=hi,
    )
    return np.empty((0, 3), np.float32) if found is None else found[0]


def circle_mask(shape, circles, shrink=0.75):
    """Filled disc mask from Hough circles.

    Shrunk from the detected radius because a Hough circle sits on the fibre's
    outer edge, and a void hugging the outside of a fibre is real. Only the
    interior, where a void cannot physically be, is claimed.
    """
    m = np.zeros(shape[:2], np.uint8)
    for x, y, r in circles:
        cv2.circle(m, (int(round(x)), int(round(y))), max(1, int(r * shrink)), 1, -1)
    return m.astype(bool)


def _features(img):
    """Per-pixel colour plus local texture, as (H*W, 6) float32.

    Raw RGB alone cannot separate a dark void from a shadowed patch of matrix.
    The local mean adds neighbourhood brightness, the local standard deviation
    adds texture - fibre edges are busy, void interiors are flat - and
    chromaticity adds the illumination-invariant blue fraction that is the
    void's most reliable signature.
    """
    x = img.astype(np.float32)
    grey = x.mean(axis=2)
    local = uniform_filter(grey, size=9)
    sq = uniform_filter(grey * grey, size=9)
    std = np.sqrt(np.clip(sq - local * local, 0, None))
    blue = x[:, :, 2] / x.sum(axis=2).clip(min=1e-4)

    f = np.stack([x[:, :, 0] / 255, x[:, :, 1] / 255, x[:, :, 2] / 255,
                  local / 255, std / 64, blue * 3], axis=-1)
    return f.reshape(-1, f.shape[-1]).astype(np.float32)


def knn_relabel(prob, img, k=7, hi=0.90, lo=0.60, max_ref=4000, rng=None):
    """Relabel low-confidence pixels by a KNN vote among high-confidence ones.

    prob is (3, H, W). Returns a refined (H, W) label map.

    Only the uncertain band is touched. Pixels the network is sure about are
    left exactly as they are - the network is better than this at the easy
    cases, and overwriting them would only add noise.
    """
    h, w = prob.shape[1:]
    flat = prob.reshape(N_CLASSES, -1)
    labels = flat.argmax(axis=0).astype(np.uint8)
    conf = flat.max(axis=0)

    sure = np.flatnonzero(conf >= hi)
    unsure = np.flatnonzero(conf < lo)
    # Nothing to vote with, or nothing to vote on. Both are common on clean
    # images and neither is an error.
    if len(unsure) == 0 or len(sure) < k or len(np.unique(labels[sure])) < 2:
        return labels.reshape(h, w)

    feats = _features(img)
    rng = rng or np.random.default_rng(0)
    if len(sure) > max_ref:
        sure = rng.choice(sure, max_ref, replace=False)

    _, idx = cKDTree(feats[sure]).query(feats[unsure], k=k, workers=-1)
    votes = labels[sure][idx]                       # (n_unsure, k)
    # Majority vote. bincount per row via a (n, 3) count matrix - faster than
    # scipy.stats.mode and gives ties to the lower class id, which is matrix.
    counts = np.stack([(votes == c).sum(axis=1) for c in range(N_CLASSES)], axis=1)
    labels[unsure] = counts.argmax(axis=1).astype(np.uint8)
    return labels.reshape(h, w)


def refine(prob, img, radius_px, use_circles=True, use_knn=True, **kw):
    """Refined (p_void, base) ready for predict.to_mask.

    p_void keeps the network's continuous probability so the threshold sweep
    still has something to sweep; refinement acts by zeroing it where a void is
    geometrically impossible, and by lifting or dropping it where the KNN vote
    disagrees with a low-confidence call.
    """
    p_void = prob[VOID_CLASS].astype(np.float32).copy()

    if use_knn:
        labels = knn_relabel(prob, img, **kw)
        conf = prob.max(axis=0)
        band = conf < kw.get("lo", 0.60)
        # Inside the uncertain band only, move the void probability to the side
        # the neighbours voted for. 0.99/0.01 rather than 1/0 so a downstream
        # threshold of exactly 1.0 cannot silently reject everything.
        p_void[band & (labels == VOID_CLASS)] = 0.99
        p_void[band & (labels != VOID_CLASS)] = 0.01

    if use_circles:
        inside = circle_mask(img.shape, fibre_circles(img, radius_px))
        p_void[inside] = 0.0   # porosity forms in resin, not inside a fibre

    base = (prob[FIBRE_CLASS] > prob[0]).astype(np.uint8)
    return p_void, base


def demo():
    """Circles must find planted discs; KNN must fix an uncertain band."""
    rng = np.random.default_rng(0)

    # Synthetic micrograph: bright fibres of a known radius on a mid matrix.
    img = np.full((200, 200, 3), 170, np.uint8)
    img[..., 2] = 160
    centres = [(50, 50), (50, 130), (130, 50), (130, 130)]
    for cy, cx in centres:
        cv2.circle(img, (cx, cy), 14, (235, 230, 225), -1)

    circles = fibre_circles(img, radius_px=14)
    assert len(circles) >= 3, f"found {len(circles)} of 4 planted fibres"
    # Every detection must land on a planted fibre, not in the matrix.
    for x, y, r in circles:
        assert min(abs(y - cy) + abs(x - cx) for cy, cx in centres) < 12, (x, y)
    assert 8 <= circles[:, 2].mean() <= 22, circles[:, 2].mean()

    # A radius the metadata does not support must return nothing rather than
    # inventing circles - a wrong prior should degrade to a no-op.
    assert len(fibre_circles(img, radius_px=0)) == 0

    inside = circle_mask(img.shape, circles)
    assert inside.any() and not inside[0, 0], "disc mask is wrong or covers everything"

    # KNN: a void-coloured patch left at low confidence must be voted void by
    # its neighbours, because confident void pixels elsewhere look like it.
    img2 = np.full((80, 80, 3), 170, np.uint8)
    img2[..., 2] = 160
    img2[10:20, 10:20] = [63, 59, 90]   # confident void
    img2[50:60, 50:60] = [63, 59, 90]   # same colour, network unsure

    prob = np.zeros((3, 80, 80), np.float32)
    prob[0] = 0.95
    prob[VOID_CLASS, 10:20, 10:20], prob[0, 10:20, 10:20] = 0.95, 0.05
    prob[:, 50:60, 50:60] = np.array([0.5, 0.05, 0.45])[:, None, None]  # unsure

    labels = knn_relabel(prob, img2)
    got = (labels[50:60, 50:60] == VOID_CLASS).mean()
    assert got > 0.7, f"KNN failed to rescue the uncertain void ({got:.2f})"
    assert (labels[10:20, 10:20] == VOID_CLASS).all(), "KNN overwrote a confident call"
    assert (labels[70:, :5] == 0).all(), "KNN invented voids in clean matrix"

    # The circle step must suppress a void predicted inside a fibre, and leave
    # one in the resin alone. That asymmetry is the entire point.
    prob3 = np.zeros((3, 200, 200), np.float32)
    prob3[0] = 0.9
    prob3[VOID_CLASS, 46:54, 46:54] = 0.9   # on top of the fibre at (50,50)
    prob3[VOID_CLASS, 90:98, 90:98] = 0.9   # in the resin between fibres
    p, base = refine(prob3, img, radius_px=14, use_knn=False)
    assert p[46:54, 46:54].max() == 0.0, "void inside a fibre survived"
    assert p[90:98, 90:98].max() > 0.5, "void in the resin was destroyed"
    assert base.shape == (200, 200)

    print(f"ok  {len(circles)} circles, KNN rescued {got:.0%} of the uncertain void")


if __name__ == "__main__":
    demo()
