import argparse
import cv2
import numpy as np
from common import load_images, resize_to_width, save, label

CLASSES = {
    "cone": {"hsv": [((18, 150, 120), (32, 255, 255))],
             "draw": (0, 220, 255), "fill": True, "split": False},
    "cube": {"hsv": [((112, 80, 60), (140, 255, 255))],
             "draw": (255, 0, 180), "fill": True, "split": True},
    "ring": {"hsv": [((0, 90, 110), (10, 255, 255)),
                     ((170, 90, 110), (179, 255, 255))],
             "draw": (0, 120, 255), "fill": False, "split": False},
}
MIN_AREA_FRAC = 0.0015   # ignore anything smaller than 0.15% of the image


def make_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def fill_holes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def split_touching(img, mask, peak_frac=0.5):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, seeds = cv2.threshold(dist, peak_frac * dist.max(), 255, cv2.THRESH_BINARY)
    n, markers = cv2.connectedComponents(seeds.astype(np.uint8))
    if n <= 2:                       # background + 1 seed -> nothing to split
        return [mask]
    markers = markers + 1            # watershed wants background = 1, not 0
    markers[(mask > 0) & (seeds == 0)] = 0   # 0 = "unknown", to be decided
    markers = cv2.watershed(img, markers)
    return [np.uint8(markers == lbl) * 255 for lbl in range(2, n + 1)]


def shape_features(cnt, has_hole, hole_area):
    area = cv2.contourArea(cnt) - hole_area
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
    return {
        "area": area,
        "solidity": area / hull_area if hull_area else 0,
        "extent": area / (w * h) if w * h else 0,
        "vertices": len(approx),
        "has_hole": has_hole,
        "aspect": max(rw, rh) / max(min(rw, rh), 1),
    }


def fits(cls, f):
    if cls == "ring":
        return f["has_hole"]        # the hole is what makes a ring a ring
    if cls == "cube":
        return f["extent"] > 0.55 and f["solidity"] > 0.85
    if cls == "cone":
        return f["extent"] < 0.8 and f["solidity"] < 0.95
    return True


def contours_for_class(cls, mask, min_area):
    """Find outer contours (and their holes) in one mask and keep those that fit `cls`."""
    found = []
    # RETR_CCOMP gives a 2-level hierarchy: outer contours and the holes inside them.
    # hierarchy[i] = [next, prev, first_child, parent]; parent == -1 means outer contour
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return found
    hier = hier[0]
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:          # skip holes themselves
            continue
        child = hier[i][2]
        hole_area, has_hole = 0, False
        while child != -1:            # sum all holes inside this contour
            a = cv2.contourArea(cnts[child])
            if a > 0.03 * cv2.contourArea(c):
                has_hole = True
                hole_area += a
            child = hier[child][0]
        f = shape_features(c, has_hole, hole_area)
        if f["area"] >= min_area and fits(cls, f):
            found.append((cls, c, f))
    return found


def detect(img, debug_dir=None, name=""):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    min_area = MIN_AREA_FRAC * img.shape[0] * img.shape[1]
    found = []
    for cls, cfg in CLASSES.items():
        mask = make_mask(hsv, cfg["hsv"])
        if cfg["fill"]:
            mask = fill_holes(mask)
        if debug_dir:
            save(debug_dir, f"mask_{cls}_{name}", mask)
        pieces = split_touching(img, mask) if cfg["split"] else [mask]
        for piece in pieces:
            found += contours_for_class(cls, piece, min_area)
    return found


def draw(img, found):
    out = img.copy()
    counts = {k: 0 for k in CLASSES}
    for cls, c, f in found:
        color = CLASSES[cls]["draw"]
        counts[cls] += 1
        cv2.drawContours(out, [c], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
        label(out, cls, (x, max(y - 8, 20)), color)
    label(out, "  ".join(f"{k}:{v}" for k, v in counts.items()), (10, 30), (255, 255, 255))
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--out", default="output/challenge2")
    ap.add_argument("--debug", action="store_true", help="save per-class masks + print features")
    args = ap.parse_args()

    for name, img in load_images(args.images, args.filter):
        img = resize_to_width(img)
        found = detect(img, args.out + "/debug" if args.debug else None, name)
        out, counts = draw(img, found)
        print(f"{name}: {counts}")
        if args.debug:
            for cls, _, f in found:
                print(f"   {cls:5s} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in f.items()))
        save(args.out, f"classified_{name}", out)


if __name__ == "__main__":
    main()
