import argparse
import cv2
import numpy as np
from common import load_images, resize_to_width, save, label

# HSV ranges for OpenCV: H is 0-179 (not 0-359), S and V are 0-255.
# Red wraps around 0, so it needs two ranges. Tune with hsv_tuner.py.
COLOR_RANGES = {
    "red":    [((0, 100, 70), (4, 255, 255)), ((165, 100, 70), (179, 255, 255))],
    "orange": [((5, 100, 100), (22, 255, 255))],
    "yellow": [((22, 80, 100), (35, 255, 255))],
    "green":  [((36, 60, 50), (85, 255, 255))],
    "blue":   [((90, 80, 50), (130, 255, 255))],
    "purple": [((130, 50, 40), (160, 255, 255))],
    "black":  [((0, 0, 0), (179, 255, 60))],
    "white":  [((0, 0, 200), (179, 40, 255))],
}


def build_detector(min_area, max_area, blob_color=None, min_circularity=0.6):
    p = cv2.SimpleBlobDetector_Params()
    # Threshold sweep
    p.minThreshold, p.maxThreshold, p.thresholdStep = 10, 220, 10
    # Size filter (area in pixels)
    p.filterByArea = True
    p.minArea, p.maxArea = min_area, max_area
    # Shape filters: circularity = 4*pi*area / perimeter^2 (1.0 = perfect circle)
    p.filterByCircularity = True
    p.minCircularity = min_circularity
    # Convexity = area / convex-hull area (dents lower it)
    p.filterByConvexity = True
    p.minConvexity = 0.8
    # Inertia ratio = how elongated (1 = round, 0 = line)
    p.filterByInertia = True
    p.minInertiaRatio = 0.4
    # Intensity "color": 0 = dark blobs, 255 = light blobs
    p.filterByColor = blob_color is not None
    if blob_color is not None:
        p.blobColor = blob_color
    return cv2.SimpleBlobDetector_create(p)


def color_mask(bgr, color):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    # Opening removes speckle noise; closing fills small gaps inside dots
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def merge_keypoints(kps):
    kept = []
    for kp in sorted(kps, key=lambda k: -k.size):          # biggest first
        x, y, r = kp.pt[0], kp.pt[1], kp.size / 2
        if all((x - k.pt[0]) ** 2 + (y - k.pt[1]) ** 2 > (max(r, k.size / 2)) ** 2 for k in kept):
            kept.append(kp)
    return kept


def detect_auto(img, args):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sat = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    kps = []
    kps += build_detector(args.min_area, args.max_area, 0, args.min_circ).detect(gray)
    kps += build_detector(args.min_area, args.max_area, 255, args.min_circ).detect(gray)
    kps += build_detector(args.min_area, args.max_area, 255, args.min_circ).detect(sat)
    kps += build_detector(args.min_area, args.max_area, 255, args.min_circ).detect(bg_distance(img))
    return merge_keypoints(kps)


def bg_distance(img, saturate_at=60, l_weight=0.4):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    q = (lab // 16).reshape(-1, 3).astype(np.int32)           # group similar colors into bins
    codes = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    top = np.bincount(codes).argmax()                        # most common bin
    bg = lab.reshape(-1, 3)[codes == top].mean(axis=0)
    diff = lab.astype(np.float32) - bg
    diff[:, :, 0] *= l_weight
    dist = np.linalg.norm(diff, axis=2)
    return np.clip(dist * 255.0 / saturate_at, 0, 255).astype(np.uint8)


def detect_color(img, args):
    mask = color_mask(img, args.color)
    det = build_detector(args.min_area, args.max_area, 255, args.min_circ)
    return det.detect(mask), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--filter", default=None, help="only files whose name contains this")
    ap.add_argument("--out", default="output/part1")
    ap.add_argument("--min-area", type=float, default=80)
    ap.add_argument("--max-area", type=float, default=20000)
    ap.add_argument("--min-circ", type=float, default=0.6)
    ap.add_argument("--color", choices=list(COLOR_RANGES), default=None,
                    help="only keep dots of this color")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    for name, img in load_images(args.images, args.filter):
        img = resize_to_width(img)
        if args.color:
            kps, mask = detect_color(img, args)
            save(args.out, f"mask_{args.color}_{name}", mask)
        else:
            kps = detect_auto(img, args)

        # Draw each blob: black outline + white inner ring so it's visible on any color.
        # kp.pt = (x, y) center, kp.size = blob diameter
        out = img.copy()
        for kp in kps:
            c, r = (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2)
            cv2.circle(out, c, r, (0, 0, 0), 4)
            cv2.circle(out, c, r, (255, 255, 255), 2)
        tag = f"{args.color} " if args.color else ""
        label(out, f"{len(kps)} {tag}blobs")
        print(f"{name}: {len(kps)} blobs")
        save(args.out, f"blobs_{name}", out)
        if args.show:
            cv2.imshow(name, out); cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
