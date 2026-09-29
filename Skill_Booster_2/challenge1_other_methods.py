import argparse
import math
import time
import cv2
import numpy as np
from skimage.feature import blob_log, blob_dog, blob_doh
from common import load_images, resize_to_width, save, label
from part1_simple_blob import bg_distance


def prep_gray(img):
    _, fg = cv2.threshold(bg_distance(img, l_weight=0.6), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(fg, (0, 0), 2).astype(np.float32) / 255.0


def keep_real_dots(blobs, gray, inside_min=0.7, ring_max=0.6):
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    keep = []
    for y, x, r in blobs:
        r = max(r, 2)
        y0, y1 = int(max(y - 1.7 * r, 0)), int(min(y + 1.7 * r + 1, h))
        x0, x1 = int(max(x - 1.7 * r, 0)), int(min(x + 1.7 * r + 1, w))
        d = np.hypot(yy[y0:y1, x0:x1] - y, xx[y0:y1, x0:x1] - x)   # distance to center
        patch = gray[y0:y1, x0:x1]
        inside, ring = patch[d <= r * 0.8], patch[(d > r * 1.25) & (d <= r * 1.65)]
        if inside.size and ring.size and inside.mean() > inside_min and ring.mean() < ring_max:
            keep.append((y, x, r))
    return np.array(keep).reshape(-1, 3)


def run_skimage(gray, method, min_sigma, max_sigma):
    t = time.time()
    if method == "LoG":
        # threshold: minimum response strength; num_sigma: how many scales to try
        blobs = blob_log(gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=0.3)
        blobs[:, 2] *= math.sqrt(2)          # sigma -> radius
    elif method == "DoG":
        blobs = blob_dog(gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=0.3)
        blobs[:, 2] *= math.sqrt(2)
    else:  # DoH: sigma is already ~radius
        blobs = blob_doh(gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=0.005)
    blobs = keep_real_dots(blobs, gray)
    return blobs, (time.time() - t) * 1000  # rows are (y, x, radius)


def run_contours(img, min_area, max_area, min_circ):
    t = time.time()
    _, bw = cv2.threshold(bg_distance(img, l_weight=0.6), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = []
    for c in cnts:
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if per == 0 or not (min_area <= area <= max_area):
            continue
        circ = 4 * math.pi * area / per ** 2
        if circ >= min_circ:
            keep.append(c)
    return keep, (time.time() - t) * 1000


def draw_circles(img, blobs, color):
    out = img.copy()
    for y, x, r in blobs:
        cv2.circle(out, (int(x), int(y)), max(int(r), 2), color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--out", default="output/challenge1")
    ap.add_argument("--min-sigma", type=float, default=3)
    ap.add_argument("--max-sigma", type=float, default=40)
    ap.add_argument("--min-area", type=float, default=80)
    ap.add_argument("--max-area", type=float, default=20000)
    ap.add_argument("--min-circ", type=float, default=0.6)
    args = ap.parse_args()

    for name, img in load_images(args.images, args.filter):
        img = resize_to_width(img, 700) 
        gray = prep_gray(img)
        panels = []
        for method, color in (("LoG", (0, 0, 255)), ("DoG", (0, 200, 0)), ("DoH", (255, 0, 0))):
            blobs, ms = run_skimage(gray, method, args.min_sigma, args.max_sigma)
            p = draw_circles(img, blobs, color)
            label(p, f"{method}: {len(blobs)} ({ms:.0f} ms)", color=color)
            panels.append(p)
            print(f"{name} {method}: {len(blobs)} blobs in {ms:.0f} ms")

        cnts, ms = run_contours(img, args.min_area, args.max_area, args.min_circ)
        p = img.copy()
        cv2.drawContours(p, cnts, -1, (0, 200, 255), 2)
        label(p, f"Contours: {len(cnts)} ({ms:.0f} ms)", color=(0, 200, 255))
        panels.append(p)
        print(f"{name} Contours: {len(cnts)} blobs in {ms:.0f} ms")

        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        save(args.out, f"compare_{name}", grid)


if __name__ == "__main__":
    main()
