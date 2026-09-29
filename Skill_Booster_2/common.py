import cv2
import glob
import os

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_images(folder, name_filter=None):
    """Return a list of (filename, BGR image) for every image in `folder`.
    name_filter: optional substring, e.g. "dot" only loads files containing "dot"."""
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                   if p.lower().endswith(IMG_EXTS))
    if name_filter:
        paths = [p for p in paths if name_filter.lower() in os.path.basename(p).lower()]
    images = []
    for p in paths:
        img = cv2.imread(p)  # BGR order (OpenCV default), not RGB
        if img is None:
            print(f"[skip] could not read {p}")
            continue
        images.append((os.path.basename(p), img))
    if not images:
        raise SystemExit(f"No images found in '{folder}'")
    return images


def resize_to_width(img, width=900):
    """Shrink big phone photos so detection params behave consistently."""
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def save(out_dir, name, img):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img)
    print(f"  saved -> {path}")


def label(img, text, org=(10, 30), color=(0, 0, 255)):
    """Draw readable text (black outline + colored fill)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
