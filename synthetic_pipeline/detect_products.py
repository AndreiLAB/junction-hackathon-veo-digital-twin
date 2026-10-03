"""
Find products on the cabinets and place them in 3D.

Two sources of evidence, both from the sideways scan photos:
  1. the relay detector (YOLO model trained on synthetic data) -> "ABB 615"
  2. OCR product words printed on the cabinets: "UniGear", "VD4", "REX615"

Writes labels_out/products.json, which build_tags.py uses to pick manuals.

  python detect_products.py --model C:/Users/bhand/runs/detect/relay615/weights/best.pt
  python detect_products.py --no-model          # OCR words only
"""
import argparse
import json
from collections import Counter

import numpy as np
import cv2
import pye57
from rapidocr_onnxruntime import RapidOCR

import find_labels as FL   # reuse the tested scan helpers (same folder)

# OCR words that identify a product -> product name used in the knowledge base
PRODUCT_WORDS = {
    "UNIGEAR": "UniGear ZS2",
    "VD4": "VD4",
    "REX615": "ABB 615",
}
DETECTOR_PRODUCT = {"abb_relion_615": "ABB 615"}   # detector class -> product


def camera_convention(e57, images, scan_pos):
    """Same procedure as find_labels.py: test the axis variants on the middle capture point."""
    conv_scan = e57.scan_count // 2
    xyz, rgb = FL.load_scan(e57, conv_scan)
    votes, results = Counter(), []
    for j in range(images.childCount()):
        R, C = FL.pose(images[j])
        if np.linalg.norm(C - scan_pos[conv_scan]) > 0.5:
            continue
        img, rep = FL.read_photo(images, j)
        f, cx, cy = FL.intrinsics(rep, img.shape[1], img.shape[0])
        err, inv, flip = FL.find_convention(xyz, rgb, img, R, C, f, cx, cy)
        results.append((err, inv, flip))
        if err < 25:
            votes[(inv, flip)] += 1
    (inv, flip) = votes.most_common(1)[0][0] if votes else min(results)[1:]
    return inv, np.array(flip, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e57", default="cloud_0.e57")
    ap.add_argument("--model", help="path to best.pt from training")
    ap.add_argument("--no-model", action="store_true", help="skip the relay detector")
    ap.add_argument("--no-ocr", action="store_true", help="skip product words")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="labels_out/products.json")
    args = ap.parse_args()

    detector = None
    if args.model and not args.no_model:
        from ultralytics import YOLO
        detector = YOLO(args.model)
    ocr = None if args.no_ocr else RapidOCR()
    if detector is None and ocr is None:
        raise SystemExit("Nothing to do: give --model or allow OCR.")

    e57 = pye57.E57(args.e57)
    images = e57.root["images2D"]
    scan_pos = [np.array(e57.get_header(i).translation, float) for i in range(e57.scan_count)]
    print("[1/3] Working out camera convention ...")
    inv, flip = camera_convention(e57, images, scan_pos)

    print("[2/3] Looking for products on sideways photos ...")
    finds = []
    for j in range(images.childCount()):
        R, C = FL.pose(images[j])
        Rc2w = R.T if inv else R
        if abs((Rc2w @ (flip * np.array([0, 0, 1.0])))[2]) > 0.7:
            continue                                     # looks up or down
        img, _ = FL.read_photo(images, j)
        if detector is not None:
            res = detector.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                product = DETECTOR_PRODUCT.get(detector.names[int(cls)])
                if product:
                    x0, y0, x1, y1 = box
                    finds.append({"product": product, "source": "detector", "text": detector.names[int(cls)],
                                  "conf": round(float(conf), 3), "photo": j,
                                  "box": [int(x0), int(y0), int(x1), int(y1)],
                                  "u": (x0 + x1) / 2, "v": (y0 + y1) / 2})
        if ocr is not None:
            r, _ = ocr(img)
            for quad, text, conf in (r or []):
                t = FL.norm(text)
                for word, product in PRODUCT_WORDS.items():
                    if word in t:
                        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
                        finds.append({"product": product, "source": "ocr", "text": text,
                                      "conf": round(float(conf), 3), "photo": j,
                                      "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                      "u": (min(xs) + max(xs)) / 2, "v": (min(ys) + max(ys)) / 2})
        n_here = sum(1 for f in finds if f["photo"] == j)
        if n_here:
            print(f"      photo {j:3d}: {n_here} product finds")

    print("[3/3] Converting to 3D ...")
    by_scan = {}
    for f in finds:
        R, C = FL.pose(images[f["photo"]])
        s = int(np.argmin([np.linalg.norm(C - p) for p in scan_pos]))
        by_scan.setdefault(s, []).append(f)
    for s, fs in sorted(by_scan.items()):
        xyz, _ = FL.load_scan(e57, s)
        for j in sorted({f["photo"] for f in fs}):
            img, rep = FL.read_photo(images, j)
            H, W = img.shape[:2]
            fx, cx, cy = FL.intrinsics(rep, W, H)
            R, C = FL.pose(images[j])
            Rc2w = R.T if inv else R
            cam = ((xyz - C) @ Rc2w) * flip
            u, v, ok = FL.project(cam, fx, cx, cy, W, H)
            rad = max(4, 0.004 * W)
            for f in [f for f in fs if f["photo"] == j]:
                near = ok & (np.abs(u - f["u"]) < rad) & (np.abs(v - f["v"]) < rad)
                if near.sum() == 0:
                    continue
                z = cam[near, 2]
                front = z < z.min() * 1.05 + 0.03
                f["xyz"] = np.median(xyz[near][front], axis=0).round(4).tolist()
        del xyz

    placed = [f for f in finds if "xyz" in f]
    with open(args.out, "w") as fh:
        json.dump(placed, fh, indent=1)
    print(f"\nPlaced {len(placed)} of {len(finds)} product finds -> {args.out}")
    for p, n in Counter(f["product"] for f in placed).most_common():
        print(f"  {p:12s} {n} sightings")


if __name__ == "__main__":
    main()