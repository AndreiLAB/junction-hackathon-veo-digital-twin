import sys
import csv
import json
import difflib
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
import pye57
from rapidocr_onnxruntime import RapidOCR

E57_FILE = "cloud_0.e57"
OUT = Path("labels_out")

# Known cabinet tags from VEO's twin. Each tag lists words that identify it
# (designation + function text printed on its nameplate).
TAGS = {
    "H01": ("H01 - PT1", ["H01", "PT1"]),
    "H02": ("H02 - STATION TRANSFORMER", ["H02", "STATIONTRANSFORMER"]),
    "H03": ("H03 - METERING", ["H03", "METERING"]),
    "H04": ("H04 - SOLAR 1", ["H04", "SOLAR1"]),
    "H05": ("H05 - SOLAR 2", ["H05", "SOLAR2"]),
    "VLK": ("VLK", ["VLK"]),
    "OT1": ("OT1", ["OT1"]),
    "TSK1": ("TSK1", ["TSK1"]),
    "TSK2": ("TSK2", ["TSK2"]),
    "OKK1": ("OKK1", ["OKK1"]),
}
MIN_SCORE = 0.65          # how similar OCR text must be to a known name


# ---------------------------------------------------------------- helpers
def val(node, name, default=None):
    try:
        return node[name].value() if node.isDefined(name) else default
    except Exception:
        return default


def pose(node):
    q, t = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if node.isDefined("pose"):
        p = node["pose"]
        if p.isDefined("rotation"):
            q = [val(p["rotation"], k, 0.0) for k in ("w", "x", "y", "z")]
        if p.isDefined("translation"):
            t = [val(p["translation"], k, 0.0) for k in ("x", "y", "z")]
    w, x, y, z = q
    R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
    return R, np.array(t, float)


def intrinsics(rep, W, H):
    f = val(rep, "focalLength", 0.0) / val(rep, "pixelWidth", 1.0)
    cx, cy = val(rep, "principalPointX"), val(rep, "principalPointY")
    if cx is None or cx < 1: cx = W / 2
    if cy is None or cy < 1: cy = H / 2
    return f, cx, cy


def project(cam, f, cx, cy, W, H):
    z = cam[:, 2]
    ok = z > 0.3
    u = np.full(len(cam), -1.0); v = np.full(len(cam), -1.0)
    u[ok] = f * cam[ok, 0] / z[ok] + cx
    v[ok] = f * cam[ok, 1] / z[ok] + cy
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, ok


def load_scan(e57, i):
    d = e57.read_scan(i, colors=True, transform=True, ignore_missing_fields=True)
    xyz = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
    rgb = np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], 1).astype(np.float32)
    if "cartesianInvalidState" in d:
        ok = np.asarray(d["cartesianInvalidState"]) == 0
        xyz, rgb = xyz[ok], rgb[ok]
    if rgb.max() > 255:
        rgb *= 255.0 / rgb.max()
    return xyz, rgb


def find_convention(xyz, rgb, img, R, C, f, cx, cy):
    """Which way the camera axes are stored: pick the variant whose projected
    point colours match the photo best."""
    H, W = img.shape[:2]
    sub = np.random.default_rng(0).choice(len(xyz), min(300_000, len(xyz)), replace=False)
    img_rgb = img[:, :, ::-1].astype(np.float32)
    best = None
    for inv in (False, True):
        Rc2w = R.T if inv else R
        local = (xyz[sub] - C) @ Rc2w
        for flip in itertools.product((1, -1), repeat=3):
            u, v, ok = project(local * np.array(flip), f, cx, cy, W, H)
            if ok.sum() < 2000:
                continue
            err = np.abs(img_rgb[v[ok].astype(int), u[ok].astype(int)] - rgb[sub][ok]).mean()
            if best is None or err < best[0]:
                best = (err, inv, flip)
    return best


def read_photo(images, j):
    rep = images[j]["pinholeRepresentation"]
    img = cv2.imdecode(np.asarray(rep["jpegImage"].read_buffer(), np.uint8), cv2.IMREAD_COLOR)
    return img, rep


# ---------------------------------------------------------------- text matching
CONFUSE = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8"})


def norm(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())


def match_tag(text, designations_only=False):
    """Return (tag, score) if the OCR text clearly refers to one known cabinet.
    Orange tapes carry only designations, so they skip the function words."""
    t = norm(text)
    if len(t) < 2:
        return None, 0.0
    tc = t.translate(CONFUSE)
    scores = {}
    for tag, (_, aliases) in TAGS.items():
        best = 0.0
        for a in (aliases[:1] if designations_only else aliases):
            ac = a.translate(CONFUSE)
            if tc == ac:
                s = 1.0
            elif len(ac) >= 3 and ac in tc:
                s = 0.95
            else:
                s = difflib.SequenceMatcher(None, tc, ac).ratio()
            best = max(best, s)
        scores[tag] = best
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (tag1, s1), (tag2, s2) = ranked[0], ranked[1]
    if s1 < MIN_SCORE or s1 - s2 < 0.05:
        return None, s1
    return tag1, s1


# ---------------------------------------------------------------- label finding in one photo
def orange_tapes(img):
    H, W = img.shape[:2]
    k = W / 4096.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 120, 150), (22, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 800 * k * k or w < 1.6 * h or w > 400 * k or h > 150 * k:
            continue
        boxes.append((x, y, w, h))
    return boxes


def labels_in_photo(ocr, img):
    found = []
    res, _ = ocr(img)
    for box, text, conf in (res or []):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        found.append({"text": text, "conf": float(conf), "source": "print",
                      "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
    for (x, y, w, h) in orange_tapes(img):
        crop = img[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6]
        big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        best = None
        for variant in (big, cv2.cvtColor(binar, cv2.COLOR_GRAY2BGR)):
            r, _ = ocr(variant)
            for _, t, c in (r or []):
                if best is None or float(c) > best[1]:
                    best = (t, float(c))
        if best:
            found.append({"text": best[0], "conf": best[1], "source": "orange tape",
                          "box": [int(x), int(y), int(x + w), int(y + h)]})
    return found


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    ocr = RapidOCR()
    e57 = pye57.E57(E57_FILE)
    images = e57.root["images2D"]
    n_img = images.childCount()
    scan_pos = [np.array(e57.get_header(i).translation, float) for i in range(e57.scan_count)]

    # A) camera convention, worked out once on a capture point in the middle
    conv_scan = e57.scan_count // 2
    print(f"[1/4] Working out camera convention on capture point {conv_scan} ...")
    xyz, rgb = load_scan(e57, conv_scan)
    votes, all_results = Counter(), []
    for j in range(n_img):
        R, C = pose(images[j])
        if np.linalg.norm(C - scan_pos[conv_scan]) > 0.5:
            continue
        img, rep = read_photo(images, j)
        f, cx, cy = intrinsics(rep, img.shape[1], img.shape[0])
        err, inv, flip = find_convention(xyz, rgb, img, R, C, f, cx, cy)
        all_results.append((err, inv, flip))
        if err < 25:
            votes[(inv, flip)] += 1
    if votes:
        (inv, flip), _ = votes.most_common(1)[0]
    else:
        _, inv, flip = min(all_results)
    best_err = min(e for e, i, fl in all_results if (i, fl) == (inv, flip))
    flip = np.array(flip, float)
    print(f"      convention: inverse={inv}, flips={flip.astype(int).tolist()}, colour error {best_err:.1f}")
    del xyz, rgb

    # B) read labels on every sideways photo
    print("[2/4] Reading labels on sideways photos (a few seconds per photo) ...")
    hits = []
    for j in range(n_img):
        R, C = pose(images[j])
        Rc2w = R.T if inv else R
        fwd = Rc2w @ (flip * np.array([0, 0, 1.0]))
        if abs(fwd[2]) > 0.7:
            continue                                   # looks up or down
        img, rep = read_photo(images, j)
        W = img.shape[1]
        for lab in labels_in_photo(ocr, img):
            tag, score = match_tag(lab["text"], designations_only=lab["source"] == "orange tape")
            if tag is None:
                continue
            x0, y0, x1, y1 = lab["box"]
            pad = int(0.03 * W)
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            crop_name = f"{tag}_photo{j:03d}_{x0}_{y0}.jpg"
            cv2.imwrite(str(OUT / "crops" / crop_name), crop)
            lab.update(tag=tag, match=round(score, 2), photo=j,
                       u=(x0 + x1) / 2, v=(y0 + y1) / 2, crop=crop_name)
            hits.append(lab)
            print(f"      photo {j:3d}: '{lab['text']}' ({lab['source']}, conf {lab['conf']:.2f}) -> {tag}")
        if j % 6 == 5:
            print(f"      ... {j + 1}/{n_img} photos checked, {len(hits)} label hits so far")

    # C) 3D position of each hit, using the points of the photo's own capture point
    print("[3/4] Converting label positions to 3D ...")
    by_scan = {}
    for h in hits:
        R, C = pose(images[h["photo"]])
        s = int(np.argmin([np.linalg.norm(C - p) for p in scan_pos]))
        by_scan.setdefault(s, []).append(h)
    for s, hs in sorted(by_scan.items()):
        xyz, _ = load_scan(e57, s)
        for j in sorted({h["photo"] for h in hs}):
            img, rep = read_photo(images, j)
            Hh, Ww = img.shape[:2]
            f, cx, cy = intrinsics(rep, Ww, Hh)
            R, C = pose(images[j])
            Rc2w = R.T if inv else R
            cam = ((xyz - C) @ Rc2w) * flip
            u, v, ok = project(cam, f, cx, cy, Ww, Hh)
            rad = max(4, 0.004 * Ww)
            for h in [h for h in hs if h["photo"] == j]:
                near = ok & (np.abs(u - h["u"]) < rad) & (np.abs(v - h["v"]) < rad)
                if near.sum() == 0:
                    continue
                z = cam[near, 2]
                front = z < z.min() * 1.05 + 0.03         # the visible surface, not points behind it
                h["xyz"] = np.median(xyz[near][front], axis=0).round(4).tolist()
                h["distance_m"] = round(float(np.median(z[front])), 2)
        print(f"      capture point {s}: {len(hs)} hits placed")
        del xyz

    # D) merge all sightings of the same cabinet into one tag
    print("[4/4] Merging sightings into tags ...")
    tags = []
    for tag, (display, _) in TAGS.items():
        obs = [h for h in hits if h["tag"] == tag and "xyz" in h]
        if not obs:
            continue
        P = np.array([o["xyz"] for o in obs])
        wts = np.array([o["conf"] * o["match"] for o in obs])
        support = [(wts * (np.linalg.norm(P - p, axis=1) < 0.5)).sum() for p in P]
        centre = P[int(np.argmax(support))]
        keep = np.linalg.norm(P - centre, axis=1) < 0.5
        pos = np.average(P[keep], axis=0, weights=wts[keep])
        best = max((o for o, k in zip(obs, keep) if k), key=lambda o: o["conf"] * o["match"])
        tags.append({"tag": tag, "name": display, "folder": "Cabinets",
                     "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "z": round(float(pos[2]), 3),
                     "sightings": int(keep.sum()), "confidence": round(float(wts[keep].max()), 2),
                     "photos": sorted({o["photo"] for o, k in zip(obs, keep) if k}),
                     "evidence_crop": best["crop"], "read_as": best["text"], "source": best["source"]})

    (OUT / "tags.json").write_text(json.dumps(tags, indent=2))
    (OUT / "label_hits.json").write_text(json.dumps(hits, indent=1))
    with open(OUT / "tags.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "name", "folder", "x", "y", "z", "sightings",
                                           "confidence", "read_as", "source", "photos", "evidence_crop"])
        w.writeheader()
        for t in tags:
            w.writerow({**t, "photos": " ".join(map(str, t["photos"]))})

    print("\n================ RESULT ================")
    print(f"{'tag':6s} {'X':>8s} {'Y':>8s} {'Z':>7s}  sightings  read as")
    for t in tags:
        print(f"{t['tag']:6s} {t['x']:8.3f} {t['y']:8.3f} {t['z']:7.3f}  {t['sightings']:9d}  '{t['read_as']}' ({t['source']})")
    missing = [t for t in TAGS if t not in {x['tag'] for x in tags}]
    print(f"\nFound {len(tags)} of {len(TAGS)} cabinets. Not found: {', '.join(missing) if missing else 'none'}")
    print(f"Saved {OUT}/tags.csv, {OUT}/tags.json and label crops in {OUT}/crops")


if __name__ == "__main__":
    main()