"""Draw candidate asset boxes on the real photos of a cabinet, to VERIFY geometry before generating a dataset.

    python synthetic_geo/verify_boxes.py --cabinet OT1 --images "VEO Images" --cameras "VEO Images/cameras.json" \
        --rect "A:-0.155:0.145:0.177:0.177" --rect "B:-0.155:-0.160:0.177:0.177" --front 0.035 --n 8 --out ot1_check.jpg

Each --rect is  name:dy:dz:width_m:height_m  where (dy, dz) is the box centre relative to the cabinet position in the registry (metres, world axes),
width/height are the asset frame size. --front is how far the front plane stands out from the door plane towards the corridor (metres).
Several --front values can be given (e.g. --front 0 --front 0.035 --front 0.07); each gets its own colour, so the best fit is visible in oblique views.
The cabinet's images are those listed in image_cabinet_view.json, spread from near to far. Acceptance rule: the box must cover the real asset tightly in at
least 8 photos, including oblique ones (angle > 45 degrees). Look at the sheet; this is a visual check, not a metric.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import door_normal, load_cameras, load_registry  # noqa: E402

COLOURS = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255)]   # BGR
NAMES = ["green", "yellow", "blue", "red", "magenta"]            # same order, for the legend


def quad(cam, P, n, dy, dz, w, h, front):
    x = P["x"] + n[0] * front
    pts = [(x, P["y"] + dy - w / 2, P["z"] + dz + h / 2), (x, P["y"] + dy + w / 2, P["z"] + dz + h / 2),
           (x, P["y"] + dy + w / 2, P["z"] + dz - h / 2), (x, P["y"] + dy - w / 2, P["z"] + dz - h / 2)]
    pr = [cam.project(np.array(p)) for p in pts]
    return None if any(p is None for p in pr) else np.array([[p[0], p[1]] for p in pr], np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cabinet", required=True)
    ap.add_argument("--images", default="VEO Images")
    ap.add_argument("--cameras", default="VEO Images/cameras.json")
    ap.add_argument("--rect", action="append", required=True, help="name:dy:dz:width_m:height_m")
    ap.add_argument("--front", action="append", type=float, help="front-plane offset in m (default 0)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="verify_boxes.jpg")
    ap.add_argument("--tile", type=int, default=420)
    a = ap.parse_args()
    fronts = a.front or [0.0]
    rects = []
    for r in a.rect:
        name, dy, dz, w, h = r.split(":")
        rects.append((name, float(dy), float(dz), float(w), float(h)))
    reg = load_registry()
    if a.cabinet not in reg:
        sys.exit(f"unknown cabinet {a.cabinet}; registry has {list(reg)}")
    P, n = reg[a.cabinet]["position"], door_normal(reg[a.cabinet]["position"])
    cams = {c.file: c for c in load_cameras(a.cameras)}
    view = json.load(open(HERE / "image_cabinet_view.json", encoding="utf-8"))
    rows = sorted([(c["depth_m"], c["view_angle_deg"], e["file"]) for e in view for c in e["view"]["cabinets"] if c["cabinet"] == a.cabinet])
    if not rows:
        sys.exit(f"no photo sees {a.cabinet}")
    idx = np.unique(np.linspace(0, len(rows) - 1, min(a.n, len(rows))).round().astype(int))
    tiles = []
    for k in idx:
        depth, ang, f = rows[k]
        cam = cams[f]
        im = cv2.imread(str(Path(a.images) / f))
        if im is None:
            sys.exit(f"cannot read {Path(a.images) / f}")
        t = max(2, im.shape[0] // 600)
        for fi, front in enumerate(fronts):
            for name, dy, dz, w, h in rects:
                q = quad(cam, P, n, dy, dz, w, h, front)
                if q is not None:
                    cv2.polylines(im, [q], True, COLOURS[fi % len(COLOURS)], t)
        u, v, Z = cam.project(np.array([P["x"], P["y"], P["z"]]))
        r = int(2048 * 0.5 / Z)
        xa, ya = max(0, int(u - r)), max(0, int(v - r))
        c = cv2.resize(im[ya:ya + 2 * r, xa:xa + 2 * r], (a.tile, a.tile))
        cv2.putText(c, f"{f} {depth:.1f}m {ang:.0f}deg", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        tiles.append(c)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.putText(sheet, "front plane m: " + "  ".join(f"{f}={NAMES[i % 5]}" for i, f in enumerate(fronts)), (6, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {a.out}: {len(idx)} photos of {a.cabinet} (depth {rows[idx[0]][0]:.1f}-{rows[idx[-1]][0]:.1f} m), rects {[r[0] for r in rects]}, front planes {fronts}")


if __name__ == "__main__":
    main()
