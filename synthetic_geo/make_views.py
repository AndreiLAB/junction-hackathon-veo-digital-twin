"""Which cabinets does each image see?  Writes image_cabinet_view.json (cameras.json + a 'view' block per image).

    python synthetic_geo/make_views.py --cameras "VEO Images/cameras.json" --out synthetic_geo/image_cabinet_view.json

A cabinet counts as visible when its nameplate/tape position projects inside the image (with a margin), is
between MIN_DEPTH and MAX_DEPTH metres in front of the camera, and the camera is on the door side
(viewing angle below MAX_ANGLE). Occlusion by walls is NOT checked. Images with no visible cabinet get
category "none" and must produce no tag.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import door_normal, load_cameras, load_registry  # noqa: E402

MIN_DEPTH, MAX_DEPTH = 0.4, 8.0     # metres
MARGIN = 150                        # px from the image border the label must stay inside
MAX_ANGLE = 70.0                    # degrees between the door normal and the direction to the camera
CORRIDOR_X = (-5.5, -2.3)           # camera must stand between the two cabinet rows (scan 1 and scans 17-18 are behind walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="VEO Images/cameras.json")
    ap.add_argument("--out", default="synthetic_geo/image_cabinet_view.json")
    a = ap.parse_args()

    reg = load_registry()
    raw = json.load(open(a.cameras, encoding="utf-8"))
    cams = load_cameras(a.cameras)
    out, cats = [], Counter()
    for entry, cam in zip(raw, cams):
        visible = []
        if not (CORRIDOR_X[0] <= cam.C[0] <= CORRIDOR_X[1]):
            cats['none'] += 1
            out.append({**entry, 'scan_index': int(entry['file'].split('_')[1].split('.')[0]) // 6 + 1,
                        'view': {'category': 'none', 'cabinets': [], 'note': 'camera outside the corridor'}})
            continue
        for cid, c in reg.items():
            P = np.array([c["position"]["x"], c["position"]["y"], c["position"]["z"]])
            pr = cam.project(P)
            if pr is None:
                continue
            u, v, d = pr
            if not (MIN_DEPTH <= d <= MAX_DEPTH and MARGIN <= u <= cam.W - MARGIN and MARGIN <= v <= cam.H - MARGIN):
                continue
            to_cam = (cam.C - P) / np.linalg.norm(cam.C - P)
            ang = float(np.degrees(np.arccos(np.clip(door_normal(c["position"]) @ to_cam, -1, 1))))
            if ang > MAX_ANGLE:
                continue
            visible.append({"cabinet": cid, "u": round(u, 1), "v": round(v, 1), "depth_m": round(d, 3),
                            "view_angle_deg": round(ang, 1), "panel_model": c["panel_model"],
                            "expects_relay": c["expects_relay"]})
        visible.sort(key=lambda r: r["depth_m"])
        cat = "none" if not visible else "single" if len(visible) == 1 else "multiple"
        cats[cat] += 1
        scan = int(entry["file"].split("_")[1].split(".")[0]) // 6 + 1
        out.append({**entry, "scan_index": scan, "view": {"category": cat, "cabinets": visible}})
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(out)} images -> {dict(cats)}")
    per = Counter(c["cabinet"] for e in out for c in e["view"]["cabinets"])
    print("images per cabinet:", dict(sorted(per.items())))
    h = [e["file"] for e in out if any(c["expects_relay"] for c in e["view"]["cabinets"])]
    print(f"images showing at least one H01-H05 cabinet: {len(h)}")
    print("TSK1 images:", [e["file"] for e in out if any(c["cabinet"] == "TSK1" for c in e["view"]["cabinets"])])


if __name__ == "__main__":
    main()
