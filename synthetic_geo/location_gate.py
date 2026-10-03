"""Location-aware gate: accept a detection only if it is where the geometry says a relay must be.

The detector (YOLO) only sees pixels. Given the camera pose of the image, the cabinet registry and the measured
relay geometry, this module predicts WHERE and HOW BIG each H01-H05 relay must appear, then:
  * accepts a detection that matches a predicted relay in position and size -> returns the cabinet it belongs to,
  * rejects/flags one that matches nothing (look-alike HMIs such as the Cerdex displays of TSK1/TSK2,
    objects on a cabinet that does not expect a relay, anything outside a known cabinet).

    from location_gate import Gate
    gate = Gate("VEO Images/cameras.json")
    gate.expected("img_067.jpg")                              # predicted relay boxes {cabinet, xyxy, depth_m}
    gate.assign("img_067.jpg", [{"class": "abb_relion_615", "conf": 0.9, "xyxy": [635, 546, 1468, 1126]}])

Boxes are in FULL-IMAGE pixels (4096 x 4096). If the detector ran on tiles, add the tile origin first.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_geo_synthetic as G  # noqa: E402  (reuses the exact geometry used to build the dataset)
from geometry import load_cameras  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Gate:
    def __init__(self, cameras_json, min_iou=0.30, size_ratio=(0.6, 1.6), frame=(4096, 4096)):
        self.cams = {c.file: c for c in load_cameras(cameras_json)}
        self.min_iou, self.size_ratio, self.frame = min_iou, size_ratio, frame

    def expected(self, image_name):
        cam = self.cams[image_name]
        out = []
        for cid in ("H01", "H02", "H03", "H04", "H05"):
            g = G.relay_geometry(cam, cid)
            if g is None or g["depth"] > G.MAX_RELAY_DEPTH:
                continue
            x0, y0, x1, y1 = g["bbox"]
            if x1 < 0 or y1 < 0 or x0 > self.frame[0] or y0 > self.frame[1]:
                continue
            out.append({"cabinet": cid, "xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "depth_m": round(g["depth"], 2)})
        return out

    def assign(self, image_name, detections):
        """Each detection {class, conf, xyxy} -> same dict + 'cabinet' (or None) + 'accepted' + 'reason'."""
        exp = self.expected(image_name)
        used, out = set(), []
        for d in sorted(detections, key=lambda d: -d.get("conf", 0)):
            r = dict(d, cabinet=None, accepted=False)
            if d["class"] != "abb_relion_615":
                r["reason"] = f"class {d['class']} is not a relay"
                out.append(r)
                continue
            w = d["xyxy"][2] - d["xyxy"][0]
            best, bi = None, 0.0
            for e in exp:
                if e["cabinet"] in used:
                    continue
                i = iou(d["xyxy"], e["xyxy"])
                ratio = w / max(1e-6, e["xyxy"][2] - e["xyxy"][0])
                if i > bi and self.size_ratio[0] <= ratio <= self.size_ratio[1]:
                    best, bi = e, i
            if best and bi >= self.min_iou:
                used.add(best["cabinet"])
                r.update(cabinet=best["cabinet"], accepted=True, iou=round(bi, 2), reason="matches predicted relay position and size")
            else:
                r["reason"] = "no expected relay at this position/size (look-alike or unknown cabinet): needs_review"
            out.append(r)
        return out


if __name__ == "__main__":      # self-test of the logic: a detection exactly where one is expected is accepted, a displaced one is not
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="VEO Images/cameras.json")
    ap.add_argument("--image", default="img_067.jpg")
    a = ap.parse_args()
    gate = Gate(a.cameras)
    exp = gate.expected(a.image)
    print("expected:", json.dumps(exp))
    if exp:
        e = exp[0]["xyxy"]
        ok = {"class": "abb_relion_615", "conf": 0.9, "xyxy": e}
        far = {"class": "abb_relion_615", "conf": 0.8, "xyxy": [e[0] + 1500, e[1] + 900, e[2] + 1500, e[3] + 900]}
        hmi = {"class": "other_hmi", "conf": 0.9, "xyxy": e}
        for r in gate.assign(a.image, [ok, far, hmi]):
            print(r["class"], "accepted" if r["accepted"] else "REJECTED", r["cabinet"], "|", r["reason"])
