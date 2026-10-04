"""Run the trained YOLO model on the real photos and write the file the backend ingests (POST /detections).

    python synthetic_geo/export_detections.py --weights runs_relay/geo_synth/weights/best.pt \
        --images "VEO Images" --cameras "VEO Images/cameras.json" --out detections.json [--conf 0.25]
    python synthetic_geo/export_detections.py --selftest          # logic test with a stub model, no weights/GPU needed

Pipeline per 4096x4096 photo: tile (1280 px, stride 960, native resolution) -> YOLO -> boxes back to FULL-image pixels ->
per-class NMS -> location gate (camera pose says where a relay must be) -> backend format.

Outputs
  detections.json         what the backend takes in (POST /detections): a list of
                          {"image","class","conf","box":[x,y,w,h],"ocr":[],"ocr_conf":null,"position":{x,y,z}|null}
                          * classes "abb_relion_615" and "vd4_breaker_window" (other_hmi is evidence for rejection, never a device);
                          * box = x, y of the top-left corner, width, height, in full-image pixels;
                          * position = the detection's centre ray intersected with the relay front plane of the cabinet the gate
                            assigned (a measurement from the detection, not the prediction); null when the gate rejected it,
                            so the backend flags it needs_review ("no position");
                          * low-confidence and rejected detections are KEPT (the backend flags them, nothing is silently dropped).
  detections_detail.json  the same plus cabinet, gate decision/reason, IoU and the other_hmi detections (for the report / debugging).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_geo_synthetic as G  # noqa: E402
from geometry import door_normal, load_registry  # noqa: E402
from location_gate import Gate  # noqa: E402

TILE, STRIDE, FRAME = 1280, 960, 4096
REL_CLASS = "abb_relion_615"
VD4_CLASS = "vd4_breaker_window"      # exported as a device too (backend: VD4 circuit breaker + VD4 manual)
EXPORT_CLASSES = (REL_CLASS, VD4_CLASS)


def tile_origins(size=FRAME, tile=TILE, stride=STRIDE):
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return [(x, y) for y in xs for x in xs]


def nms(boxes, scores, iou_thr=0.5):
    """Plain numpy NMS on [x0,y0,x1,y1]. Returns kept indices, best score first."""
    if not len(boxes):
        return []
    b, order, keep = np.asarray(boxes, float), np.argsort(-np.asarray(scores)), []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        r = order[1:]
        ix = np.maximum(0, np.minimum(b[i, 2], b[r, 2]) - np.maximum(b[i, 0], b[r, 0]))
        iy = np.maximum(0, np.minimum(b[i, 3], b[r, 3]) - np.maximum(b[i, 1], b[r, 1]))
        inter = ix * iy
        union = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[r, 2] - b[r, 0]) * (b[r, 3] - b[r, 1]) - inter
        order = r[inter / np.maximum(union, 1e-9) <= iou_thr]
    return keep


def yolo_predictor(weights, conf, device=None):
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names

    def predict(tiles):  # tiles: list of HxWx3 BGR arrays -> per tile list of (class_name, conf, [x0,y0,x1,y1])
        out = []
        for r in model.predict(tiles, imgsz=TILE, conf=conf, device=device, verbose=False):
            out.append([(names[int(c)], float(s), [float(v) for v in b]) for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())])
        return out
    return predict


def detect_image(img, predict):
    """Tile the photo, predict, map to full-image pixels, per-class NMS. -> [(class, conf, xyxy)]"""
    raw = []
    origins = tile_origins(img.shape[1] if img is not None else FRAME)
    for k in range(0, len(origins), 4):
        chunk = origins[k:k + 4]
        tiles = [img[y:y + TILE, x:x + TILE] for x, y in chunk]
        for (x, y), dets in zip(chunk, predict(tiles)):
            raw += [(c, s, [b[0] + x, b[1] + y, b[2] + x, b[3] + y]) for c, s, b in dets]
    out = []
    for cls in {c for c, _, _ in raw}:
        sel = [r for r in raw if r[0] == cls]
        out += [sel[i] for i in nms([r[2] for r in sel], [r[1] for r in sel])]
    return out


def front_plane_position(cam, cabinet, xyxy, reg, front=None):
    """Centre ray of the box intersected with the asset plane of that cabinet: x = cabinet x + n_x * front
    (front = the relay front plane, 0.07 m, by default; 0 for assets on the door plane such as the VD4 window)."""
    n = door_normal(reg[cabinet]["position"])
    front = G.GEO["relay"]["front_plane_toward_corridor_m"] if front is None else front
    xp = reg[cabinet]["position"]["x"] + n[0] * front
    o, d = cam.ray((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
    if abs(d[0]) < 1e-6:
        return None
    t = (xp - o[0]) / d[0]
    if t <= 0:
        return None
    p = o + t * d
    return {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "z": round(float(p[2]), 3)}


def vd4_expected(cam, reg):
    """Predicted VD4 window boxes (measured geometry in relay_geometry.json: vd4_breaker_window). Only cabinets listed in `present_on`:
    a window on a cabinet where none is expected (H03 has a warning triangle there; H01 is unmeasured) gets no expectation."""
    v = G.GEO.get("vd4_breaker_window")
    out = []
    for cid in (v or {}).get("present_on", []):
        P = reg[cid]["position"]
        y, z, w, h = P["y"] + v["offset_from_nameplate_m"]["dy"], P["z"] + v["offset_from_nameplate_m"]["dz"], v["size_m"]["w"], v["size_m"]["h"]
        pr = [cam.project(np.array(p)) for p in [(P["x"], y - w / 2, z + h / 2), (P["x"], y + w / 2, z + h / 2), (P["x"], y + w / 2, z - h / 2), (P["x"], y - w / 2, z - h / 2)]]
        if any(p is None or p[2] < 0.3 for p in pr):
            continue
        xs, ys = [p[0] for p in pr], [p[1] for p in pr]
        if max(xs) < 0 or max(ys) < 0 or min(xs) > FRAME or min(ys) > FRAME:
            continue
        out.append({"cabinet": cid, "xyxy": [min(xs), min(ys), max(xs), max(ys)]})
    return out


def assign_vd4(cam, reg, dets, min_iou=0.30, size_ratio=(0.6, 1.6)):
    """Same rule as the relay gate (IoU >= 0.30, width ratio 0.6-1.6), against the predicted VD4 windows."""
    from location_gate import iou
    exp, used, out = vd4_expected(cam, reg), set(), []
    for c, s, b in sorted(dets, key=lambda d: -d[1]):
        best, bi = None, 0.0
        for e in exp:
            if e["cabinet"] in used:
                continue
            i = iou(b, e["xyxy"])
            ratio = (b[2] - b[0]) / max(1e-6, e["xyxy"][2] - e["xyxy"][0])
            if i > bi and size_ratio[0] <= ratio <= size_ratio[1]:
                best, bi = e, i
        ok = bool(best) and bi >= min_iou
        if ok:
            used.add(best["cabinet"])
        out.append({"class": c, "conf": s, "xyxy": b, "cabinet": best["cabinet"] if ok else None, "accepted": ok, "iou": round(bi, 2) if ok else None,
                    "reason": "matches predicted VD4 window position and size" if ok else "no expected VD4 window here (absent on H03, unmeasured on H01) or wrong size: needs_review"})
    return out


def build_records(image_name, dets, gate, reg, use_gate=True):
    """YOLO detections of one photo -> (backend records, detail records)."""
    relays = [(c, s, b) for c, s, b in dets if c == REL_CLASS]
    vd4 = [(c, s, b) for c, s, b in dets if c == VD4_CLASS]
    others = [(c, s, b) for c, s, b in dets if c not in EXPORT_CLASSES]
    gated = gate.assign(image_name, [{"class": c, "conf": s, "xyxy": b} for c, s, b in relays]) if use_gate else \
        [{"class": c, "conf": s, "xyxy": b, "cabinet": None, "accepted": False, "reason": "gate disabled"} for c, s, b in relays]
    cam = gate.cams[image_name]
    backend, detail = [], []
    for g in gated:
        b = g["xyxy"]
        pos = front_plane_position(cam, g["cabinet"], b, reg) if g["accepted"] else None
        backend.append({"image": image_name, "class": REL_CLASS, "conf": round(float(min(max(g["conf"], 0.0), 1.0)), 4),
                        "box": [round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1)],
                        "ocr": [], "ocr_conf": None, "position": pos})
        detail.append({**backend[-1], "cabinet": g.get("cabinet"), "gate_accepted": g["accepted"], "gate_reason": g["reason"], "gate_iou": g.get("iou")})
    cam_ = gate.cams[image_name]
    for g in (assign_vd4(cam_, reg, vd4) if use_gate else [{"class": c, "conf": s, "xyxy": b, "cabinet": None, "accepted": False, "reason": "gate disabled"} for c, s, b in vd4]):
        b = g["xyxy"]
        pos = front_plane_position(cam_, g["cabinet"], b, reg, front=0.0) if g["accepted"] else None
        backend.append({"image": image_name, "class": VD4_CLASS, "conf": round(float(min(max(g["conf"], 0.0), 1.0)), 4),
                        "box": [round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1)], "ocr": [], "ocr_conf": None, "position": pos})
        detail.append({**backend[-1], "cabinet": g.get("cabinet"), "gate_accepted": g["accepted"], "gate_reason": g["reason"], "gate_iou": g.get("iou")})
    for c, s, b in others:
        detail.append({"image": image_name, "class": c, "conf": round(float(s), 4), "box": [round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1)],
                       "cabinet": None, "gate_accepted": False, "gate_reason": "look-alike class (other_hmi), not exported"})
    return backend, detail


def validate(records):
    """Fail loudly if a record would be refused by the backend (same rules as backend.main.DetectionIn)."""
    for r in records:
        assert r["class"] in EXPORT_CLASSES and isinstance(r["image"], str)
        assert 0 <= r["conf"] <= 1 and len(r["box"]) == 4 and r["box"][2] > 0 and r["box"][3] > 0
        assert r["ocr"] == [] and r["ocr_conf"] is None
        assert r["position"] is None or set(r["position"]) == {"x", "y", "z"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--images", default="VEO Images")
    ap.add_argument("--cameras", default="VEO Images/cameras.json")
    ap.add_argument("--out", default="detections.json")
    ap.add_argument("--conf", type=float, default=0.25, help="low on purpose: the backend flags low confidence instead of dropping")
    ap.add_argument("--device")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.weights:
        sys.exit("--weights is required (path to best.pt)")
    import cv2
    gate, reg, predict = Gate(a.cameras), load_registry(), yolo_predictor(a.weights, a.conf, a.device)
    backend, detail = [], []
    for name in sorted(gate.cams):
        img = cv2.imread(str(Path(a.images) / name))
        if img is None:
            sys.exit(f"cannot read {Path(a.images) / name}")
        b, d = build_records(name, detect_image(img, predict), gate, reg, not a.no_gate)
        backend, detail = backend + b, detail + d
        print(f"{name}: {len(b)} relay detections ({sum(1 for r in b if r['position'])} placed by the gate)", flush=True)
    validate(backend)
    Path(a.out).write_text(json.dumps(backend, indent=1))
    Path(a.out).with_name(Path(a.out).stem + "_detail.json").write_text(json.dumps(detail, indent=1))
    print(f"wrote {a.out}: {len(backend)} detections, {sum(1 for r in backend if r['position'])} with a position")


def selftest():
    """No weights needed: a stub 'model' that finds the expected relay in img_067 and a Cerdex-like false positive."""
    gate, reg = Gate(str(HERE.parent.parent / "VEO Images" / "cameras.json") if (HERE.parent.parent / "VEO Images").exists() else "VEO Images/cameras.json"), load_registry()
    name = "img_067.jpg"
    exp = gate.expected(name)[0]
    x0, y0, x1, y1 = exp["xyxy"]
    vexp = vd4_expected(gate.cams[name], reg)[0]["xyxy"]
    full = [("vd4_breaker_window", 0.85, list(vexp)), ("vd4_breaker_window", 0.6, [vexp[0] + 900, vexp[1] - 700, vexp[2] + 900, vexp[3] - 700]),
            ("abb_relion_615", 0.9, [x0, y0, x1, y1]), ("abb_relion_615", 0.8, [x0 + 1500, y0 + 900, x1 + 1500, y1 + 900]), ("other_hmi", 0.7, [100, 100, 300, 250])]
    # tile it exactly like the real run, so tiling + NMS + coordinate mapping are exercised
    origins, calls = tile_origins(), []

    def predict(tiles):
        out = []
        for t in tiles:
            k = len(calls)
            calls.append(k)
            ox, oy = origins[k]
            out.append([(c, s, [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]) for c, s, b in full if ox <= (b[0] + b[2]) / 2 < ox + TILE and oy <= (b[1] + b[3]) / 2 < oy + TILE])
        return out
    dets = detect_image(np.zeros((FRAME, FRAME, 3), np.uint8), predict)
    backend, detail = build_records(name, dets, gate, reg)
    validate(backend)
    print(json.dumps(backend, indent=1))
    ok = [r for r in backend if r["position"]]
    assert len(backend) == 4 and len(ok) == 2, "expected 2 relay + 2 VD4 detections, one of each accepted"
    assert sorted(r["class"] for r in ok) == ["abb_relion_615", "vd4_breaker_window"]
    cab = [d for d in detail if d["gate_accepted"]][0]["cabinet"]
    print(f"selftest OK: relay and VD4 window accepted on {cab} (positions {[r['position'] for r in ok]}); displaced ones have position null; other_hmi not exported")


if __name__ == "__main__":
    main()
