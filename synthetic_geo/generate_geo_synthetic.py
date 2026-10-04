"""Geometry-driven synthetic dataset for the ABB REX615 relay (YOLO format).

Every sample is a 1280x1280 tile of a REAL cube-face photo (VEO Images) in which
  * the real relay(s) of H01-H05 are removed (inpainted) and replaced by the reference render placed at the
    MEASURED 3D position (relay_geometry.json) through the camera pose (cameras.json): position, size and
    perspective are determined by geometry, not random;
  * the real Cerdex displays of TSK1/TSK2 stay in place and are labelled `other_hmi` (look-alike negatives);
  * appearance is matched to the photo: colour gain, sharpness and noise are measured on the real relay /
    surrounding door, then applied to the render.
Only small deterministic variations are used (tile placement grid, LCD content, +-3 mm / +-0.3 deg pose error,
brightness, JPEG quality). No flips, no random rotation, no random sizes.

    python synthetic_geo/generate_geo_synthetic.py --images "VEO Images" --cameras "VEO Images/cameras.json" \
        --out synthetic_geo_dataset [--dry-run] [--limit N]

Classes: 0 abb_relion_615, 1 other_hmi.  Train/val are split by SCAN (held-out scans never feed training).
"""
import argparse
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import door_normal, load_cameras, load_registry  # noqa: E402

GEO = json.load(open(HERE / "relay_geometry.json", encoding="utf-8"))
REG = load_registry()
RENDER = HERE.parent / "synthetic_pipeline" / "refs" / "9PAA00000215623_master.jpg"
TILE = 1280
HOLDOUT_SCANS = {6, 8, 13}
MAX_RELAY_DEPTH = 6.2
GRID = [(0.5, 0.5), (0.35, 0.5), (0.65, 0.5), (0.5, 0.35), (0.5, 0.65), (0.35, 0.35), (0.65, 0.65), (0.35, 0.65), (0.65, 0.35)]
POSE_JIT = [(0, 0, 0), (3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0), (3, 3, 0.3), (-3, -3, -0.3), (3, -3, 0.3), (-3, 3, -0.3)]  # mm, mm, deg
JPEG_Q = [88, 92, 95]
BRIGHT = [0.97, 1.0, 1.03]
BAND_VARIANTS = {"near": 24, "mid": 24, "far": 32}   # variants per (image, cabinet) pair, by camera-relay distance
EMPTY_TILES = {"train": 150, "val": 12}


# ------------------------------------------------------------------ render texture
def load_render():
    im = cv2.imread(str(RENDER))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    m = cv2.morphologyEx((g < 240).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    x, y, w, h = cv2.boundingRect(m)
    tex = im[y:y + h, x:x + w].copy()
    alpha = cv2.erode(m[y:y + h, x:x + w], np.ones((5, 5), np.uint8)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0)
    return tex, alpha


def lcd_variants(tex):
    """The reference LCD ('SLD page 1/2') plus two deterministic alternatives; LCD box measured on the render."""
    h, w = tex.shape[:2]
    x0, y0, x1, y1 = int(w * 0.479), int(h * 0.171), int(w * 0.725), int(h * 0.535)
    out = [tex]
    blank = tex.copy()
    blank[y0:y1, x0:x1] = np.median(tex[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(np.uint8)
    out.append(blank)
    meas = blank.copy()
    for k, t in enumerate(["IL1 = 12 A", "IL2 = 11 A", "IL3 = 12 A", "U12 = 0.4 kV", "f = 50.00 Hz"]):
        cv2.putText(meas, t, (x0 + 40, y0 + 90 + k * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (60, 50, 40), 4, cv2.LINE_AA)
    out.append(meas)
    return out


# ------------------------------------------------------------------ geometry
def right_vec(cam, p):
    a, b = cam.project(p), cam.project(p + np.array([0, 0.1, 0]))
    if a is None or b is None:
        return None
    return np.array([0.0, 1.0 if b[0] > a[0] else -1.0, 0.0])


def relay_geometry(cam, cid, jit=(0, 0, 0)):
    if cid not in GEO.get("cabinets", GEO.get("relay_offset_from_nameplate_m")): return []
    out = []
    P = np.array([REG[cid]["position"][a] for a in "xyz"])
    n = door_normal(REG[cid]["position"])
    
    c_geo = GEO.get("cabinets", {}).get(cid, {})
    relays = c_geo.get("relays", [])
    if not relays:
        # Fallback to old format if using unpatched geometry
        off = GEO.get("relay_offset_from_nameplate_m", {}).get(cid)
        if off:
            relays = [{"dy": off["dy"], "dz": off["dz"], "width_m": GEO["relay"]["frame_width_m"], "height_m": GEO["relay"]["frame_height_m"]}]
            
    for i, rel in enumerate(relays):
        c = P + np.array([0, rel["dy"] + jit[0] / 1000, rel["dz"] + jit[1] / 1000]) + n * rel.get("front_plane_m", GEO["relay"]["front_plane_toward_corridor_m"])
        if float(n @ (cam.C - c)) <= 0:
            continue
        r, up = right_vec(cam, c), np.array([0, 0, 1.0])
        if r is None:
            continue
        ang = np.radians(jit[2])
        r, up = r * np.cos(ang) + up * np.sin(ang), up * np.cos(ang) - r * np.sin(ang)
        w, h, d = rel["width_m"], rel["height_m"], rel.get("front_plane_m", GEO["relay"]["body_depth_m"])
        front = [c - r * w / 2 + up * h / 2, c + r * w / 2 + up * h / 2, c + r * w / 2 - up * h / 2, c - r * w / 2 - up * h / 2]
        back = [p - n * d for p in front]
        pr = [cam.project(p) for p in front + back]
        if any(p is None or p[2] < 0.3 for p in pr):
            continue
        pts = np.array([[p[0], p[1]] for p in pr])
        out.append({"cabinet": cid, "relay_idx": i, "front": pts[:4], "back": pts[4:], "depth": pr[0][2],
                "bbox": (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())})
    return out


def cerdex_geometry(cam, cid):
    d = GEO["other_hmi"]
    if cid not in d:
        return None
    xp = REG[cid]["position"]["x"]
    r = right_vec(cam, np.array([xp, d[cid]["y"], d[cid]["z"]]))
    if r is None:
        return None
    c = np.array([xp, d[cid]["y"], d[cid]["z"]])
    w, h = d["width_m"], d["height_m"]
    pts = [c - r * w / 2 + [0, 0, h / 2], c + r * w / 2 + [0, 0, h / 2], c + r * w / 2 - [0, 0, h / 2], c - r * w / 2 - [0, 0, h / 2]]
    pr = [cam.project(p) for p in pts]
    n = door_normal(REG[cid]["position"])
    if any(p is None or p[2] < 0.3 for p in pr) or float(n @ (cam.C - c)) <= 0:
        return None
    pts2 = np.array([[p[0], p[1]] for p in pr])
    return {"cabinet": cid, "poly": pts2, "depth": pr[0][2], "bbox": (pts2[:, 0].min(), pts2[:, 1].min(), pts2[:, 0].max(), pts2[:, 1].max())}


def clip_box(b, x0, y0, s=TILE):
    bx0, by0, bx1, by1 = b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0
    full = max(1e-6, (bx1 - bx0) * (by1 - by0))
    cx0, cy0, cx1, cy1 = max(0, bx0), max(0, by0), min(s, bx1), min(s, by1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, 0.0
    return (cx0, cy0, cx1, cy1), ((cx1 - cx0) * (cy1 - cy0)) / full


# ------------------------------------------------------------------ image operations
class ImageCache:
    def __init__(self, folder, size=3):
        self.folder, self.size, self.d = Path(folder), size, OrderedDict()

    def get(self, name):
        if name in self.d:
            self.d.move_to_end(name)
            return self.d[name]
        im = cv2.imread(str(self.folder / name))
        self.d[name] = im
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return im


def hull_mask(shape, relay, grow=0):
    m = np.zeros(shape[:2], np.uint8)
    pts = np.vstack([relay["front"], relay["back"]]).astype(np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts).astype(np.int32), 255)
    if grow > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m


def door_noise(gray, mask):
    ring = cv2.dilate(mask, np.ones((61, 61), np.uint8)) & ~cv2.dilate(mask, np.ones((21, 21), np.uint8))
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 3)
    v = hp[ring > 0]
    return float(np.std(v)) if v.size > 50 else 1.5


def paste_relay(R, relay, tex, ox, oy, real_mean, real_lap, noise_std, shadow=0.22):
    """Paste the render (with side faces and a soft shadow) into R; (ox, oy) = R's origin in image px."""
    S = 2
    x0, y0, x1, y1 = relay["bbox"]
    pad = 8
    bx0, by0 = int(x0 - ox) - pad, int(y0 - oy) - pad
    bw, bh = int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad
    ax0, ay0, ax1, ay1 = max(0, bx0), max(0, by0), min(R.shape[1], bx0 + bw), min(R.shape[0], by0 + bh)
    if ax1 <= ax0 + 4 or ay1 <= ay0 + 4:
        return
    gw, gh = (ax1 - ax0) * S, (ay1 - ay0) * S
    sh = lambda a: ((a - np.array([ox + ax0, oy + ay0])) * S).astype(np.float32)  # noqa: E731
    front, back = sh(relay["front"]), sh(relay["back"])
    layer = np.zeros((gh, gw, 3), np.float32)
    alpha = np.zeros((gh, gw), np.float32)
    # side faces first (flat grey, darker left/right), then the textured front face
    side_col = {0: (214, 214, 212), 1: (190, 190, 188), 2: (205, 205, 203), 3: (190, 190, 188)}
    for i in range(4):
        poly = np.array([front[i], front[(i + 1) % 4], back[(i + 1) % 4], back[i]], np.int32)
        tmp = np.zeros((gh, gw), np.uint8)
        cv2.fillConvexPoly(tmp, poly, 255)
        layer[tmp > 0] = side_col[i]
        alpha = np.maximum(alpha, tmp.astype(np.float32) / 255)
    wpx = np.linalg.norm(front[1] - front[0])
    sc = max(0.02, min(1.0, 1.6 * wpx / tex.shape[1]))
    small = cv2.resize(tex, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    a_small = cv2.resize(RENDER_ALPHA, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    sh_, sw_ = small.shape[:2]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [sw_, 0], [sw_, sh_], [0, sh_]]), front)
    wt = cv2.warpPerspective(small, Hm, (gw, gh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    wa = cv2.warpPerspective(a_small, Hm, (gw, gh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
    layer = layer * (1 - wa) + wt * wa
    alpha = np.maximum(alpha, wa[..., 0])
    # downsample the supersampled layer
    layer = cv2.resize(layer, (ax1 - ax0, ay1 - ay0), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (ax1 - ax0, ay1 - ay0), interpolation=cv2.INTER_AREA)
    # appearance matching measured on the real relay: colour gain, sharpness, noise
    m = alpha > 0.5
    if m.sum() > 20 and real_mean is not None:
        gain = np.clip(real_mean / np.maximum(layer[m].mean(0), 1), 0.75, 1.25)
        layer *= gain
    if (relay["bbox"][2] - relay["bbox"][0]) >= 60 and real_lap:
        best, bs = None, 0.0
        for s_ in (0.0, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5):
            t = cv2.GaussianBlur(layer, (0, 0), s_) if s_ > 0 else layer
            lap = float(cv2.Laplacian(cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)[m].var()) if m.sum() > 20 else 0
            err = abs(np.log((lap + 1) / (real_lap + 1)))
            if best is None or err < best:
                best, bs = err, s_
        sigma = bs
    else:
        sigma = 0.7
    if sigma > 0:
        layer = cv2.GaussianBlur(layer, (0, 0), sigma)
        alpha = cv2.GaussianBlur(alpha, (0, 0), min(sigma, 1.0))
    layer += np.random.RandomState(1).normal(0, min(noise_std, 10.0) * 0.6, layer.shape).astype(np.float32)
    roi = R[ay0:ay1, ax0:ax1].astype(np.float32)
    # soft contact shadow below the relay (light comes from the ceiling)
    sh_off = max(1, int(0.03 * (ay1 - ay0)))
    sha = np.roll(cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.025 * (ax1 - ax0))), sh_off, axis=0)
    roi *= (1 - shadow * sha * (1 - alpha))[..., None]
    out = roi * (1 - alpha[..., None]) + layer * alpha[..., None]
    R[ay0:ay1, ax0:ax1] = np.clip(out, 0, 255).astype(np.uint8)


def band(depth):
    return "near" if depth < 1.5 else "mid" if depth < 3.5 else "far"


# ------------------------------------------------------------------ one sample
def make_sample(full, cam, scan, relays, cerdex, x0, y0, variant, tex_variants):
    M = 450
    rx0, ry0 = max(0, x0 - M), max(0, y0 - M)
    rx1, ry1 = min(full.shape[1], x0 + TILE + M), min(full.shape[0], y0 + TILE + M)
    R = full[ry0:ry1, rx0:rx1].copy()
    gray = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    local = []
    for g in relays:
        bx = (g["bbox"][0] - rx0, g["bbox"][1] - ry0, g["bbox"][2] - rx0, g["bbox"][3] - ry0)   # everything in R-local px
        gl = {**g, "front": g["front"] - [rx0, ry0], "back": g["back"] - [rx0, ry0], "bbox": bx}
        if bx[2] < 0 or bx[3] < 0 or bx[0] > R.shape[1] or bx[1] > R.shape[0]:
            continue
        local.append(gl)
    stats, masks = [], []
    for g in local:
        mk = hull_mask(R.shape, g)
        masks.append(mk)
        real_mean = R[mk > 0].mean(0) if (mk > 0).sum() > 50 else None
        real_lap = float(cv2.Laplacian(gray, cv2.CV_32F)[mk > 0].var()) if (mk > 0).sum() > 50 else None
        stats.append((real_mean, real_lap, door_noise(gray, mk)))
    if masks:
        grow = max(6, int(0.12 * np.mean([g["bbox"][2] - g["bbox"][0] for g in local])))
        union = np.zeros(R.shape[:2], np.uint8)
        for mk in masks:
            union |= cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        small = cv2.resize(R, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(union, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        fill = cv2.resize(cv2.inpaint(small, msmall, 7, cv2.INPAINT_TELEA), (R.shape[1], R.shape[0]), interpolation=cv2.INTER_CUBIC)
        nz = float(np.mean([s[2] for s in stats]))
        fill = np.clip(fill.astype(np.float32) + np.random.RandomState(2).normal(0, nz, fill.shape), 0, 255)
        soft = cv2.GaussianBlur(union.astype(np.float32) / 255, (0, 0), 3)[..., None]
        R = np.clip(R * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
    tex = tex_variants[variant % len(tex_variants)]
    boxes = []
    for g, (rm, rl, ns) in zip(local, stats):
        full_g = next(r for r in relays if r["cabinet"] == g["cabinet"])
        box, vis = clip_box(full_g["bbox"], x0, y0)
        if box is None or vis < 0.5:
            continue
        paste_relay(R, g, tex, 0, 0, rm, rl, ns)
        boxes.append((0, box, g["cabinet"], full_g["depth"], vis))
    for c in cerdex:
        box, vis = clip_box(c["bbox"], x0, y0)
        if box is not None and vis >= 0.5:
            boxes.append((1, box, c["cabinet"], c["depth"], vis))
    tile = R[y0 - ry0:y0 - ry0 + TILE, x0 - rx0:x0 - rx0 + TILE].astype(np.float32) * BRIGHT[variant % 3]
    return np.clip(tile, 0, 255).astype(np.uint8), boxes


def plan(view, cams, a):
    """Deterministic list of tiles to generate: (split, image idx, tile origin, variant, kind)."""
    jobs, pairs, empties = [], Counter(), {"train": [], "val": []}
    for e in view:
        i = int(e["file"][4:7]); cam = cams[i]; split = "val" if e["scan_index"] in HOLDOUT_SCANS else "train"
        if False: # Removed far-relay filtering
            continue
        relays = {f"{c}_{i}": g for c in ("H01", "H02", "H03", "H04", "H05", "OT1", "VLK", "OKK1") for i, g in enumerate(relay_geometry(cam, c))}
        relays = {k: v for k, v in relays.items() if v["depth"] <= MAX_RELAY_DEPTH}
        occupied = [g["bbox"] for g in relays.values()] + [g["bbox"] for g in (cerdex_geometry(cam, c) for c in ("TSK1", "TSK2")) if g]
        for gx in (0, 1408, 2816):
            for gy in (0, 1408, 2816):
                if all(clip_box(b, gx, gy)[0] is None for b in occupied):
                    empties[split].append((split, i, (gx, gy), 0, "empty", "-"))
        for cid, g in relays.items():
            cx, cy = (g["bbox"][0] + g["bbox"][2]) / 2, (g["bbox"][1] + g["bbox"][3]) / 2
            if not (200 < cx < 3896 and 200 < cy < 3896):
                continue
            n = BAND_VARIANTS[band(g["depth"])]
            n = n if split == "train" else max(4, n // 5)
            pairs[(split, band(g["depth"]))] += 1
            for v in range(n):
                fx, fy = GRID[v % 9]
                jobs.append((split, i, (int(np.clip(cx - fx * TILE, 0, 4096 - TILE)), int(np.clip(cy - fy * TILE, 0, 4096 - TILE))), v, "relay", cid))
        for cid in ("TSK1", "TSK2"):
            c = cerdex_geometry(cam, cid)
            if not c or c["depth"] > 5.0:
                continue
            cx, cy = (c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2
            if not (150 < cx < 3946 and 150 < cy < 3946):
                continue
            for v in range(9 if split == "train" else 2):
                fx, fy = GRID[v % 9]
                jobs.append((split, i, (int(np.clip(cx - fx * TILE, 0, 4096 - TILE)), int(np.clip(cy - fy * TILE, 0, 4096 - TILE))), v, "other_hmi", cid))
    for sp, cand in empties.items():                       # deterministic, evenly spread over the candidates
        step = max(1, len(cand) // EMPTY_TILES[sp])
        jobs += cand[::step][:EMPTY_TILES[sp]]
    return jobs, pairs


def main():
    global RENDER_ALPHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="VEO Images")
    ap.add_argument("--cameras", default="VEO Images/cameras.json")
    ap.add_argument("--out", default="synthetic_geo_dataset")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="testing: only the first N jobs (spread over the list)")
    ap.add_argument("--shard", help="K/N: process only the images with index %% N == K (run N processes in parallel)")
    a = ap.parse_args()

    cams = load_cameras(a.cameras)
    view = json.load(open(HERE / "image_cabinet_view.json", encoding="utf-8"))
    jobs, pairs = plan(view, cams, a)
    print("pairs:", dict(pairs))
    print("jobs:", Counter((j[0], j[4]) for j in jobs))
    if a.dry_run:
        return
    tex, RENDER_ALPHA = load_render()
    variants = lcd_variants(tex)
    cache = ImageCache(a.images, size=1)
    out = Path(a.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    shard = None
    if a.shard:
        k_, n_ = map(int, a.shard.split("/"))
        jobs = [j for j in jobs if j[1] % n_ == k_]
        shard = k_
    if a.limit:
        step = max(1, len(jobs) // a.limit)
        jobs = jobs[::step][:a.limit]
    meta = open(out / (f"meta_{shard}.jsonl" if shard is not None else "meta.jsonl"), "w", encoding="utf-8")
    counts, n = Counter(), 0
    jobs.sort(key=lambda j: j[1])
    for k, (split, i, (x0, y0), v, kind, cid) in enumerate(jobs):
        cam = cams[i]
        full = cache.get(f"img_{i:03d}.jpg")
        relays = [g for c in ("H01", "H02", "H03", "H04", "H05", "OT1", "VLK", "OKK1") for g in relay_geometry(cam, c, POSE_JIT[v % 9])]
        cerdex = [g for g in (cerdex_geometry(cam, c) for c in ("TSK1", "TSK2")) if g]
        tile, boxes = make_sample(full, cam, view[i]["scan_index"], relays, cerdex, x0, y0, v, variants)
        name = f"geo_{split}_{i:03d}_{cid}_{x0}_{y0}_{v:02d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q[v % 3]])
        with open(out / "labels" / split / f"{name}.txt", "w") as f:
            for cls, b, cab, dep, vis in boxes:
                f.write(f"{cls} {(b[0] + b[2]) / 2 / TILE:.6f} {(b[1] + b[3]) / 2 / TILE:.6f} {(b[2] - b[0]) / TILE:.6f} {(b[3] - b[1]) / TILE:.6f}\n")
        meta.write(json.dumps({"image": f"{name}.jpg", "split": split, "source": cam.file, "scan": view[i]["scan_index"],
                               "tile_origin": [x0, y0], "kind": kind, "variant": v, "camera": {"position": cam.C.tolist(), "rotation_wxyz": cam.entry["rotation_wxyz"], "focal_px": cam.fx},
                               "boxes": [{"class": CLASSES[c], "cabinet": cb, "depth_m": round(d, 2), "visible": round(vs, 2), "xyxy": [round(x, 1) for x in b]} for c, b, cb, d, vs in boxes]}) + "\n")
        for c, *_ in boxes:
            counts[(split, CLASSES[c])] += 1
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(jobs)} tiles", flush=True)
    meta.close()
    (out / "data.yaml").write_text("path: {}\n".format(out.resolve().as_posix()) + "train: images/train\nval: images/val\nnames:\n  0: abb_relion_615\n  1: other_hmi\n")
    json.dump({"tiles": n, "boxes": {f"{s}/{c}": v for (s, c), v in counts.items()}}, open(out / (f"stats_{shard}.json" if shard is not None else "stats.json"), "w"), indent=1)
    print("done:", n, "tiles |", dict(counts))


CLASSES = ["abb_relion_615", "other_hmi"]
RENDER_ALPHA = None

if __name__ == "__main__":
    main()
