import sys
import itertools
import numpy as np
import cv2
import pye57

E57_FILE = "cloud_0.e57"
SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 9     # a capture point inside the aisles

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

def project(cam, f, cx, cy, W, H):
    z = cam[:, 2]
    ok = z > 0.3
    u = np.full(len(cam), -1.0); v = np.full(len(cam), -1.0)
    u[ok] = f * cam[ok, 0] / z[ok] + cx
    v[ok] = f * cam[ok, 1] / z[ok] + cy
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, ok

def depth_map(cam, f, cx, cy, W, H):
    out = None
    for s in (1, 2, 4, 8):
        u, v, ok = project(cam, f, cx, cy, W, H)
        ui, vi, z = (u[ok] / s).astype(int), (v[ok] / s).astype(int), cam[ok, 2]
        Ws, Hs = W // s, H // s
        o = np.argsort(-z)
        d = np.zeros(Ws * Hs, np.float32)
        d[np.minimum(vi[o], Hs - 1) * Ws + np.minimum(ui[o], Ws - 1)] = z[o]
        d = cv2.resize(d.reshape(Hs, Ws), (W, H), interpolation=cv2.INTER_NEAREST)
        if out is None:
            out = d
        else:
            bad = ((out == 0) | (out > d * 1.08 + 0.05)) & (d > 0)
            out[bad] = d[bad]
    return out

e57 = pye57.E57(E57_FILE)
h = e57.get_header(SCAN)
scan_pos = np.array(h.translation, float)
print(f"Capture point {SCAN} at {np.round(scan_pos, 3).tolist()}. Loading its points (about a minute) ...")
d = e57.read_scan(SCAN, colors=True, transform=True, ignore_missing_fields=True)
xyz = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
rgb = np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], 1).astype(np.float32)
if "cartesianInvalidState" in d:
    ok = np.asarray(d["cartesianInvalidState"]) == 0
    xyz, rgb = xyz[ok], rgb[ok]
if rgb.max() > 255: rgb *= 255.0 / rgb.max()
del d
sub = np.random.default_rng(0).choice(len(xyz), min(300_000, len(xyz)), replace=False)

# photos taken at this capture point
images = e57.root["images2D"]
photos = []
for j in range(images.childCount()):
    R, C = pose(images[j])
    if np.linalg.norm(C - scan_pos) < 0.5:
        photos.append(j)
print(f"Photos at this capture point: {photos}")

results = []
for j in photos:
    im = images[j]
    R, C = pose(im)
    rep = im["pinholeRepresentation"]
    img = cv2.imdecode(np.asarray(rep["jpegImage"].read_buffer(), np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    f = val(rep, "focalLength", 0.0) / val(rep, "pixelWidth", 1.0)
    cx, cy = val(rep, "principalPointX"), val(rep, "principalPointY")
    if cx is None or cx < 1: cx = W / 2
    if cy is None or cy < 1: cy = H / 2
    img_rgb = img[:, :, ::-1].astype(np.float32)
    best = None
    for inv in (False, True):
        Rc2w = R.T if inv else R
        local = (xyz[sub] - C) @ Rc2w
        for flip in itertools.product((1, -1), repeat=3):
            cam = local * np.array(flip)
            u, v, ok = project(cam, f, cx, cy, W, H)
            if ok.sum() < 2000:
                continue
            err = np.abs(img_rgb[v[ok].astype(int), u[ok].astype(int)] - rgb[sub][ok]).mean()
            if best is None or err < best[0]:
                best = (err, inv, flip)
    err, inv, flip = best
    Rc2w = R.T if inv else R
    forward = Rc2w @ (np.array(flip, float) * np.array([0, 0, 1.0]))
    kind = "looks UP" if forward[2] > 0.7 else "looks DOWN" if forward[2] < -0.7 else "looks sideways"
    print(f"  photo {j:3d}: colour error {err:5.1f}  ({kind})")
    results.append((err, j, img, R, C, f, cx, cy, W, H, inv, flip, forward))

# use a sideways-looking photo with the best colour match
side = [r for r in results if abs(r[12][2]) < 0.7] or results
err, j, img, R, C, f, cx, cy, W, H, inv, flip, forward = min(side, key=lambda r: r[0])
Rc2w = R.T if inv else R
flip = np.array(flip, float)
depth = depth_map(((xyz - C) @ Rc2w) * flip, f, cx, cy, W, H)

def pixel_to_3d(u, v):
    win = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    vals = win[win > 0]
    if vals.size == 0:
        return None
    z = float(np.median(vals))
    return C + Rc2w @ (np.array([(u - cx) / f * z, (v - cy) / f * z, z]) * flip)

print(f"\nChecking photo {j} (img_{j:03d}.jpg), colour error {err:.1f}")
checks = [("1 centre of photo", W // 2, H // 2),
          ("2 lower middle", W // 2, int(H * 0.85)),
          ("3 left middle", int(W * 0.2), H // 2),
          ("4 right middle", int(W * 0.8), H // 2)]
view = img.copy()
for name, u, v in checks:
    p = pixel_to_3d(u, v)
    txt = "no depth" if p is None else f"X={p[0]:.3f}  Y={p[1]:.3f}  Z={p[2]:.3f}"
    print(f"  point {name:18s} pixel ({u},{v}) -> {txt}")
    cv2.circle(view, (u, v), 40, (0, 0, 255), 12)
    cv2.putText(view, name.split()[0], (u + 50, v - 50), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
print(f"Camera height: Z={C[2]:.3f}")
small = cv2.resize(view, (1024, int(1024 * H / W)), interpolation=cv2.INTER_AREA)
cv2.imwrite("check_view.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("Saved check_view.jpg (the photo with the 4 checked points marked)")