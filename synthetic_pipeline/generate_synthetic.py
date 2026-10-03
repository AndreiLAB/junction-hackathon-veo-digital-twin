#!/usr/bin/env python3
"""
Synthetic training data generator for the VEO360 product-tagging challenge.

Takes clean product reference images (white background) and produces a
YOLO-format detection dataset that mimics what the device looks like inside
a Matterport scan of a switchgear room: mounted in a cabinet door, seen at an
angle, small, badly lit, partly occluded, with worn labels and camera noise.

Classes
  0 abb_relion_615   the protection relay (front bezel)
  1 nameplate        text label / designation plate (target for OCR)
"""
import argparse
import json
import math
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["abb_relion_615", "nameplate"]
OUT_W, OUT_H = 1024, 768          # output frame size (YOLO resizes anyway)
CANVAS_W, CANVAS_H = 3200, 2400
POINTCLOUD_PROB = 0.4            # share of samples rendered in "point cloud" style


# ----------------------------------------------------------------------------
# Reference preparation
# ----------------------------------------------------------------------------
def extract_object(img_bgr, thresh=6):
    """Cut the product out of a white background -> tightly cropped BGRA."""
    diff = 255 - img_bgr.min(axis=2)
    mask = (diff > thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    full = np.zeros_like(mask)
    cv2.drawContours(full, [c], -1, 255, -1)          # fill holes (white slots)
    full = cv2.GaussianBlur(full, (3, 3), 0)          # soft edge
    x, y, w, h = cv2.boundingRect(c)
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = full
    return rgba[y:y + h, x:x + w].copy()


def find_label_slots(rgba):
    """Find the blank white label strips (F-key slots, LED strip) on the bezel.
    In real installations these are filled with printed text, so we fill
    them randomly during augmentation."""
    bgr = rgba[..., :3]
    white = ((bgr.min(axis=2) > 245) & (rgba[..., 3] > 250)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    h, w = white.shape
    slots = []
    for i in range(1, n):
        x, y, sw, sh, area = stats[i]
        if area < 0.0004 * w * h or area > 0.02 * w * h:
            continue
        if area / float(sw * sh) < 0.75:   # must be roughly rectangular
            continue
        slots.append((x, y, sw, sh))
    return slots


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]
WORDS = ["INCOMER", "FEEDER", "OUTGOING", "TRAFO", "BUS COUPLER", "MEASURING",
         "SPARE", "AUX SUPPLY", "EARTH FAULT", "TRIP", "ALARM", "BLOCK",
         "CB OPEN", "CB CLOSED", "LOCAL", "REMOTE", "PROT. 1", "AR ON"]


def rand_text():
    r = random.random()
    if r < 0.35:
        return f"=J{random.randint(1, 20):02d} -{random.choice('KQFPT')}{random.randint(1, 9)}"
    if r < 0.65:
        w = random.choice(WORDS)
        return w + (f" {random.randint(1, 12)}" if random.random() < 0.5 else "")
    if r < 0.8:
        return f"{random.choice('+=')}{random.choice('ABH')}{random.randint(1, 30):02d}"
    return "".join(random.choices("ABCDEFGHJKLMNPRSTUVXYZ0123456789-", k=random.randint(4, 9)))


def draw_text_fit(img, rect, text, color, thickness=None):
    x, y, w, h = rect
    font = random.choice(FONTS)
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 1)
    scale = min(0.9 * w / max(tw, 1), 0.65 * h / max(th, 1))
    if scale <= 0.05:
        return
    th_px = thickness or max(1, int(round(scale * 1.6)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, th_px)
    org = (int(x + (w - tw) / 2), int(y + (h + th) / 2))
    cv2.putText(img, text, org, font, scale, color, th_px, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Relay appearance augmentation (label text + wear)
# ----------------------------------------------------------------------------
def augment_relay(rgba, slots):
    out = rgba.copy()
    bgr = np.ascontiguousarray(out[..., :3])
    # 1) fill the blank label strips with printed text (as in real installs)
    for (x, y, w, h) in slots:
        if random.random() < 0.7:
            draw_text_fit(bgr, (x + 2, y + 1, w - 4, h - 2), rand_text(),
                          (random.randint(0, 60),) * 3)
    # 2) label wear: fading, smudges, dirt, scratches
    H, W = bgr.shape[:2]
    for _ in range(random.randint(0, 6)):
        pw, ph = random.randint(W // 20, W // 5), random.randint(H // 20, H // 5)
        px, py = random.randint(0, W - pw), random.randint(0, H - ph)
        roi = bgr[py:py + ph, px:px + pw].astype(np.float32)
        mode = random.choice(["fade", "blur", "dirt"])
        if mode == "fade":
            roi = roi * 0.4 + roi.mean() * 0.6
        elif mode == "blur":
            roi = cv2.GaussianBlur(roi, (0, 0), random.uniform(1.5, 4))
        else:
            dirt = np.random.normal(0, 25, roi.shape[:2])[..., None]
            roi = roi * random.uniform(0.7, 0.95) + dirt
        bgr[py:py + ph, px:px + pw] = np.clip(roi, 0, 255).astype(np.uint8)
    for _ in range(random.randint(0, 8)):
        p1 = (random.randint(0, W), random.randint(0, H))
        p2 = (p1[0] + random.randint(-W // 4, W // 4), p1[1] + random.randint(-H // 8, H // 8))
        c = random.choice([200, 230, 90])
        cv2.line(bgr, p1, p2, (c, c, c), 1, cv2.LINE_AA)
    out[..., :3] = bgr
    return out


# ----------------------------------------------------------------------------
# Procedural switchgear panel background
# ----------------------------------------------------------------------------
PANEL_COLORS = [(210, 213, 208), (225, 226, 222), (196, 200, 199),
                (235, 235, 232), (150, 152, 150), (170, 180, 185)]


def rand_rect_in(region, w, h, occupied, tries=40, pad=10):
    rx, ry, rw, rh = region
    for _ in range(tries):
        if w >= rw or h >= rh:
            return None
        x = random.randint(rx, rx + rw - w)
        y = random.randint(ry, ry + rh - h)
        if all(x + w + pad < ox or ox + ow + pad < x or y + h + pad < oy or oy + oh + pad < y
               for ox, oy, ow, oh in occupied):
            occupied.append((x, y, w, h))
            return (x, y, w, h)
    return None


def draw_device(img, r, kind):
    x, y, w, h = r
    if kind == "meter":
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        lx, ly, lw, lh = x + w // 8, y + h // 6, w * 3 // 4, h // 2
        lcd = random.choice([(90, 160, 120), (60, 60, 50), (200, 170, 120)])
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), lcd, -1)
        draw_text_fit(img, (lx, ly, lw, lh), f"{random.uniform(0, 999):.1f}", (20, 20, 20))
    elif kind == "lamps":
        n = random.randint(2, 5)
        rad = min(h // 2, w // (2 * n + 1))
        for i in range(n):
            c = random.choice([(40, 40, 200), (40, 170, 40), (30, 200, 230), (230, 230, 230), (30, 30, 30)])
            cx = x + rad + i * (w // n)
            cv2.circle(img, (cx, y + h // 2), rad, (60, 60, 60), -1)
            cv2.circle(img, (cx, y + h // 2), int(rad * 0.75), c, -1)
    elif kind == "handle":
        c = random.choice([(25, 25, 25), (30, 30, 170)])
        cv2.rectangle(img, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 10, 10), 2)
    elif kind == "vent":
        for yy in range(y, y + h, max(4, h // 12)):
            cv2.line(img, (x, yy), (x + w, yy), (70, 70, 70), 2)
    elif kind == "warning":
        pts = np.array([[x + w // 2, y], [x, y + h], [x + w, y + h]], np.int32)
        cv2.fillPoly(img, [pts], (0, 210, 250))
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    elif kind == "other_ied":
        # hard negative: a different-looking protection / control device
        body = random.choice([(60, 60, 60), (120, 120, 120), (230, 230, 230), (90, 70, 40)])
        cv2.rectangle(img, (x, y), (x + w, y + h), body, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), 2)
        sx, sy = x + random.randint(w // 10, w // 3), y + h // 8
        sw, sh = w // 2, h // 3
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), random.choice([(20, 20, 20), (150, 200, 120), (200, 120, 60)]), -1)
        for i in range(random.randint(3, 8)):
            bx = x + w // 10 + i * (w // 9)
            cv2.rectangle(img, (bx, y + h * 2 // 3), (bx + w // 14, y + h * 2 // 3 + h // 10), (180, 180, 180), -1)
        draw_text_fit(img, (x + 5, y + h - h // 7, w // 3, h // 8),
                      "".join(random.choices("ABCDEFGHKMNPRSTUVXZ", k=3)), (240, 240, 240))


def draw_nameplate(img, r, inst, inst_id):
    x, y, w, h = r
    bg, fg = random.choice([((250, 250, 250), (0, 0, 0)), ((0, 220, 250), (0, 0, 0)),
                            ((20, 20, 20), (240, 240, 240)), ((200, 200, 200), (20, 20, 20))])
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    draw_text_fit(img, (x + 3, y + 2, w - 6, h - 4), rand_text(), fg)
    inst[y:y + h, x:x + w] = inst_id


def make_panel(real_bgs):
    """Return (canvas, door_regions). Uses a real background if provided."""
    if real_bgs and random.random() < 0.5:
        bg = cv2.imread(str(random.choice(real_bgs)))
        if bg is not None:
            bg = cv2.resize(bg, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
            return bg, [(0, 0, CANVAS_W, CANVAS_H)]
    base = np.array(random.choice(PANEL_COLORS), np.float32)
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), np.float32) * base
    grad = np.linspace(random.uniform(0.85, 1.05), random.uniform(0.9, 1.1), CANVAS_H)[:, None, None]
    canvas = np.clip(canvas * grad + np.random.normal(0, 3, canvas.shape), 0, 255).astype(np.uint8)
    n_doors = random.randint(2, 6)
    edges = np.linspace(0, CANVAS_W, n_doors + 1).astype(int)
    doors = []
    for i in range(n_doors):
        x0, x1 = edges[i], edges[i + 1]
        cv2.line(canvas, (x0, 0), (x0, CANVAS_H), (60, 60, 60), random.randint(3, 9))
        # horizontal compartment split (LV compartment on top)
        split = random.randint(int(CANVAS_H * 0.45), int(CANVAS_H * 0.7))
        cv2.line(canvas, (x0, split), (x1, split), (70, 70, 70), random.randint(3, 7))
        doors.append((x0 + 25, 25, x1 - x0 - 50, split - 50))
        doors.append((x0 + 25, split + 25, x1 - x0 - 50, CANVAS_H - split - 50))
    return canvas, doors


# ----------------------------------------------------------------------------
# Compositing
# ----------------------------------------------------------------------------
def paste_rgba(dst, rgba, x, y, shadow=True):
    h, w = rgba.shape[:2]
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    if shadow:
        dx, dy = random.randint(-12, 12), random.randint(4, 18)
        sx, sy = max(0, x + dx), max(0, y + dy)
        sh = a[:min(h, dst.shape[0] - sy), :min(w, dst.shape[1] - sx)]
        sh = cv2.GaussianBlur(sh, (0, 0), 8)[..., None] if sh.ndim == 3 else sh
        roi = dst[sy:sy + sh.shape[0], sx:sx + sh.shape[1]].astype(np.float32)
        dst[sy:sy + sh.shape[0], sx:sx + sh.shape[1]] = (roi * (1 - 0.4 * sh)).astype(np.uint8)
    roi = dst[y:y + h, x:x + w].astype(np.float32)
    dst[y:y + h, x:x + w] = (a * rgba[..., :3] + (1 - a) * roi).astype(np.uint8)


def lighting(img):
    f = img.astype(np.float32)
    h, w = img.shape[:2]
    # global brightness / contrast / colour temperature
    f = (f - 128) * random.uniform(0.7, 1.25) + 128 + random.uniform(-40, 30)
    f *= np.array([random.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    # spatial lighting gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = random.uniform(0, 2 * math.pi)
    g = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    f *= (1 + random.uniform(-0.4, 0.4) * (g - g.mean()))[..., None]
    # specular glare blobs (glossy bezels + fluorescent lights)
    for _ in range(random.choice([0, 0, 1, 1, 2, 3])):
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        sx, sy = random.uniform(15, 150), random.uniform(10, 90)
        blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        f += blob[..., None] * random.uniform(60, 180)
    # vignette
    if random.random() < 0.5:
        r = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
        f *= (1 - random.uniform(0.2, 0.6) * r ** 2)[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def add_occluders(img):
    """Cables, door edges, random objects. Returns occlusion mask."""
    h, w = img.shape[:2]
    occ = np.zeros((h, w), np.uint8)
    for _ in range(random.choice([0, 0, 0, 1, 1, 2])):
        kind = random.choice(["cable", "cable", "rect", "blob"])
        layer = np.zeros((h, w), np.uint8)
        if kind == "cable":
            pts = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(random.randint(2, 5))], np.int32)
            cv2.polylines(layer, [pts], False, 255, random.randint(6, 28))
        elif kind == "rect":
            x, y = random.randint(0, w), random.randint(0, h)
            cv2.rectangle(layer, (x, y), (x + random.randint(30, w // 4), y + random.randint(30, h // 3)), 255, -1)
        else:
            cv2.ellipse(layer, (random.randint(0, w), random.randint(0, h)),
                        (random.randint(20, 120), random.randint(20, 120)), random.randint(0, 180), 0, 360, 255, -1)
        if random.random() < 0.75:   # mostly dark / neutral (cables, doors, people)
            v = random.randint(10, 120)
            color = np.array([v + random.randint(-10, 10) for _ in range(3)], np.float32)
        else:
            color = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
        tex = color + np.random.normal(0, 10, (h, w, 3))
        m = layer.astype(bool)
        img[m] = np.clip(tex[m], 0, 255).astype(np.uint8)
        occ |= layer
    return occ


def pointcloud_style(img):
    """Mimic views rendered from a coloured point cloud (E57): blocky splats,
    speckle, median smoothing, small holes filled with neighbouring colour."""
    h, w = img.shape[:2]
    f = random.choice([2, 3, 4])
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_NEAREST)
    holes = np.random.random(small.shape[:2]) < random.uniform(0.0, 0.08)
    if holes.any():
        filled = cv2.dilate(small, np.ones((3, 3), np.uint8))
        small[holes] = filled[holes]
    small = np.clip(small.astype(np.float32) + np.random.normal(0, 6, small.shape), 0, 255).astype(np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.medianBlur(img, 3)


def camera_degrade(img):
    h, w = img.shape[:2]
    if random.random() < POINTCLOUD_PROB:
        img = pointcloud_style(img)
    f = random.uniform(1.0, 3.5)   # Matterport frames are soft at distance
    if f > 1.1:
        small = cv2.resize(img, (int(w / f), int(h / f)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.6))
    if random.random() < 0.2:
        k = random.choice([3, 5, 7, 9])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, M, (k, k)))
    img = np.clip(img.astype(np.float32) + np.random.normal(0, random.uniform(1, 9), img.shape), 0, 255).astype(np.uint8)
    q = random.randint(35, 95)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------
# One sample
# ----------------------------------------------------------------------------
def make_sample(refs, weights, real_bgs, debug=None):
    canvas, doors = make_panel(real_bgs)
    inst = np.zeros(canvas.shape[:2], np.int32)  # 1..99 relays, 100+ nameplates
    occupied = []
    relay_rects = []

    n_relays = random.choices([0, 1, 2, 3], weights=[0.1, 0.55, 0.25, 0.1])[0]
    first_ref = None
    for i in range(n_relays):
        ref = random.choices(refs, weights=weights)[0]
        upper = [d for d in doors if d[1] < CANVAS_H * 0.2]
        door = random.choice(upper if upper and random.random() < 0.8 else doors)
        rw = int(min(door[2] * random.uniform(0.45, 0.85), CANVAS_W * 0.14))
        rgba = cv2.resize(ref["rgba"], (rw, int(rw * ref["rgba"].shape[0] / ref["rgba"].shape[1])),
                          interpolation=cv2.INTER_AREA)
        scale = rw / ref["rgba"].shape[1]
        slots = [tuple(int(v * scale) for v in s) for s in ref["slots"]]
        aug = augment_relay(rgba, slots)
        r = rand_rect_in(door, aug.shape[1], aug.shape[0], occupied)
        if r is None:
            continue
        x, y = r[0], r[1]
        paste_rgba(canvas, aug, x, y)
        inst[y:y + aug.shape[0], x:x + aug.shape[1]][aug[..., 3] > 127] = i + 1
        relay_rects.append(r)
        if first_ref is None:
            first_ref = (ref, rgba, aug)
        # designation plate just above / below the relay (very common)
        if random.random() < 0.7:
            pw, ph = int(aug.shape[1] * random.uniform(0.4, 0.8)), int(aug.shape[0] * random.uniform(0.08, 0.14))
            py = y - ph - random.randint(8, 30) if random.random() < 0.6 else y + aug.shape[0] + random.randint(8, 30)
            px = x + (aug.shape[1] - pw) // 2
            if 0 <= py and py + ph < CANVAS_H:
                occupied.append((px, py, pw, ph))
                draw_nameplate(canvas, (px, py, pw, ph), inst, 100 + len(occupied))

    # clutter: other devices + extra nameplates + hard negatives
    for _ in range(random.randint(8, 30)):
        kind = random.choice(["meter", "lamps", "handle", "vent", "warning", "other_ied", "other_ied", "nameplate"])
        door = random.choice(doors)
        size = {"meter": (220, 160), "lamps": (300, 70), "handle": (60, 200), "vent": (300, 160),
                "warning": (90, 80), "other_ied": (380, 280), "nameplate": (260, 50)}[kind]
        s = random.uniform(0.6, 1.4) * 1.3
        r = rand_rect_in(door, int(size[0] * s), int(size[1] * s), occupied)
        if r is None:
            continue
        if kind == "nameplate":
            draw_nameplate(canvas, r, inst, 100 + len(occupied))
        else:
            draw_device(canvas, r, kind)

    if debug is not None and first_ref:
        debug["1_reference"] = first_ref[0]["orig"]
        debug["2_cutout"] = checker_composite(first_ref[1])
        debug["3_label_text_and_wear"] = checker_composite(first_ref[2])
        debug["4_placed_in_panel"] = canvas.copy()

    # ---- virtual camera: zoom + perspective + roll ------------------------
    if relay_rects:
        rx, ry, rw_, rh_ = random.choice(relay_rects)
        target_w = OUT_W * math.exp(random.uniform(math.log(0.05), math.log(0.6)))  # bias to small
        zoom = target_w / rw_
        zoom = max(zoom, OUT_W / (0.95 * CANVAS_W))   # never look past the scene
        cx = rx + rw_ / 2 + random.uniform(-0.4, 0.4) * OUT_W / zoom
        cy = ry + rh_ / 2 + random.uniform(-0.4, 0.4) * OUT_H / zoom
    else:
        zoom = random.uniform(OUT_W / (0.95 * CANVAS_W), 2.0)
        cx, cy = random.uniform(0, CANVAS_W), random.uniform(0, CANVAS_H)
    cw, ch = OUT_W / zoom, OUT_H / zoom
    cx = min(max(cx, cw / 2), CANVAS_W - cw / 2)
    cy = min(max(cy, ch / 2), CANVAS_H - ch / 2)
    quad = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]], np.float32)
    roll = math.radians(random.uniform(-6, 6))
    R = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]], np.float32)
    quad = quad @ R.T
    p = random.uniform(0.0, 0.22)   # perspective strength (viewing angle)
    quad += np.random.uniform(-p, p, quad.shape).astype(np.float32) * np.array([cw, ch], np.float32)
    quad += np.array([cx, cy], np.float32)
    dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)

    img = cv2.warpPerspective(canvas, H, (OUT_W, OUT_H), flags=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(int(v) for v in np.random.randint(40, 200, 3)))
    # instance masks on an extended canvas so we know the un-clipped area
    pad = OUT_W
    T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], np.float64)
    inst_ext = cv2.warpPerspective(inst.astype(np.float32), T @ H, (OUT_W + 2 * pad, OUT_H + 2 * pad),
                                   flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int32)
    inst_out = inst_ext[pad:pad + OUT_H, pad:pad + OUT_W]
    if debug is not None:
        debug["5_viewpoint_and_perspective"] = img.copy()

    img = lighting(img)
    if debug is not None:
        debug["6_lighting_and_glare"] = img.copy()
    occ = add_occluders(img)
    if debug is not None:
        debug["7_occlusions"] = img.copy()
    img = camera_degrade(img)

    # ---- labels ------------------------------------------------------------
    labels = []
    for iid in np.unique(inst_out):
        if iid == 0:
            continue
        full = int((inst_ext == iid).sum())
        m = inst_out == iid
        vis = int((m & (occ == 0)).sum())
        if full == 0 or (vis / full < 0.2 and vis < 1500):
            continue
        ys, xs = np.where(m)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if min(x1 - x0, y1 - y0) < 6:
            continue
        cls = 0 if iid < 100 else 1
        labels.append((cls, (x0 + x1) / 2 / OUT_W, (y0 + y1) / 2 / OUT_H, (x1 - x0) / OUT_W, (y1 - y0) / OUT_H))
    if debug is not None:
        debug["8_camera_noise_final"] = draw_boxes(img.copy(), labels)
    return img, labels


# ----------------------------------------------------------------------------
# Visual helpers (for the demo)
# ----------------------------------------------------------------------------
def checker_composite(rgba, sq=24):
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where(((xx // sq + yy // sq) % 2)[..., None] == 0, 200, 240).astype(np.uint8).repeat(3, 2)
    a = rgba[..., 3:4] / 255.0
    return (a * rgba[..., :3] + (1 - a) * bg).astype(np.uint8)


def draw_boxes(img, labels):
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh in labels:
        c = (0, 200, 0) if cls == 0 else (255, 120, 0)
        p0 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
        p1 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        cv2.rectangle(img, p0, p1, c, 2)
        cv2.putText(img, CLASSES[cls], (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return img


def tile(images, cols, cell_w=480, titles=None):
    cells = []
    for i, im in enumerate(images):
        cell_h = int(cell_w * 0.75)
        s = min(cell_w / im.shape[1], cell_h / im.shape[0])
        r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        c = np.full((cell_h + (30 if titles else 0), cell_w, 3), 255, np.uint8)
        oy = 30 if titles else 0
        c[oy + (cell_h - r.shape[0]) // 2: oy + (cell_h - r.shape[0]) // 2 + r.shape[0],
          (cell_w - r.shape[1]) // 2:(cell_w - r.shape[1]) // 2 + r.shape[1]] = r
        if titles:
            cv2.putText(c, titles[i], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        cells.append(c)
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


# ----------------------------------------------------------------------------
_CTX = None


def _init_worker(ctx, pc_prob):
    global _CTX, POINTCLOUD_PROB
    _CTX, POINTCLOUD_PROB = ctx, pc_prob


def _worker(i):
    refs, weights, real_bgs, out, val, seed = _CTX
    random.seed(seed * 1_000_003 + i)
    np.random.seed((seed * 1_000_003 + i) % 2**32)
    split = "val" if random.random() < val else "train"
    img, labels = make_sample(refs, weights, real_bgs)
    name = f"syn_{i:06d}"
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "labels" / split / f"{name}.txt", "w") as f:
        for l in labels:
            f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
    return split, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", required=True, help="PATH[:WEIGHT]")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--bg-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pointcloud-prob", type=float, default=0.4,
                    help="share of samples styled like E57 point-cloud renders")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    global POINTCLOUD_PROB
    POINTCLOUD_PROB = args.pointcloud_prob
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    refs, weights = [], []
    for spec in args.ref:
        path, _, wt = spec.partition(":")
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"Cannot read {path}")
        rgba = extract_object(img)
        refs.append({"path": path, "orig": img, "rgba": rgba, "slots": find_label_slots(rgba)})
        weights.append(float(wt) if wt else 1.0)
        print(f"ref {path}: {rgba.shape[1]}x{rgba.shape[0]}, {len(refs[-1]['slots'])} label slots, weight {weights[-1]}")

    real_bgs = sorted(p for p in Path(args.bg_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if args.bg_dir else []

    ctx = (refs, weights, real_bgs, out, args.val, args.seed)
    stats = {"images": 0, "relay_boxes": 0, "nameplate_boxes": 0, "empty_images": 0}
    with Pool(args.workers, initializer=_init_worker, initargs=(ctx, args.pointcloud_prob)) as pool:
        for k, (split, labels) in enumerate(pool.imap_unordered(_worker, range(args.n), chunksize=8)):
            stats["images"] += 1
            stats["relay_boxes"] += sum(l[0] == 0 for l in labels)
            stats["nameplate_boxes"] += sum(l[0] == 1 for l in labels)
            stats["empty_images"] += not labels
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{args.n}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    previews = []
    for i in range(16):
        for split in ("train", "val"):
            ip = out / "images" / split / f"syn_{i:06d}.jpg"
            if ip.exists():
                lab = [tuple([int(v.split()[0])] + [float(x) for x in v.split()[1:]])
                       for v in (out / "labels" / split / f"syn_{i:06d}.txt").read_text().splitlines()]
                previews.append(draw_boxes(cv2.imread(str(ip)), lab))

    # demo assets
    cv2.imwrite(str(out / "preview_grid.jpg"), tile(previews, 4, 400))
    for attempt in range(200):
        dbg = {}
        _, lab = make_sample(refs, weights, real_bgs, debug=dbg)
        if len(dbg) == 8 and any(l[0] == 0 and 0.2 < l[3] < 0.5 and 0.2 < l[1] < 0.8 for l in lab):
            break
    cv2.imwrite(str(out / "pipeline_stages.jpg"), tile(list(dbg.values()), 4, 420, titles=list(dbg.keys())))

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()