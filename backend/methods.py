"""The two ways to tag, as the frontend uses them, plus the export formats.

  * IMAGE method: a photo (or a pose entered by hand: position + orientation) -> which cabinets are in view -> where each asset
    must appear (expected boxes from the measured geometry). No cabinet in view -> no tag.
  * E57 method: the cabinets the E57 pipeline already found (no upload; the scan is already processed).
  * Exports: a manual tagging sheet (JSON/CSV) and Matterport-compatible tag JSON (Model API and Showcase SDK shapes).

Geometry is `backend/asset_geometry.json` (hand-measured, +-0.02 m). Camera convention verified in synthetic_geo/geometry.py:
camera_point = diag(1,-1,-1) @ R.T @ (P - C), u = f*x/z + cx.
"""
import csv
import io
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

GEO = json.loads((Path(__file__).resolve().parent / "asset_geometry.json").read_text(encoding="utf-8"))
FRAME, FOCAL = GEO["frame_px"], GEO["focal_px"]
FLIP = np.diag([1.0, -1.0, -1.0])


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Cam:
    """90 degree pinhole cube face, 4096 px, as in cameras.json."""

    def __init__(self, position, rotation_wxyz):
        n = float(np.linalg.norm(rotation_wxyz))
        if n < 1e-9:
            raise ValueError("rotation_wxyz must be a non-zero quaternion [w, x, y, z]")
        self.R, self.C = rot(np.asarray(rotation_wxyz, float) / n), np.asarray(position, float)

    def project(self, P):
        c = FLIP @ (self.R.T @ (np.asarray(P, float) - self.C))
        if c[2] <= 0.05:
            return None
        return (FOCAL * c[0] / c[2] + FRAME / 2, FOCAL * c[1] / c[2] + FRAME / 2, float(c[2]))


def _quad(cam, x, y0, y1, z0, z1):
    pr = [cam.project((x, y, z)) for y, z in ((y0, z1), (y1, z1), (y1, z0), (y0, z0))]
    if any(p is None or p[2] < 0.3 for p in pr):
        return None
    xs, ys = [p[0] for p in pr], [p[1] for p in pr]
    return min(xs), min(ys), max(xs), max(ys)


def _clip(b):
    """-> (clipped box, visible fraction) or (None, 0)."""
    full = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    c = [max(0, b[0]), max(0, b[1]), min(FRAME, b[2]), min(FRAME, b[3])]
    if c[2] <= c[0] or c[3] <= c[1]:
        return None, 0.0
    return c, ((c[2] - c[0]) * (c[3] - c[1])) / full


def expected_assets(cam, cid):
    """Where each measured asset of the cabinet must appear in this view (full-image px, clipped to the frame)."""
    cab, out = GEO["cabinets"][cid], []
    P, n = cab["position"], cab["door_normal"]
    for a in cab["assets"]:
        x = P["x"] + n[0] * a.get("front_m", 0.0)
        if a["kind"] == "panel":
            b = _quad(cam, P["x"], a["y_min"], a["y_max"], a["z_min"], a["z_max"])
        else:
            y, z = P["y"] + a["dy"], P["z"] + a["dz"]
            b = _quad(cam, x, y - a["w"] / 2, y + a["w"] / 2, z - a["h"] / 2, z + a["h"] / 2)
        if b is None:
            continue
        c, vis = _clip(b)
        if c is not None and vis >= 0.3:
            out.append({"class": a["class"], "label": a["label"], "kind": a["kind"], "xyxy": [round(v, 1) for v in c],
                        "visible_fraction": round(vis, 2), "status": "expected"})
    return out


def visible_cabinets(cam):
    """Cabinets in view (same rule as synthetic_geo/make_views.py: depth, margin, viewing angle, camera in the corridor)."""
    vis, V = [], GEO["visibility"]
    if not (GEO["corridor_x"][0] <= cam.C[0] <= GEO["corridor_x"][1]):
        return vis, "camera outside the corridor between the two cabinet rows (looks through walls)"
    for cid, cab in GEO["cabinets"].items():
        P = cab["position"]
        pr = cam.project((P["x"], P["y"], P["z"]))
        if pr is None:
            continue
        u, v, d = pr
        m = V["margin_px"]
        if not (V["min_depth_m"] <= d <= V["max_depth_m"] and m <= u <= FRAME - m and m <= v <= FRAME - m):
            continue
        to_cam = (cam.C - np.array([P["x"], P["y"], P["z"]]))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(cab["door_normal"], to_cam / np.linalg.norm(to_cam)))))))
        if ang > V["max_angle_deg"]:
            continue
        vis.append({"cabinet": cid, "u": round(u, 1), "v": round(v, 1), "depth_m": round(d, 3), "view_angle_deg": round(ang, 1)})
    vis.sort(key=lambda r: r["depth_m"])
    return vis, None


def load_cameras(photos_dir: Path):
    """cameras.json of the real photos -> {file: entry}; {} if the folder is not available."""
    f = Path(photos_dir) / "cameras.json"
    if not f.is_file():
        return {}
    return {e["file"]: e for e in json.loads(f.read_text(encoding="utf-8"))}


def photo_summary(entry):
    cam = Cam(entry["position"], entry["rotation_wxyz"])
    vis, why = visible_cabinets(cam)
    scan = int(entry["file"].split("_")[1].split(".")[0]) // 6 + 1
    return {"name": entry["file"], "scan": scan, "position": dict(zip("xyz", entry["position"])), "rotation_wxyz": entry["rotation_wxyz"],
            "category": "none" if not vis else "single" if len(vis) == 1 else "multiple",
            "cabinets": [v["cabinet"] for v in vis], "image_url": f"/photos/{entry['file']}/image"}


# ---------------------------------------------------------------- export formats
def _label_description(t, base):
    lines = []
    if t.get("panel_model"):
        lines.append(f"{t['panel_model']} panel ({t.get('panel_source') or 'assumed'}" + (f", confidence {t['panel_confidence']:.2f}" if t.get("panel_confidence") is not None else "") + ")")
    if t.get("devices"):
        lines.append("Devices: " + "; ".join(f"{d['type']}" + (f" (conf {d['confidence']:.2f})" if d.get("confidence") is not None else "") + (" [needs review]" if d.get("needs_review") else "") for d in t["devices"]))
    if t.get("documents"):
        lines.append("Documents: " + " | ".join(f"{d['title']}: {base}{d['url']}" for d in t["documents"]))
    for d in t.get("devices", []):
        for doc in d.get("documents", []):
            if doc["id"] not in {x["id"] for x in t.get("documents", [])}:
                lines.append(f"{d['type']} manual: {doc['title']}: {base}{doc['url']}")
    return "\n".join(lines) or t["name"]


def manual_sheet(tags, method, base, note=None):
    rows = []
    for i, t in enumerate(tags, 1):
        docs = list(t.get("documents", []))
        seen = {d["id"] for d in docs}
        for dev in t.get("devices", []):
            docs += [d for d in dev.get("documents", []) if d["id"] not in seen and not seen.add(d["id"])]
        rows.append({"n": i, "tag_name": t["name"], "folder": t["folder"], "position_e57": t["position"],
                     "panel_model": t.get("panel_model"), "panel_source": t.get("panel_source"),
                     "devices": [d["type"] for d in t.get("devices", [])],
                     "picture_url": (base + t["image"]) if t.get("image") else None,
                     "documents": [{"title": d["title"], "pages": d.get("pages"), "url": base + d["url"]} for d in docs],
                     "needs_review": t.get("needs_review", False)})
    return {"format": "manual-tagging-sheet", "method": method, "count": len(rows), "note": note,
            "instructions": ["In the digital twin, create one tag per row at the given position (E57 coordinates; convert to the model's coordinates if they differ).",
                             "Name the tag exactly as tag_name and put it in the folder `folder`.", "Attach the picture (picture_url) and every document listed (open the URL, save the PDF, attach it).",
                             "Rows with needs_review = true should be checked on site first."], "rows": rows}


def manual_csv(sheet):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["n", "tag_name", "folder", "x", "y", "z", "panel_model", "panel_source", "devices", "picture_url", "documents", "needs_review"])
    for r in sheet["rows"]:
        p = r["position_e57"] or {}
        w.writerow([r["n"], r["tag_name"], r["folder"], p.get("x", ""), p.get("y", ""), p.get("z", ""), r["panel_model"] or "", r["panel_source"] or "",
                    "; ".join(r["devices"]), r["picture_url"] or "", " | ".join(f"{d['title']} <{d['url']}>" for d in r["documents"]), r["needs_review"]])
    return buf.getvalue()


def matterport_export(tags, method, base, fmt="model_api"):
    """Tags in the shape Matterport takes. Positions are E57 coordinates: NOT transformed to the Matterport model frame (unknown).
    Cabinets without a position are skipped and listed. Fields marked unconfirmed in `notes` must be checked against Matterport's schema."""
    out, skipped = [], []
    for t in tags:
        pos = t.get("position")
        if not pos:
            skipped.append({"tag": t["id"], "reason": "no position yet"})
            continue
        n = GEO["cabinets"].get(t["id"], {}).get("door_normal", [0, 0, 1])
        desc = _label_description(t, base)
        docs = [base + d["url"] for d in t.get("documents", [])]
        if fmt == "sdk":
            out.append({"label": t["name"], "description": desc, "anchorPosition": pos, "stemVector": {"x": n[0] * 0.4, "y": n[1] * 0.4, "z": n[2] * 0.4},
                        "attachmentUrls": ([base + t["image"]] if t.get("image") else []) + docs})
        else:
            out.append({"label": t["name"], "description": desc, "enabled": True, "anchorPosition": pos, "stemEnabled": True,
                        "stemNormal": {"x": n[0], "y": n[1], "z": n[2]}, "stemLength": 0.4, "mediaType": None, "mediaUrl": docs[0] if docs else None})
    common = {"method": method, "count": len(out), "skipped": skipped, "coordinate_frame": "E57 file coordinates (not transformed to the Matterport model frame)"}
    if fmt == "sdk":
        return {"format": "matterport-showcase-sdk", **common, "tags": out,
                "usage": "Tag.registerAttachment(...attachmentUrls) then Tag.add({label, description, anchorPosition, stemVector, attachments}) in the embedded Showcase.",
                "notes": ["Tags added through the SDK exist only in the browser session (they vanish on reload).", "Requires an SDK key and an allowed domain."]}
    mutation = ("mutation addTag($modelId: ID!, $floorId: ID!, $tag: MattertagInput!) {\n  addMattertag(modelId: $modelId, mattertag: $tag) { id }\n}")
    return {"format": "matterport-model-api", **common, "modelId": "<MATTERPORT_MODEL_ID>", "floorId": "<FLOOR_ID>", "tags": out,
            "graphql": {"mutation": mutation, "variables": [{"modelId": "<MATTERPORT_MODEL_ID>", "floorId": "<FLOOR_ID>", "tag": {**t_, "floorId": "<FLOOR_ID>"}} for t_ in out]},
            "notes": ["Model ID, floor ID and an API token (Basic auth) come from the account that owns the model; none are available yet. Dry run only: nothing is sent.",
                      "mediaType is null: its allowed values were not confirmed; check Matterport's schema reference before sending (one mediaUrl per tag).",
                      "The input type name in the mutation and the optional color/icon fields are unconfirmed; check the schema reference."]}
