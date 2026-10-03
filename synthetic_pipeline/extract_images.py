import sys
import json
from pathlib import Path
import numpy as np
import pye57

path = sys.argv[1]
out = Path("e57_images")
out.mkdir(exist_ok=True)

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
    return q, t

e57 = pye57.E57(path)
images = e57.root["images2D"]
n = images.childCount()
print(f"Extracting {n} images ...")
cams = []
for j in range(n):
    im = images[j]
    q, t = pose(im)
    for rep_name in ("pinholeRepresentation", "sphericalRepresentation",
                     "cylindricalRepresentation", "visualReferenceRepresentation"):
        if not im.isDefined(rep_name):
            continue
        rep = im[rep_name]
        for blob, ext in (("jpegImage", "jpg"), ("pngImage", "png")):
            if rep.isDefined(blob):
                data = np.asarray(rep[blob].read_buffer(), np.uint8).tobytes()
                fname = f"img_{j:03d}.{ext}"
                (out / fname).write_bytes(data)
                cam = {"file": fname, "name": val(im, "name", ""), "type": rep_name,
                       "associated_scan_guid": val(im, "associatedData3DGuid", ""),
                       "rotation_wxyz": q, "position": t,
                       "width": val(rep, "imageWidth"), "height": val(rep, "imageHeight")}
                if rep_name == "pinholeRepresentation":
                    pw, ph = val(rep, "pixelWidth", 1.0), val(rep, "pixelHeight", 1.0)
                    cam.update(focal_px_x=val(rep, "focalLength", 0.0) / pw,
                               focal_px_y=val(rep, "focalLength", 0.0) / ph,
                               cx=val(rep, "principalPointX"), cy=val(rep, "principalPointY"))
                cams.append(cam)
                break
        break
    if (j + 1) % 10 == 0:
        print(f"  {j + 1}/{n}")

(out / "cameras.json").write_text(json.dumps(cams, indent=1))
print(f"Done: {len(cams)} images saved in the folder e57_images")
if cams:
    c = cams[0]
    print("First image:", c["file"], c["type"], f'{c["width"]}x{c["height"]}', "at position", c["position"])