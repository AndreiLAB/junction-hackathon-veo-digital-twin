"""
Combine everything into finished tags for VEO360.

Inputs
  labels_out/tags.json        cabinets, names and 3D positions      (find_labels.py)
  labels_out/label_hits.json  where each label was read             (find_labels.py)
  labels_out/products.json    relays and product words in 3D        (detect_products.py, optional)
  <kb>/data/knowledge.db      manuals per product                   (colleague's knowledge base)
  e57_images/                 scan photos                           (extract_images.py)

Output folder (default veo360_package/)
  tags_for_veo360.json        all tags: name, folder, position, documents, images, evidence
  review.html                 review page: open in a browser, approve tags
  <TAG>/                      one folder per tag: tag.json, label.jpg, photo.jpg, documents.txt

  python build_tags.py --kb C:/Users/bhand/veo360-repo
"""
import argparse
import base64
import html
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path

import cv2
import numpy as np

MAX_DIST = 0.6                 # m: a product belongs to the nearest cabinet within this distance (seen from above)
LINEUP_PRODUCTS = {"UniGear ZS2", "VD4"}   # cabinet-type products shared by a straight row of same-prefix cabinets
STRONG_CONF = 0.75             # a detector sighting counts as strong at this confidence
MIN_STRONG = 3                 # a cabinet needs this many strong sightings for the relay manual


def load_json(path, default):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def load_documents(kb):
    db = Path(kb) / "data" / "knowledge.db"
    con = sqlite3.connect(db)
    docs = {}
    for title, model, doc_type, doc_number, file_path in con.execute(
            "SELECT title, model, doc_type, doc_number, file_path FROM documents"):
        docs.setdefault(model, []).append({
            "title": title, "product": model, "type": doc_type, "doc_number": doc_number,
            "file": str((Path(kb) / "docs" / file_path).resolve())})
    return docs


def assign_products(tags, products):
    """Each product sighting goes to the nearest cabinet (top view) within MAX_DIST."""
    for t in tags:
        t["evidence"] = {}
    if not tags:
        return
    P = np.array([[t["x"], t["y"]] for t in tags])
    for f in products:
        d = np.linalg.norm(P - np.array(f["xyz"][:2]), axis=1)
        i = int(np.argmin(d))
        if d[i] <= MAX_DIST:
            tags[i]["evidence"].setdefault(f["product"], []).append(
                {"source": f["source"], "text": f["text"], "conf": f["conf"], "photo": f["photo"],
                 "distance_m": round(float(d[i]), 2)})


def confirmed(evidence):
    """Products with enough evidence to attach their manual."""
    out = []
    for product, ev in evidence.items():
        det = [e for e in ev if e["source"] == "detector"]
        ocr = [e for e in ev if e["source"] in ("ocr", "lineup")]
        if ocr or sum(1 for e in det if e["conf"] >= STRONG_CONF) >= MIN_STRONG:
            out.append(product)
    return sorted(out)


def apply_lineup(tags):
    """UniGear panels stand in a straight row with the same name prefix (H01..H05).
    If one panel shows 'UniGear' or 'VD4', its row neighbours get it too, marked as inferred."""
    prefix = lambda name: re.match(r"[A-Z]+", name).group(0) if re.match(r"[A-Z]+", name) else name
    for a in tags:
        for product in LINEUP_PRODUCTS & set(confirmed(a["evidence"])):
            for b in tags:
                if b is a or prefix(b["tag"]) != prefix(a["tag"]) or product in b["evidence"]:
                    continue
                aligned = abs(a["x"] - b["x"]) < 0.25 or abs(a["y"] - b["y"]) < 0.25
                if aligned:
                    b["evidence"][product] = [{"source": "lineup", "text": f"same row as {a['tag']}",
                                               "conf": None, "photo": None, "distance_m": None}]


def best_label_hit(tag, hits):
    own = [h for h in hits if h.get("tag") == tag and "u" in h]
    return max(own, key=lambda h: h["conf"] * h.get("match", 1)) if own else None


def marked_photo(photos_dir, hit, tag_name, products, width=1000):
    path = Path(photos_dir) / f"img_{hit['photo']:03d}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        return None, None
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    for f in products:
        if f["photo"] == hit["photo"] and "box" in f:
            x0, y0, x1, y1 = [int(v * s) for v in f["box"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 160, 0), 2)
            cv2.putText(img, f["product"], (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1, cv2.LINE_AA)
    u, v = int(hit["u"] * s), int(hit["v"] * s)
    cv2.circle(img, (u, v), 22, (40, 200, 40), 3)
    cv2.putText(img, tag_name, (u + 26, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 200, 40), 2, cv2.LINE_AA)
    return img, (u, v)


def b64(img):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode() if ok else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_out")
    ap.add_argument("--kb", required=True, help="repository folder with data/knowledge.db and docs/")
    ap.add_argument("--photos", default="e57_images")
    ap.add_argument("--out", default="veo360_package")
    ap.add_argument("--copy-docs", action="store_true", help="copy the PDFs into each tag folder (large)")
    args = ap.parse_args()

    labels = Path(args.labels)
    tags = load_json(labels / "tags.json", [])
    hits = load_json(labels / "label_hits.json", [])
    products = [p for p in load_json(labels / "products.json", []) if "xyz" in p]
    docs = load_documents(args.kb)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"{len(tags)} cabinet tags, {len(products)} product sightings, "
          f"{sum(len(v) for v in docs.values())} manuals for {', '.join(docs)}")

    assign_products(tags, products)
    apply_lineup(tags)

    records = []
    for t in tags:
        found = confirmed(t["evidence"])
        documents = [d for p in found for d in docs.get(p, [])]
        folder = out / t["tag"]
        folder.mkdir(exist_ok=True)
        images, label_img, photo_img = [], None, None
        crop = labels / "crops" / t.get("evidence_crop", "")
        if crop.is_file():
            shutil.copy(crop, folder / "label.jpg")
            images.append(f"{t['tag']}/label.jpg")
            label_img = cv2.imread(str(crop))
        hit = best_label_hit(t["tag"], hits)
        marker = None
        if hit:
            photo_img, marker = marked_photo(args.photos, hit, t["tag"], products)
            if photo_img is not None:
                cv2.imwrite(str(folder / "photo.jpg"), photo_img)
                images.append(f"{t['tag']}/photo.jpg")
        if args.copy_docs:
            for d in documents:
                shutil.copy(d["file"], folder / Path(d["file"]).name)
        (folder / "documents.txt").write_text("\n".join(d["file"] for d in documents) + "\n")
        rec = {"tag": t["tag"], "name": t["name"], "folder": t.get("folder", "Cabinets"),
               "position": {"x": t["x"], "y": t["y"], "z": t["z"]},
               "products": found, "documents": documents, "images": images,
               "marker": {"x": marker[0], "y": marker[1], "photo_width": 1000} if marker else None,
               "evidence": {"label_read_as": t.get("read_as"), "label_source": t.get("source"),
                            "label_sightings": t.get("sightings"), "products": t["evidence"]}}
        (folder / "tag.json").write_text(json.dumps(rec, indent=2))
        records.append((rec, label_img, photo_img))
        print(f"  {t['tag']:5s} {t['name']:28s} products: {', '.join(found) or '-':32s} manuals: {len(documents)}")

    (out / "tags_for_veo360.json").write_text(json.dumps([r[0] for r in records], indent=2))
    write_review_page(out, records)
    print(f"\nWrote {out}/tags_for_veo360.json, {out}/review.html and one folder per tag.")


def write_review_page(out, records):
    cards, items = [], []
    for i, (r, label_img, photo_img) in enumerate(records):
        p = r["position"]
        docs = "".join(
            f'<li><a href="{html.escape(Path(os.path.relpath(d["file"], out)).as_posix())}" target="_blank">'
            f'{html.escape(d["title"])}</a><span>{html.escape(d["product"])} · {html.escape(d["type"] or "")}</span></li>'
            for d in r["documents"]) or '<li class="none">No manual matched yet</li>'
        ev = []
        for product, lst in r["evidence"]["products"].items():
            if all(e["source"] == "lineup" for e in lst):
                why = f"inferred, {lst[0]['text']}"
            else:
                parts = []
                det = [e for e in lst if e["source"] == "detector"]
                ocr = [e for e in lst if e["source"] == "ocr"]
                if det:
                    parts.append(f"relay detector in {len(det)} photo(s), best {max(e['conf'] for e in det):.0%} sure")
                if ocr:
                    parts.append(f"text “{ocr[0]['text']}” read in {len(ocr)} photo(s)")
                why = "; ".join(parts)
            ev.append(f"<li>{html.escape(product)}: {html.escape(why)}</li>")
        ev_html = "".join(ev) or "<li>No products found near this cabinet</li>"
        imgs = ""
        if photo_img is not None:
            imgs += f'<figure><img src="{b64(photo_img)}"><figcaption>Scan photo, tag position circled</figcaption></figure>'
        if label_img is not None:
            imgs += f'<figure class="small"><img src="{b64(label_img)}"><figcaption>Label read as “{html.escape(str(r["evidence"]["label_read_as"]))}”</figcaption></figure>'
        items.append(f'<li data-i="{i}"><span class="dot"></span>{html.escape(r["name"])}</li>')
        cards.append(f'''<section class="card" id="c{i}" hidden>
  <h2>{html.escape(r["name"])}</h2><div class="sub">Tag · folder {html.escape(r["folder"])}</div>
  <div class="imgs">{imgs}</div>
  <h3>Documents ({len(r["documents"])})</h3><ul class="docs">{docs}</ul>
  <h3>Position</h3><p class="mono">X {p["x"]:.3f} · Y {p["y"]:.3f} · Z {p["z"]:.3f} m</p>
  <h3>Why this tag</h3><ul class="ev"><li>Label read as “{html.escape(str(r["evidence"]["label_read_as"]))}” ({html.escape(str(r["evidence"]["label_source"]))}), seen in {r["evidence"]["label_sightings"]} photos</li>{ev_html}</ul>
  <div class="actions"><button class="ok" data-i="{i}">Approve</button><button class="no" data-i="{i}">Reject</button></div>
</section>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>AutoTag review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#16161d;--panel:#24242e;--line:#34343f;--text:#ececf1;--muted:#9a9aa8;--accent:#6c5ce7;--green:#39b54a;--red:#e05555}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh}}
nav{{width:300px;background:var(--panel);border-right:1px solid var(--line);padding:16px;overflow:auto}}
nav h1{{font-size:18px;margin:0 0 4px}}nav .sub{{color:var(--muted);font-size:13px;margin-bottom:16px}}
nav h4{{font-size:13px;color:var(--muted);margin:12px 0 6px;display:flex;justify-content:space-between}}
nav ul{{list-style:none;margin:0;padding:0}}nav li{{padding:9px 10px;border-radius:6px;cursor:pointer;display:flex;align-items:center;gap:10px;font-size:14px}}
nav li:hover{{background:#2e2e3a}}nav li.active{{background:var(--accent)}}
.dot{{width:12px;height:12px;border-radius:3px;background:var(--green);transform:rotate(45deg)}}
nav li.approved .dot{{box-shadow:0 0 0 2px #fff}}nav li.approved::after{{content:'Approved';margin-left:auto;font-size:11px;color:var(--green)}}nav li.rejected::after{{content:'Rejected';margin-left:auto;font-size:11px;color:var(--red)}}nav li.rejected{{opacity:.45;text-decoration:line-through}}
main{{flex:1;overflow:auto;padding:24px 32px}}h2{{margin:0;font-size:24px}}.sub{{color:var(--muted);font-size:13px}}
h3{{font-size:14px;color:var(--muted);margin:22px 0 8px;font-weight:600}}
.imgs{{display:flex;gap:16px;flex-wrap:wrap;margin-top:16px}}figure{{margin:0;max-width:640px}}figure.small{{max-width:260px}}
figure img{{width:100%;border-radius:8px;display:block}}figcaption{{color:var(--muted);font-size:12px;margin-top:4px}}
ul.docs,ul.ev{{list-style:none;padding:0;margin:0}}ul.docs li{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:8px;display:flex;flex-direction:column}}
ul.docs a{{color:var(--text);font-weight:600;text-decoration:none}}ul.docs a:hover{{text-decoration:underline}}ul.docs span{{color:var(--muted);font-size:12px}}
ul.ev li{{font-size:13px;color:var(--muted);margin:4px 0}}.mono{{font-family:Consolas,monospace;font-size:13px}}
.actions{{margin-top:24px;display:flex;gap:10px}}button{{border:0;border-radius:6px;padding:9px 18px;font-size:14px;cursor:pointer;color:#fff}}
button.ok{{background:var(--green)}}button.no{{background:var(--red)}}#export{{background:var(--accent);margin-top:16px;width:100%}}
.none{{color:var(--muted)}}
</style></head><body>
<nav><h1>AutoTag review</h1><div class="sub">Proposed tags from the scan. Approve or reject each one.</div>
<h4><span>Cabinets</span><span>{len(records)}</span></h4><ul id="list">{"".join(items)}</ul>
<button id="export">Export approved tags</button></nav>
<main>{"".join(cards)}</main>
<script>
const data = {json.dumps([r[0] for r in records])};
const state = {{}};
function show(i) {{
  document.querySelectorAll('.card').forEach(c => c.hidden = true);
  document.getElementById('c' + i).hidden = false;
  document.querySelectorAll('#list li').forEach(l => l.classList.toggle('active', l.dataset.i == i));
}}
document.querySelectorAll('#list li').forEach(l => l.onclick = () => show(l.dataset.i));
document.querySelectorAll('button.ok, button.no').forEach(b => b.onclick = () => {{
  const i = b.dataset.i, li = document.querySelector('#list li[data-i="' + i + '"]');
  state[i] = b.classList.contains('ok') ? 'approved' : 'rejected';
  li.classList.toggle('approved', state[i] === 'approved');
  li.classList.toggle('rejected', state[i] === 'rejected');
  const next = Number(i) + 1; if (next < data.length) show(next);
}});
document.getElementById('export').onclick = () => {{
  const approved = data.filter((_, i) => state[i] === 'approved');
  const blob = new Blob([JSON.stringify(approved, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'approved_tags.json'; a.click();
}};
if (data.length) show(0);
</script></body></html>'''
    (out / "review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()