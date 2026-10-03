"""
AutoTag360 - demo window for the whole pipeline.

Tabs: Overview · Synthetic data · Model · Results · Run pipeline

Put this file in the same folder as find_labels.py, detect_products.py,
build_tags.py and extract_images.py, then run:

    python autotag_gui.py

Optional modern look:  pip install sv-ttk
"""
import base64
import csv
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


# ----------------------------------------------------------------------------- helpers
def default_model():
    runs = sorted((Path.home() / "runs" / "detect").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(runs[-1]) if runs else ""


def default_kb():
    for p in (HERE.parent / "veo360-repo", HERE.parent / "junction-hackathon-veo-digital-twin"):
        if (p / "data" / "knowledge.db").exists():
            return str(p)
    return ""


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def photo_from_bgr(img, max_w, max_h):
    """OpenCV image -> Tk PhotoImage, scaled to fit (no extra packages needed)."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode()) if ok else None


def photo_from_file(path, max_w, max_h):
    img = cv2.imread(str(path)) if path and Path(path).exists() else None
    return photo_from_bgr(img, max_w, max_h) if img is not None else None


def open_file(path):
    if os.name == "nt":
        os.startfile(path)
    else:
        webbrowser.open(Path(path).resolve().as_uri())


def load_kb_docs(kb):
    """Manuals per product from the team's knowledge base (data/knowledge.db)."""
    db = Path(kb) / "data" / "knowledge.db"
    docs = {}
    if not db.is_file():
        return docs
    con = sqlite3.connect(db)
    for title, model, doc_type, file_path in con.execute("SELECT title, model, doc_type, file_path FROM documents"):
        docs.setdefault(model, []).append({"title": title, "product": model, "type": doc_type,
                                           "file": str((Path(kb) / "docs" / file_path).resolve())})
    con.close()
    return docs


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def training_summary(model_path):
    """Read results.csv next to the trained model: epochs done and best mAP50."""
    if not model_path:
        return None
    run = Path(model_path).resolve().parent.parent
    f = run / "results.csv"
    if not f.exists():
        return {"run": run, "epochs": 0, "map50": None, "chart": run / "results.png"}
    rows = list(csv.DictReader(open(f, newline="")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    vals = [float(r["metrics/mAP50(B)"]) for r in rows if r.get("metrics/mAP50(B)")]
    return {"run": run, "epochs": len(rows), "map50": max(vals) if vals else None,
            "last_map50": vals[-1] if vals else None, "chart": run / "results.png"}


# ----------------------------------------------------------------------------- app
class App:
    STEPS = [
        ("extract", "Extract scan photos", "only needed once per scan"),
        ("labels", "Find cabinet labels", "names and 3D positions (OCR)"),
        ("products", "Find products", "relay detector + product words"),
        ("build", "Build tags and review page", "attach manuals, write package"),
    ]

    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.proc = None
        self.stop_requested = False
        self.images = {}                      # keep PhotoImage references alive
        root.title("AutoTag360")
        root.geometry("1280x820")
        root.minsize(1100, 720)
        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
            self.dark = True
        except Exception:
            self.dark = False
        style = ttk.Style()
        style.configure("H1.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("H2.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Big.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Muted.TLabel", foreground="#8a8a96")
        style.configure("Card.TFrame", relief="solid", borderwidth=1)

        self.e57 = tk.StringVar(value=str(HERE / "cloud_0.e57") if (HERE / "cloud_0.e57").exists() else "")
        self.model = tk.StringVar(value=default_model())
        self.kb = tk.StringVar(value=default_kb())
        self.conf = tk.DoubleVar(value=0.40)

        header = ttk.Frame(root, padding=(18, 12, 18, 4))
        header.pack(fill="x")
        ttk.Label(header, text="AutoTag360", style="H1.TLabel").pack(side="left")
        ttk.Label(header, text="   Automatic cabinet tags with manuals for VEO's digital twin",
                  style="Muted.TLabel").pack(side="left", pady=(8, 0))
        ttk.Button(header, text="⟳ Refresh", command=self.refresh_all).pack(side="right")

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=8)
        self.tab_overview = ttk.Frame(self.nb, padding=16)
        self.tab_synth = ttk.Frame(self.nb, padding=16)
        self.tab_model = ttk.Frame(self.nb, padding=16)
        self.tab_results = ttk.Frame(self.nb, padding=16)
        self.tab_run = ttk.Frame(self.nb, padding=16)
        for tab, name in ((self.tab_overview, "1  Overview"), (self.tab_synth, "2  Synthetic data"),
                          (self.tab_model, "3  Model"), (self.tab_results, "4  Results"),
                          (self.tab_run, "5  Run pipeline")):
            self.nb.add(tab, text=name)

        self.build_run_tab()
        self.build_model_tab()
        self.refresh_all()
        self.root.after(100, self.drain)

    # ======================================================================= tab 1: overview
    def build_overview(self):
        f = self.tab_overview
        for w in f.winfo_children():
            w.destroy()
        ttk.Label(f, text="From a 3D scan to finished tags, automatically", style="H2.TLabel").pack(anchor="w")
        ttk.Label(f, text="Today VEO places every cabinet tag by hand and uploads the manuals one by one. "
                          "This pipeline does it from the scan file.", style="Muted.TLabel",
                  wraplength=1150).pack(anchor="w", pady=(2, 14))

        stats = read_json(self.dataset_dir() / "stats.json", {}) if self.dataset_dir() else {}
        train = training_summary(self.model.get().strip())
        tags = read_json(HERE / "labels_out" / "tags.json", []) or []
        package = read_json(HERE / "veo360_package" / "tags_for_veo360.json", []) or []
        with_docs = sum(1 for t in package if t.get("documents"))

        steps = [
            ("1", "Synthetic data", f"{stats.get('images', '–')}", "training images",
             f"made from 4 ABB product photos · {stats.get('relay_boxes', '–')} relays marked"),
            ("2", "Relay detector", f"{train['map50']:.2f}" if train and train.get("map50") else "–",
             "accuracy on synthetic test images (mAP50)", f"YOLO model · {train['epochs'] if train else 0} training rounds · synthetic data only"),
            ("3", "Cabinet labels", f"{len(tags)}/10", "cabinets found",
             "OCR reads nameplates and orange tapes · 3D position from the scan"),
            ("4", "Products", f"{len(read_json(HERE / 'labels_out' / 'products.json', []) or [])}",
             "product sightings", "relay detector + words 'UniGear', 'VD4'"),
            ("5", "Tags + manuals", f"{with_docs}/{len(package)}" if package else "–", "tags with manuals",
             "manuals from the team's knowledge base · review page for approval"),
        ]
        row = ttk.Frame(f)
        row.pack(fill="x")
        for i, (num, title, big, unit, sub) in enumerate(steps):
            card = ttk.Frame(row, style="Card.TFrame", padding=12)
            card.grid(row=0, column=i * 2, sticky="nsew")
            row.columnconfigure(i * 2, weight=1)
            ttk.Label(card, text=f"Step {num}", style="Muted.TLabel").pack(anchor="w")
            ttk.Label(card, text=title, style="H2.TLabel").pack(anchor="w")
            ttk.Label(card, text=big, style="Big.TLabel").pack(anchor="w", pady=(8, 0))
            ttk.Label(card, text=unit).pack(anchor="w")
            ttk.Label(card, text=sub, style="Muted.TLabel", wraplength=190).pack(anchor="w", pady=(6, 0))
            if i < len(steps) - 1:
                ttk.Label(row, text="→", style="H2.TLabel").grid(row=0, column=i * 2 + 1, padx=4)

        bottom = ttk.Frame(f)
        bottom.pack(fill="both", expand=True, pady=(18, 0))
        left = ttk.Frame(bottom)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="How it fits together", style="H2.TLabel").pack(anchor="w")
        text = ("• The label reader answers: which cabinets exist, what are they called, where are they?\n"
                "• The relay detector answers: which devices are on each cabinet? It was trained only on\n"
                "  synthetic images, never on a real labelled photo.\n"
                "• The knowledge base answers: which manual belongs to which product?\n"
                "• Combined: each cabinet tag gets its name, position, photos and manuals. An engineer\n"
                "  approves them on the review page, then they go into VEO360.")
        ttk.Label(left, text=text, justify="left").pack(anchor="w", pady=(6, 0))

    # ======================================================================= tab 2: synthetic data
    def dataset_dir(self):
        for name in ("dataset", "dataset_test"):
            d = HERE / name
            if (d / "pipeline_stages.jpg").exists():
                return d
        return None

    def build_synth(self):
        f = self.tab_synth
        for w in f.winfo_children():
            w.destroy()
        d = self.dataset_dir()
        ttk.Label(f, text="Synthetic training data from 4 product photos", style="H2.TLabel").pack(anchor="w")
        ttk.Label(f, text="VEO has no labelled photos of relays, so we made our own: the product is cut out, its labels "
                          "filled with text and worn, placed in fake cabinets, then seen from new angles, in different "
                          "light, partly hidden and with camera noise. The box around each relay is known exactly.",
                  style="Muted.TLabel", wraplength=1150, justify="left").pack(anchor="w", pady=(2, 10))
        if not d:
            ttk.Label(f, text="No dataset found. Run generate_synthetic.py first.").pack(anchor="w")
            return
        stats = read_json(d / "stats.json", {})
        ttk.Label(f, text=f"Folder: {d.name}   ·   {stats.get('images', '?')} images   ·   "
                          f"{stats.get('relay_boxes', '?')} relays   ·   {stats.get('nameplate_boxes', '?')} nameplates   ·   "
                          f"{stats.get('empty_images', '?')} images without relays (to learn what is NOT a relay)").pack(anchor="w")
        pics = ttk.Frame(f)
        pics.pack(fill="both", expand=True, pady=(10, 0))
        for i, (fname, caption) in enumerate((("pipeline_stages.jpg", "One reference image, step by step"),
                                              ("preview_grid.jpg", "16 of the training images, boxes = labels"))):
            col = ttk.Frame(pics)
            col.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 12, 0))
            img = photo_from_file(d / fname, 600, 470)
            self.images[f"synth{i}"] = img
            if img:
                lbl = ttk.Label(col, image=img, cursor="hand2")
                lbl.pack(anchor="w")
                lbl.bind("<Button-1>", lambda e, p=d / fname: open_file(p))
            ttk.Label(col, text=caption + "  (click to enlarge)", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

    # ======================================================================= tab 3: model
    def build_model_tab(self):
        f = self.tab_model
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text="Test the relay detector on a real scan photo", style="H2.TLabel").pack(side="left")
        self.train_info = ttk.Label(top, text="", style="Muted.TLabel")
        self.train_info.pack(side="right")

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(body, width=260)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Scan photos").pack(anchor="w")
        lf = ttk.Frame(left)
        lf.pack(fill="y", expand=True, pady=4)
        self.photo_list = tk.Listbox(lf, width=26, height=22, activestyle="none", exportselection=False,
                                     bg="#1c1c22" if self.dark else "white", fg="#e6e6ee" if self.dark else "black",
                                     selectbackground="#6c5ce7", relief="flat", highlightthickness=0)
        sb = ttk.Scrollbar(lf, command=self.photo_list.yview)
        self.photo_list.configure(yscrollcommand=sb.set)
        self.photo_list.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")
        self.photo_list.bind("<<ListboxSelect>>", lambda e: self.preview_selected())
        ttk.Button(left, text="Other photo…", command=self.pick_other_photo).pack(fill="x", pady=(4, 8))
        ttk.Label(left, text="Minimum confidence").pack(anchor="w")
        cf = ttk.Frame(left)
        cf.pack(fill="x")
        self.conf_lbl = ttk.Label(cf, text="40%", width=5)
        ttk.Scale(cf, from_=0.1, to=0.9, variable=self.conf,
                  command=lambda v: self.conf_lbl.configure(text=f"{float(v):.0%}")).pack(side="left", fill="x", expand=True)
        self.conf_lbl.pack(side="left")
        self.detect_btn = ttk.Button(left, text="Detect relays", command=self.detect_selected)
        try:
            self.detect_btn.configure(style="Accent.TButton")
        except Exception:
            pass
        self.detect_btn.pack(fill="x", pady=(10, 4))
        ttk.Button(left, text="Show training chart", command=self.show_training_chart).pack(fill="x")

        mid = ttk.Frame(body)
        mid.pack(side="left", fill="both", expand=True, padx=14)
        self.photo_view = ttk.Label(mid, text="Choose a photo on the left.", anchor="center")
        self.photo_view.pack(fill="both", expand=True)
        self.photo_caption = ttk.Label(mid, text="", style="Muted.TLabel")
        self.photo_caption.pack(anchor="w", pady=(4, 0))

        right = ttk.Frame(body, width=250)
        right.pack(side="left", fill="y")
        ttk.Label(right, text="Found", style="H2.TLabel").pack(anchor="w")
        self.found = ttk.Treeview(right, columns=("conf",), height=14)
        self.found.heading("#0", text="Object")
        self.found.heading("conf", text="Sure")
        self.found.column("#0", width=150)
        self.found.column("conf", width=60, anchor="e")
        self.found.pack(fill="y", pady=(6, 6))
        self.found.bind("<<TreeviewSelect>>", self.on_found_select)
        ttk.Label(right, text="Each box: what the model thinks it is\nand how sure it is (0–100%).\n"
                              "Higher minimum = fewer boxes,\nfewer false alarms.\n\n"
                              "Click a box in the photo (or a row\nhere) for its documentation.",
                  style="Muted.TLabel", justify="left").pack(anchor="w")
        self.photo_view.bind("<Button-1>", self.on_view_click)
        self.photo_view.bind("<Motion>", self.on_view_motion)
        self.current_photo = None
        self.det_boxes, self.view_scale = [], None

    def fill_photo_list(self):
        self.photo_list.delete(0, "end")
        self.photo_files = sorted((HERE / "e57_images").glob("img_*.jpg"))
        for p in self.photo_files:
            self.photo_list.insert("end", p.name)
        good = [i for i, p in enumerate(self.photo_files) if p.name in ("img_056.jpg", "img_055.jpg")]
        if good:
            self.photo_list.selection_set(good[-1])
            self.photo_list.see(good[-1])
            self.preview_selected()
        train = training_summary(self.model.get().strip())
        if train and train.get("map50") is not None:
            self.train_info.configure(text=f"Model: {Path(self.model.get()).name} · {train['epochs']} rounds · "
                                           f"best mAP50 {train['map50']:.3f} on synthetic validation images")
        else:
            self.train_info.configure(text="Model: " + (self.model.get() or "not chosen (see Run pipeline tab)"))

    def show_image(self, path_or_img, caption, boxes=None, orig_w=None):
        img = (photo_from_bgr(path_or_img, 760, 600) if isinstance(path_or_img, np.ndarray)
               else photo_from_file(path_or_img, 760, 600))
        self.images["model_view"] = img
        self.det_boxes = boxes or []
        self.view_scale = (img.width() / orig_w) if (img and orig_w) else None
        self.photo_view.configure(image=img, text="" if img else "Could not open image.")
        self.photo_caption.configure(text=caption)

    def preview_selected(self):
        sel = self.photo_list.curselection()
        if not sel:
            return
        self.current_photo = self.photo_files[sel[0]]
        self.found.delete(*self.found.get_children())
        self.show_image(self.current_photo, f"{self.current_photo.name} · original scan photo · click 'Detect relays'")

    def pick_other_photo(self):
        p = filedialog.askopenfilename(title="Choose a photo",
                                       filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if p:
            self.photo_list.selection_clear(0, "end")
            self.current_photo = Path(p)
            self.found.delete(*self.found.get_children())
            self.show_image(self.current_photo, f"{self.current_photo.name} · original photo · click 'Detect relays'")

    def detect_selected(self):
        model = self.model.get().strip()
        if not Path(model).is_file():
            messagebox.showwarning("AutoTag360", "Choose the trained model (best.pt) on the 'Run pipeline' tab.")
            return
        if not self.current_photo:
            messagebox.showinfo("AutoTag360", "Choose a photo first.")
            return
        out_dir = HERE / "test_predictions"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / (self.current_photo.stem + "_result.png")
        conf = round(float(self.conf.get()), 2)
        code = ("import sys, json, cv2; from ultralytics import YOLO; "
                "r = YOLO(sys.argv[1]).predict(sys.argv[2], imgsz=1280, conf=float(sys.argv[4]), device='cpu', verbose=False)[0]; "
                "cv2.imwrite(sys.argv[3], r.plot()); "
                "print('JSON' + json.dumps({'w': int(r.orig_shape[1]), 'boxes': [[r.names[int(c)], float(p), [float(v) for v in b]] "
                "for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]}))")
        self.detect_btn.configure(state="disabled", text="Detecting …")
        photo = self.current_photo

        def run():
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            t0 = time.time()
            res = subprocess.run([sys.executable, "-c", code, model, str(photo), str(out), str(conf)], cwd=HERE, env=env,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 creationflags=CREATE_NO_WINDOW)
            self.q.put(("detect_done", (photo, out, conf, res, time.time() - t0)))
        threading.Thread(target=run, daemon=True).start()

    def detect_done(self, photo, out, conf, res, secs):
        self.detect_btn.configure(state="normal", text="Detect relays")
        data = {"w": None, "boxes": []}
        for line in res.stdout.splitlines():
            if line.startswith("JSON"):
                data = json.loads(line[4:])
        if res.returncode != 0 or not out.exists():
            messagebox.showerror("AutoTag360", "Detection failed:\n\n" + res.stderr[-1200:])
            return
        boxes = sorted(data["boxes"], key=lambda x: -x[1])
        self.found.delete(*self.found.get_children())
        for i, (name, p, _) in enumerate(boxes):
            self.found.insert("", "end", iid=str(i), text=name, values=(f"{p:.0%}",))
        relays = sum(1 for n, _, _ in boxes if n == "abb_relion_615")
        self.det_photo_name = photo.name
        self.show_image(out, f"{photo.name} · {relays} relay(s), {len(boxes) - relays} other · "
                             f"min. {conf:.0%} sure · {secs:.1f} s · click a box for its documentation",
                        boxes=boxes, orig_w=data["w"])

    def box_at(self, x, y):
        """Index of the detected box under a click on the displayed photo (smallest box wins)."""
        if not self.det_boxes or not self.view_scale:
            return None
        img = self.images.get("model_view")
        ox = (self.photo_view.winfo_width() - img.width()) / 2
        oy = (self.photo_view.winfo_height() - img.height()) / 2
        px, py = (x - ox) / self.view_scale, (y - oy) / self.view_scale
        hits = [(abs((b[2] - b[0]) * (b[3] - b[1])), i) for i, (_, _, b) in enumerate(self.det_boxes)
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
        return min(hits)[1] if hits else None

    def on_view_motion(self, event):
        self.photo_view.configure(cursor="hand2" if self.box_at(event.x, event.y) is not None else "")

    def on_view_click(self, event):
        i = self.box_at(event.x, event.y)
        if i is not None:
            self.detection_card(i)

    def on_found_select(self, _):
        sel = self.found.selection()
        if sel and self.det_boxes:
            self.detection_card(int(sel[0]))

    def cabinet_for_box(self, box):
        """If this is a scan photo, match the box to a relay the pipeline placed in 3D and return its cabinet tag."""
        m = re.match(r"img_(\d+)\.jpg$", getattr(self, "det_photo_name", "") or "")
        if not m:
            return None
        photo = int(m.group(1))
        finds = [f for f in (read_json(HERE / "labels_out" / "products.json", []) or [])
                 if f.get("photo") == photo and f.get("source") == "detector" and "box" in f and "xyz" in f]
        best = max(finds, key=lambda f: iou(box, f["box"]), default=None)
        if not best or iou(box, best["box"]) < 0.3:
            return None
        tags = read_json(HERE / "veo360_package" / "tags_for_veo360.json", []) or []
        if not tags:
            return None
        d = [np.hypot(t["position"]["x"] - best["xyz"][0], t["position"]["y"] - best["xyz"][1]) for t in tags]
        k = int(np.argmin(d))
        return tags[k] if d[k] <= 0.6 else None

    def detection_card(self, i):
        name, conf, box = self.det_boxes[i]
        win = tk.Toplevel(self.root)
        win.geometry("+%d+%d" % (self.root.winfo_rootx() + 160, self.root.winfo_rooty() + 140))
        f = ttk.Frame(win, padding=18)
        f.pack(fill="both", expand=True)
        src = cv2.imread(str(HERE / "test_predictions" / (Path(self.det_photo_name).stem + "_result.png")))
        if src is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = int(0.15 * max(x1 - x0, y1 - y0))
            crop = src[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            ph = photo_from_bgr(crop, 380, 260) if crop.size else None
            if ph:
                self.images["det_card"] = ph
                ttk.Label(f, image=ph).pack(anchor="w", pady=(0, 10))
        if name == "abb_relion_615":
            win.title("ABB 615 relay")
            ttk.Label(f, text="ABB 615 series relay", style="H1.TLabel").pack(anchor="w")
            ttk.Label(f, text=f"Detected by the model, {conf:.0%} sure", style="Muted.TLabel").pack(anchor="w")
            tag = self.cabinet_for_box(box)
            if tag:
                ttk.Label(f, text=f"On cabinet: {tag['name']}", style="H2.TLabel").pack(anchor="w", pady=(10, 2))
                ttk.Button(f, text="Open this cabinet's tag", command=lambda t=tag: self.tag_card(t)).pack(anchor="w")
            docs = load_kb_docs(self.kb.get().strip()).get("ABB 615", [])
            ttk.Label(f, text=f"Documentation ({len(docs)})", style="H2.TLabel").pack(anchor="w", pady=(12, 4))
            for d in docs:
                ttk.Button(f, text=f"📄  {d['title']}  ({d['type'] or d['product']})",
                           command=lambda p=d["file"]: open_file(p) if Path(p).exists()
                           else messagebox.showinfo("AutoTag360", f"File not found:\n{p}")).pack(fill="x", pady=2)
            if not docs:
                ttk.Label(f, text="No manual found: check the manuals repository on the Run pipeline tab.",
                          style="Muted.TLabel").pack(anchor="w")
        else:
            win.title("Nameplate")
            ttk.Label(f, text="Nameplate / label", style="H1.TLabel").pack(anchor="w")
            ttk.Label(f, text=f"Detected by the model, {conf:.0%} sure", style="Muted.TLabel").pack(anchor="w")
            ttk.Label(f, text="A text label on a cabinet or device. The label reader (OCR) reads labels like this\n"
                              "to name the cabinet tags, e.g. \"H05 – SOLAR 2\". See the Results tab.",
                      justify="left").pack(anchor="w", pady=(10, 0))
        ttk.Button(f, text="Close", command=win.destroy).pack(anchor="e", pady=(14, 0))

    def show_training_chart(self):
        train = training_summary(self.model.get().strip())
        if not train or not Path(train["chart"]).exists():
            messagebox.showinfo("AutoTag360", "No training chart yet (results.png appears when training finishes).")
            return
        self.found.delete(*self.found.get_children())
        self.show_image(train["chart"], "Training curves: losses go down, precision / recall / mAP go up = the model is learning")

    # ======================================================================= tab 4: results
    def build_results(self):
        f = self.tab_results
        for w in f.winfo_children():
            w.destroy()
        package = read_json(HERE / "veo360_package" / "tags_for_veo360.json", None)
        tags = package if package is not None else [
            {"tag": t["tag"], "name": t["name"], "position": {"x": t["x"], "y": t["y"], "z": t["z"]},
             "products": [], "documents": [], "images": []}
            for t in (read_json(HERE / "labels_out" / "tags.json", []) or [])]
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text=f"Proposed tags: {len(tags)} cabinets", style="H2.TLabel").pack(side="left")
        b = ttk.Button(top, text="Open review page (for approval)", command=self.open_review)
        try:
            b.configure(style="Accent.TButton")
        except Exception:
            pass
        b.pack(side="right")
        if not tags:
            ttk.Label(f, text="No tags yet. Run the pipeline first.").pack(anchor="w", pady=10)
            return

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        cols = ("x", "y", "z", "products", "manuals")
        tv = ttk.Treeview(left, columns=cols, height=11)
        tv.heading("#0", text="Tag")
        for c, w, t in (("x", 62, "X (m)"), ("y", 62, "Y (m)"), ("z", 56, "Z (m)"),
                        ("products", 210, "Products found"), ("manuals", 64, "Manuals")):
            tv.heading(c, text=t)
            tv.column(c, width=w, anchor="w" if c == "products" else "e")
        tv.column("#0", width=250)
        for i, t in enumerate(tags):
            p = t["position"]
            tv.insert("", "end", iid=str(i), text=t["name"],
                      values=(f"{p['x']:.2f}", f"{p['y']:.2f}", f"{p['z']:.2f}",
                              ", ".join(t.get("products", [])) or "–", len(t.get("documents", []))))
        tv.pack(fill="x")

        detail = ttk.Frame(left)
        detail.pack(fill="both", expand=True, pady=(12, 0))
        imgs = ttk.Frame(detail)
        imgs.pack(side="left", anchor="n")
        self.det_photo = ttk.Label(imgs, cursor="hand2")
        self.det_photo.pack(anchor="w")
        self.det_img = ttk.Label(imgs, cursor="hand2")
        self.det_img.pack(anchor="w", pady=(6, 0))
        ttk.Label(imgs, text="Click the green tag for its documents,\nanywhere else to enlarge", style="Muted.TLabel",
                  justify="left").pack(anchor="w")
        dr = ttk.Frame(detail)
        dr.pack(side="left", fill="both", expand=True, padx=12)
        self.det_title = ttk.Label(dr, text="", style="H2.TLabel")
        self.det_title.pack(anchor="w")
        self.det_text = ttk.Label(dr, text="Select a tag above.", style="Muted.TLabel", justify="left", wraplength=380)
        self.det_text.pack(anchor="w", pady=(4, 6))
        self.det_docs = ttk.Frame(dr)
        self.det_docs.pack(anchor="w", fill="x")

        mapf = ttk.Frame(body)
        mapf.pack(side="left", fill="y", padx=(16, 0))
        ttk.Label(mapf, text="Room map (seen from above) · click a tag").pack(anchor="w")
        self.map = tk.Canvas(mapf, width=380, height=520, bg="#1c1c22" if self.dark else "#f4f4f6", highlightthickness=0)
        self.map.pack(pady=(4, 0))
        self.result_tags, self.result_tv = tags, tv
        self.draw_map(tags, None)
        self.map.bind("<Button-1>", self.on_map_click)
        self.map.bind("<Motion>", self.on_map_motion)

        def on_select(_):
            sel = tv.selection()
            if sel:
                self.show_tag(tags[int(sel[0])])
                self.draw_map(tags, int(sel[0]))
        tv.bind("<<TreeviewSelect>>", on_select)
        tv.selection_set("0")

    def tag_at(self, x, y):
        for px, py, i in getattr(self, "map_pts", []):
            if abs(x - px) <= 12 and abs(y - py) <= 12:
                return i
        return None

    def on_map_click(self, event):
        i = self.tag_at(event.x, event.y)
        if i is not None:
            self.result_tv.selection_set(str(i))
            self.result_tv.see(str(i))

    def on_map_motion(self, event):
        self.map.configure(cursor="hand2" if self.tag_at(event.x, event.y) is not None else "")

    def show_tag(self, t):
        pkg = HERE / "veo360_package"
        label_path = next((pkg / r for r in t.get("images", []) if r.endswith("label.jpg")), None)
        photo_path = next((pkg / r for r in t.get("images", []) if r.endswith("photo.jpg")), None)
        img = photo_from_file(label_path, 300, 110) if label_path else None
        ph = photo_from_file(photo_path, 300, 230) if photo_path else None
        self.images["tag_label"], self.images["tag_photo"] = img, ph
        self.det_img.configure(image=img if img else "", text="" if img else "(no label photo)")
        self.det_photo.configure(image=ph if ph else "", text="" if ph else "(no scan photo)")
        self.det_img.bind("<Button-1>", lambda e, p=label_path: p and open_file(p))
        self.cur_tag, self.cur_photo_path = t, photo_path
        self.cur_photo_scale = (ph.width() / 1000.0) if ph else None
        self.det_photo.bind("<Button-1>", self.on_photo_click)
        self.det_photo.bind("<Motion>", self.on_photo_motion)
        self.det_title.configure(text=t["name"])
        ev = t.get("evidence", {})
        lines = [f"Folder: {t.get('folder', 'Cabinets')}"]
        if ev.get("label_read_as"):
            lines.append(f"Name read as “{ev['label_read_as']}” ({ev.get('label_source', '')}), "
                         f"seen in {ev.get('label_sightings', '?')} photos")
        for product, lst in (ev.get("products") or {}).items():
            src = sorted({e["source"] for e in lst})
            lines.append(f"{product}: " + ("inferred, " + lst[0]["text"] if src == ["lineup"]
                                           else f"{len(lst)} sighting(s) by {', '.join(src)}"))
        self.det_text.configure(text="\n".join(lines))
        for w in self.det_docs.winfo_children():
            w.destroy()
        docs = t.get("documents", [])
        ttk.Label(self.det_docs, text=f"Documents ({len(docs)})", style="H2.TLabel").pack(anchor="w", pady=(4, 0))
        if docs:
            ttk.Label(self.det_docs, text="Click a document to open it", style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        for d in docs:
            ttk.Button(self.det_docs, text=f"📄  {d['title']}  ({d['product']})",
                       command=lambda p=d["file"]: open_file(p) if Path(p).exists()
                       else messagebox.showinfo("AutoTag360", f"File not found:\n{p}")).pack(anchor="w", pady=1, fill="x")
        if not docs:
            ttk.Label(self.det_docs, text="No manual matched.", style="Muted.TLabel").pack(anchor="w")

    def on_tag_marker(self, x, y):
        m, s = (self.cur_tag or {}).get("marker"), self.cur_photo_scale
        if not m or not s:
            return False
        return (x - m["x"] * s) ** 2 + (y - m["y"] * s) ** 2 <= max(14, 30 * s) ** 2

    def on_photo_motion(self, event):
        self.det_photo.configure(cursor="hand2" if self.on_tag_marker(event.x, event.y) else "plus")

    def on_photo_click(self, event):
        if self.on_tag_marker(event.x, event.y):
            self.tag_card(self.cur_tag)
        elif self.cur_photo_path:
            open_file(self.cur_photo_path)

    def tag_card(self, t):
        """A small panel like VEO360's tag view: name, folder, documents to open."""
        win = tk.Toplevel(self.root)
        win.title(t["name"])
        win.geometry("+%d+%d" % (self.root.winfo_rootx() + 120, self.root.winfo_rooty() + 120))
        f = ttk.Frame(win, padding=18)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text=t["name"], style="H1.TLabel").pack(anchor="w")
        ttk.Label(f, text=f"Tag · {t.get('folder', 'Cabinets')}", style="Muted.TLabel").pack(anchor="w", pady=(0, 10))
        pkg = HERE / "veo360_package"
        for rel in t.get("images", []):
            if rel.endswith("photo.jpg"):
                ph = photo_from_file(pkg / rel, 420, 300)
                if ph:
                    self.images["card_photo"] = ph
                    lbl = ttk.Label(f, image=ph, cursor="hand2")
                    lbl.pack(anchor="w")
                    lbl.bind("<Button-1>", lambda e, p=pkg / rel: open_file(p))
        docs = t.get("documents", [])
        ttk.Label(f, text=f"Documents ({len(docs)})", style="H2.TLabel").pack(anchor="w", pady=(12, 4))
        for d in docs:
            ttk.Button(f, text=f"📄  {d['title']}  ({d['product']})",
                       command=lambda p=d["file"]: open_file(p) if Path(p).exists()
                       else messagebox.showinfo("AutoTag360", f"File not found:\n{p}")).pack(fill="x", pady=2)
        if not docs:
            ttk.Label(f, text="No manual matched for this cabinet.", style="Muted.TLabel").pack(anchor="w")
        ttk.Button(f, text="Close", command=win.destroy).pack(anchor="e", pady=(14, 0))

    def draw_map(self, tags, selected):
        c = self.map
        c.delete("all")
        W, H, pad = int(c["width"]), int(c["height"]), 30
        pts = [(t["position"]["x"], t["position"]["y"]) for t in tags] + list(self.capture_points())
        if not pts:
            return
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5
        s = min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0))
        to = lambda x, y: (pad + (x - x0) * s, H - pad - (y - y0) * s)
        fg = "#e6e6ee" if self.dark else "#222"
        for (x, y) in self.capture_points():
            px, py = to(x, y)
            c.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#6c6c7a", outline="")
        self.map_pts = []
        for i, t in enumerate(tags):
            px, py = to(t["position"]["x"], t["position"]["y"])
            self.map_pts.append((px, py, i))
            has_docs = bool(t.get("documents"))
            r = 8 if i == selected else 6
            col = "#6c5ce7" if i == selected else ("#39b54a" if has_docs else "#e0a030")
            c.create_rectangle(px - r, py - r, px + r, py + r, fill=col, outline="")
            c.create_text(px + 12, py, text=t["tag"], anchor="w", fill=fg, font=("Segoe UI", 9, "bold" if i == selected else "normal"))
        lx = pad
        for col, label, shape in (("#39b54a", "manuals attached", "rect"), ("#e0a030", "no manual", "rect"),
                                  ("#6c6c7a", "camera position", "oval")):
            (c.create_rectangle if shape == "rect" else c.create_oval)(lx, H - 15, lx + 9, H - 6, fill=col, outline="")
            c.create_text(lx + 13, H - 10, anchor="w", fill="#8a8a96", font=("Segoe UI", 8), text=label)
            lx += 118

    def capture_points(self):
        if hasattr(self, "_cps"):
            return self._cps
        self._cps = []
        try:
            import pye57
            e = pye57.E57(self.e57.get())
            for i in range(e.scan_count):
                h = e.get_header(i)
                if h.has_pose:
                    t = h.translation
                    if abs(t[2] - 1.45) < 0.3:              # indoor capture points (camera at ~1.45 m)
                        self._cps.append((float(t[0]), float(t[1])))
        except Exception:
            pass
        return self._cps

    # ======================================================================= tab 5: run pipeline
    def build_run_tab(self):
        f = self.tab_run
        ttk.Label(f, text="Run the pipeline on a scan", style="H2.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        for r, (label, var, kind) in enumerate((("Scan file (E57)", self.e57, "e57"),
                                                ("Trained model (best.pt)", self.model, "pt"),
                                                ("Manuals repository", self.kb, "dir")), start=1):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(f, textvariable=var).grid(row=r, column=1, sticky="ew", padx=8)
            ttk.Button(f, text="Browse…", command=lambda v=var, k=kind: self.browse(v, k)).grid(row=r, column=2)
        f.columnconfigure(1, weight=1)
        steps = ttk.LabelFrame(f, text="Steps", padding=10)
        steps.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        self.step_on = {}
        have_photos = any((HERE / "e57_images").glob("img_*.jpg")) if (HERE / "e57_images").is_dir() else False
        for i, (key, label, hint) in enumerate(self.STEPS):
            var = tk.BooleanVar(value=not (key == "extract" and have_photos))
            self.step_on[key] = var
            ttk.Checkbutton(steps, text=label, variable=var).grid(row=i, column=0, sticky="w", pady=1)
            ttk.Label(steps, text=hint, style="Muted.TLabel").grid(row=i, column=1, sticky="w", padx=12)
        bar = ttk.Frame(f)
        bar.grid(row=5, column=0, columnspan=3, sticky="ew")
        self.run_btn = ttk.Button(bar, text="Run", command=self.start)
        try:
            self.run_btn.configure(style="Accent.TButton")
        except Exception:
            pass
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=12)
        self.progress = ttk.Progressbar(f, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        self.log = tk.Text(f, height=16, wrap="none", font=("Consolas", 9), bg="#16161d", fg="#e6e6ee",
                           insertbackground="#fff", relief="flat")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(7, weight=1)

    def browse(self, var, kind):
        if kind == "dir":
            p = filedialog.askdirectory()
        else:
            types = [("E57 scan", "*.e57")] if kind == "e57" else [("YOLO model", "*.pt")]
            p = filedialog.askopenfilename(filetypes=types + [("All", "*.*")])
        if p:
            var.set(p)
            if kind == "pt":
                self.fill_photo_list()

    def commands(self):
        py, e57, model, kb = sys.executable, self.e57.get().strip(), self.model.get().strip(), self.kb.get().strip()
        cmds = []
        if self.step_on["extract"].get():
            cmds.append(("Extracting scan photos", [py, str(HERE / "extract_images.py"), e57]))
        if self.step_on["labels"].get():
            code = (f"import sys; sys.path.insert(0, r'{HERE}'); import find_labels as F; "
                    f"F.E57_FILE = r'{e57}'; F.main()")
            cmds.append(("Finding cabinet labels", [py, "-c", code]))
        if self.step_on["products"].get():
            extra = ["--model", model] if model else ["--no-model"]
            cmds.append(("Finding products", [py, str(HERE / "detect_products.py"), "--e57", e57] + extra))
        if self.step_on["build"].get():
            cmds.append(("Building tags and review page", [py, str(HERE / "build_tags.py"), "--kb", kb]))
        return cmds

    def check(self):
        e57, model, kb = self.e57.get().strip(), self.model.get().strip(), self.kb.get().strip()
        if any(self.step_on[k].get() for k in ("extract", "labels", "products")) and not Path(e57).is_file():
            return "Choose the scan file (E57)."
        if self.step_on["products"].get() and model and not Path(model).is_file():
            return "The model file does not exist. Clear the field to run without the model."
        if self.step_on["build"].get():
            if not (Path(kb) / "data" / "knowledge.db").is_file():
                return "Choose the repository folder that contains data/knowledge.db."
            if not self.step_on["labels"].get() and not (HERE / "labels_out" / "tags.json").is_file():
                return "No cabinet tags yet: also tick 'Find cabinet labels'."
        if not any(v.get() for v in self.step_on.values()):
            return "Tick at least one step."
        return None

    def start(self):
        problem = self.check()
        if problem:
            messagebox.showwarning("AutoTag360", problem)
            return
        self.stop_requested = False
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log.delete("1.0", "end")
        cmds = self.commands()
        self.progress.configure(maximum=len(cmds), value=0)
        threading.Thread(target=self.worker, args=(cmds,), daemon=True).start()

    def worker(self, cmds):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        t0 = time.time()
        for i, (title, cmd) in enumerate(cmds):
            self.q.put(("status", f"Step {i + 1}/{len(cmds)}: {title} …"))
            self.q.put(("log", f"\n=== {title} ===\n"))
            self.proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", creationflags=CREATE_NO_WINDOW)
            for line in self.proc.stdout:
                if "Warning" not in line and "warn(" not in line:
                    self.q.put(("log", line))
            code = self.proc.wait()
            if self.stop_requested:
                self.q.put(("done", "Stopped."))
                return
            if code != 0:
                self.q.put(("done", f"Step '{title}' failed (exit code {code}). See the log."))
                return
            self.q.put(("progress", i + 1))
        self.q.put(("done", f"Finished in {(time.time() - t0) / 60:.1f} min."))
        self.q.put(("refresh", None))

    def stop(self):
        self.stop_requested = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    # ======================================================================= plumbing
    def drain(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val)
                    self.log.see("end")
                elif kind == "status":
                    self.status.set(val)
                elif kind == "progress":
                    self.progress.configure(value=val)
                elif kind == "done":
                    self.status.set(val)
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                elif kind == "refresh":
                    self.refresh_all()
                    self.nb.select(self.tab_results)
                elif kind == "detect_done":
                    self.detect_done(*val)
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def refresh_all(self):
        if hasattr(self, "_cps"):
            del self._cps
        self.build_overview()
        self.build_synth()
        self.fill_photo_list()
        self.build_results()

    def open_review(self):
        page = HERE / "veo360_package" / "review.html"
        if page.exists():
            open_file(page)
        else:
            messagebox.showinfo("AutoTag360", "No review page yet. Run the pipeline first.")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()