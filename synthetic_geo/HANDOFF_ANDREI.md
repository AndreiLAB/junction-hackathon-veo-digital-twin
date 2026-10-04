# Hand-off for Andrei's machine (next training round). Deadline: Sunday 4 Oct, 12:00

You already have a working pipeline (branch `trained-relay-model-updated`: relay + `other_hmi`, YOLOv8s, 15 epochs). This file lists, **in priority order**, what to change so
the next run is better and covers the extra assets. Do the steps in order; **stop adding scope when time runs short, but always finish steps 1-4 and 7-8.**
Suggested clock (now ~08:15): steps 1-3 by 09:00, train 09:00-10:30, evaluate + export 10:30-11:15, push by 11:15.

## What we found in your results (so the changes make sense)
* Real held-out result: precision 0.75, recall **0.25**, AP50 ~0.35 (about 30 boxes: small). Synthetic val (AP50 0.995) does not predict this.
* Your own note names a cause (the generator's camera filter left far/small relays out of training). A second cause is **OT1 geometry** (below).
* `detections.json` on your branch (`{image, boxes}`) is **not** the backend's format; use `export_detections.py` (step 7).
* `eval_real.py` has hard-coded paths (`C:/Users/gagna/...`, `E:/OSses/...`): make them command-line arguments.

## Step 1: bring in the new code and merge the geometry carefully
`git fetch origin && git merge origin/main` into your working branch (do **not** merge Pragati's model/GUI into main; keep only your own commit when you push).
Conflicts will be in `relay_geometry.json` / generator / gate, because you restructured them (`cabinets -> relays[]`). Resolve by keeping **your structure** and adding from `main`:
* `unigear_panel` and `vd4_breaker_window` blocks (measured, see `ANTIGRAVITY_PROMPT.md` section 5b);
* **H01 relay `dy = -0.07`** (was 0.0), `dz = +0.55`;
* the **corrected OT1 values** (step 2).

## Step 2: OT1 geometry (priority 1, verified on 8 photos)
Both OT1 relays are the **narrow standard case**, not the wide case. Use (full details and evidence in `OT1_CORRECTION.md`):
```json
"OT1": { "relays": [
  {"dy": -0.155, "dz":  0.145, "width_m": 0.177, "height_m": 0.177, "front_plane_m": 0.035},
  {"dy": -0.155, "dz": -0.160, "width_m": 0.177, "height_m": 0.177, "front_plane_m": 0.035}
]}
```
Make `front_plane_m` (and the body depth, same value) **per relay**, default 0.07 for H01-H05. Your first OT1 values (dy -0.10, relay A 0.262 wide) were about 5 cm off and too wide: they leave ghost relay edges around the pasted render.

## Step 3: verify BEFORE regenerating (1 minute, mandatory)
```bash
python synthetic_geo/verify_boxes.py --cabinet OT1 --images "<VEO Images>" --cameras "<VEO Images>/cameras.json" \
  --rect "A:-0.155:0.145:0.177:0.177" --rect "B:-0.155:-0.160:0.177:0.177" --front 0 --front 0.035 --front 0.07 --n 8 --out ot1_check.jpg
```
The 0.035 box (2nd colour in the legend) must hug the relay in all photos including the oblique ones. Also check H01 (`--cabinet H01 --rect "relay:-0.07:0.55:0.262:0.177" --front 0.07`).

## Step 4: regenerate the whole dataset (v3) and fix the far-relay gap
* Extend the hard-coded cabinet lists (`generate_geo_synthetic.py`: `plan`, `main`; `location_gate.py`: `Gate.expected`) to include **OT1**; set `expects_relay: true` for OT1 in `cabinet_registry.json`.
* Keep the corridor filter (camera x in [-5.5, -2.3]: scan 1 and scans 17-18 look through walls), **but make sure far pairs (3.5-6.2 m) are in TRAIN** and that val also has some small relays (your note says far relays were held out of training).
* Run `check_dataset.py` and **look at the sheets**: no ghost relay edges, boxes on the relays, OT1 relays now labelled.

## Step 5: train, then (only if time allows) add the other assets
1. **Relay + Cerdex model (must-have):** same recipe as before, `yolov8s.pt`, `imgsz=1280`, **at least 30 epochs** (15 was short; early stopping 25). If recall is still low, try `yolov8m.pt`.
2. **Extra assets (stretch):** classes `unigear_zs2_panel` (train/infer on the **full face downscaled to 1280**, a second model: a panel is 2.29 x 1.0 m) and `vd4_breaker_window`
   (native tiles; present on H02, H04, H05; **absent on H03**; H01 unmeasured: measure it with `verify_boxes.py` or leave it out). Labels are geometry-only on REAL photos. Numbers: `relay_geometry.json`, `ANTIGRAVITY_PROMPT.md` section 5b. Assumption: 36 kV UniGear.

## Step 6: evaluate honestly on REAL photos
Add OT1's two relays (and, if trained, panel/VD4 boxes) to the real evaluation boxes and **verify each box by eye**. Report precision / recall / AP50 for the relay, the Cerdex false-positive rate,
detector alone vs detector + gate, and v2 (15 epochs) vs v3. Be explicit: one site, a small test set.

## Step 7: export for the backend (use the script, do not hand-roll)
```bash
python synthetic_geo/export_detections.py --selftest
python synthetic_geo/export_detections.py --weights <best.pt> --images "<VEO Images>" --cameras "<VEO Images>/cameras.json" --out detections.json
```
Format = `POST /detections`: flat list of `{image, class, conf, box:[x,y,w,h], ocr:[], ocr_conf:null, position:{x,y,z}|null}`; `position` is null if the gate rejected it. If you trained the extra classes, extend the exporter (`vd4_breaker_window` as a device; panel as cabinet evidence).

## Step 8: push (this is what we need back)
Push to a **new branch** (e.g. `training-results-v3`), never to `main`. Commit code, `results.md`, the training log, the verified real-eval boxes, `detections.json`, and `best.pt` (< 50 MB). Do not commit photos, the dataset, or Pragati's model/GUI files.
Tell the team the moment `best.pt` and `detections.json` are on GitHub, even if the stretch classes are not done.

## Fallback if time runs out
Steps 1-4 + the relay model + export (7) are enough for a working demo: relay devices on H01-H05 **and OT1**, `needs_review` for rejected ones, UniGear panel type shown as "assumed" in the tags.
