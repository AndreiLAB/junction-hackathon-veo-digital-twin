# Task brief for the training machine (Claude Code prompt)

You are continuing a hackathon project (Junction X Vaasa, VEO360 auto-tagging). **Deadline: Sunday 4 Oct 2026, 12:00.**
The goal of this part: train a detector for the **ABB REX615 relay** (and a look-alike display) on a geometry-driven synthetic
dataset, then measure it. Read this whole file first. Work in small steps, run what you write, show real output, and
**say plainly when something fails or an assumption is wrong. Never invent data, boxes, or results.**

## 0. Why this machine
The first machine (laptop, 7.7 GB RAM) cannot generate or train this comfortably. Use this one for: (1) generating the dataset,
(2) checking it, (3) training, (4) evaluating. Do not train anywhere else.

## 1. What exists (branch `synthetic-geo`, folder `synthetic_geo/`)
| File | Purpose |
|---|---|
| `geometry.py` | Camera model for the cube-face photos. Verified convention: `cam = diag(1,-1,-1) @ R.T @ (P - C)`, `u = f*x/z + cx`. R from `cameras.json` `rotation_wxyz`. |
| `cabinet_registry.json` | The 10 cabinets (H01-H05, VLK, OT1, TSK1, TSK2, OKK1) with xyz in the E57 frame. TSK1 was triangulated from images. H01-H05 are assumed `UniGear ZS2` (confirmed only for H05). |
| `relay_geometry.json` | Measured constants: relay frame 262 x 177 mm (wide case, from the REX615 manual), relay offset from each nameplate, front plane 0.07 m in front of the door, Cerdex display positions for TSK1/TSK2. **H01's offset comes from a single image (least certain).** |
| `make_views.py`, `image_cabinet_view.json` | Which cabinets each image sees (none / single / multiple). Cameras outside the corridor (x not in [-5.5, -2.3]) are excluded: they sit behind walls. |
| `generate_geo_synthetic.py` | Builds the YOLO dataset (see section 3). |
| `location_gate.py` | Accepts a detection only where geometry predicts a relay (position and size). **Written but not tested yet.** |
| `train_yolo.py` | Ultralytics YOLO training script. **Never run yet.** |

## 2. Data you must obtain first (not in git)
* `VEO Images/` (108 JPEG cube faces, 4096x4096, about 197 MB) and `VEO Images/cameras.json`. Get them from the team Drive.
  Do not commit them (VEO's site data).
* `synthetic_pipeline/refs/` (the 4 ABB renders) is in git. The front render `9PAA00000215623_master.jpg` is the texture source.

## 3. Step 1: generate the dataset
```bash
pip install opencv-python-headless numpy
python synthetic_geo/generate_geo_synthetic.py --images "VEO Images" --cameras "VEO Images/cameras.json" --dry-run   # expect ~1759 tiles
python synthetic_geo/generate_geo_synthetic.py --images "VEO Images" --cameras "VEO Images/cameras.json" --out synthetic_geo_dataset --shard 0/4 &
# ... shards 1/4, 2/4, 3/4 the same way (each uses up to ~1 GB RAM; use fewer shards if memory is short), then wait
```
Each shard writes `meta_K.jsonl` and `stats_K.json`; merge them (`cat meta_*.jsonl > meta.jsonl`, sum the stats). `data.yaml` is written by each shard.

Design (do not change without telling the user):
* Every sample is a **1280x1280 tile of a REAL photo**. The real H01-H05 relays are inpainted away and replaced by the render, placed by
  geometry (camera pose + cabinet position + measured offset), with appearance matched to the photo (colour gain, sharpness, noise).
* Classes: `0 abb_relion_615`, `1 other_hmi` (the Cerdex displays of TSK1/TSK2, real, kept in place, labelled). Empty tiles are background.
* Plan: about 1360 relay tiles + 171 other_hmi tiles + 150 empty tiles for **train**, 54 + 12 + 12 for **val**.
* **Split by scan**: scans 6, 8, 13 are held out (val); their photos never feed training.
* Allowed variation is deterministic and small: 9-point tile placement grid, 3 LCD contents, +-3 mm / +-0.3 deg pose error, brightness +-3%,
  JPEG 88/92/95. **No flips, no random rotation, no random sizes.**

## 4. Step 2: check it before training (mandatory)
Write `synthetic_geo/check_dataset.py` that (a) draws the labels on 40 random tiles into a contact sheet and (b) reports class counts and box-size
histogram per split. **Look at the sheet.** Specifically look for: leftover real relays that are not labelled (ghosts), double edges around a pasted
relay, boxes that do not match the relay, relays that look pasted-on. Known risks: H01 geometry; relays in cabinets outside the 10 would be
unlabelled in backgrounds. If a defect is systematic, fix the generator (not the labels) and regenerate. Report what you saw.

## 5. Step 3: real held-out evaluation set
Create `synthetic_geo/make_real_eval.py` that, for the real photos of the held-out scans (6, 8, 13), writes the geometry-predicted relay boxes and
Cerdex boxes (`real_eval_boxes.json`) plus a contact sheet. **These boxes are predictions, not ground truth: verify each by eye and drop or correct
wrong ones before using them.** Evaluate on the real (unmodified) photos, tiled the same way (1280 tiles, native resolution).

## 6. Step 4: train (GPU)
```bash
pip install ultralytics
python synthetic_geo/train_yolo.py --data synthetic_geo_dataset/data.yaml
```
YOLOv8s at 1280 px, 100 epochs, early stopping 25, **no flips / rotation / perspective** (already set in the script). If accuracy is short, try
`--model yolov8m.pt`. Keep the best weights (`runs_relay/geo_synth/weights/best.pt`); do not commit weights above 50 MB.

## 7. Step 5: evaluate honestly
Report on the REAL held-out set (not on synthetic val): precision / recall for `abb_relion_615`, and **how often the TSK1/TSK2 Cerdex display is
mistaken for a relay** (false positives). Then apply `location_gate.py` and report again, so we see the detector alone and detector + geometry
separately. Test the gate first (`python synthetic_geo/location_gate.py`). Also train/compare a baseline on the 4 raw renders only if time allows.
Be explicit that all data comes from **one site**, so the numbers show in-site performance, not generalisation to other sites.

## 8. Rules
* Python 3.11+; keep dependencies light; no hardcoded absolute paths; scripts print what they did and exit non-zero on failure.
* Never commit images, the dataset, weights over 50 MB, or secrets. Commit code and small JSON only.
* If reality contradicts this brief (a library, a file, a measurement), stop and tell the user instead of working around it quietly.
* After each step: run it, show real output, say what the next step is.

## 9. Hand-off to the backend
The backend (`backend/`, FastAPI) accepts detections at `POST /detections`:
`{image, class, conf, box:[x,y,w,h], ocr, ocr_conf, position?}` with `class` = `abb_relion_615` (the `other_hmi` class is ignored). Boxes must be in
**full-image pixels** (add the tile origin back). Produce a `detections.json` in that shape from the trained model for the 108 images, and report it.
