# Self-contained task brief for the training machine (paste into Antigravity)

You are an engineering agent joining a hackathon project mid-flight. **You have no prior context: everything you need is in this file and in
the repo.** Read all of it before acting. Work in small steps, run what you write, show real command output, and **tell the user plainly when
something fails or an assumption turns out wrong. Never invent data, boxes, measurements, or results.** If reality contradicts this brief, stop and
say so instead of working around it.

**Deadline: Sunday 4 Oct 2026, 12:00** (solution + a 5-7 minute video). **Training happens on THIS machine only** (not the author's laptop).

---
## 1. Project in one paragraph
VEO delivers electrical switchgear. Customers see the site as a digital twin (VEO360 on a Matterport scan) where a person manually creates a tag on
every cabinet (name, photo, PDF manuals). We automate this. Site: the "eHouse" switchgear hall, 10 tagged cabinets
(H01-H05, VLK, OT1, TSK1, TSK2, OKK1). **Only the ABB REX615 protection relay ("615", ABB logo) is the target device.** The challenge REQUIRES training on
synthetic data made from the organisers' reference renders, and the model must work on the real scan photos. Your part: **build/check the synthetic
dataset, train the detector, evaluate it honestly on real photos, and hand detections to the backend.**

Two solutions exist in the pitch: (A) E57 file -> point cloud + photos (a teammate, Andrei); (B) **image + the camera pose it was taken from**
(this work). In (B), *location is known*, so the model is location-aware: geometry says where a relay must be and how big it must look.

## 2. Repository and data
* Repo: `https://github.com/AndreiLAB/junction-hackathon-veo-digital-twin`, branch **`synthetic-geo`** (code is in `synthetic_geo/`; read that folder first).
* **Not in git, you must obtain from the team Drive:** folder `VEO Images/` = 108 JPEG cube faces (**4096 x 4096**, ~197 MB) + `cameras.json`.
  Never commit them (VEO's site data). Reference renders ARE in git: `synthetic_pipeline/refs/9PAA00000215623_master.jpg` (front, the texture source),
  `...619` / `...621` (angled left/right), `...617` (rear terminals, unused here).
* Python 3.11+. `pip install opencv-python-headless numpy` for generation; `pip install ultralytics` (+ CUDA torch) for training.
* RAM: generation needs ~1 GB per process (the author's 7.7 GB laptop could only run 2). Use as many shards as your RAM allows.

## 3. Camera model (verified; do not re-derive)
`cameras.json` is a list of 108 entries: `file, name, type ("pinholeRepresentation"), associated_scan_guid, rotation_wxyz [w,x,y,z],
position [x,y,z], width, height (4096), focal_px_x = focal_px_y = 2048, cx = cy = 2048`  -> a 90 degree FOV pinhole cube face.
Six faces per scan; file `img_NNN.jpg` belongs to scan `NNN // 6 + 1`.

* R = rotation matrix of the quaternion `rotation_wxyz` (standard w,x,y,z).
* **Projection:** `c = diag(1,-1,-1) @ R.T @ (P - C)`; if `c.z > 0.05`: `u = fx*c.x/c.z + cx`, `v = fy*c.y/c.z + cy`.
* **Ray through pixel (u,v):** `d = R @ diag(1,-1,-1) @ [(u-cx)/fx, (v-cy)/fy, 1]`, normalised; origin = C.
* Evidence this convention is right: it is the only one of 96 tested axis conventions that reproduces four independently found label pixels, error <= 29 px.
* World frame = E57 file frame; the same frame as the cabinet positions below. Implementation: `synthetic_geo/geometry.py` (`Camera.project`, `Camera.ray`).

## 4. Cabinet registry (`synthetic_geo/cabinet_registry.json`, metres, E57 frame)
| Cabinet | x | y | z | Notes |
|---|---|---|---|---|
| H01 | -5.728 | -5.782 | 1.313 | UniGear ZS2 (assumed), expects relay |
| H02 | -5.746 | -4.513 | 1.404 | " (a photo shows the VD4 breaker window below the relay) |
| H03 | -5.737 | -3.518 | 1.402 | " |
| H04 | -5.757 | -2.516 | 1.401 | " |
| H05 | -5.760 | -1.517 | 1.401 | " (only cabinet whose manuals VEO's own tag confirmed: UniGear ZS2, VD4, REX615) |
| VLK | -2.718 | -6.550 | 1.419 | no relay expected |
| OT1 | -2.720 | -5.766 | 1.579 | " |
| TSK1 | -2.729 | -4.878 | 1.477 | ELCON cabinet, Cerdex display. Position triangulated from the orange tape in img_037/063/069/075/082 (residual <= 7 mm) |
| TSK2 | -2.726 | -4.091 | 1.552 | ELCON cabinet, Cerdex display |
| OKK1 | -2.328 | -1.948 | 2.361 | " |

The position is the **nameplate / orange tape** location, not the cabinet centre. Row H01-H05 is at x ~ -5.75; its doors face **+x** (towards the corridor).
Row VLK/OT1/TSK1/TSK2/OKK1 is at x ~ -2.7; its doors face **-x**. Along the H row, image-right = +y for a camera looking at it from the corridor.
**H01-H05 -> panel model "UniGear ZS2" is an assumption** (confirmed only for H05); keep it flagged as assumed.

## 5. Measured relay and display geometry (`synthetic_geo/relay_geometry.json`)
* **Relay = REX615 wide case:** frame **0.262 m wide x 0.177 m high** (manual: wide case 262 mm, 4U = 177 mm; the reference render's aspect 1.49 = 262/177).
  Body depth 0.07 m; **front plane 0.07 m in front of the door plane**, towards the corridor. (This reproduces the ~290 mm apparent width measured in img_067.)
* **Relay centre relative to the cabinet nameplate position, world metres:** H02-H05: `dy = -0.271, dz = +0.451`; **H01: `dy = 0.0, dz = +0.55`
  (read from ONE image, img_061: least certain)**. dx = 0 (plane of the door).
  Confirmed by eye on H02/H03/H04/H05 at 0.7 m and H03 at 2.0 m (img_067, 073, 079, 085, 025). Automatic template matching FAILED (NCC 0.3-0.4): do not use it.
* **Relay corners (3D):** `c = P_nameplate + [0, dy, dz] + n*0.07` with n = door normal (+x for H row); right-vector r = +/-y chosen per camera so that u increases;
  up = +z; front corners = `c +/- r*w/2 +/- up*h/2`; back corners = front - n*0.07. Project all 8; bbox = bbox of the 8 pixels.
* **Look-alike display (class `other_hmi`):** "cerdex HP" HMI (black bezel, small screen, RJ45 port) on TSK1 and TSK2. Size **0.157 x 0.095 m**, on the door plane
  (x = cabinet x), centres **TSK1 (y=-5.119, z=1.563)**, **TSK2 (y=-4.318, z=1.565)** (0.80 m apart = the cabinet width). Verified by projection into 14 of 16 other views.
* **Valid cameras:** corridor only, camera x in **[-5.5, -2.3]**. Scan 1 (camera at the origin, behind a steel door) and scans 17-18 look through walls: excluded.
* Visibility rule (`make_views.py`): depth 0.4-8.0 m, label inside the frame with a 150 px margin, view angle (door normal vs camera) <= 70 degrees. Occlusion is NOT checked.
  Relay used for training only if depth <= 6.2 m. Real coverage: **57 (image, H-cabinet) pairs from 28 images**, depth 0.69-6.14 m, angles 4-70 degrees.

## 6. Dataset design (`generate_geo_synthetic.py`; do not change without telling the user)
**Every sample = a 1280x1280 tile of a REAL photo** in which (a) all real H01-H05 relays are removed (OpenCV Telea inpaint at half resolution, mask dilated by
max(6 px, 12% of relay width), feathered 3 px, noise re-added) and (b) replaced by the reference render warped by the exact homography from the geometry in section 5
(side faces drawn grey: left/right ~(190,190,188), top (214), bottom (205); soft contact shadow 0.22 below). The Cerdex displays stay real and are labelled `other_hmi`.
Appearance is matched to each photo: per-channel colour gain from the real relay region (clip 0.75-1.25), blur sigma chosen from {0,.6,.9,1.2,1.6,2,2.5} to match the real
Laplacian variance, Gaussian noise 0.9 x the measured door noise.

* **Classes:** `0 abb_relion_615`, `1 other_hmi`. Label a relay only if >= 50% of its box is inside the tile; otherwise it is removed without a label.
* **Counts (train / val):** relay tiles 1360 / 54 (variants per (image, cabinet) pair: near < 1.5 m: 24; mid 1.5-3.5 m: 24; far > 3.5 m: 32; val = max(4, n/5));
  other_hmi tiles 171 / 12; empty background tiles 150 / 12 (3x3 grid tiles with no relay/display). **Total ~1759.**
* **Split by SCAN:** scans **6, 8, 13 are held out (val)**; their photos never feed training. 7 near, 31 mid, 14 far train pairs.
* **Deterministic variation only:** 9-point tile placement grid `(0.5,0.5),(0.35,0.5),(0.65,0.5),(0.5,0.35),(0.5,0.65),(0.35,0.35),(0.65,0.65),(0.35,0.65),(0.65,0.35)`;
  3 LCD contents (reference "SLD page 1/2", blank, measurement text); pose error `(dy mm, dz mm, deg)` in `(0,0,0),(3,0,0),(-3,0,0),(0,3,0),(0,-3,0),(3,3,.3),(-3,-3,-.3),(3,-3,.3),(-3,3,-.3)`;
  brightness x {0.97, 1.0, 1.03}; JPEG quality {88, 92, 95}. **No flips (mirrored text does not exist), no random rotation, no random sizes.**
* Outputs: `images/{train,val}`, `labels/{train,val}` (YOLO txt), `data.yaml`, `meta.jsonl` (source image, scan, tile origin, camera pose, expected boxes with cabinet and depth), `stats.json`.

```bash
python synthetic_geo/generate_geo_synthetic.py --images "VEO Images" --cameras "VEO Images/cameras.json" --dry-run      # expect ~1759 jobs
python synthetic_geo/generate_geo_synthetic.py --images "VEO Images" --cameras "VEO Images/cameras.json" --out synthetic_geo_dataset --shard 0/4 &   # ... 1/4, 2/4, 3/4
# then: cat meta_*.jsonl > meta.jsonl ; sum stats_*.json ; (data.yaml is written by every shard)
```
A 16-tile sample was reviewed by eye on the author's laptop: relays appear at all distances, correct perspective on oblique views, boxes follow the foreshortened shape.
**The full set has NOT been reviewed.** (Environment note: on the author's laptop an 8-process run crashed with out-of-memory.)

## 7. Step 2: check the dataset before training (mandatory)
Write `synthetic_geo/check_dataset.py`: draw labels on 40 random tiles per split into contact sheets; print class counts and box-width histograms per split; fail if any label is
outside [0,1]. **Look at the sheets.** Reject/fix systematically-bad generation (fix the generator, not the labels): ghosts (a leftover real relay with no label), double edges around a
pasted relay, boxes not matching the relay, relays that look pasted on, relays larger/smaller than neighbouring real ones. Known risks: H01 geometry (one measurement);
real relays in cabinets outside the 10 would be unlabelled background. Report what you saw, with counts.

## 8. Step 3: real held-out evaluation set
Write `synthetic_geo/make_real_eval.py`: for the real photos of scans 6, 8, 13, compute the geometry-predicted relay boxes (reuse `relay_geometry`) and Cerdex boxes, write
`real_eval_boxes.json` + a contact sheet. **These are predictions, not ground truth. Verify each box by eye; drop or correct wrong ones; record how many you dropped.** Evaluate on the
real, unmodified photos, tiled like training (1280 px, native resolution, stride 960) with boxes converted back to full-image pixels.

## 9. Step 4: train (GPU) - `synthetic_geo/train_yolo.py`
**Algorithm: Ultralytics YOLO object detector** (`yolov8s.pt`, COCO-pretrained). Parameters: `imgsz=1280, epochs=100, batch=8, patience=25, fliplr=0, flipud=0, degrees=0, shear=0,
perspective=0, scale=0.15, translate=0.05, hsv_h=0.01, hsv_s=0.25, hsv_v=0.25, mosaic=0.3, close_mosaic=10`. Rationale: the data already follows the real geometry, so geometric
augmentation would only add unrealistic cases. If recall is short, try `yolov8m.pt` or more epochs. Keep `best.pt`; do not commit weights over 50 MB. **Do not train on the real photos'
unmodified held-out scans.**

## 10. Location awareness (`synthetic_geo/location_gate.py`) - how "the model knows where it is"
The detector sees pixels only. The gate predicts, from the camera pose of the image, where each H01-H05 relay must appear and how big (box from section 5), then accepts a detection
only if it matches a predicted box (**IoU >= 0.30, width ratio 0.6-1.6**) and assigns the cabinet. A look-alike (Cerdex display, anything on a cabinet that expects no relay, anything
outside a known cabinet) is rejected / flagged `needs_review`. Test it first (`python synthetic_geo/location_gate.py`; the self-test expects: exact box accepted, displaced box rejected,
`other_hmi` rejected). **Optional second experiment, only after the baseline is reported:** a position-conditioned detector (extra input channels: predicted-relay heatmap and expected
width); `meta.jsonl` already contains each tile's pose and expected boxes for this.

## 11. Step 5: evaluate honestly (on the REAL verified held-out set, not synthetic val)
Report: precision, recall, AP50 for `abb_relion_615`; **how often the TSK1/TSK2 Cerdex display is predicted as a relay (false positives)**; then the same numbers **after the gate**, so detector-only and
detector + geometry are shown separately (the gate must not hide model weakness). If time allows, a baseline trained only on the 4 raw renders. State clearly that everything comes
from **one site**, so the numbers show in-site performance, not generalisation to other sites. Include 2-3 failure cases (images) and how `needs_review` handles them.

## 12. Hand-off to the backend
Produce `detections.json` for the 108 images: a list of `{"image": "img_067.jpg", "class": "abb_relion_615", "conf": 0.91, "box": [x, y, w, h], "ocr": [], "ocr_conf": null, "position": null}`
(`box` in FULL-image pixels = tile detections + tile origin, merged with NMS). The FastAPI backend ingests it at `POST /detections`; `class` must be exactly `abb_relion_615`
(`other_hmi` is ignored). Cabinet assignment and `needs_review` are done by the backend and the gate.

## 13. Deliverables, in order
1. Dataset generated + `check_dataset.py` report (counts, sheets, defects found/fixed).
2. `real_eval_boxes.json` verified, with the count of dropped/corrected boxes.
3. `best.pt` + training log (loss/mAP curves).
4. `results.json` / short report: detector-only vs detector+gate, false-positive rate on the Cerdex displays, failure cases.
5. `detections.json`.
Commit code and small JSON only. Push to a new branch (e.g. `training-results`), never to `main`.

## 14. Rules
* No flips, no rotation, no random sizes in generation or training. Never mirror text.
* Never commit images, the dataset, weights > 50 MB, `VEO Images/`, or secrets.
* Unknown values are `null`; never guess a box, a label or a metric.
* No hardcoded absolute paths; scripts print what they did and exit non-zero on failure.
* After each step: run it, show real output, say what the next step is.
