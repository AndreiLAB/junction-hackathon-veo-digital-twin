# Training-only brief (paste into Antigravity on the GPU machine)

You are an engineering agent. **You have no prior context.** This task is ONLY to validate a supplied dataset, train a detector, evaluate it honestly, and
export detections. **Do not regenerate the dataset** unless it is missing or fails step 1. Work in small steps, run what you write, show real output, and
**say plainly when something fails. Never invent data, boxes, or metrics.** Deadline: Sunday 4 Oct 2026, 12:00. Train on THIS machine only.

## 0. Context (short)
Hackathon project: auto-tag cabinets in a VEO360 digital twin. The target device is the **ABB REX615 relay** (white box, "615", ABB logo, LCD, F-buttons, READY/START/TRIP LEDs).
A second class, **`other_hmi`**, is a look-alike: the black-bezel "cerdex HP" display on cabinets TSK1/TSK2. The detector must tell them apart.
Repo: `https://github.com/AndreiLAB/junction-hackathon-veo-digital-twin` (branch `main`, code in `synthetic_geo/`). The full geometry and dataset-design
reference is `synthetic_geo/ANTIGRAVITY_PROMPT.md` (read it only if you need the geometry, e.g. for the location gate).

## 1. Inputs you are given (not in git)
* **`synthetic_geo_dataset/`** (zip, ~380 MB, handed over via the team Drive): YOLO format
  `images/{train,val}/*.jpg` (1280x1280), `labels/{train,val}/*.txt`, `data.yaml`, `meta.jsonl`, `stats.json`.
  Expected: **train ~1681** tiles (relay ~1360, other_hmi ~171, empty ~150), **val ~78**. Classes: `0 abb_relion_615`, `1 other_hmi`.
* **`VEO Images/`** (108 real cube-face photos 4096x4096 + `cameras.json`), only needed for the real evaluation in step 4.
Ask the user for the paths; do not guess.

## 1b. READ FIRST: the supplied dataset is v1 and has a KNOWN DEFECT
Read `synthetic_geo/DATASET_CARD.md` (also inside the dataset zip). In short: real ABB relays on cabinet **OT1** (and probably VLK/OKK1) are **unlabelled** in the tiles, which is systematic label noise.
**Do the remedy in section 4 of that card (measure those relays, extend the geometry, regenerate the whole dataset as v2) before the final training run**, and train v1 only as a baseline for comparison,
unless the user tells you otherwise. Report v1 vs v2 on the real held-out photos.
**v2 also adds the other assets on the cabinet** (UniGear panel, VD4 breaker window, Cerdex display as a real asset): see `DATASET_CARD.md` section 6 and `ANTIGRAVITY_PROMPT.md` section 5b for the geometry, the two-scale rule for the panel (full face downscaled to 1280) and the identification rule. Extend `export_detections.py` for the new classes.

## 2. Step 1: validate the dataset (mandatory, before any training)
Write `synthetic_geo/check_dataset.py` and run it. It must: count images/labels per split and per class; fail if an image has no label file, a label has a class other than 0/1,
or a coordinate is outside [0,1]; print box-width histograms per class; save a contact sheet of 40 random tiles per split with boxes drawn (`check_train.jpg`, `check_val.jpg`).
**Look at the sheets** and report: unlabelled leftover real relays (ghosts), double edges around a pasted relay, boxes not on the device, relays that look pasted on.
If counts are far from the expected numbers or defects are systematic, STOP and tell the user (the generator is `synthetic_geo/generate_geo_synthetic.py`).

Facts about the data, to judge it: every tile is a crop of a REAL photo with the real relays removed and replaced by a geometry-placed render; the Cerdex displays are real;
validation tiles come from held-out scans 6, 8, 13 (no leakage); relays range from ~40 px wide (far, oblique) to ~830 px (0.7 m); there are NO flips or rotations in the data.

## 3. Step 2: train (GPU)
```bash
pip install ultralytics            # plus CUDA-enabled torch if not present
python synthetic_geo/train_yolo.py --data synthetic_geo_dataset/data.yaml
```
**Algorithm: Ultralytics YOLO object detection.** Parameters (already in `train_yolo.py`): `model=yolov8s.pt` (COCO-pretrained), `imgsz=1280, epochs=100, batch=8, patience=25`,
`fliplr=0, flipud=0, degrees=0, shear=0, perspective=0, scale=0.15, translate=0.05, hsv_h=0.01, hsv_s=0.25, hsv_v=0.25, mosaic=0.3, close_mosaic=10`, project `runs_relay`, name `geo_synth`.
Why: the data already follows the real geometry, so geometric augmentation would only add unrealistic cases; text must never be mirrored. Lower `batch` if you run out of GPU memory.
If validation recall of `abb_relion_615` is below ~0.9, try `--model yolov8m.pt` or more epochs, and report both runs. Keep `best.pt`; **do not commit weights above 50 MB**.
Report the training log: final box/cls loss, precision, recall, mAP50, mAP50-95 per class on the synthetic val set. **Synthetic val numbers will look good; they do not prove real-world performance.**

## 4. Step 3: honest evaluation on REAL photos
The real photos of the held-out scans 6, 8, 13 (from `VEO Images/`), unmodified, are the real test. Build `synthetic_geo/make_real_eval.py` to write geometry-predicted relay and Cerdex boxes
(`synthetic_geo/geometry.py`, `generate_geo_synthetic.py::relay_geometry / cerdex_geometry`) as `real_eval_boxes.json` plus a contact sheet. **These are predictions, not ground truth:
verify every box by eye, drop or correct wrong ones, and report how many.** Run the detector on the real photos tiled like training (1280 px tiles, native resolution, stride 960), convert
boxes to full-image pixels, NMS, and report: precision, recall, AP50 for `abb_relion_615`; and **how often a Cerdex display is predicted as a relay (false positives)**.

## 5. Step 4: location-aware gate
`synthetic_geo/location_gate.py` predicts from each photo's camera pose where each H01-H05 relay must appear and how big, and accepts a detection only if it matches
(IoU >= 0.30, width ratio 0.6-1.6), assigning the cabinet. Test it first: `python synthetic_geo/location_gate.py` (expected: exact box accepted, displaced box rejected,
`other_hmi` rejected). Then report the real-set numbers **twice: detector alone, and detector + gate**, so the gate cannot hide model weakness.

## 6. Step 5: export for the backend (the model's output IS the backend's input)
Use `synthetic_geo/export_detections.py` (do not hand-roll this; it is tested against the backend):
```bash
python synthetic_geo/export_detections.py --selftest        # logic test, no weights needed: must print "selftest OK"
python synthetic_geo/export_detections.py --weights runs_relay/geo_synth/weights/best.pt --images "<VEO Images>" --cameras "<VEO Images>/cameras.json" --out detections.json
```
Pipeline per 4096x4096 photo: 1280 px tiles (stride 960, native resolution) -> YOLO -> boxes back to FULL-image pixels -> per-class NMS -> location gate -> backend format.
**`detections.json` is exactly what the backend takes at `POST /detections`:** a list of
`{"image": "img_067.jpg", "class": "abb_relion_615", "conf": 0.91, "box": [x, y, w, h], "ocr": [], "ocr_conf": null, "position": {"x": -5.676, "y": -4.778, "z": 1.841}}`
* `box` = top-left x, y, width, height in full-image pixels; `conf` in [0,1]; `class` must be exactly `abb_relion_615` (`other_hmi` is never exported, it is only used to reject look-alikes).
* `position` = the detection's centre ray intersected with the relay front plane of the cabinet the gate assigned (a measurement from the detection); **`null` when the gate rejected it**, so the backend flags it `needs_review`.
* Keep low-confidence and rejected detections in the file (default `--conf 0.25`): the backend flags them, nothing is silently dropped. Do not filter by confidence yourself.
* `detections_detail.json` (written next to it) adds cabinet, gate decision/reason, IoU and the `other_hmi` detections for the report.
Verified behaviour (stub model, real backend): an accepted detection became a device on cabinet H02 with the REX615 manual; a displaced one and a 0.30-confidence one were created with `needs_review` and reasons.
Report: number of detections, how many placed by the gate, how many per cabinet. If you extend the geometry (remedy), the gate and the exporter pick up the new cabinets only if `location_gate.py` and the cabinet loops are extended too (see the card, remedy step 2).

## 7. Deliverables, in order
1. `check_dataset.py` output and the two contact sheets, with the defects you saw (or "none").
2. Training log and `best.pt` (not committed if > 50 MB).
3. `real_eval_boxes.json` (verified; counts of dropped/corrected boxes).
4. A short `results.md`: synthetic val vs real held-out; detector alone vs detector + gate; Cerdex false-positive rate; 2-3 failure images.
5. `detections.json`.
Commit code and small JSON/markdown only, to a new branch (e.g. `training-results`), never to `main`.

## 8. Rules and caveats to state in your report
* No flips, rotation or random sizes anywhere; never mirror text. No images, datasets, secrets or weights > 50 MB in git.
* Unknown values are `null`; never guess a box, label or metric. No hardcoded absolute paths; scripts print what they did and exit non-zero on failure.
* **All data comes from ONE site and 28 real photos show the relay cabinets**, so results show in-site performance, not generalisation. H01's relay position was measured from a single
  image, so H01 boxes are the least certain.
