# AutoTag360 – How to run and test

AutoTag360 is a desktop app that shows the automatic cabinet tagging pipeline: synthetic training
data, the trained relay detector, and the finished tags with their manuals. It runs on your own
laptop; nothing goes online.

## What you need

| What | Where to get it |
| --- | --- |
| Python 3.11 or 3.12 | python.org |
| The code and trained model | This branch, folder `synthetic_pipeline/` (model in `synthetic_pipeline/model/best.pt`) |
| The manuals | This repository: `data/knowledge.db` and `docs/` |
| The demo data zip (`autotag_data.zip`) | Ask bhand, shared privately (not on GitHub: it contains VEO site photos) |
| The scan file `cloud_0.e57` (optional) | Only needed to re-run the full pipeline or show camera positions on the room map |

## Setup (about 10 minutes)

All commands run in PowerShell or the VS Code terminal, in the repository folder.

1. Get this branch:

```
   git fetch
   git checkout trained-relay-model
```

2. Create and activate a Python environment (the prompt then starts with `(.venv)`):

```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
```

   If PowerShell says running scripts is disabled, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

3. Install the packages (a few minutes, PyTorch is large):

```
   pip install pye57 numpy opencv-python rapidocr_onnxruntime ultralytics sv-ttk
```

4. Unzip `autotag_data.zip` into the `synthetic_pipeline` folder, so you get
   `synthetic_pipeline\e57_images`, `synthetic_pipeline\labels_out`,
   `synthetic_pipeline\veo360_package` and `synthetic_pipeline\dataset`.

5. Start the app:

```
   cd synthetic_pipeline
   python autotag_gui.py
```

## First run: point the app at the model and manuals

Do this once, so the manual links point to files on your own laptop.

1. Open tab **5 Run pipeline**.
2. **Trained model** → Browse… → `synthetic_pipeline\model\best.pt`.
3. **Manuals repository** → Browse… → the repository's top folder (the one containing `data\knowledge.db`).
4. Untick every step except **Build tags and review page**, then click **Run**.
   It takes a few seconds and opens the **Results** tab.

The full pipeline (all four steps) needs `cloud_0.e57` and takes about 10 minutes;
for testing, the results in the zip are enough.

## What to try in each tab

| Tab | Try this | What you should see |
| --- | --- | --- |
| 1 Overview | Read the five step cards | 2000 training images, about 0.99 accuracy on synthetic tests, 10/10 cabinets, 6/10 tags with manuals |
| 2 Synthetic data | Click the stage picture | One ABB product photo turned into a training image, step by step |
| 3 Model | Select `img_056.jpg`, click **Detect relays**, then click a relay box | Boxes on the 615 relays; a card with the cabinet it belongs to and the REX615 manual |
| 3 Model | Move the confidence slider to 75% and detect again | Fewer boxes, fewer false alarms |
| 4 Results | Click **H05** on the room map, then the green circle in its scan photo | A tag card with 3 manuals (REX615, UniGear ZS2, VD4), the same VEO attached by hand |
| 4 Results | Click **Open review page** | The approval page in your browser: Approve, Reject, Export approved tags |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'cv2'` or `'yolo' is not recognized` | The environment isn't active: run `.venv\Scripts\Activate.ps1` (prompt must start with `(.venv)`) |
| "Choose the trained model" | Tab 5: Browse to `synthetic_pipeline\model\best.pt` |
| Manual button says "File not found" | Do the First run section once (rebuilds the links for your laptop) |
| "Choose the scan file (E57)" when clicking Run | Untick the steps that need the scan; only Build tags works without it |
| Empty photo list in tab 3 | The zip wasn't unzipped into `synthetic_pipeline` (needs `synthetic_pipeline\e57_images`) |

## Scripts in this folder

| Script | What it does |
| --- | --- |
| `generate_synthetic.py` | Turns the ABB reference images into a YOLO training dataset |
| `find_labels.py` | Reads cabinet labels with OCR and places them in 3D (10 of 10 cabinets) |
| `detect_products.py` | Finds relays (trained model) and product words ("UniGear", "VD4"), placed in 3D |
| `build_tags.py` | Attaches manuals per cabinet (relay manual needs ≥ 3 sightings at ≥ 75%), writes the review page |
| `analyze_relays.py` | Prints relay detections per cabinet |
| `autotag_gui.py` | The AutoTag360 app |
| `model/best.pt` | The trained relay detector (YOLO11n, 2,000 synthetic images, 25 epochs) |

## Data stays off GitHub

The demo zip, `e57_images`, `veo360_package` and `cloud_0.e57` contain VEO's site photos and scan.
Share them privately only.