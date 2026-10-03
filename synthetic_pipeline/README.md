\# Synthetic data and label-based tagging pipeline



Scripts for the VEO360 product-tagging challenge: synthetic training data from the

ABB REX615 reference images, plus automatic cabinet tags from the E57 scan.



\## Scripts



| Script | What it does |

| --- | --- |

| `generate\_synthetic.py` | Turns the reference images in `refs/` into a YOLO training dataset (relay + nameplate boxes) with lighting, viewpoint, occlusion, label wear and camera noise. Also writes `pipeline\_stages.jpg` and `preview\_grid.jpg` for the demo. |

| `inspect\_e57.py` | Prints what the E57 file contains (scans, points, colour, positions, photos). |

| `extract\_images.py` | Extracts the 108 embedded photos (6 per capture point) and their camera poses. |

| `contact\_sheet.py` | Makes one overview image of all extracted photos. |

| `check\_coordinates.py` | Verifies pixel-to-3D conversion on one capture point. |

| `find\_labels.py` | Reads cabinet labels (OCR + orange tapes), matches them to the 10 cabinet names, places each in 3D and writes `labels\_out/tags.csv` and `tags.json`. |



\## Folders



\- `refs/` – the four ABB reference images from the organizers.

\- `synthetic\_dataset\_sample/` – 40 generated training images with YOLO labels, plus `pipeline\_stages.jpg` and `preview\_grid.jpg`.



\## How to run



&#x20;   pip install pye57 numpy opencv-python rapidocr\_onnxruntime



&#x20;   # synthetic data (sample of 40; use --n 2000 for training)

&#x20;   python generate\_synthetic.py --ref refs/9PAA00000215623\_master.jpg:0.6 --ref refs/9PAA00000215621\_master.jpg:0.2 --ref refs/9PAA00000215619\_master.jpg:0.2 --out dataset\_test --n 40



&#x20;   # automatic tags from the scan (put cloud\_0.e57 in the same folder; it is not in git)

&#x20;   python find\_labels.py



\## Result on the eHouse scan



`find\_labels.py` found all 10 cabinets (H01–H05, VLK, OT1, TSK1, TSK2, OKK1) and placed them in 3D.



\## Not in git (on purpose)



`cloud\_0.e57` (2.3 GB, over GitHub's limit) and everything extracted from it (scan photos,

label crops), because it is VEO's site data.

