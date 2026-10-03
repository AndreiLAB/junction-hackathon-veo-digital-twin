# junction-hackathon-veo-digital-twin

Digital twin platform for VEO company - Junction Hackathon Team JSC

## VEO Visual Localization Pipeline

This repository contains the complete research and prototype visual localization pipeline built for ASTM E57 digital twins (Matterport / VEO360 scans).

Given a 2D query image, the pipeline predicts:
- Most likely scan sweep / viewpoint
- 3D Camera Position $(X, Y, Z)$
- Viewing Direction $(\text{Yaw}, \text{Pitch}, \text{Roll})$ and forward camera vector
- Top-K nearest visual candidates with cosine similarity scores

---

### Key Capabilities

1. **Streaming E57 Node Extraction (`extract_e57.py`)**:
   - Parses `Data3D` scans and `Images2D` nodes without loading multi-gigabyte point clouds into memory.
   - Decodes original embedded skybox images (JPEG/PNG).
   - Extracts camera translation and quaternion rotation $(q_w, q_x, q_y, q_z)$.
   - Computes derived 3D rotation matrices, forward vectors, and spherical orientation (Yaw/Pitch/Roll).

2. **Metadata & Relational Storage (`db/database.py`)**:
   - Structured SQLite storage (`dataset/veo_localization.db`) with unique GUID constraints for scans, images, and feature vectors.

3. **Direction Calibration (`calibrate_direction.py`)**:
   - Generates contact sheets of skybox faces $(0\text{--}5)$ alongside computed pose coordinates and forward vectors to verify camera coordinate axes.

4. **Deep Vision Embeddings (`build_embeddings.py`)**:
   - Extracts semantic visual features using pretrained **DINOv2** (`dinov2_vits14`), $L_2$-normalized for cosine similarity retrieval.

5. **Multi-Task & Leave-One-Scan-Out Modeling (`train_model.py`)**:
   - Lightweight neural head predicting scan locations and discrete direction sectors.
   - Evaluates generalization across unseen scans.

6. **Rigorous Retrieval Evaluation (`evaluate.py`)**:
   - Tests visual localization using dynamically generated aggressive augmentations (Color jitter, random perspective, random resized crop, Gaussian blur).
   - Generates retrieval contact sheets (`QUERY | TOP 1 | TOP 2 | TOP 3 | TOP 4 | TOP 5`) and comprehensive metrics.

7. **Inference & Visualization (`localize.py`, `visualize_localization.py`)**:
   - CLI tool for querying any input image against the indexed digital twin and plotting visual localization sheets.

---

### Project Structure

```
├── config.py                     # Central configuration (paths, axes, models)
├── extract_e57.py                # E57 image, pose, and metadata extractor
├── inspect_dataset.py            # Quality check & HTML dataset visualizer
├── calibrate_direction.py        # Direction calibration & skybox contact sheet
├── build_embeddings.py           # DINOv2 feature extraction pipeline
├── train_model.py                # Multi-task orientation & scan classifier
├── evaluate.py                   # Rigorous retrieval benchmark suite
├── localize.py                   # Single-image query visual localization
├── visualize_localization.py     # Visual query debugger
├── pipeline.py                   # End-to-end pipeline runner
├── requirements.txt              # Dependencies
├── db/
│   └── database.py               # SQLite schema & array adapters
├── e57/
│   └── geometry.py               # Quaternion & 3D vector transformations
├── vision/                       # Vision modules
├── outputs/                      # Evaluation reports, models, & visualizations
└── dataset/                      # Extracted images & SQLite database
```

---

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AndreiLAB/junction-hackathon-veo-digital-twin.git
   cd junction-hackathon-veo-digital-twin
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

#### 1. Full Pipeline Execution
To run the complete extraction, calibration, embedding generation, training, and evaluation:
```bash
python pipeline.py --e57 path/to/cloud.e57
```

#### 2. Query Localization
To localize an arbitrary photo against the digital twin:
```bash
python localize.py --image path/to/query.jpg
```

To render an annotated visual comparison contact sheet:
```bash
python visualize_localization.py --image path/to/query.jpg
```

#### 3. Rigorous Evaluation
To evaluate localization robustness against synthetic perturbations:
```bash
python evaluate.py
```
Outputs:
- `outputs/evaluation_report.md`
- `outputs/evaluation_results.json`
- `outputs/retrieval_contact_sheet.jpg`


---

## Manuals knowledge base (REX615, UniGear ZS2, VD4)

Manuals converted to section-based Markdown and a searchable SQLite database, so tags can return their PDFs and an optional `/ask` can answer with page citations.

- `data/knowledge.db`: `documents` (title, model, doc number, PDF path, pages), `chunks` (section path, page range, text) and an FTS5 index `chunks_fts`.
- `docs_md/<folder>/<manual>.md`: readable Markdown with `<!-- p.N -->` page markers.
- The PDFs are **not in git** (about 140 MB). Put them in `docs/abb_615/`, `docs/unigear_zs2/`, `docs/vd4/` (shared via Drive).

```bash
pip install pymupdf
python scripts/ingest_docs.py                                   # rebuild (about 5 min for the 1,985-page REX615 manual)
python scripts/kb_search.py "READY LED" --model "ABB 615"       # search with citations, prints "not found" if nothing matches
python scripts/kb_eval.py                                       # retrieval test: 11/11 in top 3
```
Library use: `from kb_search import search; search(con, question, k=5, model="ABB 615")` returns chunks with document, section path and page range.
