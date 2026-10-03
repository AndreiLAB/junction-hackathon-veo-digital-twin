import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_DIR = DATASET_DIR / "metadata"
AUGMENTED_DIR = DATASET_DIR / "augmented"
DB_PATH = DATASET_DIR / "veo_localization.db"

# Model Config
EMBEDDING_MODEL = "dinov2_vits14"  # Using torch hub for DINOv2
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 16

# Vision Optimization Configs
OCR_MAX_SIDE = 1536
VISION_WORKERS = 2
OCR_ORT_THREADS = 2

# This allows calibration of which quaternion axis means "camera forward"
# Expected values can be adjusted based on calibration.
# Matterport usually uses -Z or +Z for forward, but we will test.
FORWARD_AXIS = "-z" 

# Make sure directories exist
for d in [DATASET_DIR, OUTPUTS_DIR, IMAGES_DIR, METADATA_DIR, AUGMENTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
