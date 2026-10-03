import argparse
import logging
import cv2
import numpy as np
import pye57
from pathlib import Path
import os
import json
import time

from config import IMAGES_DIR, OCR_MAX_SIDE, VISION_WORKERS, OCR_ORT_THREADS

os.environ["OMP_NUM_THREADS"] = str(OCR_ORT_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OCR_ORT_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(OCR_ORT_THREADS)

from extract_e57 import extract_e57
from db.database import get_db_connection, init_db
from vision.detector import MockDetector
from vision.label_detector import detect_labels
from e57.raycast import raycast_2d_to_3d, pose_matrix
from tagging.tagger import generate_tag
from knowledge.adapter import find_documents
from exporters.json_exporter import export_tags_to_json
from exporters.csv_exporter import export_tags_to_csv
from models import LocalizedDetection, Tag, PhysicalTag
from typing import List

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_scan(e57, i):
    d = e57.read_scan(i, colors=True, transform=True, ignore_missing_fields=True)
    xyz = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
    if "cartesianInvalidState" in d:
        ok = np.asarray(d["cartesianInvalidState"]) == 0
        xyz = xyz[ok]
    return xyz

def run_pipeline(e57_path: str, detector_type: str = "mock", high_accuracy: bool = False) -> List[PhysicalTag]:
    """
    Core engine to convert an E57 scan into structured Physical Tags.
    Can be imported and called directly by FastAPI.
    """
    ocr_res = 4096 if high_accuracy else OCR_MAX_SIDE
    logging.info(f"Starting pipeline in {'HIGH ACCURACY' if high_accuracy else 'FAST'} mode (OCR Res: {ocr_res})")
    
    # 1. Extract metadata and images to DB
    logging.info("[1/5] Extracting E57 metadata and images...")
    extract_e57(e57_path)

    # Prepare outputs
    os.makedirs("outputs", exist_ok=True)
    
    # 2. Open E57 file and DB for processing
    logging.info("[2/5] Running vision detectors...")
    e57_file = pye57.E57(e57_path)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM images WHERE file_path IS NOT NULL")
    images_rows = cursor.fetchall()
    
    from vision.detector import get_detector
    detector = get_detector(detector_type)
    all_detections = []
    
    # 3. Detect Labels and Objects
    import concurrent.futures
    import threading
    
    valid_images = [row for row in images_rows if row["forward_z"] is None or abs(row["forward_z"]) <= 0.7]
    valid_images = [row for row in valid_images if os.path.exists(row["file_path"])]
    
    total_imgs = len(valid_images)
    processed_count = 0
    count_lock = threading.Lock()
    
    start_time = time.time()

    def process_image(row):
        nonlocal processed_count
        img_id = row["id"]
        path = row["file_path"]
        
        img = cv2.imread(path)
        if img is None:
            return []
            
        orig_h, orig_w = img.shape[:2]
        scale = min(ocr_res / orig_w, ocr_res / orig_h)
        if scale < 1.0:
            resized = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
        else:
            resized = img
            scale = 1.0
            
        # Detect cabinet labels (via OCR & orange tape)
        labels = detect_labels(resized, img_id)
        # Detect ABB equipment (via mock object detector)
        objects = detector.detect(resized, img_id)
        
        all_det = labels + objects
        
        # Scale bounding boxes back to original coordinates
        for det in all_det:
            x0, y0, x1, y1 = det.bbox
            det.bbox = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
            
        with count_lock:
            processed_count += 1
            logging.info(f"[{processed_count:02d}/{total_imgs}] processing {os.path.basename(path)}... (orig: {orig_w}x{orig_h}, ocr: {resized.shape[1]}x{resized.shape[0]})")
            
        return all_det

    with concurrent.futures.ThreadPoolExecutor(max_workers=VISION_WORKERS) as executor:
        results = list(executor.map(process_image, valid_images))
        
    end_time = time.time()
    elapsed = end_time - start_time
    avg_time = elapsed / total_imgs if total_imgs > 0 else 0
    logging.info(f"Vision Processing completed in {elapsed:.2f}s ({avg_time:.2f}s per image) with {VISION_WORKERS} workers")
        
    for dets in results:
        for det in dets:
            all_detections.append(det)
            bbox_list = [float(x) for x in det.bbox]
            cursor.execute('''
                INSERT INTO detections (image_id, class_name, confidence, bbox, ocr_text, ocr_conf)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (det.image_id, det.class_name, float(det.confidence), json.dumps(bbox_list), det.ocr_text, float(det.ocr_conf) if det.ocr_conf else None))
    
    conn.commit()
    logging.info(f"      Found {len(all_detections)} total detections.")

    # 4. Raycast 2D to 3D
    logging.info("[3/5] Raycasting to 3D point cloud...")
    tags = []
    
    # Group detections by scan_id to minimize E57 point cloud loading
    cursor.execute("SELECT * FROM scans")
    scans_dict = {row["id"]: dict(row) for row in cursor.fetchall()}
    
    img_dict = {row["id"]: dict(row) for row in images_rows}
    
    dets_by_scan = {}
    for det in all_detections:
        img_data = img_dict[det.image_id]
        scan_id = img_data["scan_id"]
        if scan_id is not None:
            dets_by_scan.setdefault(scan_id, []).append((det, img_data))

    # Determine convention (hardcoded from find_labels.py logic to simplify)
    # inv=False, flip=[1, -1, -1] was typically used
    inv = False
    flip = [1, -1, -1]

    for scan_id, det_tuples in dets_by_scan.items():
        logging.info(f"      Loading scan {scan_id}...")
        # read_scan uses 0-based index. scan_id in DB is 1-based, but guid is 'scan_N'
        # Safest way is to parse 'scan_N' guid if possible.
        scan_guid = scans_dict[scan_id]["guid"]
        try:
            scan_index = int(scan_guid.replace("scan_", ""))
        except:
            scan_index = scan_id - 1
            
        xyz = load_scan(e57_file, scan_index)
        
        for det, img_data in det_tuples:
            R, C = pose_matrix(
                img_data["qw"], img_data["qx"], img_data["qy"], img_data["qz"],
                img_data["x"], img_data["y"], img_data["z"]
            )
            
            # Approximated focal length logic from find_labels.py
            W, H = img_data["width"], img_data["height"]
            # Assume focalLength is approximately W/2 if missing
            f = W / 2.0
            cx, cy = W / 2.0, H / 2.0
            intrinsics = (f, cx, cy)
            
            anchor_xyz = raycast_2d_to_3d(
                det.bbox, xyz, (R, C), intrinsics, W, H, flip_convention=flip, inverse_rotation=inv
            )
            
            loc_det = LocalizedDetection(
                detection=det,
                scan_id=scan_id,
                camera_position=(img_data["x"], img_data["y"], img_data["z"]),
                camera_orientation=(img_data["qw"], img_data["qx"], img_data["qy"], img_data["qz"]),
                estimated_world_position=anchor_xyz
            )
            
            docs = find_documents(det.class_name)
            tag = generate_tag(loc_det, docs=docs)
            tags.append(tag)
            
            cursor.execute('''
                INSERT INTO tags (tag_id, parent_tag_id, label, asset_type, confidence, x, y, z, source_image_id, source_scan_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (tag.tag_id, tag.parent_tag_id, tag.label, tag.asset_type, tag.confidence, tag.x, tag.y, tag.z, tag.source_image_id, tag.source_scan_id))
            
        del xyz # Free memory

    conn.commit()

    # 5. Export
    logging.info("[4/5] Exporting raw detections to CSV and JSON...")
    export_tags_to_json(tags, "outputs/raw_detections.json")
    export_tags_to_csv(tags, "outputs/raw_detections.csv")
    
    # 6. Aggregate physical tags
    logging.info("[5/5] Aggregating raw detections into physical tags...")
    from tagging.aggregator import aggregate_tags
    from exporters.json_exporter import export_physical_tags_to_json
    from exporters.csv_exporter import export_physical_tags_to_csv
    
    physical_tags = aggregate_tags(tags, dist_threshold=1.5)
    logging.info(f"      Aggregated {len(tags)} raw detections into {len(physical_tags)} unique physical tags.")
    
    export_physical_tags_to_json(physical_tags, "outputs/physical_tags.json")
    export_physical_tags_to_csv(physical_tags, "outputs/physical_tags.csv")

    # 7. Generate Verification Report
    logging.info("Generating verification report...")
    with open("outputs/verification_report.md", "w") as f:
        f.write("# Vision Detector Verification Report\n\n")
        f.write("| Source Image | BBox | Class | Conf | Estimated XYZ | Physical Tag ID |\n")
        f.write("|---|---|---|---|---|---|\n")
        for t in tags:
            img_path = img_dict[t.source_image_id]["file_path"] if t.source_image_id in img_dict else "Unknown"
            bbox_str = f"[{t.bbox[0]:.1f}, {t.bbox[1]:.1f}, {t.bbox[2]:.1f}, {t.bbox[3]:.1f}]"
            # Extract filename from path
            img_name = os.path.basename(img_path)
            f.write(f"| {img_name} | {bbox_str} | {t.label} | {t.confidence:.2f} | ({t.x:.2f}, {t.y:.2f}, {t.z:.2f}) | {t.physical_tag_id} |\n")

    logging.info("[6/6] Done! Outputs saved to outputs/")
    return physical_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VEO E57 Post-Processing Pipeline")
    parser.add_argument("--e57", type=str, required=True, help="Path to the E57 file")
    parser.add_argument("--detector", type=str, choices=["mock", "abb"], default="mock", help="Detector backend to use")
    parser.add_argument("--high-accuracy", action="store_true", help="Run in high accuracy mode (4096px, slower)")
    args = parser.parse_args()
    run_pipeline(args.e57, args.detector, args.high_accuracy)
