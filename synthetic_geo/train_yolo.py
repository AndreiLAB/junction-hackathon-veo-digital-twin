"""Train the relay / look-alike-HMI detector on the geometry-driven synthetic dataset (run on the training computer).

    pip install ultralytics
    python synthetic_geo/train_yolo.py --data synthetic_geo_dataset/data.yaml

Algorithm: Ultralytics YOLO object detector (small variant, 1280 px). Classes: 0 abb_relion_615, 1 other_hmi.
Location awareness is added at inference by location_gate.py (camera pose -> expected relay position and size);
each dataset image also has its camera pose and expected boxes in meta.jsonl if a position-conditioned model is tried later.

Augmentation is deliberately restrained: the synthetic images already follow the real geometry, so NO flips
(mirrored text does not exist), NO rotation, NO perspective warp; only mild colour and scale jitter.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic_geo_dataset/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="start from COCO weights; yolov8m.pt if accuracy is short")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    YOLO(a.model).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, patience=25,
        fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        scale=0.15, translate=0.05, hsv_h=0.01, hsv_s=0.25, hsv_v=0.25, mosaic=0.3, close_mosaic=10,
        project="runs_relay", name="geo_synth")


if __name__ == "__main__":
    main()
