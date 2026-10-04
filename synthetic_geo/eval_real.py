import json
import cv2
import os
import sys
import glob
from ultralytics import YOLO

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def evaluate():
    runs = sorted(glob.glob("C:/Users/gagna/runs/detect/runs_relay/geo_synth*/weights/best.pt"), key=os.path.getmtime)
    if not runs:
        print("No model found!")
        return
    model = YOLO(runs[-1])
    
    with open("E:/OSses/VEO Images/cameras.json") as f:
        cams = json.load(f)
        
    GRID = [(0.5, 0.5), (0.35, 0.5), (0.65, 0.5), (0.5, 0.35), (0.5, 0.65), (0.35, 0.35), (0.65, 0.65), (0.35, 0.65), (0.65, 0.35)]
    TILE = 1280
    
    detections = []
    
    for c in cams:
        img_name = c["file"]
        img_path = os.path.join("E:/OSses/VEO Images", img_name)
        if not os.path.exists(img_path): continue
        img = cv2.imread(img_path)
        if img is None: continue
        
        # We run the detector on tiles?
        # "In the final system, inference will also run on tiles or downscaled images..."
        # But wait, the backend currently just uses `OCR_MAX_SIDE=1536` resizing.
        # If we just resize the image to 1536 and run YOLO, relays will be very small and YOLO might miss them.
        # Let's run on 1280x1280 tiles using a fixed grid like training.
        h, w = img.shape[:2]
        file_boxes = []
        for x0 in range(0, w, 1024):
            for y0 in range(0, h, 1024):
                if x0 + TILE > w: x0 = w - TILE
                if y0 + TILE > h: y0 = h - TILE
                
                tile = img[y0:y0+TILE, x0:x0+TILE]
                res = model.predict(tile, imgsz=TILE, conf=0.25, verbose=False)[0]
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    class_name = res.names[cls_id]
                    # Map to full image
                    fx1, fy1, fx2, fy2 = bx1 + x0, by1 + y0, bx2 + x0, by2 + y0
                    # Deduplicate using NMS later, but for now just add
                    file_boxes.append({
                        "class": class_name,
                        "conf": round(conf, 3),
                        "xyxy": [round(fx1, 1), round(fy1, 1), round(fx2, 1), round(fy2, 1)]
                    })
        
        # Simple NMS to deduplicate boxes from overlapping tiles
        file_boxes = sorted(file_boxes, key=lambda x: -x["conf"])
        keep = []
        for b in file_boxes:
            if not any(iou(b["xyxy"], k["xyxy"]) > 0.4 and b["class"] == k["class"] for k in keep):
                keep.append(b)
                
        detections.append({
            "image": img_name,
            "boxes": keep
        })
        print(f"Processed {img_name}: {len(keep)} detections")
        
    with open("detections.json", "w") as f:
        json.dump(detections, f, indent=2)

if __name__ == "__main__":
    evaluate()
