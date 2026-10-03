import cv2
import numpy as np
import difflib
from typing import List
from models import Detection
from vision.ocr import run_ocr

TAGS = {
    "H01": ("H01 - PT1", ["H01", "PT1"]),
    "H02": ("H02 - STATION TRANSFORMER", ["H02", "STATIONTRANSFORMER"]),
    "H03": ("H03 - METERING", ["H03", "METERING"]),
    "H04": ("H04 - SOLAR 1", ["H04", "SOLAR1"]),
    "H05": ("H05 - SOLAR 2", ["H05", "SOLAR2"]),
    "VLK": ("VLK", ["VLK"]),
    "OT1": ("OT1", ["OT1"]),
    "TSK1": ("TSK1", ["TSK1"]),
    "TSK2": ("TSK2", ["TSK2"]),
    "OKK1": ("OKK1", ["OKK1"]),
}
CONFUSE = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8"})
MIN_SCORE = 0.65

def norm(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())

def match_tag(text, designations_only=False):
    t = norm(text)
    if len(t) < 2:
        return None, 0.0
    tc = t.translate(CONFUSE)
    scores = {}
    for tag, (_, aliases) in TAGS.items():
        best = 0.0
        for a in (aliases[:1] if designations_only else aliases):
            ac = a.translate(CONFUSE)
            if tc == ac:
                s = 1.0
            elif len(ac) >= 3 and ac in tc:
                s = 0.95
            else:
                s = difflib.SequenceMatcher(None, tc, ac).ratio()
            best = max(best, s)
        scores[tag] = best
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (tag1, s1), (tag2, s2) = ranked[0], ranked[1]
    if s1 < MIN_SCORE or s1 - s2 < 0.05:
        return None, s1
    return tag1, s1

def orange_tapes(img):
    H, W = img.shape[:2]
    k = W / 4096.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 120, 150), (22, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 800 * k * k or w < 1.6 * h or w > 400 * k or h > 150 * k:
            continue
        boxes.append((x, y, w, h))
    return boxes

def detect_labels(img, image_id: int) -> List[Detection]:
    detections = []
    
    # 1. OCR over entire image
    ocr_results = run_ocr(img)
    for res in ocr_results:
        tag, score = match_tag(res.text, designations_only=False)
        if tag is not None:
            detections.append(Detection(
                image_id=image_id,
                class_name="cabinet_label",
                confidence=score,
                bbox=res.bbox,
                ocr_text=tag,
                ocr_conf=res.confidence
            ))
            
    # 2. Orange tape crops
    for (x, y, w, h) in orange_tapes(img):
        crop = img[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        best = None
        for variant in (big, cv2.cvtColor(binar, cv2.COLOR_GRAY2BGR)):
            r = run_ocr(variant)
            for res in r:
                if best is None or res.confidence > best[1]:
                    best = (res.text, res.confidence)
        
        if best:
            text, conf = best
            tag, score = match_tag(text, designations_only=True)
            if tag is not None:
                detections.append(Detection(
                    image_id=image_id,
                    class_name="cabinet_label",
                    confidence=score,
                    bbox=(x, y, x + w, y + h),
                    ocr_text=tag,
                    ocr_conf=conf
                ))
    
    return detections
