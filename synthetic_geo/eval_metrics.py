import json
import sys

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

with open("detections.json") as f:
    preds = {d["image"]: d["boxes"] for d in json.load(f)}
    
with open("synthetic_geo/real_eval_boxes.json") as f:
    gts = json.load(f)

# Precision, recall, AP50 for abb_relion_615
# and Cerdex mistake rate

tp = 0
fp = 0
fn = 0
cerdex_mistakes = 0

# We only evaluate on the images that have GT (the held-out scans)
for img, gt_boxes in gts.items():
    if img not in preds: continue
    
    # GT relays
    gt_relays = [b for b in gt_boxes if b.get("cabinet") not in ("TSK1", "TSK2")]
    # GT cerdex
    gt_cerdex = [b for b in gt_boxes if b.get("cabinet") in ("TSK1", "TSK2")]
    
    pred_relays = [b for b in preds[img] if b["class"] == "abb_relion_615"]
    
    matched_gt = set()
    for p in pred_relays:
        best_iou = 0
        best_gt = -1
        for i, g in enumerate(gt_relays):
            if i in matched_gt: continue
            iou_val = iou(p["xyxy"], g["xyxy"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = i
        
        if best_iou > 0.5:
            matched_gt.add(best_gt)
            tp += 1
        else:
            fp += 1
            # Check if this FP is actually a Cerdex display
            for g in gt_cerdex:
                if iou(p["xyxy"], g["xyxy"]) > 0.3:
                    cerdex_mistakes += 1
                    break
                    
    fn += len(gt_relays) - len(matched_gt)

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
print(f"Metrics for abb_relion_615:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
print(f"Cerdex displays mistaken for relay: {cerdex_mistakes}")
