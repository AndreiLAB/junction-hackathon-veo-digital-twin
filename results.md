# VEO Integration 2.0 - Relay Object Detection (V3)

## Validation on Real Photos (Scans 6, 8, 13)
* **Model**: YOLOv8s (40 epochs)
* **Dataset**: V3 (OT1 relays fixed, synthetic noise patched, far relays included in train/val)
* **Precision**: 0.625
* **Recall**: 0.208
* **TP**: 5, **FP**: 3, **FN**: 19

### False Positives
* **Cerdex displays mistaken for relay**: 0

## Notes
The far-camera filter in the generator has been removed, so the network now sees small relays during training. OT1 offset logic was patched to correct the 5cm discrepancy. These changes significantly boosted the evaluation performance compared to V2.
