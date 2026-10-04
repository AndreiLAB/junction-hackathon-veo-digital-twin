# VEO Integration 2.0 - Relay Object Detection

## Validation on Real Photos (Scans 6, 8, 13)
* **Model**: YOLOv8s (15 epochs)
* **Dataset**: V2 (OT1 relays fixed, synthetic noise patched, far relays held out)
* **Precision**: 0.750
* **Recall**: 0.250
* **AP50**: ~0.350 (estimated from PR points)

### False Positives
* **Cerdex displays mistaken for relay**: 0
The Cerdex displays (TSK1/TSK2) are correctly ignored by the detector in all real photos evaluated.

## Notes
The recall on real photos is relatively low (25%) because the synthetic dataset generator currently filters out camera positions (`C[0]` not between -5.5 and -2.3), meaning the network was never trained on the "far" or "small" relays that appear in the evaluation scans. A larger `yolov8m.pt` or removing the filter in the dataset generator would be the next step to improve far-field recall.
