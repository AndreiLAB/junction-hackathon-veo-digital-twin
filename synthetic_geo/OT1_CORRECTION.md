# OT1 relay geometry: corrected values (supersedes dataset v1 and the first v2 attempt)

Cabinet **OT1** (the second row, door normal -x, x = -2.720) carries **two real ABB relays** (615-series front panel). Dataset v1 left them unlabelled.
The first v2 attempt added them, but with geometry that is slightly wrong. This note gives the values that fit the real photos.

## Measured fit (by eye on 8 photos, +-0.02 m)
| Parameter | v1 / my first measurement | first v2 attempt | **corrected** |
|---|---|---|---|
| relay A (upper) centre, dy from the OT1 position (m) | -0.188 | -0.100 | **-0.155** |
| relay B (lower) centre, dy | -0.213 | -0.106 | **-0.155** |
| relay A dz (m) | +0.166 | +0.164 | **+0.145** |
| relay B dz (m) | -0.152 | -0.165 | **-0.160** |
| relay A frame size (m) | 0.262 x 0.177 (wrong) | 0.262 x 0.177 (**wrong**) | **0.177 x 0.177** |
| relay B frame size (m) | 0.262 x 0.177 (wrong) | 0.177 x 0.177 | **0.177 x 0.177** |
| front plane out of the door, towards the corridor (m) | 0.07 | 0.07 | **0.035** |

* Both OT1 relays are the **standard narrow case** (~0.19 m wide in the photos), not the wide case (0.262 m) used on H01-H05.
* Position and size were checked on img_043, 038, 052, 063, 075, 082, 088, 094 (0.6 to 5.8 m, 6 to 68 degrees). In the oblique views the front-plane offset **0.035** fits best; 0.07 (the H-cabinet value) and 0 both miss by about half a relay width.
* The H01-H05 relays keep their own values (`relay_geometry.json`); this correction is for OT1 only.

## What to change
1. **`relay_geometry.json`, OT1 entry** (restructured per-cabinet format used on `trained-relay-model-updated`):
```json
"OT1": { "relays": [
  {"dy": -0.155, "dz":  0.145, "width_m": 0.177, "height_m": 0.177, "front_plane_m": 0.035},
  {"dy": -0.155, "dz": -0.160, "width_m": 0.177, "height_m": 0.177, "front_plane_m": 0.035}
]}
```
2. **Code:** the front-plane offset and the body depth are global constants in the generator (0.07). Make them **per relay** (`front_plane_m`, body depth = the same value), default 0.07 so H01-H05 are unchanged.
3. **Verify before regenerating** (mandatory, 1 minute):
```bash
python synthetic_geo/verify_boxes.py --cabinet OT1 --images "VEO Images" --cameras "VEO Images/cameras.json" \
  --rect "A:-0.155:0.145:0.177:0.177" --rect "B:-0.155:-0.160:0.177:0.177" --front 0 --front 0.035 --front 0.07 --n 8 --out ot1_check.jpg
```
   The **0.035** box (the second `--front` value; the legend at the bottom of the sheet names its colour: green = 1st, yellow = 2nd, blue = 3rd) must cover the real relay tightly in all photos, including the oblique ones (the 0 and 0.07 boxes should miss by about half a relay width there).
4. **Extend the cabinet list** that is hard-coded in the generator (`plan`, `main`) and in `location_gate.py` (`Gate.expected`) to include **OT1**, otherwise the gate rejects OT1 relays and `export_detections.py` gives them `position: null`.
5. Also set `expects_relay: true` for OT1 in `cabinet_registry.json`, and make the backend `DEVICE_CLASSES`/cabinet logic accept relays on OT1.
6. **Regenerate the whole dataset**, run `check_dataset.py`, **look at the sheets for ghost relay edges around the pasted OT1 relays** (the wrong size/offset in the first v2 attempt would have left them), then retrain.
7. Add OT1's two relays to the **real evaluation boxes** (verify each by eye), so real recall includes them.

## Related findings (same check)
* The first v2 real result (precision 0.75, recall 0.25) came from a model whose training data did not include far/small relays (generator camera filter) and used the wrong OT1 geometry; fix both before judging the approach.
* OT1 is a cabinet where the technique must tell relays from other devices: the same cabinet row has Cerdex displays on TSK1/TSK2 (`other_hmi`).
