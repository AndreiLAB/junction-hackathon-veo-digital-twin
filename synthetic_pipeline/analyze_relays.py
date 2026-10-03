import json
import numpy as np

tags = json.load(open("labels_out/tags.json"))
finds = [f for f in json.load(open("labels_out/products.json"))
         if f["source"] == "detector" and "xyz" in f]
P = np.array([[t["x"], t["y"]] for t in tags])

print(f"{'cabinet':8s} {'detections':>10s} {'best':>6s} {'>=60%':>6s} {'>=75%':>6s} {'>=90%':>6s} {'photos':>7s} {'height':>7s}")
for i, t in enumerate(tags):
    mine = []
    for f in finds:
        d = np.linalg.norm(P - np.array(f["xyz"][:2]), axis=1)
        if np.argmin(d) == i and d.min() <= 0.6:
            mine.append(f)
    if not mine:
        print(f"{t['tag']:8s} {0:10d}")
        continue
    c = [f["conf"] for f in mine]
    print(f"{t['tag']:8s} {len(c):10d} {max(c):6.2f} {sum(v >= 0.6 for v in c):6d} {sum(v >= 0.75 for v in c):6d} "
          f"{sum(v >= 0.9 for v in c):6d} {len({f['photo'] for f in mine}):7d} {np.median([f['xyz'][2] for f in mine]):7.2f}")