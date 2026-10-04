import os
import cv2
import glob
import random

def check_dataset(root):
    print(f"Checking dataset at {root}")
    for split in ["train", "val"]:
        imgs = glob.glob(f"{root}/images/{split}/*.jpg")
        lbls = glob.glob(f"{root}/labels/{split}/*.txt")
        print(f"[{split}] {len(imgs)} images, {len(lbls)} label files")
        
        valid = 0
        boxes_cnt = 0
        sizes = []
        for lf in lbls:
            with open(lf, 'r') as f:
                for line in f:
                    c, x, y, w, h = map(float, line.strip().split())
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"INVALID BOX in {lf}: {c} {x} {y} {w} {h}")
                    else:
                        valid += 1
                        sizes.append(w * 1280) # rough px width
                    boxes_cnt += 1
        print(f"[{split}] {boxes_cnt} total boxes, {valid} valid boxes.")
        if sizes:
            sizes.sort()
            print(f"[{split}] Box widths: min={sizes[0]:.0f}, median={sizes[len(sizes)//2]:.0f}, max={sizes[-1]:.0f}")
            
        # Make a contact sheet of 16 images
        random.seed(42)
        sample = random.sample(imgs, min(16, len(imgs)))
        sheet_rows = []
        for i in range(0, len(sample), 4):
            row_imgs = []
            for j in range(4):
                if i+j < len(sample):
                    img = cv2.imread(sample[i+j])
                    l_file = sample[i+j].replace('images', 'labels').replace('.jpg', '.txt')
                    if os.path.exists(l_file):
                        with open(l_file, 'r') as f:
                            for line in f:
                                c, x, y, w, h = map(float, line.strip().split())
                                cx, cy, bw, bh = int(x*1280), int(y*1280), int(w*1280), int(h*1280)
                                cv2.rectangle(img, (cx-bw//2, cy-bh//2), (cx+bw//2, cy+bh//2), (0,255,0), 2)
                    img = cv2.resize(img, (320, 320))
                    row_imgs.append(img)
            import numpy as np
            sheet_rows.append(np.concatenate(row_imgs, axis=1))
        sheet = np.concatenate(sheet_rows, axis=0)
        cv2.imwrite(f"{root}/check_{split}_sheet.jpg", sheet)
        print(f"[{split}] Wrote check_{split}_sheet.jpg")

if __name__ == "__main__":
    check_dataset("synthetic_geo_dataset_v2")
