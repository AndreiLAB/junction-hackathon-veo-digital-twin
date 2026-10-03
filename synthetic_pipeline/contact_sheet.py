import cv2
import numpy as np
from pathlib import Path

folder = Path("e57_images")
files = sorted(folder.glob("img_*.jpg"))
T = 220                      # thumbnail size in pixels
COLS = 6                     # 6 photos per capture point = one row per scan
rows = (len(files) + COLS - 1) // COLS
sheet = np.full((rows * (T + 24), COLS * T + 70, 3), 255, np.uint8)
for k, f in enumerate(files):
    img = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_8)   # fast 1/8-size load
    thumb = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)
    r, c = divmod(k, COLS)
    y, x = r * (T + 24) + 22, 70 + c * T
    sheet[y:y + T, x:x + T] = thumb
    cv2.putText(sheet, f.stem.replace("img_", ""), (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    if c == 0:
        cv2.putText(sheet, f"scan {r}", (4, y + T // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
cv2.imwrite("contact_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"Saved contact_sheet.jpg with {len(files)} thumbnails")