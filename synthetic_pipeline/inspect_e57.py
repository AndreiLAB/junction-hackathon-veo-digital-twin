import sys
import pye57

path = sys.argv[1]
e57 = pye57.E57(path)
print("File:", path)
print("Number of 3D scans:", e57.scan_count)

total = 0
for i in range(e57.scan_count):
    h = e57.get_header(i)
    total += h.point_count
    print(f"\nScan {i}")
    print("  points:", f"{h.point_count:,}")
    print("  fields:", h.point_fields)
    print("  has colour:", "yes" if "colorRed" in h.point_fields else "no")
    if h.has_pose:
        print("  scanner position:", [round(float(v), 3) for v in h.translation])
    else:
        print("  scanner position: none stored")
    try:
        print(f"  size x: {h.xMinimum:.2f} to {h.xMaximum:.2f}")
        print(f"  size y: {h.yMinimum:.2f} to {h.yMaximum:.2f}")
        print(f"  size z: {h.zMinimum:.2f} to {h.zMaximum:.2f}")
    except Exception:
        print("  size: not stored in file header")

print(f"\nTotal points: {total:,}")
root = e57.root
n_img = root["images2D"].childCount() if root.isDefined("images2D") else 0
print("Embedded photos:", n_img)