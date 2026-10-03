import os
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
from config import OUTPUTS_DIR
from db.database import get_db_connection
from e57.geometry import calculate_forward_vector, vector_to_yaw_pitch_roll

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_id", type=int, help="Scan ID to calibrate", default=None)
    parser.add_argument("--forward_axis", type=str, default="-z", help="Axis representing camera forward (e.g. -z, z, x, y)")
    args = parser.parse_args()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    scan_id = args.scan_id
    if scan_id is None:
        cursor.execute("SELECT id FROM scans LIMIT 1")
        row = cursor.fetchone()
        if not row:
            print("No scans found in database.")
            return
        scan_id = row['id']
    
    # Get the 6 skybox images for this scan
    cursor.execute("""
        SELECT id, name, file_path, x, y, z, qw, qx, qy, qz 
        FROM images 
        WHERE scan_id = ? AND file_path IS NOT NULL
        ORDER BY name
        LIMIT 6
    """, (scan_id,))
    
    images = cursor.fetchall()
    
    if not images:
        print(f"No images found for scan_id {scan_id}")
        return
        
    print(f"Calibrating using {len(images)} images from scan {scan_id}")
    
    # Create contact sheet
    img_size = 300
    contact_sheet = Image.new('RGB', (img_size * 3, img_size * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(contact_sheet)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    for i, row in enumerate(images):
        col = i % 3
        r = i // 3
        
        q = (row['qw'], row['qx'], row['qy'], row['qz'])
        vec = calculate_forward_vector(q, forward_axis=args.forward_axis)
        yaw, pitch, roll = vector_to_yaw_pitch_roll(vec)
        
        img_path = row['file_path']
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Crop to square center if it's equirectangular or similar
            # For skybox it's usually square
            w, h = img.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            img = img.crop((left, top, left+size, top+size))
            img = img.resize((img_size, img_size))
            
            contact_sheet.paste(img, (col * img_size, r * img_size))
            
            text = f"{row['name']}\nPos: {row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f}\nYaw: {yaw:.1f}° P: {pitch:.1f} R: {roll:.1f}\nVec: {vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}"
            
            # Draw text with outline for readability
            text_x = col * img_size + 10
            text_y = r * img_size + 10
            
            for offset_x, offset_y in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                draw.text((text_x + offset_x, text_y + offset_y), text, fill="black", font=font)
            draw.text((text_x, text_y), text, fill="yellow", font=font)
            
    out_path = OUTPUTS_DIR / f"scan_{scan_id:03d}_directions_{args.forward_axis}.jpg"
    contact_sheet.save(out_path)
    print(f"Saved calibration contact sheet to {out_path}")
    print(f"Check if the images follow a consistent clockwise/counter-clockwise order")
    print(f"and if the yaw angle matches the expected visual direction.")

if __name__ == "__main__":
    main()
