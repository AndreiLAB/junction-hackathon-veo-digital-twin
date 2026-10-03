import os
import json
from config import METADATA_DIR, OUTPUTS_DIR
from db.database import get_db_connection

def main():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM scans")
    num_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM images")
    num_images = cursor.fetchone()[0]
    
    print(f"Number of scans: {num_scans}")
    print(f"Number of images: {num_images}")
    
    if num_scans > 0:
        print(f"Images per scan: {num_images / num_scans:.2f}")
        
    # Missing poses
    cursor.execute("SELECT COUNT(*) FROM images WHERE x IS NULL OR y IS NULL OR z IS NULL")
    missing_poses = cursor.fetchone()[0]
    print(f"Missing poses: {missing_poses}")
    
    # Missing images
    cursor.execute("SELECT COUNT(*) FROM images WHERE file_path IS NULL OR file_path = ''")
    missing_images = cursor.fetchone()[0]
    print(f"Missing images: {missing_images}")
    
    # Image dimensions
    cursor.execute("SELECT width, height, COUNT(*) as count FROM images GROUP BY width, height")
    dims = cursor.fetchall()
    print("Image dimensions:")
    for row in dims:
        print(f"  {row['width']}x{row['height']}: {row['count']} images")
        
    # Projection types
    cursor.execute("SELECT projection_type, COUNT(*) as count FROM images GROUP BY projection_type")
    projs = cursor.fetchall()
    print("Projection types:")
    for row in projs:
        print(f"  {row['projection_type']}: {row['count']} images")
        
    # Coordinate ranges
    cursor.execute("SELECT MIN(x), MAX(x), MIN(y), MAX(y), MIN(z), MAX(z) FROM images")
    coords = cursor.fetchone()
    if coords and coords[0] is not None:
        print(f"Coordinate ranges:")
        print(f"  X: {coords[0]:.2f} to {coords[1]:.2f}")
        print(f"  Y: {coords[2]:.2f} to {coords[3]:.2f}")
        print(f"  Z: {coords[4]:.2f} to {coords[5]:.2f}")
        
    # Yaw/Pitch/Roll ranges
    cursor.execute("SELECT MIN(yaw), MAX(yaw), MIN(pitch), MAX(pitch), MIN(roll), MAX(roll) FROM images")
    angles = cursor.fetchone()
    if angles and angles[0] is not None:
        print(f"Yaw range: {angles[0]:.2f} to {angles[1]:.2f}")
        print(f"Pitch range: {angles[2]:.2f} to {angles[3]:.2f}")
        print(f"Roll range: {angles[4]:.2f} to {angles[5]:.2f}")
        
    # Generate HTML overview
    generate_html(conn)
    conn.close()

def generate_html(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT images.id, images.name, images.file_path, images.x, images.y, images.z, images.yaw, images.pitch, images.roll, scans.name as scan_name
        FROM images
        LEFT JOIN scans ON images.scan_id = scans.id
        ORDER BY scans.name, images.name
    """)
    images = cursor.fetchall()
    
    html = """
    <html>
    <head>
        <title>VEO Localization Dataset Overview</title>
        <style>
            body { font-family: sans-serif; background: #f0f0f0; }
            .grid { display: flex; flex-wrap: wrap; gap: 10px; padding: 10px; }
            .card { background: white; border-radius: 5px; padding: 10px; width: 300px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .card img { max-width: 100%; height: auto; }
            .meta { font-size: 0.9em; color: #555; }
        </style>
    </head>
    <body>
        <h1>VEO Localization Dataset Overview</h1>
        <div class="grid">
    """
    
    for img in images:
        img_path = ""
        if img["file_path"]:
            # Need to get relative path for the HTML file in root
            img_path = os.path.relpath(img["file_path"], OUTPUTS_DIR)
            
        html += f"""
        <div class="card">
            <img src="{img_path}" alt="{img['name']}">
            <h3>{img['name']}</h3>
            <div class="meta">
                <p><strong>Scan:</strong> {img['scan_name']}</p>
                <p><strong>Position:</strong> X:{img['x']:.2f} Y:{img['y']:.2f} Z:{img['z']:.2f}</p>
                <p><strong>Direction:</strong> Yaw:{img['yaw']:.1f}° Pitch:{img['pitch']:.1f}° Roll:{img['roll']:.1f}°</p>
            </div>
        </div>
        """
        
    html += """
        </div>
    </body>
    </html>
    """
    
    html_path = OUTPUTS_DIR / "dataset_overview.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated HTML overview at {html_path}")

if __name__ == "__main__":
    main()
