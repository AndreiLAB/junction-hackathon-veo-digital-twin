import argparse
import pye57
import logging
from pathlib import Path
import os
import numpy as np

from config import IMAGES_DIR, FORWARD_AXIS
from db.database import init_db, get_db_connection
from e57.geometry import calculate_forward_vector, vector_to_yaw_pitch_roll

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_e57(e57_path):
    logging.info(f"Opening E57 file: {e57_path}")
    e57 = pye57.E57(e57_path)
    imf = e57.image_file
    raw_root = imf.root()
    root = raw_root
    
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    scan_guid_to_id = {}
    if root.isDefined("data3D"):
        data3d_nodes = root["data3D"]
        scan_count = data3d_nodes.childCount()
        logging.info(f"Found {scan_count} Data3D scans")
        
        for i in range(scan_count):
            scan_node = data3d_nodes[i]
            
            guid = scan_node["guid"].value() if scan_node.isDefined("guid") else f"scan_{i}"
            name = scan_node["name"].value() if scan_node.isDefined("name") else f"Scan {i}"
            
            x = y = z = 0.0
            qw = 1.0; qx = qy = qz = 0.0
            
            if scan_node.isDefined("pose"):
                pose = scan_node["pose"]
                if pose.isDefined("translation"):
                    t = pose["translation"]
                    x = t["x"].value() if t.isDefined("x") else 0.0
                    y = t["y"].value() if t.isDefined("y") else 0.0
                    z = t["z"].value() if t.isDefined("z") else 0.0
                if pose.isDefined("rotation"):
                    r = pose["rotation"]
                    qw = r["w"].value() if r.isDefined("w") else 1.0
                    qx = r["x"].value() if r.isDefined("x") else 0.0
                    qy = r["y"].value() if r.isDefined("y") else 0.0
                    qz = r["z"].value() if r.isDefined("z") else 0.0
                    
            cursor.execute("SELECT id FROM scans WHERE guid = ?", (guid,))
            row = cursor.fetchone()
            if row:
                scan_id = row['id']
                cursor.execute('''
                    UPDATE scans SET name=?, x=?, y=?, z=?, qw=?, qx=?, qy=?, qz=? WHERE id=?
                ''', (name, x, y, z, qw, qx, qy, qz, scan_id))
            else:
                cursor.execute('''
                    INSERT INTO scans (guid, name, x, y, z, qw, qx, qy, qz)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (guid, name, x, y, z, qw, qx, qy, qz))
                scan_id = cursor.lastrowid
            scan_guid_to_id[guid] = scan_id
    else:
        logging.warning("No Data3D scans found.")
        
    if not root.isDefined("images2D"):
        logging.warning("No Images2D found in E57 file.")
        conn.commit()
        conn.close()
        return
        
    image2d_nodes = root["images2D"]
    image_count = image2d_nodes.childCount()
    logging.info(f"Found {image_count} Image2D nodes")
    
    successful_extractions = 0
    missing_poses = 0
    
    for i in range(image_count):
        img_node = image2d_nodes[i]
        
        guid = img_node["guid"].value() if img_node.isDefined("guid") else f"img_{i}"
        name = img_node["name"].value() if img_node.isDefined("name") else f"Image_{i}"
        assoc_guid = img_node["associatedData3DGuid"].value() if img_node.isDefined("associatedData3DGuid") else None
        
        scan_id = scan_guid_to_id.get(assoc_guid) if assoc_guid else None
        
        x = y = z = None
        qw = 1.0; qx = qy = qz = 0.0
        
        if img_node.isDefined("pose"):
            pose = img_node["pose"]
            if pose.isDefined("translation"):
                t = pose["translation"]
                x = t["x"].value() if t.isDefined("x") else 0.0
                y = t["y"].value() if t.isDefined("y") else 0.0
                z = t["z"].value() if t.isDefined("z") else 0.0
            if pose.isDefined("rotation"):
                r = pose["rotation"]
                qw = r["w"].value() if r.isDefined("w") else 1.0
                qx = r["x"].value() if r.isDefined("x") else 0.0
                qy = r["y"].value() if r.isDefined("y") else 0.0
                qz = r["z"].value() if r.isDefined("z") else 0.0
        else:
            missing_poses += 1
            
        projection_type = "unknown"
        width = height = None
        file_path = None
        
        rep_key = None
        if img_node.isDefined("sphericalRepresentation"):
            rep_key = "sphericalRepresentation"
            projection_type = "spherical"
        elif img_node.isDefined("pinholeRepresentation"):
            rep_key = "pinholeRepresentation"
            projection_type = "pinhole"
        elif img_node.isDefined("cylindricalRepresentation"):
            rep_key = "cylindricalRepresentation"
            projection_type = "cylindrical"
            
        if rep_key:
            rep = img_node[rep_key]
            if rep.isDefined("imageWidth"):
                width = rep["imageWidth"].value()
            if rep.isDefined("imageHeight"):
                height = rep["imageHeight"].value()
                
            try:
                blob = None
                ext = ".jpg"
                if rep.isDefined("jpegImage"):
                    blob = rep["jpegImage"]
                    ext = ".jpg"
                elif rep.isDefined("pngImage"):
                    blob = rep["pngImage"]
                    ext = ".png"
                    
                if blob:
                    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    filename = f"scan_{scan_id:03d}_{safe_name}{ext}" if scan_id is not None else f"{safe_name}{ext}"
                    out_path = IMAGES_DIR / filename
                    
                    if not out_path.exists():
                        blob_size = blob.byteCount()
                        blob_data = np.zeros(shape=blob_size, dtype=np.uint8)
                        blob.read(blob_data, 0, blob_size)
                        
                        with open(out_path, "wb") as f:
                            f.write(blob_data.tobytes())
                            
                    file_path = str(out_path)
            except Exception as e:
                logging.error(f"Cannot read image for {name}: {e}")
                
        forward_x = forward_y = forward_z = None
        yaw = pitch = roll = None
        
        if x is not None:
            vec = calculate_forward_vector((qw, qx, qy, qz), forward_axis=FORWARD_AXIS)
            forward_x, forward_y, forward_z = vec
            yaw, pitch, roll = vector_to_yaw_pitch_roll(vec)
            
        cursor.execute("SELECT id FROM images WHERE guid = ?", (guid,))
        row = cursor.fetchone()
        if row:
            cursor.execute('''
                UPDATE images SET scan_id=?, name=?, file_path=?, width=?, height=?, x=?, y=?, z=?, qw=?, qx=?, qy=?, qz=?, forward_x=?, forward_y=?, forward_z=?, yaw=?, pitch=?, roll=?, projection_type=? WHERE id=?
            ''', (scan_id, name, file_path, width, height, x, y, z, qw, qx, qy, qz, forward_x, forward_y, forward_z, yaw, pitch, roll, projection_type, row['id']))
        else:
            cursor.execute('''
                INSERT INTO images (guid, scan_id, name, file_path, width, height, x, y, z, qw, qx, qy, qz, forward_x, forward_y, forward_z, yaw, pitch, roll, projection_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (guid, scan_id, name, file_path, width, height, x, y, z, qw, qx, qy, qz, forward_x, forward_y, forward_z, yaw, pitch, roll, projection_type))
        
        successful_extractions += 1
        
    conn.commit()
    conn.close()
    logging.info(f"Extraction complete. Successfully processed {successful_extractions} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e57", type=str, required=True)
    parser.add_argument("--output", type=str, default="./dataset")
    args = parser.parse_args()
    
    extract_e57(args.e57)
