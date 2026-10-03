import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import EMBEDDING_MODEL, OUTPUTS_DIR
from db.database import get_db_connection
from build_embeddings import load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Query image path")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: image {args.image} not found.")
        return
        
    model, transform = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        q_img = Image.open(args.image).convert('RGB')
        img_t = transform(q_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
        
    with torch.no_grad():
        output = model(img_t)
        query_emb = torch.nn.functional.normalize(output, p=2, dim=1).cpu().numpy()[0]
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT e.image_id, e.embedding, i.name, i.file_path, i.x, i.y, i.z, i.yaw, s.id as scan_id, s.name as scan_name
        FROM embeddings e
        JOIN images i ON e.image_id = i.id
        JOIN scans s ON i.scan_id = s.id
        WHERE e.model_name = ?
    """, (EMBEDDING_MODEL,))
    
    data = cursor.fetchall()
    
    best_matches = []
    for row in data:
        sim = np.dot(query_emb, row['embedding'])
        best_matches.append((sim, row))
        
    best_matches.sort(key=lambda x: x[0], reverse=True)
    top_5 = best_matches[:5]
    
    # Create contact sheet
    img_size = 300
    sheet = Image.new('RGB', (img_size * 6, img_size + 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
        
    # Draw query
    q_resized = q_img.resize((img_size, img_size))
    sheet.paste(q_resized, (0, 50))
    draw.text((10, 10), "QUERY", fill="black", font=font)
    
    # Draw top 5
    for i, (sim, row) in enumerate(top_5):
        col = i + 1
        x_offset = col * img_size
        
        if row['file_path'] and os.path.exists(row['file_path']):
            db_img = Image.open(row['file_path']).convert('RGB')
            # Crop to square
            w, h = db_img.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            db_img = db_img.crop((left, top, left+size, top+size))
            db_img = db_img.resize((img_size, img_size))
            
            sheet.paste(db_img, (x_offset, 50))
            
        text = f"Top {i+1}\nSim: {sim:.2f}\nScan: {row['scan_id']}\nYaw: {row['yaw']:.1f}°"
        draw.text((x_offset + 10, 10), text, fill="black", font=font)
        
    out_path = OUTPUTS_DIR / "localization_debug.jpg"
    sheet.save(out_path)
    print(f"Saved visual localization debugging to {out_path}")

if __name__ == "__main__":
    main()
