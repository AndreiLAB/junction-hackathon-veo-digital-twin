import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
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
        
    print(f"Loading query image: {args.image}")
    
    model, transform = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        img = Image.open(args.image).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
        
    print("Extracting embedding...")
    with torch.no_grad():
        output = model(img_t)
        query_emb = torch.nn.functional.normalize(output, p=2, dim=1).cpu().numpy()[0]
        
    print("Querying database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all embeddings
    cursor.execute("""
        SELECT e.image_id, e.embedding, i.name, i.x, i.y, i.z, i.yaw, i.pitch, i.roll, s.id as scan_id, s.name as scan_name
        FROM embeddings e
        JOIN images i ON e.image_id = i.id
        JOIN scans s ON i.scan_id = s.id
        WHERE e.model_name = ?
    """, (EMBEDDING_MODEL,))
    
    data = cursor.fetchall()
    if not data:
        print("No embeddings in database. Run pipeline first.")
        return
        
    best_matches = []
    
    for row in data:
        db_emb = row['embedding']
        # Cosine similarity
        sim = np.dot(query_emb, db_emb)
        best_matches.append((sim, row))
        
    # Sort descending
    best_matches.sort(key=lambda x: x[0], reverse=True)
    
    top_5 = best_matches[:5]
    
    top_match = top_5[0]
    sim = top_match[0]
    row = top_match[1]
    
    print("\n--- RESULTS ---")
    print(f"Predicted scan: {row['scan_id']} ({row['scan_name']})")
    print(f"Predicted viewpoint: {row['name']}")
    print(f"Similarity: {sim:.3f}")
    
    if row['x'] is not None:
        print(f"Position:\nX = {row['x']:.2f}\nY = {row['y']:.2f}\nZ = {row['z']:.2f}")
    else:
        print("Position: Unknown")
        
    if row['yaw'] is not None:
        print(f"Direction:\nYaw = {row['yaw']:.1f}°\nPitch = {row['pitch']:.1f}°\nRoll = {row['roll']:.1f}°")
    else:
        print("Direction: Unknown")
        
    print("\nTop 5 candidates:")
    for i, (sim, row) in enumerate(top_5):
        print(f"{i+1}. Scan {row['scan_id']} / {row['name']} (Sim: {sim:.3f})")
        
    # Save a visual result if possible
    # We will just write a simple result info
    res_path = OUTPUTS_DIR / "query_result.txt"
    with open(res_path, "w") as f:
        f.write(f"Query: {args.image}\n")
        f.write(f"Top match: {row['scan_name']} / {row['name']}\n")
        f.write(f"Similarity: {sim:.3f}\n")
    print(f"Saved result summary to {res_path}")

if __name__ == "__main__":
    main()
