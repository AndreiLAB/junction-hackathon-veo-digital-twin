import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, IMAGES_DIR
from db.database import get_db_connection
import numpy as np

def load_model():
    print(f"Loading {EMBEDDING_MODEL}...")
    # Use DINOv2 from torch hub
    model = torch.hub.load('facebookresearch/dinov2', EMBEDDING_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    # Standard imagenet transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, transform

def main():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all images
    cursor.execute("SELECT id, file_path FROM images WHERE file_path IS NOT NULL")
    images = cursor.fetchall()
    
    if not images:
        print("No images found in the database. Run extraction first.")
        return
        
    model, transform = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process in batches
    for i in tqdm(range(0, len(images), EMBEDDING_BATCH_SIZE)):
        batch = images[i:i + EMBEDDING_BATCH_SIZE]
        batch_tensors = []
        valid_ids = []
        
        for row in batch:
            img_id = row['id']
            img_path = row['file_path']
            
            if not os.path.exists(img_path):
                print(f"File missing: {img_path}")
                continue
                
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img)
                batch_tensors.append(img_t)
                valid_ids.append(img_id)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
        if not batch_tensors:
            continue
            
        # Inference
        inputs = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            
        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1).cpu().numpy()
        
        # Save to DB
        for img_id, emb in zip(valid_ids, embeddings):
            # Check if exists
            cursor.execute("SELECT 1 FROM embeddings WHERE image_id = ? AND model_name = ?", (img_id, EMBEDDING_MODEL))
            if cursor.fetchone():
                cursor.execute(
                    "UPDATE embeddings SET embedding = ? WHERE image_id = ? AND model_name = ?",
                    (emb, img_id, EMBEDDING_MODEL)
                )
            else:
                cursor.execute(
                    "INSERT INTO embeddings (image_id, model_name, embedding) VALUES (?, ?, ?)",
                    (img_id, EMBEDDING_MODEL, emb)
                )
        conn.commit()
        
    conn.close()
    print("Embeddings generation complete.")

if __name__ == "__main__":
    main()
