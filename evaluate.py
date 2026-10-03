import os
import time
import json
import torch
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from config import EMBEDDING_MODEL, OUTPUTS_DIR
from db.database import get_db_connection
from build_embeddings import load_model

def get_augmentations():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def angular_diff(a1, a2):
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)

def spatial_diff(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def main():
    print("Loading database and reference embeddings...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT e.image_id, e.embedding, i.name, i.file_path, i.x, i.y, i.z, i.yaw, i.pitch, i.roll, s.id as scan_id, s.name as scan_name
        FROM embeddings e
        JOIN images i ON e.image_id = i.id
        JOIN scans s ON i.scan_id = s.id
        WHERE e.model_name = ? AND i.file_path IS NOT NULL
    """, (EMBEDDING_MODEL,))
    
    db_data = cursor.fetchall()
    if not db_data:
        print("No embeddings found.")
        return
        
    db_embeddings = np.array([row['embedding'] for row in db_data])
    
    model, _ = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aug_transform = get_augmentations()
    
    # We will generate 3 augmented queries per image
    queries_per_image = 3
    
    spatial_errors = []
    angular_errors = []
    top1_correct = 0
    top5_correct = 0
    total_queries = 0
    inference_times = []
    
    results = []
    contact_sheet_data = []
    
    print(f"Generating {queries_per_image} augmented queries per image and evaluating...")
    
    for row in db_data:
        img_path = row['file_path']
        if not os.path.exists(img_path):
            continue
            
        orig_img = Image.open(img_path).convert('RGB')
        gt_scan = row['scan_id']
        gt_pos = (row['x'], row['y'], row['z'])
        gt_yaw = row['yaw']
        
        for q_idx in range(queries_per_image):
            # Generate augmented query
            q_tensor = aug_transform(orig_img).unsqueeze(0).to(device)
            
            # Inference
            start_t = time.time()
            with torch.no_grad():
                output = model(q_tensor)
                query_emb = torch.nn.functional.normalize(output, p=2, dim=1).cpu().numpy()[0]
            inf_time = time.time() - start_t
            inference_times.append(inf_time)
            
            # Retrieval
            sims = np.dot(db_embeddings, query_emb)
            top5_indices = np.argsort(sims)[::-1][:5]
            
            top5_rows = [db_data[idx] for idx in top5_indices]
            
            # Evaluate
            pred_top1 = top5_rows[0]
            pred_pos = (pred_top1['x'], pred_top1['y'], pred_top1['z'])
            pred_yaw = pred_top1['yaw']
            
            sp_err = spatial_diff(gt_pos, pred_pos)
            ang_err = angular_diff(gt_yaw, pred_yaw)
            
            spatial_errors.append(sp_err)
            angular_errors.append(ang_err)
            
            # We define correct localization as predicting the exact same original image
            # OR an image with very close spatial/angular bounds (e.g. same scan).
            # Let's use exact scan_id and same viewpoint as Top-1 accuracy constraint.
            if pred_top1['image_id'] == row['image_id']:
                top1_correct += 1
                
            if any(r['image_id'] == row['image_id'] for r in top5_rows):
                top5_correct += 1
                
            total_queries += 1
            
            if len(contact_sheet_data) < 5:
                # Save data for visual contact sheet (Query + Top 5)
                # We need to save the augmented image to disk temporarily or keep in memory
                # We'll just generate it again for the sheet or keep it in memory
                contact_sheet_data.append({
                    'query_img': orig_img, # Will augment specifically for drawing
                    'gt_name': row['name'],
                    'top5': top5_rows
                })
                
    # Calculate metrics
    top1_acc = top1_correct / total_queries
    top5_acc = top5_correct / total_queries
    med_spatial = np.median(spatial_errors)
    mean_spatial = np.mean(spatial_errors)
    med_angular = np.median(angular_errors)
    avg_inf = np.mean(inference_times)
    
    res_dict = {
        "total_queries": total_queries,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "median_spatial_error_m": float(med_spatial),
        "mean_spatial_error_m": float(mean_spatial),
        "median_angular_error_deg": float(med_angular),
        "average_inference_time_s": float(avg_inf)
    }
    
    with open(OUTPUTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(res_dict, f, indent=4)
        
    report = f"""# Visual Localization Evaluation Report

## Methodology
- **Reference Database**: {len(db_data)} original E57 extracted images.
- **Query Generation**: {queries_per_image} augmented queries generated dynamically per reference image.
- **Augmentations**: ColorJitter, RandomPerspective, RandomResizedCrop, GaussianBlur.
- **Embedding Model**: {EMBEDDING_MODEL} (Cosine Similarity Retrieval).
- **Note on Architecture**: The primary localization system is image retrieval, not scan-ID classification. While `train_model.py` explores a lightweight scan classifier, classification cannot generalize to unseen scans not present in the training set. Retrieval naturally handles this by simply adding new reference views.

## Results
- **Total Queries**: {total_queries}
- **Top-1 Accuracy**: {top1_acc:.2%} (Query matched its exact un-augmented source image)
- **Top-5 Accuracy**: {top5_acc:.2%}
- **Median Spatial Error**: {med_spatial:.3f} meters
- **Mean Spatial Error**: {mean_spatial:.3f} meters
- **Median Angular Error**: {med_angular:.2f} degrees
- **Avg Inference Time**: {avg_inf*1000:.1f} ms per image
"""
    with open(OUTPUTS_DIR / "evaluation_report.md", "w") as f:
        f.write(report)
        
    print(report)
    
    # Generate Contact Sheet
    if contact_sheet_data:
        print("Generating retrieval contact sheet...")
        img_size = 200
        sheet = Image.new('RGB', (img_size * 6, img_size * len(contact_sheet_data)), color=(255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            
        for r_idx, data in enumerate(contact_sheet_data):
            y_off = r_idx * img_size
            
            # Query (re-apply transform for visualization, just simple crop/color)
            q_vis = transforms.ColorJitter(brightness=0.4, contrast=0.4)(data['query_img'])
            q_vis = q_vis.resize((img_size, img_size))
            sheet.paste(q_vis, (0, y_off))
            draw.text((10, y_off + 10), f"QUERY (Aug)\nGT: {data['gt_name']}", fill="yellow", font=font)
            
            for c_idx, row in enumerate(data['top5']):
                x_off = (c_idx + 1) * img_size
                if row['file_path'] and os.path.exists(row['file_path']):
                    db_img = Image.open(row['file_path']).convert('RGB').resize((img_size, img_size))
                    sheet.paste(db_img, (x_off, y_off))
                draw.text((x_off + 10, y_off + 10), f"Top {c_idx+1}\n{row['name']}", fill="yellow", font=font)
                
        sheet.save(OUTPUTS_DIR / "retrieval_contact_sheet.jpg")

if __name__ == "__main__":
    main()
