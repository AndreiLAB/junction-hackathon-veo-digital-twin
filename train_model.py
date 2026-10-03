import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import numpy as np
from config import OUTPUTS_DIR, EMBEDDING_MODEL, EMBEDDING_DIM
from db.database import get_db_connection

class VEOModel(nn.Module):
    def __init__(self, embedding_dim, num_scans, num_directions):
        super(VEOModel, self).__init__()
        # Small MLP on top of embeddings
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Multi-task heads
        self.scan_head = nn.Linear(256, num_scans)
        self.direction_head = nn.Linear(256, num_directions)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        scan_logits = self.scan_head(x)
        dir_logits = self.direction_head(x)
        
        return scan_logits, dir_logits

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, scan_labels, dir_labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.scan_labels = torch.tensor(scan_labels, dtype=torch.long)
        self.dir_labels = torch.tensor(dir_labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.scan_labels[idx], self.dir_labels[idx]

def get_direction_sector(yaw, num_sectors=6):
    # Sectorize 0-360 degrees
    sector_size = 360.0 / num_sectors
    return int((yaw % 360.0) / sector_size)

def main():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Load all data
    cursor.execute("""
        SELECT e.embedding, i.scan_id, i.yaw
        FROM embeddings e
        JOIN images i ON e.image_id = i.id
        WHERE e.model_name = ? AND i.yaw IS NOT NULL
    """, (EMBEDDING_MODEL,))
    
    data = cursor.fetchall()
    
    if not data:
        print("No embeddings found to train on.")
        return
        
    embeddings = []
    scan_ids = []
    yaws = []
    
    for row in data:
        embeddings.append(row['embedding'])
        scan_ids.append(row['scan_id'])
        yaws.append(row['yaw'])
        
    embeddings = np.array(embeddings)
    
    # Remap scan_ids to 0..N-1
    unique_scans = sorted(list(set(scan_ids)))
    scan_to_idx = {sid: i for i, sid in enumerate(unique_scans)}
    scan_labels = np.array([scan_to_idx[sid] for sid in scan_ids])
    
    num_sectors = 6
    dir_labels = np.array([get_direction_sector(y, num_sectors) for y in yaws])
    
    print(f"Loaded {len(embeddings)} samples across {len(unique_scans)} scans.")
    
    # LEAVE-ONE-SCAN-OUT Validation
    logo = LeaveOneGroupOut()
    
    fold_accuracies_scan = []
    fold_accuracies_dir = []
    
    # We will just do 3 random folds for speed in prototype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(embeddings, groups=scan_labels)):
        # Just run a few folds
        if fold >= 3:
            break
            
        train_dataset = EmbeddingDataset(embeddings[train_idx], scan_labels[train_idx], dir_labels[train_idx])
        test_dataset = EmbeddingDataset(embeddings[test_idx], scan_labels[test_idx], dir_labels[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = VEOModel(EMBEDDING_DIM, len(unique_scans), num_sectors).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training Fold {fold+1} (Left out scan {unique_scans[test_idx[0]]})")
        
        for epoch in range(10): # 10 epochs for quick prototype
            model.train()
            for emb, s_lbl, d_lbl in train_loader:
                emb, s_lbl, d_lbl = emb.to(device), s_lbl.to(device), d_lbl.to(device)
                
                optimizer.zero_grad()
                s_out, d_out = model(emb)
                
                loss_s = criterion(s_out, s_lbl)
                loss_d = criterion(d_out, d_lbl)
                loss = loss_s + loss_d
                
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        all_s_preds = []
        all_s_lbls = []
        all_d_preds = []
        all_d_lbls = []
        
        with torch.no_grad():
            for emb, s_lbl, d_lbl in test_loader:
                emb = emb.to(device)
                s_out, d_out = model(emb)
                
                s_preds = torch.argmax(s_out, dim=1).cpu().numpy()
                d_preds = torch.argmax(d_out, dim=1).cpu().numpy()
                
                all_s_preds.extend(s_preds)
                all_s_lbls.extend(s_lbl.numpy())
                all_d_preds.extend(d_preds)
                all_d_lbls.extend(d_lbl.numpy())
                
        # For leave-one-scan-out, scan prediction will obviously be 0% since the scan is unseen in training!
        # This highlights why nearest neighbor or image-pair training is better for generalization.
        # But for direction, it should be possible to predict.
        
        s_acc = accuracy_score(all_s_lbls, all_s_preds)
        d_acc = accuracy_score(all_d_lbls, all_d_preds)
        
        print(f"Fold {fold+1} Scan Acc: {s_acc:.2f} (Expected ~0% in LOO)")
        print(f"Fold {fold+1} Dir Acc: {d_acc:.2f}")
        
        fold_accuracies_scan.append(s_acc)
        fold_accuracies_dir.append(d_acc)
        
    print("\nTraining completed.")
    
    # Save the model trained on ALL data for localization testing
    model = VEOModel(EMBEDDING_DIM, len(unique_scans), num_sectors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = EmbeddingDataset(embeddings, scan_labels, dir_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Training final model on all data...")
    for epoch in range(15):
        model.train()
        for emb, s_lbl, d_lbl in train_loader:
            emb, s_lbl, d_lbl = emb.to(device), s_lbl.to(device), d_lbl.to(device)
            optimizer.zero_grad()
            s_out, d_out = model(emb)
            loss = criterion(s_out, s_lbl) + criterion(d_out, d_lbl)
            loss.backward()
            optimizer.step()
            
    # Save model and mapping
    torch.save({
        'state_dict': model.state_dict(),
        'scan_to_idx': scan_to_idx,
        'idx_to_scan': {v: k for k, v in scan_to_idx.items()},
        'num_sectors': num_sectors
    }, OUTPUTS_DIR / "trained_model.pth")
    print("Saved final model.")
    
if __name__ == "__main__":
    main()
