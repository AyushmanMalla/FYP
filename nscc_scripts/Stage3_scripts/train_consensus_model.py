# File: train_consensus_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- 1. The Dataset for Pre-computed Features ---
class ConsensusDataset(Dataset):
    """A very fast dataset that reads pre-computed features and scores from a CSV."""
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Identify feature and label columns
        label_cols = [col for col in df.columns if col.startswith('True_')]
        score_cols = [col for col in df.columns if col.startswith('Score_')]
        triage_cols = [col for col in df.columns if col.startswith('TriageFeat_')]
        
        # Convert to numpy for faster access
        self.labels = df[label_cols].values.astype(np.float32)
        
        # Combine triage features and specialist scores into one feature matrix
        triage_features = df[triage_cols].values.astype(np.float32)
        score_features = df[score_cols].values.astype(np.float32)
        self.features = np.concatenate([triage_features, score_features], axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. The Dynamic Consensus Model (MLP) ---
class DynamicConsensusModel(nn.Module):
    def __init__(self, input_dim=2053, hidden_dim=256, output_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# --- 3. Main Training Logic ---
def train(args):
    # --- Configuration for Optimized Run ---
    CONFIG = {
        "train_csv": os.path.join(args.base_dir, "consensus_train_set.csv"),
        "val_csv": os.path.join(args.base_dir, "consensus_val_set.csv"),
        "model_save_path": os.path.join(args.base_dir, "dynamic_consensus_model.pth"),
        "learning_rate": 3e-4, # A good starting LR for MLPs
        "batch_size": 256,     # Large batch size for GPU utilization
        "num_epochs": 25,
        "num_workers": 14,     # Use most of the 16 available CPUs
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Dynamic Consensus Model on {device} ---")

    # --- Optimized DataLoader Setup ---
    train_dataset = ConsensusDataset(CONFIG["train_csv"])
    val_dataset = ConsensusDataset(CONFIG["val_csv"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True # Key optimization
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True # Key optimization
    )

    # --- Model, Loss, and Optimizer ---
    model = DynamicConsensusModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2, verbose=True)

    # --- Automatic Mixed Precision (AMP) for speed ---
    scaler = torch.cuda.amp.GradScaler()

    best_auc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        
        model.train()
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for features, labels in train_bar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use AMP's autocast
            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_bar.set_postfix(loss=loss.item())

        # --- Validation Phase ---
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        
        # Calculate mean AUC across the 5 pathologies
        val_aucs = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(5)]
        mean_val_auc = np.mean(val_aucs)
        
        print(f"Epoch {epoch+1} Mean Val AUC: {mean_val_auc:.4f}")
        scheduler.step(mean_val_auc)

        if mean_val_auc > best_auc:
            best_auc = mean_val_auc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"🎉 New best model saved with Mean AUC: {best_auc:.4f}")

    print(f"\n--- Training complete --- Best Mean AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Dynamic Consensus Model.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to base scratch directory.")
    args = parser.parse_args()
    train(args)
