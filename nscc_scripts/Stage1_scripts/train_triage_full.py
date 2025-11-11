# File: train_triage_full.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# --- 1. Configuration for Full Run ---
SCRATCH_BASE_PATH = "dataset/" # ⚠️ UPDATE THIS

CONFIG = {
    "train_csv": os.path.join(SCRATCH_BASE_PATH, "triage_train_set.csv"),
    "val_csv": os.path.join(SCRATCH_BASE_PATH, "triage_val_set.csv"),
    "image_dir": os.path.join(SCRATCH_BASE_PATH, "CXR_ALL_FLAT"),
    "model_save_path": os.path.join(SCRATCH_BASE_PATH, "triage_model_best.pth"),
    "learning_rate": 1e-4,
    "batch_size": 64,      # Increased for powerful A100 GPUs
    "num_epochs": 10,      # A solid number for a full run
    "num_workers": 16,      # Match the number of CPUs requested
}

# --- 2. Dataset Class ---
class TriageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.loc[idx, 'Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.metadata.loc[idx, 'is_abnormal']
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label.unsqueeze(0)

# --- 3. Model Definition ---
def get_triage_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# --- 4. Main Training and Evaluation Logic ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    print("Loading datasets...")
    train_dataset = TriageDataset(CONFIG["train_csv"], CONFIG["image_dir"], transform=data_transforms['train'])
    val_dataset = TriageDataset(CONFIG["val_csv"], CONFIG["image_dir"], transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    print("Initializing model...")
    model = get_triage_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.1)

    best_auc = 0.0

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_preds)
        val_accuracy = accuracy_score(np.round(all_labels), np.round(all_preds))
        print(f"Epoch {epoch+1} Validation AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"📉 Learning rate reduced to {new_lr:.1e}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"🎉 New best model saved with AUC: {best_auc:.4f}")

    print(f"\n--- Training complete --- Best AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
