# File: train_triage_smoke_test.py

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

# --- 1. Configuration ---
SCRATCH_BASE_PATH = "dataset/" # ⚠️ UPDATE THIS

CONFIG = {
    "train_csv": os.path.join(SCRATCH_BASE_PATH, "triage_train_set.csv"),
    "test_csv": os.path.join(SCRATCH_BASE_PATH, "triage_test_set.csv"),
    "image_dir": os.path.join(SCRATCH_BASE_PATH, "CXR_ALL_FLAT"),
    "model_save_path": os.path.join(SCRATCH_BASE_PATH, "triage_model_smoke_test.pth"), # Save to a different file
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 1,   # ## SMOKE TEST MODIFICATION: Only 1 epoch
    "num_workers": 4,
    "smoke_test_batches": 20 # ## SMOKE TEST MODIFICATION: Number of batches to run
}

# --- 2. Dataset Class ---
class TriageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, limit=None):
        full_metadata = pd.read_csv(csv_file)
        if limit:
            # ## SMOKE TEST MODIFICATION: Use only a small subset of the data
            self.metadata = full_metadata.head(limit)
            print(f"--- SMOKE TEST: Using a limited dataset of {limit} samples ---")
        else:
            self.metadata = full_metadata
            
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # (This function remains the same)
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
    # (This function remains the same)
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
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # ## SMOKE TEST MODIFICATION: Limit the number of samples loaded
    dataset_limit = CONFIG["batch_size"] * (CONFIG["smoke_test_batches"] + 5)

    print("Loading datasets for SMOKE TEST...")
    train_dataset = TriageDataset(CONFIG["train_csv"], CONFIG["image_dir"], transform=data_transforms['train'], limit=dataset_limit)
    test_dataset = TriageDataset(CONFIG["test_csv"], CONFIG["image_dir"], transform=data_transforms['test'], limit=dataset_limit)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    print("Initializing model...")
    model = get_triage_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} (Smoke Test) ---")
        
        model.train()
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for i, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item())
            
            # ## SMOKE TEST MODIFICATION: Stop epoch early
            if i >= CONFIG["smoke_test_batches"]:
                print(f"--- SMOKE TEST: Stopping training epoch after {i+1} batches ---")
                break

        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(test_loader, desc=f"Validating Epoch {epoch+1}")):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                # ## SMOKE TEST MODIFICATION: Stop validation early
                if i >= CONFIG["smoke_test_batches"]:
                    print(f"--- SMOKE TEST: Stopping validation after {i+1} batches ---")
                    break
        
        val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        print(f"Epoch {epoch+1} Validation AUC (on subset): {val_auc:.4f}")
        torch.save(model.state_dict(), CONFIG["model_save_path"])
        print(f"🎉 Smoke test model saved to {CONFIG['model_save_path']}")

    print("\n--- ✅ Smoke test complete! ---")

if __name__ == '__main__':
    main()
