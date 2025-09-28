# File: train_specialist_preprocessed.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
import os
import argparse
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import cv2 # Using OpenCV for advanced transforms

# --- 1. Custom Transform Classes for Feature Enhancement ---

class ApplyCLAHE:
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_gray = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

class GammaCorrection:
    """Applies Gamma Correction to adjust brightness non-linearly."""
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.invGamma = 1.0 / self.gamma
        self.table = np.array([((i / 255.0) ** self.invGamma) * 255
                               for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, img):
        img_np = np.array(img)
        # Apply the gamma correction using the lookup table
        return Image.fromarray(cv2.LUT(img_np, self.table))

# --- 2. Central Hub for Transform Selection ---

def get_pathology_transforms(pathology):
    """Returns a deterministic, pathology-specific transform pipeline."""
    if pathology in ['pneumonia', 'atelectasis']:
        # For these, we increase contrast and slightly increase brightness (gamma < 1.0)
        return transforms.Compose([
            ApplyCLAHE(clip_limit=3.0),
            GammaCorrection(gamma=0.8),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pathology == 'infiltration':
        # For this, we increase contrast and slightly decrease brightness (gamma > 1.0)
        return transforms.Compose([
            ApplyCLAHE(clip_limit=3.0),
            GammaCorrection(gamma=1.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pathology in ['cardiomegaly', 'effusion']:
        # For shape-based pathologies, we do no enhancement
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown pathology for transform selection: {pathology}")

# --- (The SpecialistDataset and get_specialist_model functions remain the same) ---
class SpecialistDataset(Dataset):
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
        label = self.metadata.loc[idx, 'label']
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor.unsqueeze(0)

def get_specialist_model(architecture_name='resnet50'):
    """Loads the specified pre-trained architecture and adapts it for binary classification."""
    print(f"Loading {architecture_name} model...")
    if architecture_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif architecture_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
    elif architecture_name == 'vit_base_patch16_224':
        model = models.vit_base_patch16_224(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 1)
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    # Fine-tune the whole model
    for param in model.parameters():
        param.requires_grad = True
    return model

# --- 3. Main Training Logic ---
def train(args):
    # --- Configuration ---
    SCRATCH_BASE_PATH = args.base_dir
    CONFIG = {
        "train_csv": os.path.join(SCRATCH_BASE_PATH, "specialist_datasets", f"{args.pathology}_specialist_train.csv"),
        "test_csv": os.path.join(SCRATCH_BASE_PATH, "specialist_datasets", f"{args.pathology}_specialist_test.csv"),
        "image_dir": os.path.join(SCRATCH_BASE_PATH, "CXR_ALL_FLAT"),
        "model_save_path": os.path.join(SCRATCH_BASE_PATH, f"specialist_{args.pathology}_{args.architecture}_preprocessed.pth"),
        "learning_rate": 1e-4, # Reverted to best performing LR
        "batch_size": 64,
        "num_epochs": 20, # Reduced epochs
        "num_workers": 8,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training PREPROCESSED Specialist: {args.pathology} ({args.architecture}) ---")

    # --- Dynamic Transform Selection ---
    train_transforms = get_pathology_transforms(args.pathology)
    # The test set should always have minimal processing
    test_transforms = get_pathology_transforms(args.pathology) # Use the simplest transform

    # --- Data and Sampler Setup ---
    train_dataset = SpecialistDataset(CONFIG["train_csv"], CONFIG["image_dir"], transform=train_transforms)
    test_dataset = SpecialistDataset(CONFIG["test_csv"], CONFIG["image_dir"], transform=test_transforms)
    
    # ## MODIFICATION: Removed WeightedRandomSampler and re-enabled shuffle=True ##
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    model = get_specialist_model(args.architecture).to(device)
    
    # --- Weighted Loss Calculation (as you specified) ---
    df_train = pd.read_csv(CONFIG["train_csv"])
    class_counts = df_train['label'].value_counts().to_dict()
    pos_count = class_counts.get(1, 0)
    neg_count = class_counts.get(0, 0)
    
    if pos_count > 0:
        pos_weight = neg_count / pos_count
        print(f"Calculated positive weight for {args.pathology}: {pos_weight:.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    else:
        print("Warning: No positive samples found, using standard BCE loss.")
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2)
    
    # --- Training Loop ---
    best_auc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        # (The training and validation loops are identical to the Triage script)
        # ...
        # --- Training Phase ---
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
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

        # --- Validation Phase ---
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Validating Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        print(f"Epoch {epoch+1} Val AUC: {val_auc:.4f}")
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"🎉 New best model saved for {args.pathology} with AUC: {best_auc:.4f}")

    print(f"\n--- Training complete for {args.pathology} --- Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specialist model for a specific pathology.")
    parser.add_argument("--pathology", type=str, required=True, help="Name of the pathology (e.g., 'pneumonia').")
    parser.add_argument("--architecture", type=str, required=True, help="Model architecture (e.g., 'resnet50').")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to the base scratch directory.")
    
    args = parser.parse_args()
    train(args)