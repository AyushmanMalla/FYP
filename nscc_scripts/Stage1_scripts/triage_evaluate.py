import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# --- 1. Configuration ---
# ⚠️ UPDATE THESE PATHS to match your environment
SCRATCH_BASE_PATH = "dataset/"
CONFIG = {
    "test_csv": os.path.join(SCRATCH_BASE_PATH, "triage_test_set.csv"),
    "image_dir": os.path.join(SCRATCH_BASE_PATH, "CXR_ALL_FLAT"),
    "model_path": os.path.join(SCRATCH_BASE_PATH, "triage_model_best.pth"), # Path to your saved model
    "batch_size": 64,
    "num_workers": 16,
}

# --- 2. Dataset and Model Definition (copy from your training script) ---
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

def get_triage_model():
    model = models.resnet50(weights=None) # Set weights to None, as we are loading our own
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# --- 3. Main Evaluation Logic ---
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the same transforms as your validation set
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading test dataset...")
    test_dataset = TriageDataset(CONFIG["test_csv"], CONFIG["image_dir"], transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print("Loading trained model...")
    model = get_triage_model()
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model = model.to(device)
    model.eval()

    all_labels, all_preds_proba, all_preds_binary = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_proba = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds_proba.extend(preds_proba)
            # Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
            all_preds_binary.extend((preds_proba > 0.5).astype(int))

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds_binary = np.array(all_preds_binary)

    # --- Calculate all metrics ---
    auc = roc_auc_score(all_labels, all_preds_proba)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary)
    recall = recall_score(all_labels, all_preds_binary)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_binary).ravel()

    print("\n--- Triage Model Final Performance ---")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (How many selected 'Abnormal' items are relevant?)")
    print(f"Recall: {recall:.4f}  (How many relevant 'Abnormal' items are selected?)")
    print("\n--- Confusion Matrix ---")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp} (Type I Error)")
    print(f"False Negatives (FN): {fn} (Type II Error - The more critical error in this case)")
    print("--------------------------------------")


if __name__ == '__main__':
    evaluate()