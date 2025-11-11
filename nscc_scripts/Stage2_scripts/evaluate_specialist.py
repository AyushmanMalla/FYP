import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from train_specialist_final import SpecialistDataset, get_specialist_model, get_pathology_transforms

def evaluate(args):
    """
    Evaluates a trained specialist model on the test set.
    """
    # --- Configuration ---
    SCRATCH_BASE_PATH = args.base_dir
    CONFIG = {
        "test_csv": os.path.join(SCRATCH_BASE_PATH, "specialist_test_set.csv"),
        "image_dir": os.path.join(SCRATCH_BASE_PATH, "CXR_ALL_FLAT"),
        "model_path": os.path.join(SCRATCH_BASE_PATH, f"specialist_{args.pathology}_{args.architecture}_preprocessed.pth"),
        "batch_size": 64,
        "num_workers": 8,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating Specialist: {args.pathology} ({args.architecture}) on {device} ---")

    # --- Data Loading ---
    test_transforms = get_pathology_transforms(args.pathology)
    test_dataset = SpecialistDataset(CONFIG["test_csv"], CONFIG["image_dir"], args.pathology, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # --- Model Loading ---
    model = get_specialist_model(args.architecture)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.to(device)
    model.eval()

    # --- Evaluation ---
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {args.pathology}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    print(f"\n--- Evaluation Complete ---")
    print(f"Pathology: {args.pathology}")
    print(f"Architecture: {args.architecture}")
    print(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained specialist model.")
    parser.add_argument("--pathology", type=str, required=True, help="Name of the pathology (e.g., 'Pneumonia').", choices=["Pneumonia", "Effusion", "Cardiomegaly", "Infiltration", "Atelectasis"])
    parser.add_argument("--architecture", type=str, required=True, help="Model architecture (e.g., 'resnet50').")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to the base scratch directory.")
    
    args = parser.parse_args()
    evaluate(args)
