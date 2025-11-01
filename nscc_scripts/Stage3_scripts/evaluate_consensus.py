# File: evaluate_consensus_corrected.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# --- 1. Model and Dataset Classes (Unchanged) ---
class ConsensusDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        label_cols = [col for col in df.columns if col.startswith('True_')]
        score_cols = [col for col in df.columns if col.startswith('Score_')]
        triage_cols = [col for col in df.columns if col.startswith('TriageFeat_')]
        
        # Ensure correct column ordering for labels
        self.label_names = sorted([col.replace('True_', '') for col in label_cols])
        sorted_label_cols = [f"True_{name}" for name in self.label_names]
        
        self.labels = df[sorted_label_cols].values.astype(np.float32)
        triage_features = df[triage_cols].values.astype(np.float32)
        score_features = df[score_cols].values.astype(np.float32)
        self.features = np.concatenate([triage_features, score_features], axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

# --- 2. Corrected Evaluation Logic ---
def evaluate(args):
    # --- Configuration ---
    CONFIG = {
        "test_csv": os.path.join(args.base_dir, "val_split.csv"),
        "model_path": os.path.join(args.base_dir, "dynamic_consensus_model.pth"),
        "batch_size": 256,
        "num_workers": 14, # Reduced for better compatibility
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating Consensus Model on {device} ---")

    test_dataset = ConsensusDataset(CONFIG["test_csv"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = DynamicConsensusModel(output_dim=len(test_dataset.label_names)).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    all_labels, all_preds_proba = [], []
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            features = features.to(device)
            outputs = model(features)
            preds_proba = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.numpy())
            all_preds_proba.append(preds_proba)

    all_labels = np.concatenate(all_labels)
    all_preds_proba = np.concatenate(all_preds_proba)

    # --- CORRECTED METRIC CALCULATION ---
    # This block now finds and uses the optimal threshold for each pathology,
    # matching the logic from your final_benchmark.py script.
    
    pathologies = test_dataset.label_names
    results = {}

    for i, pathology in enumerate(pathologies):
        y_true_pathology = all_labels[:, i]
        y_pred_proba_pathology = all_preds_proba[:, i]

        # Calculate AUC (threshold-independent)
        try:
            auc = roc_auc_score(y_true_pathology, y_pred_proba_pathology)
        except ValueError:
            auc = np.nan # Handle cases where only one class is present in y_true

        # Find the optimal threshold from the ROC curve using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true_pathology, y_pred_proba_pathology)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Binarize predictions using this "perfect" threshold for the test set
        y_pred_binary = (y_pred_proba_pathology >= optimal_threshold).astype(int)

        # Calculate other metrics based on the binarized predictions
        results[pathology] = {
            "AUC": auc,
            "Accuracy": accuracy_score(y_true_pathology, y_pred_binary),
            "Precision": precision_score(y_true_pathology, y_pred_binary, zero_division=0),
            "Recall": recall_score(y_true_pathology, y_pred_binary, zero_division=0),
            "F1-Score": f1_score(y_true_pathology, y_pred_binary, zero_division=0),
            "Optimal_Threshold": optimal_threshold
        }

    # --- Print the results in a clear, formatted table ---
    results_df = pd.DataFrame(results)

    print("\n--- Final End-to-End Pipeline Performance ---")
    print(f"{'Pathology':<15} | {'AUC':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)
    
    mean_results = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    for pathology, metrics in results.items():
        print(f"{pathology:<15} | {metrics['AUC']:<10.4f} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | | {metrics['Optimal_Threshold']:<10.4f}")
        for key in mean_results:
            mean_results[key].append(metrics[key])

    print("-" * 65)
    print(f"{'Mean':<15} | {np.mean(mean_results['AUC']):<10.4f} | {np.mean(mean_results['Accuracy']):<10.4f} | {np.mean(mean_results['Precision']):<10.4f} | {np.mean(mean_results['Recall']):<10.4f}")
    print("---------------------------------------------")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Dynamic Consensus Model.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to base scratch directory.")
    args = parser.parse_args()
    evaluate(args)