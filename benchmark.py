import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_predictions(args):
    """
    Dynamically finds common pathologies between ground truth and prediction files,
    calculates classification metrics, and prints the results.
    """
    # --- 1. Load Data ---
    try:
        ground_truth_df = pd.read_csv(args.truth_file)
        predictions_df = pd.read_csv(args.result_file)
        print("✅ Data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}. Please ensure both CSV files are at the correct path.")
        return

    # ## MODIFICATION: Dynamically find the diseases to evaluate ##
    # Find the intersection of columns, excluding the image identifier columns.
    gt_cols = set(ground_truth_df.columns) - {'Image Index', 'Patient ID', 'Finding Labels', 'View Position'}
    pred_cols = set(predictions_df.columns) - {'image_filename'}
    
    target_diseases = sorted(list(gt_cols.intersection(pred_cols)))
    
    if not target_diseases:
        print("❌ ERROR: No common pathology columns found between the two files. Please check your CSVs.")
        return
        
    print(f"Found {len(target_diseases)} common pathologies to evaluate: {target_diseases}")

    # --- 2. Merge Data ---
    # Merge based on the respective image identifier columns.
    merged_df = pd.merge(
        ground_truth_df,
        predictions_df,
        left_on='Image Index',
        right_on='image_filename'
        # Note: We don't need suffixes if column names are now unique (except for identifiers)
        # But we will add them just in case of other overlapping columns.
    )

    # --- 3. Calculate Metrics for Each Disease ---
    results = []
    for disease in target_diseases:
        # For this logic, we assume the ground truth columns are named 'Disease'
        # and prediction columns are also named 'Disease'.
        # If your prediction file from torchxrayvision has different names, adjust here.
        y_true = merged_df[f"{disease}_x"] # Pandas appends _x for left df
        y_scores = merged_df[f"{disease}_y"] # Pandas appends _y for right df

        # Calculate AUC-ROC and find the best threshold for F1-score etc.
        auc_score = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find the optimal threshold that maximizes Youden's J statistic
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred_binary = (y_scores >= optimal_threshold).astype(int)

        # Calculate other metrics at this optimal threshold
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        results.append({
            "Disease": disease,
            "AUC-ROC": auc_score,
            "Threshold": optimal_threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    # --- 4. Display Results ---
    if not results:
        print("No results to display.")
        return
        
    results_df = pd.DataFrame(results)
    mean_auc = results_df['AUC-ROC'].mean()

    print(results_df)    
    print(f"\n🩺 Model Performance Evaluation: {os.path.basename(args.result_file)}")
    print("="*90)
    print(results_df.to_string(index=False, float_format="{:.4f}"))
    print("-" * 90)
    print(f"Mean AUC-ROC across {len(results_df)} pathologies: {mean_auc:.4f}")
    print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth.")
    parser.add_argument("--truth_file", type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the inference results CSV file.")
    args = parser.parse_args()
    evaluate_predictions(args)