import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(ground_truth_csv, predictions_csv):
    """
    Merges ground truth and prediction files, calculates classification metrics
    for target diseases, and prints the results in a formatted table.

    Args:
        ground_truth_csv (str): Path to the CSV with ground truth labels.
        predictions_csv (str): Path to the CSV with model prediction scores.
    """
    # These are the 5 diseases with binary labels in your sampled_subset.csv
    target_diseases = ["Pneumonia", "Effusion", "Cardiomegaly", "Infiltration", "Atelectasis"]

    # --- 1. Load and Merge Data ---
    try:
        ground_truth_df = pd.read_csv(ground_truth_csv)
        predictions_df = pd.read_csv(predictions_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both CSV files are in the correct directory.")
        return

    # This is the key step: Merge the two dataframes.
    # We use suffixes to distinguish between ground truth (_true) and prediction (_pred) columns
    # because both files share column names like "Atelectasis".
    merged_df = pd.merge(
        ground_truth_df,
        predictions_df,
        left_on='Image Index',
        right_on='image_filename',
        suffixes=('_true', '_pred')
    )

    # --- 2. Calculate Metrics for Each Disease ---
    results = []
    for disease in target_diseases:
        true_col = f"{disease}_true"
        pred_col = f"{disease}_pred"
        
        # Check if the disease exists in both dataframes
        if true_col not in merged_df.columns or pred_col not in merged_df.columns:
            print(f"Warning: Columns for '{disease}' not found. It might not be in both CSVs. Skipping.")
            continue

        y_true = merged_df[true_col]
        y_scores = merged_df[pred_col]

        # Calculate AUC-ROC and find the best threshold
        auc_score = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr) # Youden's J statistic
        optimal_threshold = thresholds[optimal_idx]

        # Apply threshold to get binary predictions
        y_pred_binary = (y_scores >= optimal_threshold).astype(int)

        # Calculate metrics
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

    # --- 3. Display Results in a Clean Table ---
    if not results:
        print("No results to display. Please check your CSV files and column names.")
        return
        
    results_df = pd.DataFrame(results)
    print("🩺 Model Performance Evaluation (densenet121-res224-all):")
    print("="*80)
    print(results_df.to_string(index=False, float_format="{:.4f}".format))
    print("="*80)

if __name__ == "__main__":
    # Use the filenames you've specified
    GROUND_TRUTH_FILE = 'sampled_subset_metadata.csv'
    PREDICTIONS_FILE = 'inference_results.csv'

    evaluate_predictions(GROUND_TRUTH_FILE, PREDICTIONS_FILE)