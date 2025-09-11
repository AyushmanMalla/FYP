import pandas as pd
import os
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def create_dataset_split(args):
    """
    Loads the NIH dataset, curates it for specific pathologies + 'No Finding',
    and performs a patient-aware, multi-label stratified split to create
    train and test sets.

    Args:
        args (argparse.Namespace): Command-line arguments containing dataset_dir and test_size.
    """
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing the full NIH metadata...")
    src_base = args.dataset_dir
    csv_path = os.path.join(src_base, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)

    df = df[df["View Position"] == "PA"].copy()
    target_conditions = ["Pneumonia", "Effusion", "Cardiomegaly", "Infiltration", "Atelectasis"]

    def label_contains(label_str, condition):
        return int(condition in str(label_str).split("|"))

    for condition in target_conditions:
        df[condition] = df["Finding Labels"].apply(lambda x: label_contains(x, condition))

    # --- 2. Filter for the Desired Population ---
    df['has_target_pathology'] = df[target_conditions].max(axis=1)
    df['is_no_finding'] = (df['Finding Labels'] == 'No Finding').astype(int)
    df_filtered = df[(df['has_target_pathology'] == 1) | (df['is_no_finding'] == 1)].copy()
    
    print(f"Filtered dataset to {len(df_filtered)} images (Target Pathologies + No Finding).")
    print("-" * 30)

    # --- 3. Create Patient-Level Labels for Stratification ---
    print("Creating patient-level labels for stratification...")
    patient_ids_col = "Patient ID"
    patient_labels = df_filtered.groupby(patient_ids_col)[target_conditions].max()

    patients = patient_labels.index.to_numpy().reshape(-1, 1)
    labels = patient_labels.to_numpy()
    print(f"Found {len(patients)} unique patients in the filtered dataset.")
    print("-" * 30)

    # --- 4. Perform the Stratified, Patient-Aware Split ---
    print(f"Performing multi-label stratified split with a {args.test_size:.0%} test size...")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_patient_idx, test_patient_idx = next(msss.split(patients, labels))

    train_pids = patients[train_patient_idx].flatten()
    test_pids = patients[test_patient_idx].flatten()

    train_df = df_filtered[df_filtered[patient_ids_col].isin(train_pids)]
    test_df = df_filtered[df_filtered[patient_ids_col].isin(test_pids)]
    print("Split complete.")
    print("-" * 30)

    # --- 5. Verify the Split and Save ---
    print(f"Total Patients in filtered set: {len(patient_labels)}")
    print(f"Train Patients: {len(train_pids)} ({len(train_pids)/len(patient_labels):.2%})")
    print(f"Test Patients: {len(test_pids)} ({len(test_pids)/len(patient_labels):.2%})")
    print("\nVerifying condition distribution across splits (should be similar):")

    cols_to_verify = target_conditions + ['is_no_finding']
    overall_dist = df_filtered[cols_to_verify].mean().rename("Overall")
    train_dist = train_df[cols_to_verify].mean().rename("Train Set")
    test_dist = test_df[cols_to_verify].mean().rename("Test Set")
    dist_df = pd.concat([overall_dist, train_dist, test_dist], axis=1)
    dist_df = dist_df.rename(index={'is_no_finding': 'No Finding'})
    print(dist_df.to_string(float_format="{:.4f}".format))

    train_csv_path = os.path.join(src_base, "nih_curated_train_set.csv")
    test_csv_path = os.path.join(src_base, "nih_curated_test_set.csv")
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print("\n--- ✅ Rigorous and CURATED train and test sets created! ---")
    print(f"Train set metadata saved to: {train_csv_path}")
    print(f"Test set metadata saved to: {test_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a rigorous, curated, and stratified train/test split from the NIH Chest X-ray dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the root directory of the NIH dataset (containing Data_Entry_2017.csv)."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.015,
        help="Proportion of the dataset to allocate to the test set (e.g., 0.2 for 20%)."
    )

    args = parser.parse_args()
    create_dataset_split(args)