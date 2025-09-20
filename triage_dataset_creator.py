import pandas as pd
import os
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

def create_triage_dataset(df_pa, output_dir, test_size):
    """
    Creates the Triage dataset (Normal vs. Abnormal) with a patient-aware,
    stratified split.
    """
    print("\n" + "="*50)
    print("🚀 Creating Triage Dataset (Normal vs. Abnormal)")
    print("="*50)

    # --- 1. Create the binary 'is_abnormal' label ---
    df_triage = df_pa.copy()
    df_triage['is_abnormal'] = (df_triage['Finding Labels'] != 'No Finding').astype(int)
    print(f"Created 'is_abnormal' label. Distribution:\n{df_triage['is_abnormal'].value_counts(normalize=True).to_string()}")

    # --- 2. Create Patient-Level Labels for Stratification ---
    patient_ids_col = "Patient ID"
    patient_labels = df_triage.groupby(patient_ids_col)[['is_abnormal']].max()
    patients = patient_labels.index.to_numpy().reshape(-1, 1)
    labels = patient_labels.to_numpy()

    # --- 3. Perform the Stratified, Patient-Aware Split ---
    print(f"\nPerforming binary stratified split with a {test_size:.0%} test size...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_patient_idx, test_patient_idx = next(sss.split(patients, labels))
    
    train_pids = patients[train_patient_idx].flatten()
    test_pids = patients[test_patient_idx].flatten()

    train_df = df_triage[df_triage[patient_ids_col].isin(train_pids)]
    test_df = df_triage[df_triage[patient_ids_col].isin(test_pids)]
    print("Split complete.")

    # --- 4. Verify and Save ---
    overall_dist = df_triage['is_abnormal'].mean()
    train_dist = train_df['is_abnormal'].mean()
    test_dist = test_df['is_abnormal'].mean()
    print("\nVerifying 'is_abnormal' distribution:")
    print(f"  Overall: {overall_dist:.4f}\n  Train:   {train_dist:.4f}\n  Test:    {test_dist:.4f}")

    train_csv_path = os.path.join(output_dir, "triage_train_set.csv")
    test_csv_path = os.path.join(output_dir, "triage_test_set.csv")
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"\n✅ Triage dataset created!")
    print(f"   Train set saved to: {train_csv_path} ({len(train_df)} images)")
    print(f"   Test set saved to:  {test_csv_path} ({len(test_df)} images)")

def create_specialist_dataset(df_pa, output_dir, test_size):
    """
    Creates the Specialist dataset (Abnormal-Only, 5 pathologies) with a
    patient-aware, multi-label stratified split.
    """
    print("\n" + "="*50)
    print("🚀 Creating Specialist Dataset (Abnormal-Only, 5 Pathologies)")
    print("="*50)

    # --- 1. One-hot encode the 5 target conditions ---
    df_spec = df_pa.copy()
    target_conditions = ["Pneumonia", "Effusion", "Cardiomegaly", "Infiltration", "Atelectasis"]
    
    def label_contains(label_str, condition):
        return int(condition in str(label_str).split("|"))

    for condition in target_conditions:
        df_spec[condition] = df_spec["Finding Labels"].apply(lambda x: label_contains(x, condition))

    # --- 2. Filter for Abnormal-Only images with target pathologies ---
    df_spec['has_target_pathology'] = df_spec[target_conditions].max(axis=1)
    df_filtered = df_spec[df_spec['has_target_pathology'] == 1].copy()
    print(f"Filtered to {len(df_filtered)} abnormal images containing at least one of the 5 target pathologies.")

    # --- 3. Create Patient-Level Labels for Stratification ---
    patient_ids_col = "Patient ID"
    patient_labels = df_filtered.groupby(patient_ids_col)[target_conditions].max()
    patients = patient_labels.index.to_numpy().reshape(-1, 1)
    labels = patient_labels.to_numpy()
    
    # --- 4. Perform the Multi-Label Stratified, Patient-Aware Split ---
    print(f"\nPerforming multi-label stratified split with a {test_size:.0%} test size...")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_patient_idx, test_patient_idx = next(msss.split(patients, labels))

    train_pids = patients[train_patient_idx].flatten()
    test_pids = patients[test_patient_idx].flatten()

    train_df = df_filtered[df_filtered[patient_ids_col].isin(train_pids)]
    test_df = df_filtered[df_filtered[patient_ids_col].isin(test_pids)]
    print("Split complete.")

    # --- 5. Verify and Save ---
    print("\nVerifying condition distribution across splits (should be similar):")
    overall_dist = df_filtered[target_conditions].mean().rename("Overall")
    train_dist = train_df[target_conditions].mean().rename("Train Set")
    test_dist = test_df[target_conditions].mean().rename("Test Set")
    dist_df = pd.concat([overall_dist, train_dist, test_dist], axis=1)
    print(dist_df.to_string(float_format="{:.4f}".format))

    train_csv_path = os.path.join(output_dir, "specialist_train_set.csv")
    test_csv_path = os.path.join(output_dir, "specialist_test_set.csv")
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"\n✅ Specialist dataset created!")
    print(f"   Train set saved to: {train_csv_path} ({len(train_df)} images)")
    print(f"   Test set saved to:  {test_csv_path} ({len(test_df)} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Triage and Specialist datasets from the NIH Chest X-ray dataset."
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
        default=0.15, # Using a more standard 15% test size
        help="Proportion of the patients to allocate to the test set."
    )

    args = parser.parse_args()

    # --- Load and filter for PA view once ---
    print("Loading and filtering for PA-view images...")
    csv_path = os.path.join(args.dataset_dir, "Data_Entry_2017.csv")
    df_full = pd.read_csv(csv_path)
    df_pa_view = df_full[df_full["View Position"] == "PA"].copy()
    print(f"Found {len(df_pa_view)} PA-view images.")

    # --- Create both datasets ---
    create_triage_dataset(df_pa_view, args.dataset_dir, args.test_size)
    create_specialist_dataset(df_pa_view, args.dataset_dir, args.test_size)
    
    print("\n\n--- 🎉 All datasets created successfully! ---")