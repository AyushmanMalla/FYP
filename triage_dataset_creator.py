import pandas as pd
import os
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

#Usage - python triage_dataset_creator.py --dataset_dir <path-to-Data_Entry_2017.csv file>

def create_datasets(df_pa, output_dir, val_size, test_size):
    """
    Creates Triage and Specialist datasets with a patient-aware, multi-label
    stratified 3-way split (train, validation, test).
    """
    print("\n" + "="*50)
    print("🚀 Creating Unified Train, Validation, and Test Datasets")
    print("="*50)

    # --- 1. Create combined labels for stratification ---
    df_processed = df_pa.copy()
    target_conditions = ["Pneumonia", "Effusion", "Cardiomegaly", "Infiltration", "Atelectasis"]
    
    # One-hot encode the 5 target conditions
    def label_contains(label_str, condition):
        return int(condition in str(label_str).split("|"))
    for condition in target_conditions:
        df_processed[condition] = df_processed["Finding Labels"].apply(lambda x: label_contains(x, condition))
        
    # Create the binary 'is_abnormal' label
    df_processed['is_abnormal'] = (df_processed['Finding Labels'] != 'No Finding').astype(int)
    
    # Combine all labels for a single, robust stratification
    stratification_labels = ['is_abnormal'] + target_conditions

    # --- 2. Create Patient-Level Labels for Stratification ---
    patient_ids_col = "Patient ID"
    patient_labels = df_processed.groupby(patient_ids_col)[stratification_labels].max()
    patients = patient_labels.index.to_numpy()
    labels = patient_labels.to_numpy()
    print(f"Performing split on {len(patients)} unique patients.")

    # --- 3. Perform the 3-Way Stratified, Patient-Aware Split ---
    # First split: separate train from (validation + test)
    temp_size = val_size + test_size
    print(f"\nPerforming first split: {1-temp_size:.0%} train, {temp_size:.0%} temp (val+test)...")
    msss_train_temp = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=temp_size, random_state=42)
    train_patient_idx, temp_patient_idx = next(msss_train_temp.split(patients, labels))
    
    train_pids = patients[train_patient_idx]
    temp_pids = patients[temp_patient_idx]
    temp_labels = labels[temp_patient_idx]

    # Second split: separate validation and test from the temp set
    relative_test_size = test_size / temp_size
    print(f"Performing second split on temp set: {1-relative_test_size:.0%} validation, {relative_test_size:.0%} test...")
    msss_val_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=42)
    val_patient_idx, test_patient_idx = next(msss_val_test.split(temp_pids, temp_labels))

    val_pids = temp_pids[val_patient_idx]
    test_pids = temp_pids[test_patient_idx]

    # --- 4. Create the final dataframes ---
    train_df = df_processed[df_processed[patient_ids_col].isin(train_pids)]
    val_df = df_processed[df_processed[patient_ids_col].isin(val_pids)]
    test_df = df_processed[df_processed[patient_ids_col].isin(test_pids)]
    print("Split complete.")
    print(f"  Train patients: {len(train_pids)}, Val patients: {len(val_pids)}, Test patients: {len(test_pids)}")
    print(f"  Train images:   {len(train_df)}, Val images:   {len(val_df)}, Test images:   {len(test_df)}")

    # --- 5. Create and Save Triage Datasets ---
    print("\n" + "-"*20 + " Triage Datasets " + "-"*20)
    triage_cols = ['Image Index', 'Finding Labels', 'Patient ID', 'is_abnormal']
    
    # Verify distribution
    overall_dist = df_processed['is_abnormal'].mean()
    train_dist = train_df['is_abnormal'].mean()
    val_dist = val_df['is_abnormal'].mean()
    test_dist = test_df['is_abnormal'].mean()
    print("Verifying 'is_abnormal' distribution:")
    print(f"  Overall: {overall_dist:.4f}\n  Train:   {train_dist:.4f}\n  Val:     {val_dist:.4f}\n  Test:    {test_dist:.4f}")

    # Save
    for split_name, df_split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = os.path.join(output_dir, f"triage_{split_name}_set.csv")
        df_split[triage_cols].to_csv(path, index=False)
        print(f"  Saved {split_name} set to: {path} ({len(df_split)} images)")

    # --- 6. Create and Save Specialist Datasets ---
    print("\n" + "-"*20 + " Specialist Datasets " + "-"*20)
    
    # Filter for abnormal images with target pathologies
    df_spec_filtered = df_processed[df_processed[target_conditions].max(axis=1) == 1].copy()
    
    specialist_train_df = df_spec_filtered[df_spec_filtered[patient_ids_col].isin(train_pids)]
    specialist_val_df = df_spec_filtered[df_spec_filtered[patient_ids_col].isin(val_pids)]
    specialist_test_df = df_spec_filtered[df_spec_filtered[patient_ids_col].isin(test_pids)]

    # Verify distribution
    print("Verifying condition distribution across specialist splits:")
    dist_df = pd.concat([
        df_spec_filtered[target_conditions].mean().rename("Overall"),
        specialist_train_df[target_conditions].mean().rename("Train"),
        specialist_val_df[target_conditions].mean().rename("Val"),
        specialist_test_df[target_conditions].mean().rename("Test")
    ], axis=1)
    print(dist_df.to_string(float_format="{:.4f}".format))

    # Save
    for split_name, df_split in [('train', specialist_train_df), ('val', specialist_val_df), ('test', specialist_test_df)]:
        path = os.path.join(output_dir, f"specialist_{split_name}_set.csv")
        df_split.to_csv(path, index=False)
        print(f"  Saved {split_name} set to: {path} ({len(df_split)} images)")


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
        "--val_size",
        type=float,
        default=0.15,
        help="Proportion of patients for the validation set."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Proportion of patients for the test set."
    )

    args = parser.parse_args()

    # --- Load and filter for PA view once ---
    print("Loading and filtering for PA-view images...")
    csv_path = os.path.join(args.dataset_dir, "Data_Entry_2017.csv")
    df_full = pd.read_csv(csv_path)
    df_pa_view = df_full[df_full["View Position"] == "PA"].copy()
    print(f"Found {len(df_pa_view)} PA-view images.")

    # --- Create all datasets from a single split ---
    create_datasets(df_pa_view, args.dataset_dir, args.val_size, args.test_size)
    
    print("\n\n--- 🎉 All datasets created successfully! ---")