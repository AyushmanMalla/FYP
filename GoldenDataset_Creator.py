import pandas as pd
import os
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def create_full_pa_split(args):
    """
    Loads the NIH dataset, filters for PA-view images, and performs a
    patient-aware, multi-label stratified split across all 14 pathologies
    to create a comprehensive train and test set for benchmarking.
    """
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing the full NIH metadata...")
    src_base = args.dataset_dir
    csv_path = os.path.join(src_base, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)

    df_pa = df[df["View Position"] == "PA"].copy()
    print(f"Filtered to {len(df_pa)} PA-view images.")
    
    all_conditions = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    def label_contains(label_str, condition):
        return int(condition in str(label_str).split("|"))

    for condition in all_conditions:
        df_pa[condition] = df_pa["Finding Labels"].apply(lambda x: label_contains(x, condition))

    # --- 2. Create Patient-Level Labels for Stratification ---
    print("\nCreating patient-level labels for stratification...")
    patient_ids_col = "Patient ID"
    patient_labels = df_pa.groupby(patient_ids_col)[all_conditions].max()

    patients = patient_labels.index.to_numpy().reshape(-1, 1)
    labels = patient_labels.to_numpy()
    print(f"Found {len(patients)} unique patients in the PA-view dataset.")
    print("-" * 30)

    # --- 3. Perform the Stratified, Patient-Aware Split ---
    print(f"Performing multi-label stratified split with a {args.test_size:.0%} test size...")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_patient_idx, test_patient_idx = next(msss.split(patients, labels))

    train_pids = patients[train_patient_idx].flatten()
    test_pids = patients[test_patient_idx].flatten()

    train_df = df_pa[df_pa[patient_ids_col].isin(train_pids)]
    test_df = df_pa[df_pa[patient_ids_col].isin(test_pids)]
    print("Split complete.")
    print("-" * 30)

    # --- 4. Verify the Split and Save ---
    print(f"Total Patients: {len(patient_labels)}")
    print(f"Train Patients: {len(train_pids)}")
    print(f"Test Patients:  {len(test_pids)}")

    # ## MODIFICATION: Added distribution verification check ##
    print("\nVerifying condition distribution across splits (should be similar):")
    # Add a 'No Finding' column just for the verification printout
    df_pa['No Finding'] = (df_pa['Finding Labels'] == 'No Finding').astype(int)
    train_df['No Finding'] = (train_df['Finding Labels'] == 'No Finding').astype(int)
    test_df['No Finding'] = (test_df['Finding Labels'] == 'No Finding').astype(int)
    
    cols_to_verify = all_conditions + ['No Finding']
    overall_dist = df_pa[cols_to_verify].mean().rename("Overall")
    train_dist = train_df[cols_to_verify].mean().rename("Train Set")
    test_dist = test_df[cols_to_verify].mean().rename("Test Set")
    
    dist_df = pd.concat([overall_dist, train_dist, test_dist], axis=1)
    print(dist_df.to_string(float_format="{:.4f}".format))
    
    # --- 5. Save final files ---
    # Drop the temporary 'No Finding' column before saving if you don't need it
    train_df = train_df.drop(columns=['No Finding'])
    test_df = test_df.drop(columns=['No Finding'])

    train_csv_path = os.path.join(src_base, "nih_pa_view_train.csv")
    test_csv_path = os.path.join(src_base, "nih_pa_view_test.csv")
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"\n--- ✅ Full PA-view train and test sets created! ---")
    print(f"Train set metadata saved to: {train_csv_path}")
    print(f"Test set metadata saved to: {test_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a full, stratified train/test split of all PA-view images from the NIH dataset."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to the root directory of the NIH dataset."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Proportion of patients to allocate to the test set."
    )

    args = parser.parse_args()
    create_full_pa_split(args)