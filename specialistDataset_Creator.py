import pandas as pd
import os
import argparse

def create_one_vs_all_datasets(args):
    """
    Reads the master abnormal-only train/test splits and generates
    five separate one-vs-all binary classification datasets for each specialist.
    """
    print("--- 🚀 Starting Specialist Dataset Creation ---")

    # Define the 5 pathologies for our specialists
    pathologies = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Pneumonia"]

    # Load the master abnormal-only datasets
    try:
        train_master_df = pd.read_csv(args.train_csv_path)
        test_master_df = pd.read_csv(args.test_csv_path)
        print(f"Loaded master train set with {len(train_master_df)} images.")
        print(f"Loaded master test set with {len(test_master_df)} images.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Master file not found. {e}")
        return

    # Create a directory to store the new datasets
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Datasets will be saved in: {output_dir}")
    print("-" * 30)

    # Loop through each pathology to create its specific dataset
    for pathology in pathologies:
        print(f"Processing dataset for '{pathology}' specialist...")

        # --- Process Training Set ---
        train_df = train_master_df.copy()
        # Create the binary 'label' column (1 for the target pathology, 0 for others)
        train_df['label'] = (train_df[pathology] == 1).astype(int)
        # Keep only essential columns
        train_final_df = train_df[['Image Index', 'Patient ID', 'label']]
        # Save to a new CSV
        train_output_path = os.path.join(output_dir, f"{pathology.lower()}_specialist_train.csv")
        train_final_df.to_csv(train_output_path, index=False)
        
        # --- Process Test Set ---
        test_df = test_master_df.copy()
        test_df['label'] = (test_df[pathology] == 1).astype(int)
        test_final_df = test_df[['Image Index', 'Patient ID', 'label']]
        test_output_path = os.path.join(output_dir, f"{pathology.lower()}_specialist_test.csv")
        test_final_df.to_csv(test_output_path, index=False)

        # Print statistics for verification
        train_dist = train_final_df['label'].value_counts(normalize=True)[1]
        test_dist = test_final_df['label'].value_counts(normalize=True)[1]
        print(f"  Train set prevalence for {pathology}: {train_dist:.2%}")
        print(f"  Test set prevalence for {pathology}: {test_dist:.2%}")
        print(f"  Saved to '{os.path.basename(train_output_path)}' and '{os.path.basename(test_output_path)}'\n")

    print("--- 🎉 All 10 specialist datasets created successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create one-vs-all datasets for specialist models."
    )
    parser.add_argument(
        "--train_csv_path",
        type=str,
        default="specialist_train_set.csv",
        help="Path to the master abnormal-only training CSV."
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        default="specialist_test_set.csv",
        help="Path to the master abnormal-only test CSV."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="specialist_datasets",
        help="Directory to save the new one-vs-all CSV files."
    )
    args = parser.parse_args()
    create_one_vs_all_datasets(args)