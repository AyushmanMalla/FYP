# File: patch_consensus_csv.py

#once the metadataset_creator jobs are done, run this script to fix the issues of the true score labels

import pandas as pd
import os
import argparse

def patch_file(consensus_path, truth_path, pathologies):
    """
    Fixes the ground truth 'True_*' columns in a consensus dataset file
    by merging it with the correct ground truth file.
    """
    print(f"---  patching {os.path.basename(consensus_path)} ---")
    
    try:
        consensus_df = pd.read_csv(consensus_path)
        truth_df = pd.read_csv(truth_path)
        print("✅ Successfully loaded both files.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find file. {e}")
        return

    # ## MODIFICATION: Create one-hot encoded labels in the truth_df on-the-fly ##
    print("Generating one-hot encoded ground truth labels...")
    for pathology in pathologies:
        capitalized_pathology = pathology.capitalize()
        truth_df[capitalized_pathology] = truth_df['Finding Labels'].apply(
            lambda x: 1 if capitalized_pathology in str(x) else 0
        )

    # To merge, extract the filename from the full ImagePath in the consensus_df
    consensus_df['Image Index'] = consensus_df['ImagePath'].apply(os.path.basename)

    # Merge the two dataframes.
    truth_cols_to_merge = ['Image Index'] + [p.capitalize() for p in pathologies]
    merged_df = pd.merge(
        consensus_df,
        truth_df[truth_cols_to_merge],
        on='Image Index',
        how='left'
    )

    # Now, update the incorrect 'True_*' columns with the correct data
    print("Updating ground truth labels in the consensus file...")
    for p in pathologies:
        true_col_name = f"True_{p}"
        source_col_name = p.capitalize()
        
        # Update the column in the original consensus_df using the merged data
        consensus_df[true_col_name] = merged_df[source_col_name]

    # Drop the temporary 'Image Index' column we added
    consensus_df = consensus_df.drop(columns=['Image Index'])
    
    # Fill any potential NaN values that might result from a failed merge with 0
    consensus_df.fillna(0, inplace=True)

    # Save the corrected file, overwriting the old one
    consensus_df.to_csv(consensus_path, index=False)
    
    # Verification
    print("Verification: Checking a few corrected labels for 'Infiltration'...")
    # Find rows where the original Finding Labels contained Infiltration
    infiltration_check = truth_df[truth_df['Infiltration'] == 1]['Image Index']
    # Check the corresponding rows in our patched dataframe
    print(consensus_df[consensus_df['ImagePath'].str.contains(infiltration_check.iloc[0])][['ImagePath', 'True_infiltration']])
    
    print(f"✅ Successfully patched and saved file to {consensus_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix the ground truth labels in the consensus datasets.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to the base scratch directory.")
    args = parser.parse_args()

    PATHOLOGIES = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'pneumonia']

    # --- Define file paths ---
    consensus_train_path = os.path.join(args.base_dir, "consensus_train_set.csv")
    truth_train_path = os.path.join(args.base_dir, "triage_train_set.csv")
    
    consensus_test_path = os.path.join(args.base_dir, "consensus_test_set.csv")
    truth_test_path = os.path.join(args.base_dir, "triage_test_set.csv")

    # --- Run the patching process ---
    patch_file(consensus_train_path, truth_train_path, PATHOLOGIES)
    patch_file(consensus_test_path, truth_test_path, PATHOLOGIES)

    print("--- 🎉 All files have been patched successfully! ---")
