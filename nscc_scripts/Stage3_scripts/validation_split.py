import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

def simple_split(args):
    """
    Simply splits a CSV file into train and validation sets.
    """
    print("\n" + "="*50)
    print("🚀 Creating Simple Train/Validation Split")
    print("="*50)

    # 1. Load the CSV file
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {args.input_csv} with {len(df)} rows.")

    # 2. Split into train and validation
    train_df, val_df = train_test_split(
        df,
        test_size=args.validation_size,
        random_state=42,
        shuffle=True
    )

    # 3. Save the splits
    train_path = os.path.join(args.output_dir, "train_split.csv")
    val_path = os.path.join(args.output_dir, "val_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\n✅ Done!")
    print(f"  - Train set: {train_path} ({len(train_df)} rows)")
    print(f"  - Validation set: {val_path} ({len(val_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple train/validation split for a CSV file.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the directory containing consensus_train_set.csv."
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Proportion of the data to use for validation (e.g., 0.2 for 20%)."
    )

    args = parser.parse_args()

    args.input_csv = os.path.join(args.base_dir, "consensus_train_set.csv")
    args.output_dir = args.base_dir

    simple_split(args)
