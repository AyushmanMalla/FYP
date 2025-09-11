import os
import argparse
import pandas as pd
import torch
import torchxrayvision as xrv
import torchvision
import skimage.io
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ChestXRayInferenceDataset(Dataset):
    """Dataset for loading chest X-ray images for inference."""
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_filename = row["Image Index"]
        img_path = os.path.join(self.image_dir, image_filename)

        try:
            img = skimage.io.imread(img_path)
            img = xrv.datasets.normalize(img, 255)
            img = img[None, :, :]  # Add channel dimension

            if self.transform:
                img = self.transform(img)

            img_tensor = torch.from_numpy(img)
            return image_filename, img_tensor
        except Exception as e:
            print(f"Error loading image {image_filename}: {e}")
            return None, None

def collate_fn(batch):
    """Custom collate function to filter out None values from failed image loads."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

def run_benchmark(args):
    """
    Main function to run the inference benchmark.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load Model ---
    print(f"Loading model: {args.model_weights}")
    model = xrv.models.DenseNet(weights=args.model_weights)
    model = model.to(DEVICE)
    model.eval()
    disease_labels = model.pathologies

    # --- Setup Data ---
    print(f"Loading test set from: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    dataset = ChestXRayInferenceDataset(df, args.image_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn # Use custom collate function
    )

    # --- Run Inference ---
    print(f"Starting inference on {len(dataset)} images...")
    results = []
    with torch.no_grad():
        for filenames, image_batch in tqdm(dataloader):
            if filenames is None: continue # Skip empty batches
            
            image_batch = image_batch.to(DEVICE)
            preds = model(image_batch)
            preds_np = preds.cpu().numpy()

            for i, filename in enumerate(filenames):
                pred_dict = {"image_filename": filename}
                pred_dict.update({disease_labels[j]: preds_np[i, j] for j in range(len(disease_labels))})
                results.append(pred_dict)

    # --- Save Results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n--- ✅ Inference complete! ---")
    print(f"Results for {len(results_df)} images saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a chest X-ray dataset using a torchxrayvision model."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the output CSV file with model predictions."
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Name of the torchxrayvision model weights to use (e.g., 'densenet121-res224-all')."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to the test set CSV file containing image filenames."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing the test images."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to process in each batch."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel data loading."
    )

    args = parser.parse_args()
    run_benchmark(args)