import os
import argparse
import pandas as pd
import torch
import torchxrayvision as xrv
import torchvision
import skimage.io
from tqdm import tqdm

def run_inference(args):
    """
    Loads a model and a test set, runs inference on each image one-by-one,
    and saves the raw prediction scores to a CSV file.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 1. Load Model ---
    print(f"Loading model: {args.model_weights}")
    model = xrv.models.DenseNet(weights=args.model_weights)
    model = model.to(DEVICE)
    model.eval()
    disease_labels = model.pathologies

    # --- 2. Setup Data and Transforms ---
    print(f"Loading test set from: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    # --- 3. Run Inference (Row-by-Row) ---
    print(f"Starting inference on {len(df)} images...")
    results = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_filename = row["Image Index"]
            img_path = os.path.join(args.image_dir, image_filename)

            try:
                # Load and preprocess the image
                img = skimage.io.imread(img_path)
                img = xrv.datasets.normalize(img, 255)
                img = img[None, :, :]  # Add channel dimension
                img = transform(img)
                img_tensor = torch.from_numpy(img)
                
                # Add batch dimension of 1 and move to device
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

                # Run inference
                preds = model(img_tensor)
                preds_np = preds.cpu().numpy().flatten()

                # Store results
                pred_dict = {"image_filename": image_filename}
                pred_dict.update({disease_labels[j]: preds_np[j] for j in range(len(disease_labels))})
                results.append(pred_dict)
            
            except Exception as e:
                print(f"Could not process image {image_filename}: {e}")

    # --- 4. Save Results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n--- ✅ Inference complete! ---")
    print(f"Results for {len(results_df)} images saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a chest X-ray test set and save the raw predictions."
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
        help="Path to the curated test set CSV file."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing all the test images."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path for the output CSV file where predictions will be saved."
    )

    args = parser.parse_args()
    run_inference(args)