import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import pandas as pd
import os
import argparse
from PIL import Image
from tqdm import tqdm
import warnings

# Suppress common UserWarning from torchvision about weights
warnings.filterwarnings("ignore", category=UserWarning)

def generate_predictions(args):
    """
    Runs a pre-trained torchxrayvision model on a test set and saves the
    predicted probabilities to a CSV file. Designed for HPC execution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"🚀 Initializing TorchXRayVision Benchmark on device: {device}")
    print("="*60)

    # --- 1. Load the pre-trained model ---
    # The weights for this model should be pre-downloaded by running a
    # simple script on the login node before submitting the job.
    print(f"Loading model with weights: '{args.model_weights}'...")
    model = xrv.models.DenseNet(weights=args.model_weights)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- 2. Define standard image transformations ---
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
        transforms.Grayscale(num_output_channels=1) # Model expects a single channel
    ])

    # --- 3. Load the test set DataFrame ---
    try:
        df_test = pd.read_csv(args.test_csv)
        print(f"Loaded test set: {os.path.basename(args.test_csv)} ({len(df_test)} images)")
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Test CSV not found at {args.test_csv}")
        return

    # --- 4. Run inference loop ---
    results = []
    pathologies = model.pathologies

    with torch.no_grad():
        for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Running Inference"):
            image_filename = row['Image Index']
            image_path = os.path.join(args.image_dir, image_filename)

            try:
                image = Image.open(image_path)
                image_tensor = transform(image).unsqueeze(0).to(device)
                outputs = model(image_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                result_row = {'image_filename': image_filename}
                result_row.update(dict(zip(pathologies, probs)))
                results.append(result_row)
            except Exception as e:
                print(f"⚠️ Warning: Could not process {image_filename}. Error: {e}. Skipping.")

    # --- 5. Save results to CSV ---
    if not results:
        print("❌ FATAL ERROR: No images were processed. Output file will not be created.")
        return
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Benchmark inference complete!")
    print(f"   Results for {len(results_df)} images saved to: {args.output_csv}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark predictions using a torchxrayvision model on an HPC cluster.")
    parser.add_argument("--test_csv", type=str, required=True, help="Full path to the test set CSV file (e.g., triage_test_set.csv).")
    parser.add_argument("--image_dir", type=str, required=True, help="Full path to the directory containing all CXR images.")
    parser.add_argument("--output_csv", type=str, default="torchxrayvision_results.csv", help="Name of the output CSV file to save predictions.")
    parser.add_argument("--model_weights", type=str, default="densenet121-res224-all", help="Name of the pre-trained weights to use from torchxrayvision.")
    
    args = parser.parse_args()
    generate_predictions(args)