# File: create_consensus_dataset_fast.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import csv # Import the standard csv library

# --- 1. Re-used Components from Previous Scripts ---
# (ApplyCLAHE, GammaCorrection, get_pathology_transforms, get_specialist_model)
# ... (These classes and functions are unchanged) ...

class ApplyCLAHE:
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_gray = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

class GammaCorrection:
    """Applies Gamma Correction to adjust brightness non-linearly."""
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.invGamma = 1.0 / self.gamma
        self.table = np.array([((i / 255.0) ** self.invGamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, img):
        img_np = np.array(img)
        # Apply the gamma correction using the lookup table
        return Image.fromarray(cv2.LUT(img_np, self.table))

def get_pathology_transforms(pathology):
    """Returns a deterministic, pathology-specific transform pipeline."""
    if pathology in ['Pneumonia', 'Atelectasis']:
        # For these, we increase contrast and slightly increase brightness (gamma < 1.0)
        return transforms.Compose([
            ApplyCLAHE(clip_limit=3.0),
            GammaCorrection(gamma=0.8),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pathology == 'Infiltration':
        # For this, we increase contrast and slightly decrease brightness (gamma > 1.0)
        return transforms.Compose([
            ApplyCLAHE(clip_limit=3.0),
            GammaCorrection(gamma=1.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pathology in ['Cardiomegaly', 'Effusion']:
        # For shape-based pathologies, we do no enhancement
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown pathology for transform selection: {pathology}")


def get_specialist_model(architecture_name='resnet50'):
    """Loads the specified pre-trained architecture and adapts it for binary classification."""
    print(f"Loading {architecture_name} model...")
    if architecture_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif architecture_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
    elif architecture_name == 'vit_base_patch16_224':
        model = models.vit_base_patch16_224(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 1)
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    # Fine-tune the whole model
    for param in model.parameters():
        param.requires_grad = True
    return model

# --- 2. MODIFIED Helper Function to Create the Triage Feature Extractor ---

def get_triage_feature_extractor(model_path, device):
    """
    Loads the trained Triage ResNet-50, converts it into a feature extractor,
    and returns the model AND the number of output features.
    """
    model = models.resnet50() # Define the base architecture
    num_ftrs = model.fc.in_features # Get feature dimension (e.g., 2048)
    model.fc = nn.Linear(num_ftrs, 1) # Adapt it to match the saved weights
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Replace the final classification layer with an Identity layer
    # Now, the model's output will be the 2048-element feature vector
    model.fc = nn.Identity()
    
    model = model.to(device)
    model.eval()
    # Return the model and the feature dimension
    return model, num_ftrs

# --- 3. NEW Dataset Class for Parallel Loading ---

class MetaGenDataset(Dataset):
    """
    A custom PyTorch Dataset to load and transform images for meta-dataset generation.
    All I/O and CPU-heavy transforms happen here, in parallel workers.
    """
    def __init__(self, csv_path, image_dir, triage_transform, specialist_transforms, pathologies):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.triage_transform = triage_transform
        self.specialist_transforms = specialist_transforms # This is a dict
        self.pathologies = pathologies

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path_short = row['Image Index']
        image_path_full = os.path.join(self.image_dir, image_path_short)
        
        try:
            # 1. Load the single raw image
            raw_image = Image.open(image_path_full).convert('RGB')
            
            # 2. Apply all necessary transforms
            triage_input = self.triage_transform(raw_image)
            
            specialist_inputs = {
                p: self.specialist_transforms[p](raw_image) 
                for p in self.pathologies
            }
            
            # 3. Get ground truth labels
            ground_truth = {f"True_{p}": row.get(p, 0) for p in self.pathologies}

            return {
                "image_path": image_path_full,
                "triage_input": triage_input,
                "specialist_inputs": specialist_inputs,
                "ground_truth": ground_truth,
                "valid": True # Flag for success
            }

        except Exception as e:
            print(f"Skipping image {image_path_short} due to error: {e}")
            # Return a flag indicating failure
            return {"valid": False}

def collate_fn_skip_errors(batch):
    """
    A custom collate_fn that filters out samples that failed to load
    and then stacks the rest into a batch.
    """
    # Filter out invalid samples (where "valid" is False)
    batch = [item for item in batch if item["valid"]]
    
    # If the whole batch failed, return None
    if not batch:
        return None
    
    # Manually collate specialist inputs (since it's a dict of tensors)
    specialist_inputs_dict = {p: [] for p in batch[0]["specialist_inputs"].keys()}
    for item in batch:
        for p, tensor in item["specialist_inputs"].items():
            specialist_inputs_dict[p].append(tensor)
            
    collated_specialist_inputs = {p: torch.stack(tensors) for p, tensors in specialist_inputs_dict.items()}

    # Collate the rest of the data
    return {
        "image_path": [item["image_path"] for item in batch],
        "triage_input": torch.stack([item["triage_input"] for item in batch]),
        "ground_truth": {k: torch.tensor([item["ground_truth"][k] for item in batch]) for k in batch[0]["ground_truth"].keys()},
        "specialist_inputs": collated_specialist_inputs
    }


# --- 4. REFACTORED Main Data Generation Logic ---

def generate_meta_dataset(args, input_csv_path, output_csv_path):
    """
    Processes a dataset in parallel batches to generate features and scores.
    Streams results directly to a CSV file to save memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Generating meta-dataset for {os.path.basename(input_csv_path)} on {device} ---")

    # --- Load all necessary models ---
    print("Loading all trained models...")
    # Get the feature extractor AND the number of features
    triage_feature_extractor, num_triage_features = get_triage_feature_extractor(args.triage_model_path, device)
    
    specialists = {}
    for pathology, arch in zip(args.pathologies, args.architectures):
        model_path = os.path.join(args.base_dir, f"specialist_{pathology}_{arch}_preprocessed.pth")
        model = get_specialist_model(arch)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        specialists[pathology] = model
    print("✅ All models loaded.")

    # --- Define necessary transforms ---
    triage_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    specialist_transforms = {p: get_pathology_transforms(p) for p in args.pathologies}

    # --- Create Dataset and DataLoader ---
    dataset = MetaGenDataset(
        csv_path=input_csv_path,
        image_dir=args.image_dir,
        triage_transform=triage_transform,
        specialist_transforms=specialist_transforms,
        pathologies=args.pathologies
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,  # Speeds up CPU-to-GPU data transfer
        collate_fn=collate_fn_skip_errors # Use our custom collate_fn
    )

    print(f"Starting data generation with batch_size={args.batch_size} and num_workers={args.num_workers}...")

    # --- Process data and stream to CSV ---
    total_samples = 0
    # Open the output CSV file *once*
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Define and write the header
        header = ["ImagePath"]
        header.extend([f"True_{p}" for p in args.pathologies])
        header.extend([f"Score_{p}" for p in args.pathologies])
        header.extend([f"TriageFeat_{i}" for i in range(num_triage_features)])
        writer.writerow(header)
        
        with torch.no_grad():
            # Loop over batches from the DataLoader
            for batch in tqdm(dataloader, total=len(dataloader)):
                
                # Skip any batches where all images failed to load
                if batch is None:
                    continue
                
                # --- Get data from the batch ---
                # Data is already transformed and batched by the DataLoader
                image_paths = batch["image_path"]
                triage_input_batch = batch["triage_input"].to(device, non_blocking=True)
                specialist_inputs_batch = {p: t.to(device, non_blocking=True) for p, t in batch["specialist_inputs"].items()}
                ground_truth_batch = batch["ground_truth"] # Stays on CPU
                
                current_batch_size = len(image_paths)

                # 1. Get Triage Features (in a batch)
                triage_features = triage_feature_extractor(triage_input_batch).cpu().numpy()
                
                # 2. Get Specialist Scores (in a batch)
                specialist_scores = {}
                for pathology in args.pathologies:
                    model = specialists[pathology]
                    s_input = specialist_inputs_batch[pathology]
                    
                    logit = model(s_input)
                    probs = torch.sigmoid(logit).cpu().numpy().flatten()
                    specialist_scores[f"Score_{pathology}"] = probs
                
                # 3. Write all rows in this batch to the CSV
                for i in range(current_batch_size):
                    record = [
                        image_paths[i],
                    ]
                    
                    # Add ground truth
                    record.extend([ground_truth_batch[f"True_{p}"][i].item() for p in args.pathologies])
                    
                    # Add specialist scores
                    record.extend([specialist_scores[f"Score_{p}"][i] for p in args.pathologies])
                    
                    # Add triage features
                    record.extend(triage_features[i])
                    
                    # Write the single row
                    writer.writerow(record)
                
                total_samples += current_batch_size

    print(f"\n✅ Meta-dataset with {total_samples} samples saved to {output_csv_path}")

if __name__ == "__main__":
    # Define pathologies and architectures in the order they were trained
    PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Pneumonia']
    ARCHITECTURES = ['resnet50', 'resnet50', 'resnet50', 'densenet121', 'densenet121']

    parser = argparse.ArgumentParser(description="Generate meta-dataset for the consensus model.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to base scratch directory.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory with all images.")
    parser.add_argument("--triage_model_path", type=str, required=True, help="Path to the trained triage model.")
    
    # --- NEW: Add arguments for performance tuning ---
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for GPU inference.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of CPU workers for data loading. Match this to your NCPUs.")

    args = parser.parse_args()
    args.pathologies = PATHOLOGIES
    args.architectures = ARCHITECTURES
    
    # Safety check: don't request more workers than available CPUs
    max_cpus = os.cpu_count()
    if args.num_workers > max_cpus:
        print(f"Warning: Requested {args.num_workers} workers, but only {max_cpus} are available.Using {max_cpus}.")
        args.num_workers = max_cpus
    
    # Generate the training set for the consensus model
    generate_meta_dataset(args,
        input_csv_path=os.path.join(args.base_dir, "triage_train_set.csv"),
        output_csv_path=os.path.join(args.base_dir, "consensus_train_set.csv")
    )
    
    # Generate the validation set for the consensus model
    generate_meta_dataset(args,
        input_csv_path=os.path.join(args.base_dir, "triage_val_set.csv"),
        output_csv_path=os.path.join(args.base_dir, "consensus_val_set.csv")
    )
    
    # Generate the test set for the consensus model
    generate_meta_dataset(args,
        input_csv_path=os.path.join(args.base_dir, "triage_test_set.csv"),
        output_csv_path=os.path.join(args.base_dir, "consensus_test_set.csv")
    )