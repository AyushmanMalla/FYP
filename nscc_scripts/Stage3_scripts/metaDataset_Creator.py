# File: create_consensus_dataset.py

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

# --- 1. Re-used Components from Previous Scripts ---

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
    if pathology in ['pneumonia', 'atelectasis']:
        # For these, we increase contrast and slightly increase brightness (gamma < 1.0)
        return transforms.Compose([
            ApplyCLAHE(clip_limit=3.0),
            GammaCorrection(gamma=0.8),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pathology == 'infiltration':
        # For this, we increase contrast and slightly decrease brightness (gamma > 1.0)
        return transforms.Compose([
            ApplyCLAHE(clip_limit=3.0),
            GammaCorrection(gamma=1.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pathology in ['cardiomegaly', 'effusion']:
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


# --- 2. New Helper Function to Create the Triage Feature Extractor ---

def get_triage_feature_extractor(model_path, device):
    """Loads the trained Triage ResNet-50 and converts it into a feature extractor."""
    model = models.resnet50() # Define the base architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Adapt it to match the saved weights
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Replace the final classification layer with an Identity layer
    # Now, the model's output will be the 2048-element feature vector
    model.fc = nn.Identity()
    
    model = model.to(device)
    model.eval()
    return model

# --- 3. Main Data Generation Logic ---

def generate_meta_dataset(args, input_csv_path, output_csv_path):
    """
    Processes a dataset to generate features and scores for the consensus model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Generating meta-dataset for {os.path.basename(input_csv_path)} on {device} ---")

    # --- Load all necessary models ---
    print("Loading all trained models...")
    triage_feature_extractor = get_triage_feature_extractor(args.triage_model_path, device)
    
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

    df = pd.read_csv(input_csv_path)
    
    # This list will store the data for our new CSV
    meta_data = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(args.image_dir, row['Image Index'])
            
            try:
                raw_image = Image.open(image_path).convert('RGB')
                
                # 1. Get Triage Features
                triage_input = triage_transform(raw_image).unsqueeze(0).to(device)
                triage_features = triage_feature_extractor(triage_input).cpu().numpy().flatten()
                
                # 2. Get Specialist Scores
                specialist_scores = {}
                for pathology in args.pathologies:
                    transform = specialist_transforms[pathology]
                    model = specialists[pathology]
                    
                    specialist_input = transform(raw_image).unsqueeze(0).to(device)
                    logit = model(specialist_input)
                    prob = torch.sigmoid(logit).item()
                    specialist_scores[f"Score_{pathology}"] = prob
                
                # 3. Collect Ground Truth Labels
                ground_truth = {f"True_{p}": row.get(p, 0) for p in args.pathologies}

                # 4. Combine everything into one record
                record = {
                    "ImagePath": image_path,
                    **ground_truth,
                    **specialist_scores
                }
                # Add triage features separately to avoid huge column names
                for i, feat in enumerate(triage_features):
                    record[f"TriageFeat_{i}"] = feat

                meta_data.append(record)

            except Exception as e:
                print(f"Skipping image {row['Image Index']} due to error: {e}")

    # --- Save the final meta-dataset ---
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Meta-dataset with {len(meta_df)} samples saved to {output_csv_path}")

if __name__ == "__main__":
    # Define pathologies and architectures in the order they were trained
    PATHOLOGIES = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'pneumonia']
    ARCHITECTURES = ['resnet50', 'resnet50', 'resnet50', 'densenet121', 'densenet121']

    parser = argparse.ArgumentParser(description="Generate meta-dataset for the consensus model.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to base scratch directory.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory with all images.")
    parser.add_argument("--triage_model_path", type=str, required=True, help="Path to the trained triage model.")

    args = parser.parse_args()
    args.pathologies = PATHOLOGIES
    args.architectures = ARCHITECTURES
    
    # Generate the training set for the consensus model
    generate_meta_dataset(args,
        input_csv_path=os.path.join(args.base_dir, "triage_train_set.csv"),
        output_csv_path=os.path.join(args.base_dir, "consensus_train_set.csv")
    )
    
    # Generate the test set for the consensus model
    generate_meta_dataset(args,
        input_csv_path=os.path.join(args.base_dir, "triage_test_set.csv"),
        output_csv_path=os.path.join(args.base_dir, "consensus_test_set.csv")
    )