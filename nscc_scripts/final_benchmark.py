# File: final_benchmark.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import os
import argparse
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import cv2

# --- 1. Re-used Components from Previous Scripts ---
# (These must be defined for the script to load and use the models correctly)

class ApplyCLAHE:
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
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.invGamma = 1.0 / self.gamma
        self.table = np.array([((i / 255.0) ** self.invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    def __call__(self, img):
        img_np = np.array(img)
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

class DynamicConsensusModel(nn.Module):
    def __init__(self, input_dim=2053, hidden_dim=256, output_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def get_triage_classifier(model_path, device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def get_triage_feature_extractor(model_path, device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.fc = nn.Identity() # Convert to feature extractor
    return model.to(device).eval()

# --- 2. Main Evaluation Logic ---

def run_end_to_end_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting End-to-End Evaluation on {device} ---")

    # --- Load all models ---
    print("Loading all models...")
    triage_classifier = get_triage_classifier(args.triage_model_path, device)
    triage_feature_extractor = get_triage_feature_extractor(args.triage_model_path, device)
    
    specialists = {}
    for pathology, arch in zip(args.pathologies, args.architectures):
        model_path = os.path.join(args.base_dir, f"specialist_{pathology}_{arch}_preprocessed.pth")
        model = get_specialist_model(arch)
        model.load_state_dict(torch.load(model_path, map_location=device))
        specialists[pathology] = model.to(device).eval()
        
    consensus_model = DynamicConsensusModel().to(device)
    consensus_model.load_state_dict(torch.load(args.consensus_model_path, map_location=device))
    consensus_model.eval()
    print("✅ All models loaded.")

    # --- Setup Data and Transforms ---
    df_test = pd.read_csv(args.test_csv)
    # Ensure ground truth columns exist
    for p in args.pathologies:
        cap_p = p.capitalize()
        if cap_p not in df_test.columns:
            df_test[cap_p] = df_test['Finding Labels'].apply(lambda x: 1 if cap_p in str(x) else 0)

    triage_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    specialist_transforms = {p: get_pathology_transforms(p) for p in args.pathologies}

    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            image_path = os.path.join(args.image_dir, row['Image Index'])
            
            # --- Get Ground Truth for this image ---
            true_labels = [row[p.capitalize()] for p in args.pathologies]
            all_true_labels.append(true_labels)
            
            try:
                raw_image = Image.open(image_path).convert('RGB')
                
                # --- Stage 1: Triage ---
                triage_input = triage_transform(raw_image).unsqueeze(0).to(device)
                triage_logit = triage_classifier(triage_input)
                triage_prob = torch.sigmoid(triage_logit).item()
                
                # --- Conditional Logic ---
                if triage_prob < 0.17:
                    # Triage predicts NORMAL, so final prediction is all zeros
                    final_probs = np.zeros(len(args.pathologies))
                else:
                    # Triage predicts ABNORMAL, proceed to full pipeline
                    triage_features = triage_feature_extractor(triage_input)
                    
                    specialist_scores_list = []
                    for pathology in args.pathologies:
                        transform = specialist_transforms[pathology]
                        model = specialists[pathology]
                        specialist_input = transform(raw_image).unsqueeze(0).to(device)
                        logit = model(specialist_input)
                        prob = torch.sigmoid(logit).item()
                        specialist_scores_list.append(prob)
                    
                    specialist_scores = torch.tensor([specialist_scores_list], dtype=torch.float32).to(device)
                    
                    # Run Consensus Model
                    fused_features = torch.cat((triage_features, specialist_scores), dim=1)
                    consensus_logits = consensus_model(fused_features)
                    final_probs = torch.sigmoid(consensus_logits).cpu().numpy().flatten()
                
                all_pred_probs.append(final_probs)

            except Exception as e:
                print(f"Error processing {row['Image Index']}: {e}. Appending zeros.")
                all_pred_probs.append(np.zeros(len(args.pathologies)))

    # --- Calculate Final Metrics ---
    # y_true = np.array(all_true_labels)
    # y_pred_probs = np.array(all_pred_probs)

    # results = []
    # for i, pathology in enumerate(args.pathologies):
    #     auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
    #     fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred_probs[:, i])
    #     optimal_idx = np.argmax(tpr - fpr)
    #     optimal_threshold = thresholds[optimal_idx]
    #     y_pred_binary = (y_pred_probs[:, i] >= optimal_threshold).astype(int)
        
    #     results.append({
    #         "Disease": pathology.capitalize(),
    #         "AUC-ROC": auc,
    #         "Threshold": optimal_threshold,
    #         "Accuracy": accuracy_score(y_true[:, i], y_pred_binary),
    #         "Precision": precision_score(y_true[:, i], y_pred_binary, zero_division=0),
    #         "Recall": recall_score(y_true[:, i], y_pred_binary, zero_division=0),
    #         "F1-Score": f1_score(y_true[:, i], y_pred_binary, zero_division=0)
    #     })

    y_true = np.array(all_true_labels)
    y_pred_probs = np.array(all_pred_probs)
    
    # These are your constants, determined from the validation set
    FIXED_THRESHOLDS = {
        'atelectasis': 0.1100,  
        'cardiomegaly': 0.0364, 
        'effusion': 0.1298,     
        'infiltration': 0.1310, 
        'pneumonia': 0.0222     
    }
    print("\nUsing FIXED thresholds determined from validation set:")
    print(FIXED_THRESHOLDS)

    results = []
    for i, pathology in enumerate(args.pathologies):
        # Use the pre-determined, fixed threshold for this pathology
        threshold = FIXED_THRESHOLDS[pathology]
        
        auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        
        # Binarize predictions using the FIXED threshold
        y_pred_binary = (y_pred_probs[:, i] >= threshold).astype(int)
        
        # Calculate all metrics based on this unbiased prediction
        results.append({
            "Disease": pathology.capitalize(),
            "AUC-ROC": auc,
            "Threshold_Used": threshold,
            "Accuracy": accuracy_score(y_true[:, i], y_pred_binary),
            "Precision": precision_score(y_true[:, i], y_pred_binary, zero_division=0),
            "Recall": recall_score(y_true[:, i], y_pred_binary, zero_division=0),
            "F1-Score": f1_score(y_true[:, i], y_pred_binary, zero_division=0)
        })

    # --- Display Results ---
    results_df = pd.DataFrame(results)
    mean_auc = results_df['AUC-ROC'].mean()
    
    print(f"\n🩺 Final End-to-End System Performance")
    print("="*90)
    print(results_df.to_string(index=False, float_format="{:.4f}.format"))
    print("-" * 90)
    print(f"Mean AUC-ROC across {len(results_df)} pathologies: {mean_auc:.4f}")
    print("="*90)
    results_df.to_csv("hehe_final.csv")

if __name__ == "__main__":
    # Define the final configuration of your trained models
    PATHOLOGIES = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'pneumonia']
    ARCHITECTURES = ['resnet50', 'resnet50', 'resnet50', 'densenet121', 'densenet121']

    parser = argparse.ArgumentParser(description="Run end-to-end evaluation of the hierarchical model.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to where all the models, csvs are")
    # Add necessary arguments
    # ...
    args = parser.parse_args()
    args.pathologies = PATHOLOGIES
    args.architectures = ARCHITECTURES
    args.triage_model_path = os.path.join(args.base_dir, "triage_model_best.pth")
    args.consensus_model_path = os.path.join(args.base_dir, "dynamic_consensus_model.pth")
    args.test_csv = os.path.join(args.base_dir, "triage_test_set.csv")
    args.image_dir = os.path.join(args.base_dir, "CXR_ALL_FLAT")

    run_end_to_end_evaluation(args)
