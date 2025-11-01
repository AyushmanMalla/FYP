# File: generate_triage_preds.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import os
import argparse
from PIL import Image
from tqdm import tqdm

def get_triage_classifier(model_path, device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_triage_classifier(args.model_path, device)
    df_test = pd.read_csv(args.test_csv)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.4Deep06], std=[0.229, 0.224, 0.225])
    ])

    results = []
    with torch.no_grad():
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            image_path = os.path.join(args.image_dir, row['Image Index'])
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                logit = model(image_tensor)
                prob = torch.sigmoid(logit).item()
                results.append({'Image Index': row['Image Index'], 'triage_score': prob})
            except Exception as e:
                print(f"Error on {row['Image Index']}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Triage predictions saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Path to where all the models, csvs are")
    args = parser.parse_args()

    args.model_path = os.path.join(args.base_dir, "triage_model_best.pth")
    args.test_csv = os.path.join(args.base_dir, "triage_test_set.csv")
    args.image_dir = os.path.join(args.base_dir, "CXR_ALL_FLAT")
    args.output_csv = os.path.join(args.base_dir, "triage_predictions.csv")
    main(args)