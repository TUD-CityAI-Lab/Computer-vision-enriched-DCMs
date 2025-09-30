import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cvdcm import cvdcm_model
import numpy as np
import pandas as pd
import os
from PIL import Image
from pathlib import Path
from datetime import datetime
from ImageChoicedata_preprocessing import ImageChoiceDataset
from transformers import AutoImageProcessor

# Image preprocessor
image_processor = AutoImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')

# Paths
repo_root = Path(os.getcwd())
data_dir = repo_root / "data"
img_path = data_dir / "images"
path_pretrained_model = repo_root /'CVDCM' /'pretrained_models' / 'Model3_CVDCM_CE0585_retrained.pt' # Pretrained model path
print(path_pretrained_model)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Load
model = cvdcm_model(path_pretrained_model).to(device)

# List of images in the image folder
img_lst = []
for file in os.listdir(img_path):
        if file.lower().endswith("jpg") or file.lower().endswith("png") or file.lower().endswith("jpeg"):
            img_lst.append(file)
print('Number of images =', len(img_lst))

# Add image path df
df = pd.DataFrame()
df['img'] = img_lst

# Calculate V for each image
V_img = []
for i, imgx in enumerate(img_lst):
    try:
        # Print progress
        print(f"Processing image {i+1}/{len(img_lst)}", end="\r")

        # Load image
        img = Image.open(os.path.join(img_path, imgx)).convert('RGB')  # Always convert to RGB mode

        # Preprocess image
        img = image_processor(img, return_tensors="pt").pixel_values[0].to(device)

        # Add batch dimension
        img = img.unsqueeze(0)

        # Forward pass
        V_img.append(model.return_image_utility(img).cpu().detach().numpy()[0][0])

    except Exception as e:
        print(f"\nError processing image {imgx}: {str(e)}")
        continue

# Add to df
df['V'] = V_img
df.to_csv(repo_root / 'image_utilities.csv', index=False)
print(f'\nImage utilities saved to {repo_root / "image_utilities.csv"}')