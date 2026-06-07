#!/usr/bin/env python3
"""
Execute embedding generation for the visual search engine project.
This script replicates the logic from notebooks/01_encodage.ipynb
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import time
import sys

# Configuration
DATASET_ROOT = Path('./clothing-dataset-small-master')
EMBEDDINGS_DIR = Path('./embeddings')
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("=" * 60)
print("PHASE 1: IMAGE ENCODING - VISUAL SEARCH ENGINE")
print("=" * 60)

# Step 1: Explore the Dataset
print("\n[1/5] Exploring the dataset...")
image_paths = []
image_labels = []

train_dir = DATASET_ROOT / 'train'

for category_dir in sorted(train_dir.iterdir()):
    if category_dir.is_dir():
        category = category_dir.name
        category_images = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.jpeg')) + list(category_dir.glob('*.png'))
        print(f'  {category}: {len(category_images)} images')
        
        for img_path in category_images:
            image_paths.append(str(img_path))
            image_labels.append(category)

print(f'\nTotal images found: {len(image_paths)}')

# Step 2: Define the Encoder Class
print("\n[2/5] Defining ImageEncoder class...")

class ImageEncoder:
    def __init__(self, model_name='resnet18', device='cpu'):
        self.device = device
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.embedding_dim = self._get_embedding_dim()
        
    def _load_model(self, model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'mobilenetv2':
            model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f'Unknown model: {model_name}')
        
        # Remove classification layer, keep features
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        model.eval()
        return model
    
    def _get_embedding_dim(self):
        # Create dummy input to infer embedding dimension
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[1]
    
    def encode_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
            
            # Flatten and normalize
            embedding = embedding.view(embedding.size(0), -1)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f'Error encoding {image_path}: {e}')
            return None
    
    def encode_batch(self, image_paths, batch_size=32):
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f'Encoding with {self.model_name}'):
            batch_paths = image_paths[i:i+batch_size]
            batch_embeddings = []
            
            for path in batch_paths:
                emb = self.encode_image(path)
                if emb is not None:
                    batch_embeddings.append(emb)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

print("  ImageEncoder class defined ✓")

# Step 3: Generate Embeddings with ResNet-18
print("\n[3/5] Generating embeddings with ResNet-18...")
start_time = time.time()

encoder_resnet = ImageEncoder('resnet18', device=device)
print(f'  ResNet-18 embedding dimension: {encoder_resnet.embedding_dim}')

embeddings_resnet = encoder_resnet.encode_batch(image_paths, batch_size=32)
print(f'  Generated {embeddings_resnet.shape[0]} embeddings of dimension {embeddings_resnet.shape[1]}')

# Save embeddings
np.save(EMBEDDINGS_DIR / 'resnet18_embeddings.npy', embeddings_resnet)
print(f'  Saved: resnet18_embeddings.npy ({embeddings_resnet.nbytes / 1024 / 1024:.2f} MB)')

resnet_time = time.time() - start_time

# Step 4: Generate Embeddings with MobileNetV2
print("\n[4/5] Generating embeddings with MobileNetV2...")
start_time = time.time()

encoder_mobile = ImageEncoder('mobilenetv2', device=device)
print(f'  MobileNetV2 embedding dimension: {encoder_mobile.embedding_dim}')

embeddings_mobile = encoder_mobile.encode_batch(image_paths, batch_size=32)
print(f'  Generated {embeddings_mobile.shape[0]} embeddings of dimension {embeddings_mobile.shape[1]}')

# Save embeddings
np.save(EMBEDDINGS_DIR / 'mobilenetv2_embeddings.npy', embeddings_mobile)
print(f'  Saved: mobilenetv2_embeddings.npy ({embeddings_mobile.nbytes / 1024 / 1024:.2f} MB)')

mobile_time = time.time() - start_time

# Step 5: Save Metadata
print("\n[5/5] Saving metadata...")

# Create metadata file
metadata = {
    'total_images': len(image_paths),
    'image_paths': image_paths,
    'image_labels': image_labels,
    'models': {
        'resnet18': {
            'embedding_dim': int(encoder_resnet.embedding_dim),
            'embeddings_file': 'resnet18_embeddings.npy',
            'num_embeddings': int(embeddings_resnet.shape[0])
        },
        'mobilenetv2': {
            'embedding_dim': int(encoder_mobile.embedding_dim),
            'embeddings_file': 'mobilenetv2_embeddings.npy',
            'num_embeddings': int(embeddings_mobile.shape[0])
        }
    }
}

with open(EMBEDDINGS_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('  Saved: metadata.json')

# Print Summary
print("\n" + "=" * 60)
print("✅ ENCODING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f'\nSummary:')
print(f'  Total images processed: {metadata["total_images"]}')
print(f'  Number of categories: {len(set(image_labels))}')
print(f'  ResNet-18 embeddings shape: {embeddings_resnet.shape}')
print(f'  ResNet-18 file size: {embeddings_resnet.nbytes / 1024 / 1024:.2f} MB')
print(f'  MobileNetV2 embeddings shape: {embeddings_mobile.shape}')
print(f'  MobileNetV2 file size: {embeddings_mobile.nbytes / 1024 / 1024:.2f} MB')
print(f'  ResNet-18 processing time: {resnet_time:.2f}s')
print(f'  MobileNetV2 processing time: {mobile_time:.2f}s')
print(f'\nOutput files:')
print(f'  - {EMBEDDINGS_DIR / "resnet18_embeddings.npy"}')
print(f'  - {EMBEDDINGS_DIR / "mobilenetv2_embeddings.npy"}')
print(f'  - {EMBEDDINGS_DIR / "metadata.json"}')
print(f'\nNext step: Run notebook 02_requete.ipynb to test similarity search')
print("=" * 60)

sys.exit(0)
