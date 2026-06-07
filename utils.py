
# Modules utilitaires pour le moteur de recherche visuelle


import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class ImageEncoder:

    # Encodeur d'images utilisant des modèles pré-entraînés.
    
    
    AVAILABLE_MODELS = ['resnet18', 'mobilenetv2']
    
    def __init__(self, model_name: str = 'resnet18', device: str = 'cpu'):

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f'Unknown model: {model_name}. Choose from {self.AVAILABLE_MODELS}')
        
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.embedding_dim = self._get_embedding_dim()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_name: str):
        """Charge le modèle pré-entraîné."""
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'mobilenetv2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        model.eval()
        return model
    
    def _get_embedding_dim(self) -> int:

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[1]
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:

        # Encode une image en vecteur d'embedding.
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
            
            embedding = embedding.view(embedding.size(0), -1)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f'Erreur lors du codage {image_path}: {e}')
            return None


class ImageSearchEngine:

    # Moteur de recherche d'images par similarité.
    
    
    
    def __init__(self, embeddings: np.ndarray, image_paths: List[str], 
                 encoder: ImageEncoder):
        """
        Args:
            embeddings: Matrice d'embeddings (N x D)
            image_paths: Liste des chemins des images
            encoder: Instance de ImageEncoder
        """
        self.embeddings = embeddings
        self.image_paths = image_paths
        self.encoder = encoder
        
        if len(embeddings) != len(image_paths):
            raise ValueError('embeddings et image_paths doivent avoir la même longueur')
    
    def search(self, query_image_path: str, k: int = 5) -> Optional[Dict]:

        # Recherche les k images les plus similaires.
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_embedding = self.encoder.encode_image(query_image_path)
        if query_embedding is None:
            return None
        
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        top_k_paths = [self.image_paths[i] for i in top_k_indices]
        
        return {
            'query_path': query_image_path,
            'results': list(zip(top_k_paths, top_k_scores, top_k_indices))
        }


def load_metadata(embeddings_dir: Path) -> Dict:
    """Charge les métadonnées depuis metadata.json."""
    with open(embeddings_dir / 'metadata.json', 'r') as f:
        return json.load(f)


def save_metadata(embeddings_dir: Path, metadata: Dict) -> None:
    """Sauvegarde les métadonnées dans metadata.json."""
    with open(embeddings_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
