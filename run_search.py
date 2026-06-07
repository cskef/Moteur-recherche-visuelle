#!/usr/bin/env python
"""
Phase 2: Image Search Engine
Lance la recherche d'images similaires et génère les résultats visualisés.
Équivalent de 02_requete.ipynb en script standalone.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.models as models
from tqdm import tqdm

# Configuration des chemins
EMBEDDINGS_DIR = Path('embeddings')
RESULTS_DIR = Path('results')
DATASET_DIR = Path('clothing-dataset-small-master/train')

RESULTS_DIR.mkdir(exist_ok=True)

# Configuration device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✓ Device: {device}\n')

# Vérifier que les embeddings existent
if not (EMBEDDINGS_DIR / 'metadata.json').exists():
    print('✗ Erreur: metadata.json non trouvé!')
    print('  Lance d\'abord: python run_embeddings.py')
    exit(1)

if not (EMBEDDINGS_DIR / 'resnet18_embeddings.npy').exists():
    print('✗ Erreur: resnet18_embeddings.npy non trouvé!')
    print('  Lance d\'abord: python run_embeddings.py')
    exit(1)

print('✓ Embeddings trouvés\n')


# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ImageEncoder:
    """Encodeur d'images avec ResNet-18 pré-entraîné"""
    
    def __init__(self, model_name='resnet18', device='cpu'):
        self.device = device
        self.model_name = model_name
        self.model = self._load_model(model_name)
        
    def _load_model(self, model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'mobilenetv2':
            model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f'Model inconnu: {model_name}')
        
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        model.eval()
        return model
    
    def encode_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
            
            embedding = embedding.view(embedding.size(0), -1)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f'✗ Erreur: {image_path}: {e}')
            return None


class ImageSearchEngine:
    """Moteur de recherche d'images par similarité"""
    
    def __init__(self, embeddings, image_paths, encoder):
        self.embeddings = embeddings
        self.image_paths = image_paths
        self.encoder = encoder
    
    def search(self, query_image_path, k=5):
        # Encoder l'image requête
        query_embedding = self.encoder.encode_image(query_image_path)
        if query_embedding is None:
            return None
        
        # Calculer les similarités
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Top-k résultats
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        top_k_paths = [self.image_paths[i] for i in top_k_indices]
        
        return {
            'query_path': query_image_path,
            'results': list(zip(top_k_paths, top_k_scores, top_k_indices))
        }


def visualize_results(search_result, title='Résultats de Recherche', save_path=None):
    """Visualise les résultats de recherche"""
    query_path = search_result['query_path']
    results = search_result['results']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Image requête
    query_img = Image.open(query_path)
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title('Image Requête', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Masquer les autres cases de la première ligne
    for i in range(1, 3):
        axes[0, i].axis('off')
    
    # Afficher les résultats
    for idx, (img_path, score, _) in enumerate(results):
        row = (idx + 3) // 3
        col = (idx + 3) % 3
        
        if row < 2:
            try:
                result_img = Image.open(img_path)
                axes[row, col].imshow(result_img)
                axes[row, col].set_title(
                    f'Résultat {idx+1}\nSimilarité: {score:.3f}',
                    fontsize=11
                )
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Erreur: {e}',
                                   ha='center', va='center')
                axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f'✓ Visualisation sauvegardée: {save_path}')
    
    return fig


def main():
    print('=' * 70)
    print('PHASE 2: RECHERCHE D\'IMAGES SIMILAIRES')
    print('=' * 70 + '\n')
    
    # Charger les métadonnées
    print('📂 Chargement des métadonnées...')
    with open(EMBEDDINGS_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    image_paths = metadata['image_paths']
    image_labels = metadata['image_labels']
    print(f'✓ {len(image_paths)} images chargées\n')
    
    # Charger les embeddings
    print('📊 Chargement des embeddings (ResNet-18)...')
    embeddings = np.load(EMBEDDINGS_DIR / 'resnet18_embeddings.npy')
    print(f'✓ Shape: {embeddings.shape}\n')
    
    # Créer l'encodeur et le moteur de recherche
    print('🔧 Initialisation du moteur de recherche...')
    encoder = ImageEncoder('resnet18', device=device)
    search_engine = ImageSearchEngine(embeddings, image_paths, encoder)
    print('✓ Moteur prêt\n')
    
    # Tests de recherche
    print('=' * 70)
    print('TESTS DE RECHERCHE')
    print('=' * 70 + '\n')
    
    # Test 1: Image aléatoire du dataset
    print('📌 Test 1: Image aléatoire du dataset')
    test_idx = np.random.randint(0, len(image_paths))
    test_image_path = image_paths[test_idx]
    test_category = image_labels[test_idx]
    
    print(f'  Image: {Path(test_image_path).name}')
    print(f'  Catégorie: {test_category}')
    
    start_time = time.time()
    results = search_engine.search(test_image_path, k=5)
    search_time = time.time() - start_time
    
    print(f'  Temps de recherche: {search_time:.2f}s\n')
    
    if results:
        print('  Top 5 résultats similaires:')
        for i, (path, score, idx) in enumerate(results['results']):
            print(f'    {i+1}. {Path(path).name} '
                  f'(Similarité: {score:.4f}, Catégorie: {image_labels[idx]})')
        
        # Visualiser et sauvegarder
        fig = visualize_results(results, title=f'Recherche: {Path(test_image_path).name}')
        save_path = RESULTS_DIR / f'search_result_random_{test_idx}.png'
        visualize_results(results, title=f'Recherche: {Path(test_image_path).name}',
                         save_path=save_path)
    
    print()
    
    # Test 2: Images de différentes catégories
    print('📌 Test 2: Recherche par catégorie')
    categories = list(set(image_labels))[:3]  # 3 catégories aléatoires
    
    for category in categories:
        category_indices = [i for i, label in enumerate(image_labels) if label == category]
        if category_indices:
            query_idx = category_indices[0]
            query_path = image_paths[query_idx]
            
            print(f'\n  Catégorie: {category}')
            print(f'  Image requête: {Path(query_path).name}')
            
            start_time = time.time()
            results = search_engine.search(query_path, k=5)
            search_time = time.time() - start_time
            
            print(f'  Temps de recherche: {search_time:.2f}s')
            
            if results:
                # Compter les résultats de la même catégorie
                same_category_count = sum(
                    1 for _, _, idx in results['results'] 
                    if image_labels[idx] == category
                )
                print(f'  Résultats de la même catégorie: {same_category_count}/5')
                
                # Sauvegarder la visualisation
                save_path = RESULTS_DIR / f'search_result_{category}.png'
                visualize_results(results, title=f'Recherche: {category}',
                                 save_path=save_path)
    
    print()
    
    # Test 3: Performance de recherche
    print('📌 Test 3: Benchmark de performance')
    num_searches = 10
    times = []
    
    for _ in range(num_searches):
        test_idx = np.random.randint(0, len(image_paths))
        test_image_path = image_paths[test_idx]
        
        start_time = time.time()
        search_engine.search(test_image_path, k=5)
        times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f'  Nombre de recherches: {num_searches}')
    print(f'  Temps moyen: {avg_time:.3f}s')
    print(f'  Temps min: {min_time:.3f}s')
    print(f'  Temps max: {max_time:.3f}s')
    
    if avg_time < 5:
        print('  ✓ Performance: OK (< 5s)')
    else:
        print('  ⚠ Performance: À optimiser (> 5s)')
    
    print()
    
    # Résumé final
    print('=' * 70)
    print('RÉSUMÉ')
    print('=' * 70)
    print(f'✓ Images indexées: {len(image_paths)}')
    print(f'✓ Dimension embedding: {embeddings.shape[1]}')
    print(f'✓ Modèle: ResNet-18 (pré-entraîné ImageNet)')
    print(f'✓ Visualisations sauvegardées dans: {RESULTS_DIR}/')
    print(f'✓ Performance moyenne: {avg_time:.3f}s par recherche')
    print()


if __name__ == '__main__':
    try:
        main()
        print('✓ Recherche complétée avec succès!\n')
    except KeyboardInterrupt:
        print('\n⚠ Interrompu par l\'utilisateur')
    except Exception as e:
        print(f'\n✗ Erreur: {e}')
        import traceback
        traceback.print_exc()
