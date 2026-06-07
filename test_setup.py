
import sys
from pathlib import Path

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'jupyter': 'Jupyter'
    }
    
    print('Vérification des dépendances...\n')
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f'✓ {name}')
        except ImportError:
            print(f'✗ {name} - NON INSTALLÉ')
            all_ok = False
    
    return all_ok


def check_directories():
    """Vérifie la structure des dossiers du projet."""
    required_dirs = [
        'embeddings',
        'results',
        'notebooks',
        'clothing-dataset-small-master'
    ]
    
    print('\nVérification de la structure des dossiers...\n')
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f'✓ {dir_name}/')
        else:
            print(f'✗ {dir_name}/ - NON TROUVÉ')
            all_ok = False
    
    return all_ok


def check_dataset():
    """Vérifie la présence et la structure du dataset."""
    dataset_path = Path('clothing-dataset-small-master')
    
    if not dataset_path.exists():
        print('\n✗ Dataset non trouvé!')
        return False
    
    print('\nVérification du dataset...\n')
    
    splits = ['train', 'validation', 'test']
    all_ok = True
    
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            categories = [d for d in split_path.iterdir() if d.is_dir()]
            images = []
            for cat in categories:
                images.extend(cat.glob('*.jpg'))
                images.extend(cat.glob('*.jpeg'))
                images.extend(cat.glob('*.png'))
            
            print(f'✓ {split}: {len(categories)} catégories, {len(images)} images')
        else:
            print(f'✗ {split}: NON TROUVÉ')
            all_ok = False
    
    return all_ok


def main():
    """Exécute tous les tests."""
    print('=' * 60)
    print('TEST DE CONFIGURATION DU MOTEUR DE RECHERCHE VISUELLE')
    print('=' * 60)
    
    deps_ok = check_dependencies()
    dirs_ok = check_directories()
    dataset_ok = check_dataset()
    
    print('\n' + '=' * 60)
    if deps_ok and dirs_ok and dataset_ok:
        print('✓ CONFIGURATION OK - Prêt à lancer les notebooks!')
        print('=' * 60)
        return 0
    else:
        print('✗ CONFIGURATION INCOMPLÈTE')
        print('=' * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
