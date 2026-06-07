# Moteur de Recherche Visuelle - Vêtements

Un moteur de recherche d'images basé sur les embeddings profonds pour trouver des vêtements similaires.

## 🎯 Objectifs

- **BF01** : Service d'encodage acceptant JPEG/PNG, produisant des embeddings de dimension fixe
- **BF02** : Génération et stockage des embeddings pour l'ensemble du dataset (~5,000 images)
- **BF03** : Système de requête avec k-NN configurable (défaut k=5), similarité cosinus
- **BF04** : Visualisation des résultats (image requête + top-k avec scores)

## 📊 Architecture

### Phase 1 : Encodage (`01_encodage.ipynb`)
- Chargement du dataset CODAIT (10 catégories de vêtements)
- Encodage avec modèles pré-entraînés : **ResNet-18** et **MobileNetV2** (ImageNet)
- Génération des embeddings en batch (avec barre de progression)
- Stockage en format NumPy (`.npy`) + métadonnées JSON

### Phase 2 : Requête & Visualisation (`02_requete.ipynb`)
- Chargement des embeddings pré-calculés
- Encodage d'une image requête
- Calcul de similarité cosinus (scikit-learn)
- Recherche k-NN → top-k résultats avec scores
- Visualisation matplotlib

## 🛠️ Tech Stack

| Composant | Outil |
|-----------|------|
| Modèles | PyTorch + torchvision |
| Vecteurs | NumPy |
| Recherche | Scikit-learn (cosine_similarity) |
| Visualisation | Matplotlib + Pillow |
| Dev | Jupyter Notebook |

## 📁 Structure du Projet

```
2-Moteur-recherche-visuelle/
├── clothing-dataset-small-master/     # Dataset CODAIT
│   ├── train/                         # 10 catégories
│   ├── validation/
│   └── test/
├── embeddings/                        # Vecteurs générés
│   ├── resnet18_embeddings.npy
│   ├── mobilenetv2_embeddings.npy
│   └── metadata.json
├── results/                           # Résultats de requêtes (images)
├── notebooks/
│   ├── 01_encodage.ipynb              # Phase 1 : encoding pipeline
│   └── 02_requete.ipynb               # Phase 2 : query interface
├── requirements.txt                   # Dépendances
└── README.md                          # Ce fichier
```

## 🚀 Installation & Utilisation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Générer les embeddings (Phase 1)
```bash
jupyter notebook notebooks/01_encodage.ipynb
```
- Parcourt le dataset
- Génère embeddings avec ResNet-18 et MobileNetV2
- Sauvegarde en `.npy` + métadonnées

### 3. Rechercher des images similaires (Phase 2)
```bash
jupyter notebook notebooks/02_requete.ipynb
```
- Charge une image requête
- Encode et cherche les k images les plus similaires
- Affiche résultats avec scores de similarité

## 📈 Critères de Performance

- ✅ **Déterminisme** : Même image → même embedding
- ✅ **Vitesse** : Recherche < 5 secondes sur ~5,000 images (k=5)
- ✅ **Modularité** : Facile d'ajouter de nouveaux modèles d'encodage
- ✅ **Support GPU** : Optionnel (CPU par défaut)

## 📝 Dataset

**CODAIT Clothing Dataset (Small)**
- 10 catégories : dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt
- Splits : train, validation, test
- Format : JPEG/PNG

## 🔄 Comparaison des Modèles

| Modèle | Params | Dimension Embedding | Vitesse | Précision |
|--------|--------|-------------------|---------|-----------|
| ResNet-18 | ~11M | 512 | Moyenne | Haute |
| MobileNetV2 | ~3.5M | 1280 | Rapide | Bonne |

## 📚 Ressources

- [PyTorch Models](https://pytorch.org/vision/stable/models.html)
- [CODAIT Clothing Dataset](https://github.com/CODAIT/clothing_dataset)
- [Scikit-learn k-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
