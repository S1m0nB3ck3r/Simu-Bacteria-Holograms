# Architecture U-Net 3D pour Segmentation d'Hologrammes

## 📋 Vue d'ensemble

Ce document décrit l'architecture du réseau de neurones utilisé pour la segmentation volumétrique 3D d'hologrammes de bactéries, ainsi que les choix de design motivés par les contraintes spécifiques du problème.

---

## 🎯 Problème à résoudre

### Contexte
- **Entrée** : Hologrammes 2D (512×512 pixels) de bactéries en suspension
- **Objectif** : Segmenter les bactéries dans le volume 3D reconstruit (512×512×200 voxels)
- **Défi principal** : **Déséquilibre extrême des classes**
  - Volume total : ~52 millions de voxels
  - Objets : ~20 bactéries × 20 voxels = 400 voxels positifs
  - **Ratio positif : 0.00076%** (1 voxel sur 130,000)

### Défis spécifiques
1. **Bruit de diffraction** : Le volume reconstruit contient beaucoup de bruit de diffraction dans les régions "vides"
2. **Objets rares** : Les bactéries représentent une fraction infime du volume
3. **Séparation d'objets proches** : Le modèle doit gérer le cas où la diffraction de deux bactéries proches se chevauche
4. **Faux positifs critiques** : Le modèle doit rejeter le bruit tout en détectant avec certitude les vraies bactéries

---

## 🏗️ Architecture U-Net 3D

### Choix architectural : U-Net

**Pourquoi U-Net ?**
- ✅ **Skip connections** : Préservent les détails spatiaux fins (essentiel pour la localisation précise)
- ✅ **Architecture encoder-decoder** : Capture le contexte global et les détails locaux
- ✅ **Éprouvé** : Standard de facto pour la segmentation médicale et biomédicale
- ✅ **Adaptable au 3D** : Extension naturelle pour données volumétriques

### Structure générale

```
Input (1, 128, 128, 64)
    ↓
┌─────────────────────────────────────────────────────────┐
│                    ENCODER PATH                          │
│  Conv3D → ReLU → Conv3D → ReLU → MaxPool3D             │
│  Channels: 1 → 64 → 128 → 256 → 512                    │
│  Spatial dims: 128×128×64 → 64×64×32 → 32×32×16 → ...  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                    BOTTLENECK                            │
│  Conv3D → ReLU → Conv3D → ReLU                         │
│  Channels: 512 → 1024                                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                    DECODER PATH                          │
│  UpConv3D → Concat(skip) → Conv3D → ReLU              │
│  Channels: 1024 → 512 → 256 → 128 → 64                │
│  Spatial dims: ... → 32×32×16 → 64×64×32 → 128×128×64  │
└─────────────────────────────────────────────────────────┘
    ↓
Output (1, 128, 128, 64) - Segmentation binaire
```

### Détails des couches

#### Encoder
- **Conv3D blocs** : 2 convolutions 3×3×3 par niveau
- **Activation** : ReLU (standard, non-linéarité)
- **Pooling** : MaxPool3D 2×2×2 (réduction spatiale)
- **Dropout** : 0.3 (régularisation contre overfitting)
- **Canaux** : 64 → 128 → 256 → 512 (augmente avec la profondeur)

**Justification** :
- Convolutions 3×3×3 : Balance entre réceptive field et nombre de paramètres
- MaxPool : Réduit la dimension spatiale, augmente le champ récepteur
- Dropout 0.3 : Critique car peu de données d'entraînement

#### Bottleneck
- **Canaux** : 512 → 1024
- **Rôle** : Représentation la plus abstraite/compacte du volume

#### Decoder
- **UpConv3D** : Upsampling + convolution transposée
- **Skip connections** : Concaténation avec features de l'encoder
- **Conv3D blocs** : 2 convolutions après chaque concat

**Justification skip connections** :
- Récupère les détails fins perdus lors du downsampling
- Essentiel pour localisation précise des petites bactéries
- Aide le gradient à se propager (évite vanishing gradient)

#### Output
- **Conv3D finale** : 1×1×1, 1 canal (segmentation binaire)
- **Pas d'activation** : Les logits sont passés directement à la loss

---

## 📊 Fonction de Loss : SegmentationLoss

### Architecture de la loss

```python
SegmentationLoss = 0.3 × BCE_weighted + 0.7 × Dice_Loss
```

### Composant 1 : Binary Cross-Entropy (BCE) pondérée

```python
BCE_weighted = BCEWithLogitsLoss(pos_weight=10.0)
```

**Rôle** : Loss pixel-wise qui pénalise chaque erreur de classification

**Pondération (`pos_weight=10.0`)** :
- Multiplie la pénalité pour les faux négatifs par 10
- Compense partiellement le déséquilibre (mais pas complètement, car 10 << 130,000)
- Aide le modèle à "démarrer" l'apprentissage des objets

**Justification** :
- ✅ Apprend à rejeter le bruit voxel par voxel
- ✅ Force le modèle à prêter attention aux rares voxels positifs
- ❌ Seule, elle ne suffit pas (le modèle peut prédire "fond" partout et avoir 99.999% de précision)

### Composant 2 : Dice Loss

```python
Dice = (2 × |Pred ∩ Target| + ε) / (|Pred| + |Target| + ε)
Dice_Loss = 1 - Dice
```

**Rôle** : Mesure de similarité entre la prédiction et la vérité terrain

**Propriétés** :
- Insensible au déséquilibre des classes (pas de biais vers la classe majoritaire)
- Évalue la segmentation globalement, pas pixel par pixel
- Score de 0 (aucun recouvrement) à 1 (recouvrement parfait)

**Justification** :
- ✅ Optimise directement la métrique de segmentation qui nous intéresse
- ✅ Force le modèle à produire des objets cohérents (pas juste quelques pixels)
- ✅ Crucial pour des données très déséquilibrées

### Pondération 30%/70%

```
Total_Loss = 0.3 × BCE + 0.7 × Dice
```

**Justification du ratio** :
- **70% Dice** : Priorité à la segmentation globale des objets
- **30% BCE** : Affine la précision voxel par voxel et aide à rejeter le bruit

**Alternative testées** :
- 100% BCE → Prédit "fond" partout (Dice = 0)
- 100% Dice → Peut ignorer les détails fins
- 50%/50% → Moins bon que 30%/70% (empirique)

---

## 📈 Métriques de suivi

### 1. Loss (principale)
- **Utilité** : Objectif d'optimisation
- **Évolution attendue** : Décroissance progressive de ~0.7 à ~0.15-0.20

### 2. Dice Score
```
Dice = (2 × TP) / (2×TP + FP + FN)
```
- **Utilité** : Métrique de segmentation standard
- **Objectif** : > 0.70 (bon), > 0.80 (excellent)
- **Évolution attendue** :
  - Époques 1-20 : ~0.00 (apprend le fond)
  - Époques 20-50 : 0.10-0.30 (commence à détecter)
  - Époques 50-100 : 0.30-0.60 (apprentissage actif)
  - Époques 100+ : 0.60-0.80 (convergence)

### 3. Precision (Précision)
```
Precision = TP / (TP + FP)
```
- **Utilité** : Mesure le taux de faux positifs
- **Interprétation** : Proportion de prédictions positives qui sont correctes
- **Objectif** : > 0.90 (rejette bien le bruit)
- **Critique pour notre cas** : Le modèle doit **rejeter massivement le bruit de diffraction**

### 4. Recall (Sensibilité)
```
Recall = TP / (TP + FN)
```
- **Utilité** : Mesure le taux de détection
- **Interprétation** : Proportion de vraies bactéries détectées
- **Objectif** : > 0.60 (détecte la majorité), > 0.80 (excellent)
- **Critique pour notre cas** : Le modèle doit **détecter à coup sûr les bactéries**

### Trade-off Precision/Recall

```
Haute Precision (0.95) + Faible Recall (0.40)
→ Détecte peu, mais ce qu'il détecte est correct
→ Préférable en début d'entraînement

Équilibre (0.85/0.70)
→ Objectif idéal pour notre application
→ Rejette le bruit + détecte la majorité des objets

Faible Precision (0.60) + Haute Recall (0.90)
→ Détecte tout mais beaucoup de faux positifs
→ À éviter (trop de bruit accepté)
```

---

## ⚙️ Hyperparamètres et justifications

### Entraînement

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `batch_size` | 2 | Limitation VRAM + reconstruction coûteuse |
| `learning_rate` | 0.001 | 10× supérieur au standard (0.0001) car loss custom |
| `num_epochs` | 200 | Problème difficile, convergence lente attendue |
| `optimizer` | Adam | Standard, adaptatif, fonctionne bien en 3D |
| `dropout` | 0.3 | Régularisation contre overfitting |

### Learning Rate Scheduler
- **Type** : ReduceLROnPlateau
- **Patience** : 20 époques
- **Factor** : 0.5 (LR × 0.5)
- **Justification** : Affine progressivement l'apprentissage quand bloqué

### Early Stopping
- **Patience** : 50 époques
- **Métrique** : Validation Dice
- **Justification** : Large patience car convergence lente sur données déséquilibrées

### Patches 3D

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `patch_size_xy` | 128×128 | Balance contexte/mémoire |
| `patch_size_z` | 64 | Suffisant pour capturer une bactérie |
| `stride_xy` | 96 | Overlap 25% pour cohérence spatiale |
| `stride_z` | 48 | Overlap 25% en profondeur |

**Justification overlap** :
- Évite les artefacts de bord entre patches
- Augmente le nombre d'exemples d'entraînement
- Améliore la robustesse de la prédiction

---

## 🔄 Pipeline d'entraînement

### 1. Chargement des données
```python
Hologramme 2D (512×512)
    ↓
Reconstruction 3D (propagation angulaire)
    ↓
Volume intensité (512×512×200)
    ↓
Extraction patches (128×128×64)
```

### 2. Forward pass
```python
Input patch (1, 128, 128, 64)
    ↓
U-Net 3D
    ↓
Logits (1, 128, 128, 64)
    ↓
Loss (SegmentationLoss)
```

### 3. Backward pass
```python
Loss → Gradients → Optimizer.step()
```

### 4. Métriques
```python
Sigmoid(Logits) → Prédictions binaires
    ↓
Dice, Precision, Recall
```

---

## 🎯 Performances attendues

### Phase 1 : Apprentissage du fond (époques 1-20)
- **Loss** : 0.70 → 0.50
- **Dice** : 0.00 → 0.10
- **Precision** : 0.00 → 0.80
- **Recall** : 0.00 → 0.05
- **Interprétation** : Le modèle apprend à rejeter le bruit

### Phase 2 : Détection initiale (époques 20-50)
- **Loss** : 0.50 → 0.30
- **Dice** : 0.10 → 0.40
- **Precision** : 0.80 → 0.85
- **Recall** : 0.05 → 0.30
- **Interprétation** : Commence à détecter les bactéries

### Phase 3 : Amélioration (époques 50-100)
- **Loss** : 0.30 → 0.20
- **Dice** : 0.40 → 0.65
- **Precision** : 0.85 → 0.88
- **Recall** : 0.30 → 0.60
- **Interprétation** : Équilibre precision/recall s'améliore

### Phase 4 : Convergence (époques 100-200)
- **Loss** : 0.20 → 0.15
- **Dice** : 0.65 → 0.75
- **Precision** : 0.88 → 0.90
- **Recall** : 0.60 → 0.70
- **Interprétation** : Performance finale atteinte

---

## 🔍 Cas d'usage spécifiques

### 1. Bactéries isolées
- **Défi** : Signal faible noyé dans le bruit
- **Solution** : Skip connections préservent les détails fins
- **Attendu** : Haute précision de détection (Recall > 0.90)

### 2. Bactéries proches (diffraction chevauchante)
- **Défi** : Séparation de deux objets dont la diffraction se chevauche
- **Solution** : Contexte 3D capturé par le réseau (pas juste 2D)
- **Attendu** : Performance réduite mais acceptable (Recall > 0.60)

### 3. Régions de bruit intense
- **Défi** : Faux positifs dans zones bruitées
- **Solution** : BCE + poids élevé sur precision
- **Attendu** : Faible taux de faux positifs (Precision > 0.85)

---

## 📝 Notes d'implémentation

### Reconstruction à la volée
- Les hologrammes sont reconstruits pendant l'entraînement
- Cache du dernier volume pour réutilisation des patches
- Temps : ~0.6s reconstruction + 0.5s par batch

### Ordre des axes
- **CuPy/Propagation** : (Z, Y, X)
- **PyTorch/U-Net** : (C, D, H, W) = (Channel, Depth, Height, Width)
- **Conversion** : Transpose automatique dans le dataset

### Gestion mémoire GPU
- Batch size limité à 2 par la VRAM
- Volume complet ne tient pas en mémoire → approche par patches
- Trade-off : plus de patches = plus long mais meilleure couverture

---

## 🔬 Améliorations futures possibles

### 1. Attention mechanisms
- Ajouter des modules d'attention pour focaliser sur les régions d'intérêt
- Peut améliorer la détection dans les zones bruitées

### 2. Architecture plus profonde
- Tester ResNet blocks ou DenseNet blocks
- Peut améliorer l'apprentissage de features complexes

### 3. Multi-scale training
- Entraîner sur plusieurs résolutions simultanément
- Améliore la robustesse aux échelles

### 4. Data augmentation 3D
- Rotations, flips, déformations élastiques
- Actuellement non implémenté (à tester)

### 5. Test-Time Augmentation (TTA)
- Prédire sur plusieurs versions augmentées et moyenner
- Peut améliorer les performances à l'inférence

---

## 📚 Références

1. **U-Net** : Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. **Dice Loss** : Milletari et al. (2016) - "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
3. **3D U-Net** : Çiçek et al. (2016) - "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
4. **Class Imbalance** : Lin et al. (2017) - "Focal Loss for Dense Object Detection"

---

**Auteur** : Simon BECKER  
**Date** : Février 2026  
**Version** : 1.0
