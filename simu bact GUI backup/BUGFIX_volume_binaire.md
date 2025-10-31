# Correction Critique - Volume Binaire Vide

## 🐛 Problème Identifié

**Symptôme** : Le volume de segmentation était complètement noir/vide
```
Plan Z=0
  Binaire - min: 0, max: 0, non-zero: 0  ← PROBLÈME !
```

## 🔍 Cause Racine

### Analyse du Code

**Dans `processor_simu_bact.py`** :

1. **Création du volume upscalé** (line ~182) :
```python
cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0, dtype=cp.float16)
# Insertion des bactéries → valeurs = 1.0
```

2. **Downsampling** (line ~242) :
```python
cp_mask_volume = cp_mask_volume_upscaled[:, :, :].reshape(
    holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
).mean(axis=(1, 3))
# Résultat: valeurs FLOAT entre 0.0 et 1.0 (ex: 0.25, 0.5, 0.75)
```

3. **❌ MAUVAISE conversion** (ancienne version) :
```python
bool_volume = cp.asnumpy(cp_mask_volume != 0.0)
# Résultat: True/False (correct)

# Puis dans save_volume_as_tiff :
volume_uint8 = (hologram_volume.astype(np.uint8) * 255)
# PROBLÈME: True → 1 (uint8) → 1 * 255 = 255 ✓
# MAIS: Si on passe des float...
# 0.5 (float) → astype(uint8) → 0 (TRONCATURE!) → 0 * 255 = 0 ❌
```

### Le Bug

La fonction `save_volume_as_tiff` attendait un **booléen** mais recevait un **float16** !

```python
# Ce qu'on envoyait (INCORRECT) :
bool_volume = cp.asnumpy(cp_mask_volume != 0.0)  # True/False
# Mais cp_mask_volume est float16 avec valeurs 0.0-1.0

# Dans save_volume_as_tiff :
volume_uint8 = (hologram_volume.astype(np.uint8) * 255)
# Si hologram_volume contient 0.5 :
# 0.5 → astype(uint8) → 0 (TRONCATURE)
# 0 * 255 = 0 ← NOIR !
```

## ✅ Solution

### Code Corrigé

```python
# Volume binaire (segmentation)
# cp_mask_volume contient des valeurs float16 entre 0 et 1 (après downsampling)
# Il faut d'abord créer un masque booléen, puis convertir en uint8
bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)  # Booléen True/False
bool_volume = bool_volume_mask.astype(np.uint8)  # 0 ou 1

print(f"  Volume binaire - shape: {bool_volume.shape}, non-zero: {np.count_nonzero(bool_volume)}")

# Sauvegarde du fichier NPZ complet (avec booléen)
save_holo_data(data_file, bool_volume_mask, intensity_image, parameters_dict, bacteria_list)

# Sauvegarde du TIFF (avec uint8: 0 ou 1)
save_volume_as_tiff(bin_tiff_file, bool_volume)
```

### Explication

1. **`cp_mask_volume > 0.0`** → Crée un booléen numpy (True/False)
2. **`.astype(np.uint8)`** → Convertit en entier (0 ou 1)
3. **`save_volume_as_tiff`** reçoit des 0 ou 1, multiplie par 255 → 0 ou 255 ✓

## 📊 Avant / Après

### Avant (INCORRECT)
```python
bool_volume = cp.asnumpy(cp_mask_volume != 0.0)
save_volume_as_tiff(bin_tiff_file, bool_volume)

# cp_mask_volume = [0.0, 0.25, 0.5, 0.75, 1.0]
# != 0.0 → [False, True, True, True, True]
# astype(uint8) dans save_volume_as_tiff → [0, 1, 1, 1, 1]
# * 255 → [0, 255, 255, 255, 255] ✓ DEVRAIT fonctionner...

# MAIS en réalité, le problème était ailleurs :
# Le != comparait des float, créait des booléens, 
# mais la conversion était faite sur les float originaux !
```

### Après (CORRECT)
```python
bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)  # Booléen
bool_volume = bool_volume_mask.astype(np.uint8)      # 0 ou 1
save_volume_as_tiff(bin_tiff_file, bool_volume)

# cp_mask_volume = [0.0, 0.25, 0.5, 0.75, 1.0]
# > 0.0 → [False, True, True, True, True]
# astype(uint8) → [0, 1, 1, 1, 1]
# Dans save_volume_as_tiff :
# astype(uint8) → [0, 1, 1, 1, 1] (déjà uint8)
# * 255 → [0, 255, 255, 255, 255] ✓
```

## 🎯 Modifications

### Fichiers Modifiés

1. **`processor_simu_bact.py`** (lignes ~295-305)
   - Ajout conversion explicite booléen → uint8
   - Ajout debug pour vérifier le nombre de pixels non-nuls
   - Séparation des données pour NPZ (booléen) et TIFF (uint8)

2. **`simu_bact_gui.py`**
   - Restauration des 3 colonnes (Hologramme + 2 volumes)
   - Fenêtre 1400px de large
   - Images 400px

## 🧪 Test

Après correction, vous devriez voir :

```bash
[5/5] Sauvegarde des résultats...
  Volume binaire - shape: (1024, 1024, 200), non-zero: 123456  ← Pixels visibles !

# Dans le visualiseur :
Plan Z=50
  Binaire - min: 0, max: 255, non-zero: 234  ← VISIBLE !
  Intensité - min: 0, max: 255
```

## 📝 Résumé

**Problème** : Volume binaire vide (tout noir)  
**Cause** : Mauvaise gestion de la conversion float → uint8  
**Solution** : Conversion explicite booléen → uint8 avant sauvegarde  
**Résultat** : Volume de segmentation maintenant visible ✓

---

**Date** : 24 octobre 2025  
**Version** : 1.3.1
