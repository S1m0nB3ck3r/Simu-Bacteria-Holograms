# Changelog

## Version 2.0.0 - 2025-10-24

### ✨ Nouvelles Fonctionnalités Majeures

#### Applications Séparées
- **simulation_gui.py** : Interface dédiée à la simulation
- **visualizer_gui.py** : Interface dédiée à la visualisation
- Meilleure organisation et séparation des responsabilités

#### Options de Sauvegarde Personnalisées
- Section **SAUVEGARDE** avec 8 options au choix :
  - Hologramme BMP 8bits
  - Hologramme TIFF 32bits
  - Hologramme NPY 32bits
  - Volume propagé TIFF multistack
  - Volume propagé NPY
  - Volume segmentation TIFF multistack
  - Volume segmentation NPY bool
  - Positions bactéries CSV
- Économie d'espace disque en ne sauvant que ce qui est nécessaire

#### Simulations Multiples
- Paramètre **"Nombre d'hologrammes"** pour générer plusieurs hologrammes en une fois
- Nommage automatique avec suffixe `_i` (ex: `holo_0.bmp`, `holo_1.bmp`, ...)
- Barre de progression globale

#### Visualisation Améliorée
- Affichage côte à côte : Hologramme + Volume Segmenté + Volume Propagé
- Navigation intuitive avec slider Z
- Boutons de navigation rapide (début, précédent, suivant, fin)
- Chargement depuis dossier de résultats

### 🔧 Améliorations

#### Processor
- Gestion des itérations multiples
- Sauvegarde conditionnelle selon les options
- Messages de progression détaillés
- Export CSV pour les positions de bactéries

#### Interface
- Interface compacte optimisée
- Section de sauvegarde claire
- Meilleurs messages d'état
- Fichiers batch pour lancement rapide

### 📝 Documentation
- README.md complet
- QUICKSTART.md pour démarrage rapide
- CHANGELOG.md (ce fichier)
- Commentaires de code améliorés

### 🐛 Corrections
- Fix: Volume binaire maintenant visible (conversion uint8 correcte)
- Fix: Espacement vertical optimisé
- Fix: Gestion correcte des imports scipy

---

## Version 1.3.0 - 2025-10-24 (Backup)

### Fonctionnalités
- Interface unique avec visualisateur intégré
- Génération d'un hologramme à la fois
- Sauvegarde automatique BMP + TIFF
- Pop-up de visualisation avec slider Z

### Limitations
- Pas de choix des formats de sauvegarde
- Une seule itération à la fois
- Interface parfois trop haute pour certains écrans

---

## Historique Antérieur

### Version 1.2.0
- Ajout du volume d'intensité 3D
- Correction du volume binaire

### Version 1.1.0
- Interface GUI de base
- Génération d'hologrammes
- Sauvegarde TIFF multistack

### Version 1.0.0
- Scripts de ligne de commande initiaux
