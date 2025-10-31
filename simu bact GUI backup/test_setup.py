# -*- coding: utf-8 -*-

"""
Script de test rapide pour vérifier la configuration
"""

import os
import sys
import json

def test_imports():
    """Teste les imports nécessaires"""
    print("Test des imports...")
    try:
        import tkinter as tk
        print("✅ tkinter OK")
    except ImportError as e:
        print(f"❌ tkinter manquant : {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ numpy OK (version {np.__version__})")
    except ImportError as e:
        print(f"❌ numpy manquant : {e}")
        return False
    
    try:
        import cupy as cp
        print(f"✅ cupy OK (version {cp.__version__})")
    except ImportError as e:
        print(f"❌ cupy manquant : {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL OK")
    except ImportError as e:
        print(f"❌ PIL manquant : {e}")
        return False
    
    try:
        import tifffile
        print("✅ tifffile OK")
    except ImportError as e:
        print(f"❌ tifffile manquant : {e}")
        return False
    
    return True

def test_config_file():
    """Teste la présence et la validité du fichier de configuration"""
    print("\nTest du fichier de configuration...")
    config_file = "parameters_simu_bact.json"
    
    if not os.path.exists(config_file):
        print(f"❌ Fichier {config_file} introuvable")
        return False
    
    try:
        with open(config_file, 'r') as f:
            params = json.load(f)
        print(f"✅ Fichier de configuration OK ({len(params)} paramètres)")
        return True
    except Exception as e:
        print(f"❌ Erreur de lecture du fichier : {e}")
        return False

def test_parent_modules():
    """Teste l'accès aux modules du répertoire parent"""
    print("\nTest des modules parent...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    try:
        import simu_hologram
        print("✅ simu_hologram OK")
    except ImportError as e:
        print(f"❌ simu_hologram manquant : {e}")
        return False
    
    try:
        import propagation
        print("✅ propagation OK")
    except ImportError as e:
        print(f"❌ propagation manquant : {e}")
        return False
    
    try:
        import traitement_holo
        print("✅ traitement_holo OK")
    except ImportError as e:
        print(f"❌ traitement_holo manquant : {e}")
        return False
    
    return True

def main():
    print("="*60)
    print("TEST DE CONFIGURATION - Simulation Hologrammes Bactéries")
    print("="*60)
    
    all_ok = True
    all_ok &= test_imports()
    all_ok &= test_config_file()
    all_ok &= test_parent_modules()
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ TOUS LES TESTS SONT PASSÉS !")
        print("Vous pouvez lancer l'interface : python simu_bact_gui.py")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Installez les dépendances manquantes avant de continuer")
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())
