# -*- coding: utf-8 -*-
"""
Aides PDF construites à partir du code (les tableaux suivent automatiquement les GUI et les scripts) :
  HELP_GUI_GENERATE_DATASET.pdf : GUI_generate_dataset.py et simu_dataset.py ;
  HELP_TRAIN.pdf                : GUI_train.py, train.py, train_stage2.py ;
  HELP_INFERENCE.pdf            : GUI_inference.py, inference_pipeline.py.
Champs des GUI : libellé, défaut et infobulle lus dans leurs définitions (TABS, DEFAULTS, simu_dataset.LAYOUT) ;
options des scripts : lues dans leur argparse (main() appelé, parse_args intercepté).

Utilisation : python deep_learning_segmentation/doc_tools/make_help_pdfs.py
"""
import argparse
import importlib
import os
import sys

from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdf_style import DL_DIR, P, bullets, build, esc, table

sys.path.insert(0, DL_DIR)
REPO_DIR = os.path.dirname(DL_DIR)
DATE = "2026-10-22"


# =============================================================================
# LECTURE DU CODE
# =============================================================================

class _Captured(Exception):
    pass


def parser_of(module_name):
    """ArgumentParser construit par module.main(), sans l'exécuter."""
    mod = importlib.import_module(module_name)
    orig, box = argparse.ArgumentParser.parse_args, {}

    def fake(self, *a, **k):
        box['p'] = self
        raise _Captured()
    argparse.ArgumentParser.parse_args = fake
    try:
        mod.main()
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return box['p']


def short(v):
    """Valeur par défaut lisible (chemins raccourcis, tuples « a b c »)."""
    if v is None or v == '':
        return "—"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, (list, tuple)):
        return ' '.join(short(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    s = str(v)
    for root, rep in ((DL_DIR, 'deep_learning_segmentation'), (REPO_DIR, '.')):
        if s.lower().startswith(root.lower()):
            s = rep + s[len(root):].replace('\\', '/')
    return s


def options_rows(parser, skip=('-h',)):
    rows = [["Option", "Défaut", "Rôle"]]
    for a in parser._actions:
        if any(o in skip for o in a.option_strings) or isinstance(a, argparse._HelpAction):
            continue
        name = ', '.join(a.option_strings) if a.option_strings else a.dest
        if a.nargs not in (None, 0) and a.option_strings:
            name += ' ' + ' '.join([a.dest.upper()] * (a.nargs if isinstance(a.nargs, int) else 1))
        if a.choices:
            name += ' {' + ', '.join(str(c) for c in a.choices) + '}'
        if isinstance(a, argparse._StoreTrueAction):
            dflt = "non"
        elif a.required or not a.option_strings:
            dflt = "obligatoire"
        else:
            dflt = short(a.default)
        rows.append([esc(name), esc(dflt), esc(a.help or '')])
    return rows


def kind_text(kind):
    if isinstance(kind, list):
        return "choix : " + ', '.join(str(k) for k in kind)
    return {'dir': 'dossier', 'int': 'entier', 'float': 'nombre', 'str': 'texte', 'bool': 'case à cocher'}.get(kind, kind)


def fields_rows(rows, defaults, option_of=None, extra_help=None):
    head = ["Champ", "Défaut", "Option du script", "Explication"] if option_of is not None else \
        ["Champ", "Défaut", "Explication"]
    out = [head]
    for key, label, kind, help_text in rows:
        d = defaults.get(key, '')
        h = help_text or (extra_help or {}).get(key, '')
        h = (h + " — " if h else "") + kind_text(kind)
        line = [f"<b>{esc(label)}</b>", esc(short(d))]
        if option_of is not None:
            line.append(esc(option_of.get(key, '—')))
        out.append(line + [esc(h)])
    return out


W4 = [4.2 * cm, 2.6 * cm, 3.4 * cm, 7.4 * cm]
W3 = [4.6 * cm, 3.0 * cm, 10.0 * cm]
WO = [5.2 * cm, 2.6 * cm, 9.8 * cm]


# =============================================================================
# HELP_GUI_GENERATE_DATASET.pdf
# =============================================================================

def help_generate():
    import simu_dataset as sd
    story = [P("Aide — GUI_generate_dataset.py", 'title'),
             P(f"Génération d'un jeu d'hologrammes simulés pour le U-Net 3D (moteur simu_dataset.py) — état du {DATE}.", 'sub'),
             P("À quoi ça sert", 'h1'),
             P("Le GUI simule des hologrammes de bactéries (BPM, fond plat + bruit capteur, sans fond réel) et écrit, pour "
               "chacun des jeux d'entraînement, de validation et de test, les hologrammes, leurs volumes de segmentation "
               "d'instance et les positions des bactéries, plus un fichier parameters.yaml qui décrit tout. Ce dossier est "
               "celui que prend GUI_train.py."),
             P("Lancer", 'h2'),
             P("<font name='DVM'>python deep_learning_segmentation/GUI_generate_dataset.py</font> (depuis la racine du dépôt ; "
               "sur cette machine, définir PYTHONIOENCODING=utf-8 pour la ligne de commande)."),
             P("Boutons et affichage", 'h2'),
             *bullets(["<b>Lancer la génération</b> : écrit parameters.yaml dans le dossier de sortie puis lance la simulation "
                       "dans un processus séparé (l'interface reste réactive). Si le dossier contient déjà train/, val/ ou "
                       "test/, le GUI demande avant de les supprimer.",
                       "<b>Arrêter</b> : termine l'hologramme en cours ; parameters.yaml (état « interrompu ») et index.csv "
                       "restent cohérents, le jeu partiel est utilisable.",
                       "<b>Tuer</b> : arrêt immédiat du processus (le dernier hologramme peut être incomplet).",
                       "<b>Charger parameters.yaml…</b> : reprend tous les paramètres d'un jeu existant.",
                       "<b>Panneau d'information</b> (mis à jour à chaque saisie) : pixel objet, champ, profondeur, plage z des "
                       "bactéries, Δn, grille des masques ; nombre d'hologrammes, place disque et durée estimées ; erreurs de "
                       "cohérence en rouge.",
                       "<b>Aperçu</b> : le dernier hologramme écrit, bactéries entourées (couleur = z) ; journal et barre de "
                       "progression en bas.",
                       "Les réglages sont mémorisés dans GUI_generate_dataset_params.json (à côté du script)."]),
             PageBreak()]
    for sec, rows in sd.LAYOUT:
        r = [["Champ", "Clé (parameters.yaml)", "Défaut", "Explication"]]
        for key, label, kind, default, help_text in rows:
            r.append([f"<b>{esc(label)}</b>", esc(key), esc(short(default)),
                      esc((help_text + " — " if help_text else "") + kind_text(kind))])
        story += [KeepTogether([P(f"Section {sec}", 'h2'), table(r, [4.2 * cm, 4.0 * cm, 2.4 * cm, 7.0 * cm], mono_cols=(1,))]),
                  Spacer(1, 4)]
    story += [P("Contrôles de cohérence avant génération", 'h2'),
              *bullets(["taille XY multiple de 16 (et ≥ 64) ; volume plus profond qu'une bactérie ;",
                        "indice : milieu ≤ bactérie min ≤ bactérie max ; tailles et bruit : min ≤ max ;",
                        "facteur upscale de 1 à 15 (fractions sur 8 bits) ; gris du fond BMP de 1 à 254 ;",
                        "nombre de bactéries (validation, test) : entiers séparés par des espaces."]),
              PageBreak(),
              P("Ce qui est écrit", 'h1'),
              table([["Fichier", "Contenu"]] + [[esc(k), esc(v)] for k, v in sd.FORMATS.items()],
                    [4.2 * cm, 13.4 * cm], mono_cols=(0,)),
              Spacer(1, 6),
              *bullets(["<b>parameters.yaml</b> à la racine : tous les paramètres (une clé par ligne, libellé en commentaire), "
                        "valeurs dérivées (pixel objet, champ, profondeur, plage z des bactéries, Δn, z0_plane), méthode de "
                        "simulation, formats, état et date de la génération.",
                        "<b>generate.log</b> : journal ; contrôle des masques sur les premiers hologrammes de chaque jeu "
                        "(union des instances = masque traversé par la BPM, à quelques voxels d'arrondi près).",
                        "<b>Lire en Python</b> : simu_dataset.load_split(dossier/train) (même format que les anciens .npz) ; "
                        "simu_dataset.instance_masks(npz) (liste des boîtes) ; simu_dataset.instance_volume(npz) (volume "
                        "d'étiquettes complet)."]),
              P("Ligne de commande", 'h1'),
              P("La génération sans interface lit un parameters.yaml existant (écrit par le GUI, ou copié d'un autre jeu "
                "puis modifié) :"),
              table(options_rows(parser_of('simu_dataset')), WO, mono_cols=(0,)),
              Spacer(1, 6),
              P("<font name='DVM'>python deep_learning_segmentation/simu_dataset.py deep_learning_segmentation/datasets/mon_jeu</font>"),
              P("Conseils", 'h1'),
              *bullets(["<b>Indice de la bactérie</b> : garder la plage 1,362–1,390 (Δn 0,032–0,060, calée sur le film réel). "
                        "Le 1,335 de simulation_gui.py donne un signal 8,6 fois trop faible.",
                        "<b>Tailles</b> : 1,8 × 0,6 µm ± 10 % (longueur bout à bout).",
                        "<b>Pas de simulation</b> : 0,25 µm (convergé) ; c'est aussi le pas Z des masques et de la fenêtre fine "
                        "de l'étage 2.",
                        "<b>Profondeur</b> : GUI_train.py reconstruit par défaut exactement le volume du jeu (distance "
                        "volume-caméra → distance + plans × pas) ; la choisir pour couvrir les bactéries des films à analyser.",
                        "<b>Durée</b> : environ 1,5 s + 0,016 s par bactérie et par hologramme 1024 × 1024 (Quadro RTX 3000) ; "
                        "304 hologrammes de 20 à 400 bactéries ≈ 6 à 15 min, ≈ 320 Mo.",
                        "<b>Bactéries rapprochées et recouvrements</b> : la distance entre centres (0,6–4 µm par défaut) "
                        "seule n'empêche pas deux bâtonnets de s'interpénétrer (≈ 47 % à 1 µm). « Bactéries sans "
                        "recouvrement » (coché par défaut) écarte une bactérie qui en touche une autre jusqu'à ne plus en "
                        "toucher aucune (test exact entre capsules) : à 400 bactéries, 12 à 43 recouvrements supprimés, "
                        "déplacements ≤ 0,6 µm, ≈ 60 ms par hologramme ; les distances aux voisines restent les mêmes. Un "
                        "écart minimal de 0,175 µm rend aussi les masques voxelisés disjoints (sinon quelques voxels de "
                        "contact restent partagés). Les jeux antérieurs au 2026-10-22 ont été générés sans ce test.",
                        "<b>Différences avec simulation_gui.py</b> : même physique ; le vide est franchi en une propagation "
                        "(identique à 10⁻⁵ près), l'éclairage n'est pas apodisé (écart ≤ 0,001), orientation isotrope "
                        "(cos φ uniforme), BMP à gris fixe (I / I0 × gris du fond) au lieu du min/max de chaque image."])]
    return build(os.path.join(DL_DIR, 'HELP_GUI_GENERATE_DATASET.pdf'), story, "Aide GUI_generate_dataset",
                 "Aide — GUI_generate_dataset.py (deep_learning_segmentation)")


# =============================================================================
# HELP_TRAIN.pdf
# =============================================================================

OPT_TRAIN = {   # champ de GUI_train.py -> option du script qu'il alimente
    'dataset': "--train_dir / --val_dir (étage 1), --data (étage 2)", 'z_min': "--z_min_um", 'z_max': "--z_max_um",
    's1_name': "--name", 's1_out': "--out", 's1_loss': "--loss", 's1_norm': "--norm", 's1_film_holos': "--film_holos",
    's1_mesh_z': "--mad_mesh_um (Z)", 's1_mesh_xy': "--mad_mesh_um (XY)", 's1_decim': "--decim", 's1_high_pass': "--high_pass",
    's1_low_pass': "--low_pass", 's1_z_step': "--z_step", 's1_patch': "--patch", 's1_pos_frac': "--pos_fraction",
    's1_heat_xy': "--heat_sigma_xy_um", 's1_heat_z': "--heat_sigma_z_um", 's1_peak_box': "--peak_box", 's1_base': "--base",
    's1_levels': "--depth", 's1_iters': "--iters", 's1_lr': "--lr", 's1_orient': "--orient_weight 1",
    's2_name': "--name", 's2_out': "--out", 's2_decim': "--decim", 's2_high_pass': "--high_pass", 's2_low_pass': "--low_pass",
    's2_z_step': "--z_step", 's2_fine_nz': "--fine_nz", 's2_fine_nxy': "--fine_nxy", 's2_crop': "--refocus_crop",
    's2_heat_xy': "--heat_sigma_xy_um", 's2_heat_z': "--heat_sigma_z_um", 's2_tol_xy': "--tol_xy_um", 's2_tol_z': "--tol_z_um",
    's2_iters': "--iters", 's2_lr': "--lr", 's2_base': "--base", 's2_levels': "--depth", 's2_crops': "--crops_per_holo",
    's2_holos': "--holos_per_step", 's2_jit_xy': "--jitter_xy", 's2_jit_z': "--jitter_z", 's2_neg': "--neg_fraction",
    's2_mid': "--mid_fraction", 's2_mid_pair': "--mid_pair_um", 's2_vmin': "--vmin",
}


def help_train():
    import GUI_train as gt
    story = [P("Aide — entraînement du U-Net 3D", 'title'),
             P(f"GUI_train.py, train.py (étage 1, positions), train_stage2.py (étage 2, instances) — état du {DATE}.", 'sub'),
             P("Déroulement", 'h1'),
             *bullets(["<b>1. Jeu de données</b> : générer un jeu avec GUI_generate_dataset.py (HELP_GUI_GENERATE_DATASET.pdf), "
                       "ou reprendre un dossier de datasets/.",
                       "<b>2. Étage 1</b> (positions) : environ 30 à 45 min pour 6000 itérations ; donne le modèle de détection.",
                       "<b>3. Étage 2</b> (instances) : environ 55 min ; même décimation que l'étage 1 avec lequel il sera utilisé.",
                       "<b>4. Utiliser</b> : choisir les dossiers des deux runs dans GUI_inference.py (onglet Modèles), ou les "
                       "copier à la place de models/stage1 et models/stage2 pour en faire les modèles par défaut."]),
             P("Lancer", 'h2'),
             P("<font name='DVM'>python deep_learning_segmentation/GUI_train.py</font> ; en ligne de commande : "
               "<font name='DVM'>python deep_learning_segmentation/train.py --name s1 --train_dir &lt;jeu&gt;/train "
               "--val_dir &lt;jeu&gt;/val</font> et <font name='DVM'>python deep_learning_segmentation/train_stage2.py "
               "--name s2 --data &lt;jeu&gt;</font>."),
             P("Boutons et affichage", 'h2'),
             *bullets(["<b>Générer un jeu…</b> : ouvre GUI_generate_dataset.py dans une fenêtre indépendante.",
                       "<b>Entraîner l'étage 1 / l'étage 2</b> : lance le script dans un processus séparé (priorité basse, "
                       "l'interface reste réactive), avec les réglages des onglets ; demande avant d'écraser un run existant.",
                       "<b>Arrêter</b> : arrêt propre (fichier STOP dans le dossier du run) : fin de l'itération, dernière "
                       "validation, bilan enregistré.",
                       "<b>Tuer</b> : arrêt immédiat (le dernier model_best.pt enregistré reste valable).",
                       "<b>Suivre un run…</b> : reprend l'affichage d'un run en cours ou fini (son train.log).",
                       "<b>Courbes</b> : loss d'entraînement et validation (étage 1 : F1, rappel, rappel des couchées ; "
                       "étage 2 : IoU, part d'IoU > 0,5, rappel et précision des centres) ; journal et progression en bas.",
                       "<b>Fermer</b> pendant un entraînement : le laisser tourner (suivi possible ensuite) ou l'arrêter "
                       "proprement. Réglages mémorisés dans GUI_train_params.json."]),
             P("Contenu d'un run (runs/&lt;nom&gt;/)", 'h2'),
             table([["Fichier", "Contenu"],
                    ["model_best.pt", "poids du meilleur modèle en validation + toute la configuration (réglages, grille, "
                                      "optique, filtre, statistiques du film…) + seuil retenu (étage 1)"],
                    ["config.json", "la même configuration, lisible"],
                    ["history.json", "une ligne par validation (loss, métriques, seuil, temps)"],
                    ["train_summary.json", "bilan : itérations, temps, mémoire GPU, meilleure validation"],
                    ["train.log", "journal complet du script"]], [3.6 * cm, 14.0 * cm], mono_cols=(0,)),
             PageBreak()]
    for tab, rows in gt.TABS.items():
        story += [P(f"Onglet {tab}", 'h1'),
                  table(fields_rows(rows, gt.DEFAULTS, OPT_TRAIN), W4, mono_cols=(2,)), Spacer(1, 4)]
        if tab == "Données":
            story += [P("Sous les champs : résumé du jeu (parameters.yaml) et volume reconstruit retenu (automatique ou "
                        "saisi). Les défauts des onglets Étage 1 et Étage 2 sont les constantes en tête de train.py et "
                        "train_stage2.py.", 'small')]
        else:
            story += [PageBreak()]
    story += [P("train.py — étage 1 (positions)", 'h1'),
              P("Constantes en tête du script (valeurs par défaut des options) : DECIM, HIGH_PASS, LOW_PASS, Z_STEP_UM, PATCH, "
                "POS_FRACTION, HEAT_SIGMA_XY_UM, HEAT_SIGMA_Z_UM, PEAK_BOX, FILM_STATS_HOLOS, MAD_MESH_UM.", 'small'),
              table(options_rows(parser_of('train')), WO, mono_cols=(0,)),
              PageBreak(),
              P("train_stage2.py — étage 2 (instances)", 'h1'),
              P("Constantes en tête du script : DECIM, HIGH_PASS, LOW_PASS, Z_STEP_UM, FINE_NZ, FINE_NXY, REFOCUS_CROP, "
                "HEAT_SIGMA_XY_UM, HEAT_SIGMA_Z_UM, TOL_XY_UM, TOL_Z_UM, CROPS_PER_HOLO, NEG_FRACTION, MID_FRACTION, "
                "MID_PAIR_UM, JITTER_XY_UM, JITTER_Z_UM, VMIN.", 'small'),
              table(options_rows(parser_of('train_stage2')), WO, mono_cols=(0,)),
              PageBreak(),
              P("Règles et conseils", 'h1'),
              *bullets(["<b>Décimation et filtre</b> : la décimation recadre le spectre ; sans perte si le passe-bas est ≤ taille / "
                        "(2 × décimation) : 256 px pour 1024 et ×2, 128 pour ×4. Les deux étages utilisés ensemble doivent avoir "
                        "la même décimation (l'inférence refuse sinon).",
                        "<b>Tailles multiples de 2^niveaux</b> (8 pour 3 niveaux) : patchs de l'étage 1, fenêtre fine de l'étage 2 "
                        "(plans et côté). Le carré refocalisé doit contenir la fenêtre fine : côté fin ≤ ¾ × décimation × carré.",
                        "<b>Normalisation</b> : filmmad (défaut du GUI) — mesurée sur le film, constante d'une image à l'autre. "
                        "Au benchmark du 22 octobre, toutes les normalisations se valent à nombre de détections égal "
                        "(resultats/benchmark_grid_B/RAPPORT_NORMALISATION_ETAGE1.pdf).",
                        "<b>Profondeur</b> : vide = celle du jeu ; au moins la profondeur des bactéries des films à analyser.",
                        "<b>Seuil de l'étage 1</b> : choisi à la fin de l'entraînement (meilleur F1 en validation simulée) et "
                        "enregistré ; sur un film réel, il se règle à l'inférence (HELP_INFERENCE.pdf).",
                        "<b>Mémoire</b> (Quadro RTX 3000, 6 Go) : patchs 48 × 96 × 96 par 4 ≈ 5 Go ; 64 × 128 × 128 déborde. "
                        "Une décimation de 1 multiplie la mémoire et le temps de reconstruction par 4.",
                        "<b>Réentraîner à l'identique</b> : même jeu, mêmes options, même graine (--seed)."])]
    return build(os.path.join(DL_DIR, 'HELP_TRAIN.pdf'), story, "Aide entraînement U-Net 3D",
                 "Aide — entraînement du U-Net 3D (deep_learning_segmentation)")


# =============================================================================
# HELP_INFERENCE.pdf
# =============================================================================

OPT_INF = {
    'folder': "folder (1er argument)", 'ext': "--ext", 'mean_removal': "--mean_removal", 'first': "--first",
    'last': "--last", 'stage1': "--stage1", 'stage2': "--stage2", 'pix_size_um': "--dx_um (= pixel / grossissement)",
    'magnification': "--dx_um (= pixel / grossissement)", 'wavelength_um': "--wavelength_um", 'index_medium': "--index_medium",
    'z_min': "--z_min", 'z_max': "--z_max", 'z_step': "--z_step", 'decim': "--decim", 'high_pass': "--high_pass",
    'low_pass': "--low_pass", 'z_margin': "--z_margin_um", 'thr1': "--thr1", 'rounds': "--rounds", 'prior': "--prior",
    'peak_box': "--peak_box", 'film_holos': "--film_holos", 'vmin': "--vmin", 'heat_thr': "--heat_thr", 'nms2': "--nms",
    'border2': "--border", 'same_xy': "--same_xy_um", 'same_z': "--same_z_um", 'dup_iou': "--dup_iou",
    'dup_xy': "--dup_xy_um", 'dup_z': "--dup_z_um", 'out_dir': "out_dir (2e argument)", 'save_npz': "--no_npz (inverse)",
    'save_bmp': "--no_bmp (inverse)",
}


def help_inference():
    import GUI_inference as gi
    defaults = dict(gi.DEFAULTS, mean_removal=gi.REMOVAL.get(gi.DEFAULTS['mean_removal'], gi.DEFAULTS['mean_removal']))
    story = [P("Aide — inférence du U-Net 3D", 'title'),
             P(f"GUI_inference.py et inference_pipeline.py : positions 3D et volume segmenté par instance sur un film "
               f"holographique — état du {DATE}.", 'sub'),
             P("Ce que fait l'inférence", 'h1'),
             *bullets(["<b>Suppression de l'hologramme moyen</b> du film (division ou soustraction), puis <b>reconstruction</b> "
                       "par spectre angulaire sur la plage z choisie (volume grossier).",
                       "<b>Normalisation</b> du volume comme à l'entraînement du modèle (pour une normalisation « film », "
                       "statistiques mesurées sur des images du film traité).",
                       "<b>Étage 1</b> : carte des centres ; détections = maxima locaux au-dessus du seuil.",
                       "<b>Étage 2</b> : fenêtre fine refocalisée autour de chaque détection → masque de la bactérie et centres "
                       "des voisines ; second tour sur les centres non segmentés ; fusion des doublons ; mesures (barycentre, "
                       "axe, longueur, épaisseur, volume).",
                       "Les modèles enregistrent leur grille, leur filtre, leur normalisation et leurs fenêtres : laissés vides, "
                       "les champs reprennent ces valeurs ; tout écart s'affiche en orange sous les onglets."]),
             P("Lancer", 'h2'),
             P("<font name='DVM'>python deep_learning_segmentation/GUI_inference.py</font> ; sans interface : "
               "<font name='DVM'>python deep_learning_segmentation/inference_pipeline.py &lt;film&gt; &lt;sortie&gt; "
               "--z_max 100</font>."),
             P("Boutons, vues et souris", 'h2'),
             *bullets(["<b>Charger les modèles</b> : charge les deux étages (fait automatiquement au premier calcul) ; leur "
                       "description et les écarts avec l'entraînement s'affichent sous les onglets.",
                       "<b>Image n°</b> (ou Entrée dans le compteur) : calcule une image seule, sans export ; <b>◀ ▶</b> : "
                       "image précédente ou suivante.",
                       "<b>Lancer le lot</b> : traite la plage d'images (onglet Entrée) et écrit les sorties (onglet Sortie) ; "
                       "<b>Arrêter</b> : fin de l'image en cours.",
                       "<b>Vue</b> : hologramme + détections (z en µm), instances en projection XY, XZ, YZ ; flèches Haut / Bas "
                       "pour changer de vue ; « Contraste ± », « z (µm) », « détections de l'étage 1 » ; barre d'outils "
                       "matplotlib (zoom, déplacement), le zoom est conservé d'une image à l'autre.",
                       "<b>Clic sur une bactérie</b> (hors mode zoom) : fenêtre avec ses coupes XY, XZ, YZ (amplitude du champ "
                       "refocalisé, son masque en orange, ceux des voisines en pointillés).",
                       "Temps de chaque étape affichés sous la figure. Réglages mémorisés dans GUI_inference_params.json."]),
             PageBreak()]
    for tab, rows in gi.TABS.items():
        story += [P(f"Onglet {tab}", 'h1'),
                  table(fields_rows(rows, defaults, OPT_INF), W4, mono_cols=(2,)), Spacer(1, 6)]
    story += [PageBreak(),
              P("inference_pipeline.py — ligne de commande", 'h1'),
              P("Constantes en tête du script (valeurs par défaut) : DEFAULT_OPTICS, DEFAULT_RECON, Z_MARGIN_UM, PEAK_BOX, "
                "FILM_STATS_HOLOS, STAGE2, MEAN_REMOVAL. « — » : valeur enregistrée avec le modèle.", 'small'),
              table(options_rows(parser_of('inference_pipeline')), WO, mono_cols=(0,)),
              PageBreak(),
              P("Sorties (dossier de sortie)", 'h1'),
              table([["Fichier", "Contenu"],
                     ["positions_unet3d.csv", "une bactérie par ligne : HOLOGRAM NUMBER, OBJECT NUMBER, X / Y / Z POSITION (m), "
                                              "NUMBER OF VOXEL (6 premières colonnes = celles du RESULT_*.csv d'HOLOTRACKER), "
                                              "puis image, axis_x / y / z, length_um, thickness_um, volume_um3, score, round "
                                              "(tour de l'étage 2 qui l'a trouvée)"],
                     ["instances/&lt;image&gt;_instances.npz", "origin (n, 3) et masks (n, nz, nxy, nxy) de chaque instance "
                                                              "sur la grille fine ; grid : voxel (k, r, c) ↔ (z, y, x) = "
                                                              "((origin_k + k)·dz, (origin_r + r)·dx, (origin_c + c)·dx)"],
                     ["&lt;image&gt;_localisations.bmp", "hologramme, bactéries entourées, couleur et valeur de z (µm), "
                                                        "échelle de couleur"],
                     ["inference_params.json", "réglages du lot, description des modèles, optique, reconstruction, grille, "
                                               "fenêtres, réglages de l'étage 2, temps médians"]],
                    [4.6 * cm, 13.0 * cm], mono_cols=(0,)),
              P("Réglages qui comptent", 'h1'),
              *bullets(["<b>Seuil de l'étage 1</b> : c'est le réglage qui pèse le plus sur un film réel. Le seuil de validation "
                        "simulée donne, selon le modèle, de 55 à 141 détections par image sur le film de référence (≈ 49 "
                        "bactéries suivies par HOLOTRACKER). Le régler pour obtenir un nombre de détections plausible.",
                        "<b>Suppression de l'hologramme moyen</b> : division pour un film réel (retire aussi un éclairage non "
                        "uniforme) ; « aucune (÷ moyenne de l'image) » pour un hologramme isolé ou un BMP simulé ; « aucune "
                        "(fond déjà à 1) » pour un TIFF du simulateur.",
                        "<b>Plage z</b> : seuls les plans de z min − marge à z max + marge sont reconstruits ; les détections hors "
                        "de [z min, z max] sont retirées. Une plage étroite accélère d'autant.",
                        "<b>Pas Z, décimation, filtre</b> : laisser vides (valeurs d'entraînement) sauf raison précise ; un "
                        "écart change ce que voit le réseau. L'étage 2 exige la décimation de son entraînement.",
                        "<b>Doublons</b> : baisser l'IoU de fusion réduit les doublons mais peut fusionner deux bactéries "
                        "collées ; agrandir la fenêtre des maxima de l'étage 1 a le même effet sur les détections.",
                        "<b>Temps</b> (1024 × 1024, fp16) : 0–100 µm ≈ 0,7 s par image pour l'étage 1, + 0,1 à 0,5 s pour "
                        "l'étage 2 selon le nombre de bactéries."])]
    return build(os.path.join(DL_DIR, 'HELP_INFERENCE.pdf'), story, "Aide inférence U-Net 3D",
                 "Aide — inférence du U-Net 3D (deep_learning_segmentation)")


if __name__ == '__main__':
    which = sys.argv[1:] or ['generate', 'train', 'inference']
    for w in which:
        print({'generate': help_generate, 'train': help_train, 'inference': help_inference}[w]())
