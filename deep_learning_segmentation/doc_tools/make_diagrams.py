# Schémas du PDF PIPELINE_UNET3D (pipeline d'inférence, données / entraînement, U-Net 3D, bloc résiduel, étage 2)
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
INK, MUTED = '#1f2328', '#57606a'
C = dict(data='#dbeafe', phys='#e0f2e9', net='#fde7d3', post='#f3e8ff', out='#fef3c7', edge='#57606a')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def box(ax, x, y, w, h, title, body='', color='#ffffff', tsize=9.5, bsize=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=color, ec=C['edge'], lw=1.0))
    ax.text(x + w / 2, y + h - 0.022, title, ha='center', va='top', fontsize=tsize, fontweight='bold', color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.062, body, ha='center', va='top', fontsize=bsize, color=MUTED, linespacing=1.35)


def arrow(ax, p0, p1, text='', off=(0, 0), style='-|>', color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, lw=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, fontsize=7.3, color=MUTED, ha='center')


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


# ---------------------------------------------------------------- 1. pipeline d'inférence
fig, ax = canvas(11, 7.2)
W, H = 0.205, 0.17
y1, y2, y3 = 0.76, 0.46, 0.13
box(ax, 0.02, y1, W, H, "Hologramme", "1024 × 1024 px, pixel 0,175 µm\nhologramme moyen du film retiré\n(division ou soustraction)", C['data'])
box(ax, 0.265, y1, W, H, "Reconstruction", "spectre angulaire, filtre 15–250 px\ndécimation ×2 par le spectre\nplans à 1 µm sur la plage z choisie", C['phys'])
box(ax, 0.51, y1, W, H, "Volume grossier", "Re, Im du champ, 2 × Z × 512 × 512\n(0,35 × 0,35 × 1 µm), normalisé\ncomme à l'entraînement du modèle", C['data'])
box(ax, 0.755, y1, 0.225, H, "Étage 1 : U-Net 3D", "carte des centres (CenterNet)\ntuiles 256² + halo 32, fp16\nBatchNorm repliée", C['net'])
box(ax, 0.755, y2, 0.225, H, "Pics → positions", "maxima locaux (3 × 5 × 5)\nseuil choisi en validation\n(x, y, z) de chaque détection", C['post'])
box(ax, 0.51, y2, W, H, "Refocalisation locale", "fenêtre 64² du plan le plus proche\npropagation Δz ≤ 4,5 µm, ×2 spectral\nexacte à < 1 % près", C['phys'])
box(ax, 0.265, y2, W, H, "Étage 2 : U-Net 3D", "fenêtres 32 × 40 × 40 voxels\n0,175 × 0,175 × 0,25 µm (8 × 7 × 7 µm)\n→ self, autres, centres", C['net'])
box(ax, 0.02, y2, W, H, "Second tour", "centres vus et non segmentés\n(bactéries fusionnées à l'étage 1)\n→ nouvelles fenêtres", C['post'])
box(ax, 0.02, y3, W, H, "Doublons", "masques qui se recouvrent\n(IoU > 0,5) : on garde le plus gros\nun seul arbre de recherche", C['post'])
box(ax, 0.265, y3, W, H, "Instances", "un masque fin par bactérie\n(un voxel peut appartenir à deux\nbactéries qui se croisent)", C['out'])
box(ax, 0.51, y3, W, H, "Mesures", "barycentre (x, y, z), axe (ACP)\nlongueur, épaisseur, volume\nscore, tour d'origine", C['out'])
box(ax, 0.755, y3, 0.225, H, "Sorties", "CSV (colonnes d'HOLOTRACKER)\ninstances .npz (origine + masque)\naffichage GUI_inference", C['out'])
arrow(ax, (0.225, y1 + H / 2), (0.265, y1 + H / 2))
arrow(ax, (0.47, y1 + H / 2), (0.51, y1 + H / 2))
arrow(ax, (0.715, y1 + H / 2), (0.755, y1 + H / 2))
arrow(ax, (0.867, y1), (0.867, y2 + H))
arrow(ax, (0.755, y2 + H / 2), (0.715, y2 + H / 2), "positions", (0, 0.012))
arrow(ax, (0.6125, y1), (0.6125, y2 + H), "même volume", (0.055, 0))
arrow(ax, (0.51, y2 + H / 2), (0.47, y2 + H / 2))
arrow(ax, (0.265, y2 + H * 0.35), (0.225, y2 + H * 0.35), "centres", (0, -0.03))
arrow(ax, (0.225, y2 + H * 0.78), (0.265, y2 + H * 0.78), "", rad=0.0, color='#7a5bd6')
ax.text(0.245, y2 + H * 0.83, "fenêtres", fontsize=7.3, color='#7a5bd6', ha='center')
arrow(ax, (0.3675, y2), (0.1225, y3 + H), "masques self", (0.02, 0.01), rad=-0.15)
arrow(ax, (0.225, y3 + H / 2), (0.265, y3 + H / 2))
arrow(ax, (0.47, y3 + H / 2), (0.51, y3 + H / 2))
arrow(ax, (0.715, y3 + H / 2), (0.755, y3 + H / 2))
ax.text(0.5, 0.985, "Inférence : positions 3D puis segmentation d'instance", ha='center', va='top', fontsize=13, fontweight='bold')
ax.text(0.5, 0.045, "Temps (Quadro RTX 3000, fp16) : reconstruction 31 ms · étage 1 1,23 s (208 plans) · étage 2 de 77 ms (N = 50) à 540 ms (N = 400)",
        ha='center', fontsize=8.5, color=MUTED)
for lab, col, x in (("données", C['data'], 0.13), ("physique", C['phys'], 0.26), ("réseau", C['net'], 0.39),
                    ("post-traitement", C['post'], 0.52), ("sorties", C['out'], 0.68)):
    ax.add_patch(Rectangle((x, 0.005), 0.018, 0.018, fc=col, ec=C['edge'], lw=0.6))
    ax.text(x + 0.024, 0.014, lab, fontsize=7.5, va='center', color=MUTED)
fig.savefig(os.path.join(OUT, '1_pipeline_inference.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. données et entraînement
fig, ax = canvas(11, 6.4)
H = 0.2
yA, yB = 0.66, 0.24
box(ax, 0.02, yA, 0.22, H, "Tirage des bactéries", "GUI_generate_dataset.py : N, profondeur\n1,8 × 0,6 µm ± 10 %, orientation\nisotrope, 30 % rapprochées\n(ou films : dynamique du film réel)", C['data'])
box(ax, 0.275, yA, 0.22, H, "BPM (simulateur)", "tranches occupées seulement\npas 0,25 µm, Δn 0,032–0,060\n(calé sur le film : 0,046)\nvide franchi en un saut", C['phys'])
box(ax, 0.53, yA, 0.2, H, "Hologramme", "BMP 8 bits (I / I0 × 128)\nfond plat + bruit capteur\nσ = 0,015, pas de fond réel\n+ positions CSV", C['data'])
box(ax, 0.765, yA, 0.215, H, "Masques d'instance", "écrits par le simulateur (.npz)\n(même noyau, même grille)\nunion = masque traversé par\nla BPM (contrôle)", C['out'])
box(ax, 0.02, yB, 0.3, H, "Étage 1 : patchs 48 × 96 × 96", "volume grossier sur la profondeur du jeu\nnormalisé (MAD du film par défaut)\n70 % des patchs centrés sur une bactérie\ncible : gaussiennes des centres\nloss focale CenterNet, 6000 it (0,3–0,45 s/it)", C['net'])
box(ax, 0.35, yB, 0.3, H, "Étage 2 : fenêtres 32 × 40 × 40", "refocalisation locale autour d'une bactérie\n(+ décalage σ 0,25 / 0,5 µm), 15 % entre deux\nvoisines, 10 % au hasard ; cibles self, autres\n(fractions du simulateur), centres ; BCE + Dice\n+ CenterNet, 6000 it (0,56 s/it)", C['net'])
box(ax, 0.68, yB, 0.3, H, "Validation, meilleur modèle", "étage 1 : F1 sur volumes entiers,\nseuil choisi sur une grille\nétage 2 : IoU exacte (instance entière)\n+ F1 des centres\nhistory.json, model_best.pt", C['post'])
arrow(ax, (0.24, yA + H / 2), (0.275, yA + H / 2))
arrow(ax, (0.495, yA + H / 2), (0.53, yA + H / 2))
arrow(ax, (0.43, yA), (0.80, yA), "masques écrits par le simulateur", (0, -0.075), rad=0.35, color='#b07d00')
arrow(ax, (0.63, yA), (0.17, yB + H), "hologrammes", (-0.05, 0.02), rad=0.1)
arrow(ax, (0.63, yA), (0.5, yB + H), "", rad=0.05)
arrow(ax, (0.87, yA), (0.5, yB + H), "masques (étage 2)", (0.1, 0.04), rad=-0.1)
arrow(ax, (0.32, yB + H / 2), (0.35, yB + H / 2))
arrow(ax, (0.65, yB + H / 2), (0.68, yB + H / 2))
ax.text(0.5, 0.97, "Données simulées et entraînement", ha='center', va='top', fontsize=13, fontweight='bold')
ax.text(0.5, 0.1, "GUI_generate_dataset.py / simu_dataset.py → datasets/<jeu> (parameters.yaml, train/, val/, test/) · generate_sequences.py (films)\n"
        "train.py (étage 1) · train_stage2.py (étage 2) · ou GUI_train.py, qui prend le dossier du jeu", ha='center', fontsize=8.5, color=MUTED, linespacing=1.5)
fig.savefig(os.path.join(OUT, '2_donnees_entrainement.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. architecture du U-Net 3D
fig, ax = canvas(11, 6.6)
levels = [(16, '48 × 96 × 96', '32 × 40 × 40'), (32, '24 × 48 × 48', '16 × 20 × 20'),
          (64, '12 × 24 × 24', '8 × 10 × 10'), (128, '6 × 12 × 12', '4 × 5 × 5')]
xe, xd = 0.16, 0.54
ys = [0.74, 0.54, 0.34, 0.14]
bw = 0.2
for i, (ch, s1, s2) in enumerate(levels):
    hgt = 0.13
    box(ax, xe, ys[i], bw, hgt, f"ResBlock → {ch} canaux", f"étage 1 : {s1}\nétage 2 : {s2}", C['net'], tsize=9, bsize=7.5)
    if i < 3:
        box(ax, xd, ys[i], bw, hgt, f"ResBlock → {ch} canaux", f"entrée : concat. {ch} + {ch}\n= {2 * ch} canaux", C['net'], tsize=9, bsize=7.5)
        arrow(ax, (xe + bw, ys[i] + hgt * 0.55), (xd, ys[i] + hgt * 0.55), "liaison (copie)", (0, 0.012), color='#2a78d6')
        arrow(ax, (xe + bw * 0.5, ys[i]), (xe + bw * 0.5, ys[i + 1] + hgt), "max-pool 2", (0.055, 0))
        nxt = xd + bw * 0.5 if i + 1 < 3 else xe + bw
        if i + 1 < 3:
            arrow(ax, (xd + bw * 0.5, ys[i + 1] + hgt), (xd + bw * 0.5, ys[i]), "conv. transposée 2³", (0.075, 0))
        else:
            arrow(ax, (xe + bw, ys[i + 1] + hgt * 0.5), (xd + bw * 0.5, ys[i]), "conv. transposée 2³ (128 → 64)", (0.12, -0.02), rad=0.25)
box(ax, 0.01, 0.76, 0.12, 0.1, "Entrée", "Re, Im : 2 canaux", C['data'], tsize=8.5, bsize=7.3)
arrow(ax, (0.13, 0.805), (xe, ys[0] + 0.065))
box(ax, 0.79, 0.74, 0.17, 0.13, "Tête 1 × 1 × 1", "étage 1 : 1 canal\n(centres)\nétage 2 : 3 canaux\n(self, autres, centres)", C['out'], tsize=8.5, bsize=7.2)
arrow(ax, (xd + bw, ys[0] + 0.065), (0.79, ys[0] + 0.065))
ax.text(0.5, 0.985, "U-Net 3D résiduel (les deux étages : même architecture)", ha='center', va='top', fontsize=13, fontweight='bold')
ax.text(0.79, 0.4, "entrée normalisée : fixe, MAD\nou écart-type (hologramme / film)\nbase 16 canaux, 3 niveaux\n1,42 M paramètres\nnoyaux 3³ isotropes\nBatchNorm (repliée dans\nla convolution à l'inférence)\nLeakyReLU 0,1, fp16\nbiais de sortie −4 (objets rares)\n\nétage 1 à l'inférence : tuiles\n208 × 320 × 320 (halo 32)\nétage 2 : lots de 64 fenêtres", fontsize=8.3, color=INK,
        va='center', ha='left', bbox=dict(boxstyle='round,pad=0.5', fc='#f6f8fa', ec='#d0d7de'))
fig.savefig(os.path.join(OUT, '3_unet3d.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. bloc résiduel
fig, ax = canvas(11, 2.8)
xs = [0.02, 0.165, 0.31, 0.455, 0.6, 0.745, 0.87]
labs = [("x", "C_in canaux"), ("Conv 3³", "C_in → C_out"), ("BatchNorm", "+ LeakyReLU"), ("Conv 3³", "C_out → C_out"),
        ("BatchNorm", ""), ("+", "somme"), ("LeakyReLU", "sortie")]
for (t, b), x in zip(labs, xs):
    ax.add_patch(FancyBboxPatch((x, 0.5), 0.11, 0.24, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=C['net'] if t not in ('x', '+') else '#ffffff', ec=C['edge'], lw=1.0))
    ax.text(x + 0.055, 0.645, t, ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
    ax.text(x + 0.055, 0.565, b, ha='center', va='center', fontsize=7.5, color=MUTED)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + 0.11, 0.62), (b, 0.62))
ax.add_patch(FancyArrowPatch((0.075, 0.5), (0.8, 0.5), arrowstyle='-|>', mutation_scale=11, lw=1.2, color='#2a78d6',
                             connectionstyle="arc3,rad=0.18"))
ax.text(0.44, 0.13, "raccourci : identité si C_in = C_out, sinon conv. 1³", fontsize=8.5, color='#2a78d6', ha='center')
ax.text(0.5, 0.96, "Bloc résiduel (ResBlock)", ha='center', va='top', fontsize=12, fontweight='bold')
fig.savefig(os.path.join(OUT, '4_resblock.png'), dpi=200)
plt.close(fig)
print('ok')
