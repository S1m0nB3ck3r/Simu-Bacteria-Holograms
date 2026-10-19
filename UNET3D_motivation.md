# Pourquoi un U-Net 3D — justification mesurée

Ce document explique **pourquoi** la segmentation volumétrique par réseau de neurones
est la bonne réponse au problème de l'holographie inline en milieu dense, en s'appuyant
sur des mesures faites sur le simulateur de ce dépôt, et non sur des arguments généraux.
Les mesures du 2026-09-04 utilisaient une bactérie de 3 × 1 µm, trop grosse ; elles ont été
refaites le 2026-10-13 avec des bactéries réalistes (1,5–2 × 0,8 µm), et ce sont ces valeurs qui
figurent ci-dessous (détail : `CLAUDE_HOLOTRACKER_TO_DO.md` § 1). **Les effets d'orientation et de
densité sont nettement plus faibles qu'annoncé en 2026-09** ; la section 2 en tient compte.

Il traite aussi la question qui décide de l'utilisabilité réelle : **le pipeline
classique segmente un hologramme 1024×1024 en ≈ 400 ms. Le réseau peut-il tenir dans
cet ordre de grandeur ?**

---

## 1. Le vrai mécanisme du problème

L'holographie inline est un problème mal posé, mais il faut être précis sur la cause,
parce que le remède en dépend.

### Ce que ce n'est pas

**Ce n'est pas l'image jumelle.** Si elle était en cause, un montage hors-axe la
supprimerait et le problème disparaîtrait. Or les deux mécanismes réellement mesurés
ci-dessous survivraient tous les deux à un passage en hors-axe.

**Ce n'est pas la diffusion multiple.** À Δn = 0,005 (indice 1,335 contre 1,33), le
déphasage à la traversée d'une bactérie de 0,8 µm vaut 0,038 rad, soit 2,2°. Il faudrait
**83 bactéries parfaitement alignées** sur un même rayon pour accumuler π. Les objets
sont des objets de phase très faibles ; BPM et Born du premier ordre y donnent presque
le même résultat.

### Ce que c'est — deux effets, tous deux mesurés

**(a) Le signal dépend de l'orientation d'un facteur 1,5.**

Le déphasage vaut `2π·Δn·(trajet dans l'objet)/λ`. Un bâtonnet de 1,75 µm aligné sur l'axe
optique présente 1,75 µm de matière au rayon qui le traverse ; couché, seulement 0,8 µm.

| φ | pic du focus TENEGRAD |
|---:|---:|
| 0° (parallèle à l'illumination) | 0,066 |
| 40° | 0,051 |
| 90° (couchée) | 0,045 |

Effet **projectif**, monotone, indépendant de la géométrie du montage.

**(b) La réponse est 2,5 fois plus large en Z qu'en XY, et deux objets proches
s'additionnent de façon cohérente.**

Largeurs à mi-hauteur (filtre 15–250) : X 2,1 µm, Y 2,2 µm, **Z 5,3 µm**.

Deux bactéries empilées selon z, pic du focus rapporté au cas d'une seule :

| écart | latéral (X) | axial (Z) |
|---:|---:|---:|
| 20 µm | 1,05× | 1,08× |
| 8 µm | 1,08× | 1,18× |
| 4 µm | 1,03× | 1,35× |
| 2 µm | 1,03× | 1,65× |
| 1 µm | 1,05× | **1,81×** |

Selon Z, croissance monotone jusqu'à 1,81× : une tache unique, 80 % plus brillante.
C'est la « grosse patate ». Latéralement, rien. **Addition cohérente d'amplitudes**,
strictement linéaire.

---

## 2. Pourquoi un pipeline à seuil ne peut pas y arriver

### 2.1 Le seuillage amplifie l'écart d'orientation

À seuil global figé (moyenne + 10σ, calibré sur la bactérie couchée) :

| φ | pic / seuil | nb voxels | plus grosse composante |
|---:|---:|---:|---:|
| 0° | 1,86× | 1 324 | 1 263 |
| 90° | 1,26× | 262 | 237 |

**×1,47 sur le pic → ×5,1 sur les voxels → ×5,3 sur la composante.** Le nombre de voxels reste
un descripteur d'orientation. Surtout, **une bactérie couchée isolée n'est qu'à 1,26× le seuil** :
elle est déjà à la limite de détection avant toute question de densité.

### 2.2 Le seuil global se couple à la densité, mais modérément

Une bactérie témoin couchée, toujours au même endroit, charge croissante autour (3 tirages) :

| nb bactéries | seuil 10 σ | pic du témoin | détecté |
|---:|---:|---:|:--|
| 1 | 0,029 | 0,033 | 3/3 |
| 41 | 0,032 | 0,035 | 3/3 |
| 101 | 0,035 | 0,032 | **0/3** |
| 201 | 0,040 | 0,035 | **0/3** |

Le seuil est multiplié par 1,37 entre 1 et 201 bactéries (par 7,3 entre 1 et 41 avec la bactérie
de 3 × 1 µm de 2026-09). Le témoin n'est perdu qu'au-delà d'une centaine de bactéries. Médiane +
15·MAD ne fait pas mieux : il rate le témoin même seul. En benchmark dense (5 à 200 bactéries,
K au F1 maximal, TENEGRAD), le rappel à N = 200 vaut 0,79 avec le seuil global, 0,83 avec le seuil
local et 0,89 avec les maxima locaux ; avec SUM_OF_VARIANCE, 0,90 à 0,92 pour les trois.

**Conséquence pour l'argumentaire.** L'argument « aucun seuil global ne peut fonctionner en
milieu dense » était surévalué : un pipeline classique bien réglé (SUM_OF_VARIANCE ou maxima
locaux) retrouve environ 90 % des bactéries à N = 200. Ce qui reste au réseau, ce sont les 10 %
manquants, surtout des bactéries couchées (15 à 20 % d'entre elles sont perdues), et les fusions
axiales (b), qui ne dépendent pas de la taille des bactéries. Le gain du U-Net est donc à démontrer
par la mesure sur ces cas, pas à postuler.

> **Conclusion : aucune décision globale ne peut trancher.** La bonne décision dépend du
> contexte local, et l'information nécessaire est locale. C'est exactement la classe de
> problème que résout une convolution apprise.

---

## 3. Ce que le réseau doit apprendre

Un CNN ne fixe pas de seuil : il produit une décision **locale et contextuelle**. Les
trois modes d'échec mesurés sont précisément ce qu'il doit absorber :

1. **Amplitude dépendante de l'orientation** — apprendre qu'une réponse faible et
   allongée dans une zone calme est une bactérie couchée, pas du bruit.
2. **Fusion axiale** — apprendre qu'une tache 1,8× trop brillante et trop étendue en z
   contient deux objets, en s'appuyant sur la forme de la réponse et non sur son
   intensité.
3. **Fond variable avec la densité** — normaliser implicitement par le contexte, ce
   qu'aucun seuil global ne peut faire.

Le réseau ne « devine » pas : il exploite un a priori appris sur ce qui est
physiquement plausible, pour régulariser une inversion mal posée. C'est légitime **à
condition que le jeu d'entraînement contienne les cas difficiles**.

---

## 4. Ce que le dataset doit contenir

| exigence | état actuel | à corriger |
|---|---|---|
| orientations isotropes | `phi = 90·rand()`, `theta = 90·rand()` | `phi = degrees(arccos(u))`, `theta ∈ [0°, 360°)` |
| densités variées | `nb_objects` fixe | balayer la densité |
| paires à écart axial imposé | absent | générer 1, 2, 4, 8 µm en Z — l'axe où la fusion se produit |
| patchs contenant des objets | tirage uniforme | échantillonnage biaisé positifs |

Le premier point est le plus insidieux : un tirage **uniforme** en φ sur-représente les
bactéries proches de l'axe optique, c'est-à-dire les 1,5× plus brillantes. Pour une
orientation isotrope, φ doit suivre `p(φ) ∝ sin φ`. En l'état, le réseau s'entraînerait
sur une population plus facile que la réalité et sous-représenterait exactement le cas
que le pipeline classique rate déjà.

---

## 5. Faisabilité en temps de calcul

**Contrainte : le pipeline classique fait 1024×1024 en ≈ 400 ms. L'objectif est de
rester dans cet ordre de grandeur, au pire 10×.**

Mesures sur **Quadro RTX 3000, 6 Go**, U-Net 3D du dépôt (5,65 M paramètres) :

| élément | fp32 | fp16 (AMP) |
|---|---:|---:|
| patch 64×128×128 (1,05 Mvoxel) | 67,3 ms | 23,4 ms |
| patch 64×256×256 (4,19 Mvoxel) | 270,2 ms | 93,5 ms |

Le coût est **linéaire en nombre de voxels** (4,0× de voxels → 4,0× de temps) :
≈ **64 ms par million de voxels en fp32, 22 ms en fp16**.

Rétro-propagation seule, 1024×1024 : 111 ms pour 200 plans, 53 ms pour 100, 24,5 ms
pour 50.

### Budget total, patchs sans recouvrement

| configuration | voxels | temps réseau | + reconstruction | vs 400 ms |
|---|---:|---:|---:|---:|
| 1024×1024×200, fp32 | 210 M | 17,2 s | 17,3 s | **43×** |
| 1024×1024×200, fp16 | 210 M | 6,0 s | 6,1 s | **15×** |
| 1024×1024×50, fp16 | 52 M | 1,5 s | 1,53 s | **3,8×** |
| 512×512×50, fp16 *(estimé)* | 13 M | ≈ 290 ms | ≈ 300 ms | **0,75×** |

### Le levier décisif : le volume est massivement sur-échantillonné

La largeur axiale mesurée de la réponse est de **5,3 µm** (bactérie réaliste, 2026-10-13 ;
12 µm avec l'ancienne bactérie de 3 × 1 µm), pour un pas Z de **0,5 µm** — un facteur **11** de
sur-échantillonnage. Passer à 2 µm (50 plans) garde 2,7 plans par largeur à mi-hauteur : peu
d'information perdue, un peu de précision de centroïde. **Gain immédiat : ×4.** La marge est
plus mince que ne le laissait croire la mesure de 2026-09 : à vérifier sur la précision en Z.

Latéralement, le voxel objet fait 0,1375 µm pour une résolution optique de l'ordre de
0,6 µm : une décimation par 2 reste à ~2× Nyquist. **Gain supplémentaire : ×4**, au prix
d'une précision de localisation latérale réduite.

### Verdict

**L'objectif est atteignable, mais pas avec la configuration actuelle.**

- En l'état (fp32, 200 plans, patchs 128×128×64 avec recouvrement) : **plus de 40× le
  budget**. Hors-jeu.
- Avec fp16 + pas Z à 2 µm + patchs jointifs : **1,5 s, soit 3,8×**. Dans la tolérance
  « pas 10 fois pire ».
- En ajoutant une décimation latérale par 2 : **≈ 300 ms**, sous le budget.

Leviers non encore exploités : export TensorRT (typiquement 1,5 à 2× de plus sur du
fp16), traitement des patchs par lots plutôt qu'un par un, et réduction du nombre de
canaux si la précision le permet. La VRAM de 6 Go impose de rester en traitement par
patchs — le volume 1024×1024×200 en float32 pèse à lui seul 800 Mo.

Le compromis se joue donc entre **précision de localisation** et **débit**, et il se
règle par le pas d'échantillonnage, pas par l'architecture. Ce n'est pas perdu d'avance.

---

## 6. Où sont les mesures

Toutes les valeurs de ce document viennent de scripts exécutés sur la chaîne du dépôt
(`libs/propagation.py`, `libs/focus.py`, `libs/CCL3D.py`, `deep_learning_segmentation/model.py`),
sans modification du code. Le protocole est décrit dans `HOLOTRACKER_TO_DO.md`, section 4,
avec les écueils rencontrés — notamment le fait qu'une analyse de profil 1D ne permet
pas de mesurer la distance de fusion, les franges de diffraction produisant une dizaine
de maxima parasites.
