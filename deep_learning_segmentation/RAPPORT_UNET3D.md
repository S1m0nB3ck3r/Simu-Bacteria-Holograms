# U-Net 3D : détection et segmentation d'instance de bactéries en holographie inline

Rapport du 2026-10-18. Bactérie : bâtonnet transparent de 1,8 × 0,6 µm (± 10 %). Machine : Quadro RTX 3000, 6 Go.
Structure du répertoire : [README.md](README.md) ; schémas du pipeline et du réseau : [PIPELINE_UNET3D.pdf](PIPELINE_UNET3D.pdf) ;
figures : [RAPPORT_UNET3D_figures/](RAPPORT_UNET3D_figures/).

## 1. Résumé

1. **La chaîne retenue a deux étages, entraînés sur des hologrammes 100 % simulés, sans fond** (fond plat + bruit
   capteur) : `models/stage1` (positions 3D) et `models/stage2` (masque fin de chaque bactérie, deux tours).
2. **Sur le film réel**, l'étage 1 retrouve **93 %** des bactéries suivies par HOLOTRACKER, au seuil choisi en
   validation. Les mêmes réseaux entraînés sur fonds réels hybrides n'en retrouvaient que 34 %. Sur ses
   100 détections par image, 46 correspondent aux trajectoires, et 34 sont très probablement de vraies bactéries
   non suivies. Les vraies fausses détections sont au plus une vingtaine par image.
3. **En simulation** (200 µm, N = 50 à 400, 30 % de bactéries collées), l'étage 1 obtient un F1 de 0,938, contre
   0,826 pour la méthode classique. Les bactéries côte à côte sont séparées dès 1,5 µm (classique : 3 µm), mais
   l'une au-dessus de l'autre seulement à partir de 6 µm (classique : 4 µm).
4. **Segmentation** : le second tour fait passer le rappel de 0,89 à 0,91. Position affinée à 0,12 µm en XY et
   0,21 µm en Z, axe à 11° près. L'IoU voxel par voxel plafonne vers 0,53, même sur les vraies positions, et la
   taille (± 10 %) n'est pas mesurable : la forme fine est sous la résolution optique.
5. **Simulateur corrigé** : longueur bout à bout du bâtonnet égale à L (elle valait L + T/2), Δn recalibré à
   0,046, masque d'instance exporté par le simulateur lui-même.
6. **Dynamique** du film réel : 40 % de nageuses, à 0,86 µm/image en ligne presque droite ; 60 % de bactéries
   immobiles qui diffusent. Elle sert à générer des films 100 % simulés en mouvement, avec l'identité de chaque
   bactérie, à la cadence conventionnelle de 100 Hz.
7. **Temps** (1024 × 1024, fp16), pour 200 µm :
   - **étage 1** : 1,23 s ;
   - **étage 2** : de 0,08 s (N = 50) à 0,54 s (N = 400) ;
   - **méthode classique** : 2,6 s pour 400 plans.

   À 100 µm, l'étage 1 descend à 0,29 s avec un pas Z de 2 µm.

## 2. Ancien code (supprimé le 2026-10-18 ; dans le dernier commit git et dans `archive/`)

Défauts relevés, par gravité :
1. cibles retournées en Z : masque enregistré après le flip Z ;
2. optique fausse : pixel objet de 3,4 nm, longueur d'onde du vide ; aucun objet n'est remis au point ;
3. hologramme brut, et normalisation différente à l'entraînement et au test ;
4. trois scripts qui plantent : volumes alloués en (X, Y, Z) au lieu de (Z, Y, X) ;
5. fuite entre entraînement et test ;
6. 40 plans sur 200 jamais vus, et des patchs tirés sans biais vers les positifs ;
7. masque exact comme cible, sous la résolution, sans métrique objet ;
8. BatchNorm sur des lots de 1–2 patchs.

## 3. Simulateur : corrections, calibration, masques

| Point | Avant | Après | Vérification |
|---|---|---|---|
| Longueur bout à bout | L + T/2 | **L** (segment L − T) | 1,80 µm pour L = 1,8 µm ; volume −16 % |
| Δn | 0,043 | **0,046** | amplitude réelle / simulée = 1,07 avec 0,043 (1451 bactéries, 30 images) |
| Pas BPM | 0,5 µm | 0,25 µm | énergie −3,5 % ; 0,125 µm identique |
| Taille | fixe | ± 10 % par bactérie | |
| Bactéries rapprochées | jamais | 30 % à 0,6–4 µm d'une autre | |
| Masque | aucun | **masque d'instance du simulateur** (même noyau, même grille que la BPM) | union = masque traversé par la BPM, à 1 voxel d'arrondi près sur 24 000 |

Le code d'apprentissage ne connaît pas la forme des objets : il lit le masque écrit par le simulateur. Déduire
les instances à l'entraînement, à partir du masque binaire et des positions, a été testé et écarté : 8 à 12 % des
voxels sont mal attribués quand deux bactéries se touchent.

## 4. La chaîne

Le détail des étapes est schématisé dans [PIPELINE_UNET3D.pdf](PIPELINE_UNET3D.pdf).

- **Reconstruction** : spectre angulaire, filtre 15–250 px, décimation ×2 par le spectre (toute la bande est
  conservée) ; volume grossier de 208 plans à 1 µm, voxel 0,35 × 0,35 × 1 µm, Re et Im en unités de bruit capteur.
- **Étage 1** : U-Net 3D résiduel (base 16, 3 niveaux, 1,42 M paramètres), qui produit une carte de chaleur des
  centres (loss focale CenterNet). Inférence par tuiles de 256 px avec 32 px de halo, en fp16, BatchNorm repliée.
  Maxima locaux, puis positions.
- **Étage 2** : même architecture, sur des fenêtres de 32 × 40 × 40 voxels (0,175 × 0,175 × 0,25 µm, la grille
  du masque du simulateur), refocalisées localement à partir du volume grossier (exact à moins de 1 % près).
  Trois sorties : la bactérie de la fenêtre, les autres, et les centres. Deux tours : les centres vus et non
  segmentés deviennent de nouvelles fenêtres. Les doublons sont retirés par recouvrement des masques.
- **Données d'entraînement** (`data/seg_sim`) : 240 / 24 / 40 hologrammes et 48 de paires, 200 µm, N = 20 à 400,
  Δn de 0,032 à 0,060, fond plat + bruit capteur σ = 0,015.

## 5. Modèles retenus (entraînement sans fond)

### 5.1 Détection (étage 1), test simulé, 40 hologrammes

| | N = 50 | 100 | 200 | 300 | 400 | Tous : rappel | précision | couchées | σxy | σz |
|---|---|---|---|---|---|---|---|---|---|---|
| **U-Net, étage 1** | 0,945 | 0,951 | 0,941 | 0,939 | 0,931 | 0,890 | 0,992 | 0,858 | 0,11 µm | 0,31 µm |
| Classique (400 plans) | 0,803 | 0,798 | 0,816 | 0,837 | 0,834 | 0,777 | 0,881 | 0,738 | 0,22 µm | 0,55 µm |

Les cinq premières colonnes donnent le F1 selon le nombre de bactéries. Les manques de l'étage 1 sont surtout
des bactéries collées, que la carte des centres fusionne ; l'étage 2 en rattrape une partie (5.3).

| Paires détectées (2 détections) | 1 µm | 1,5 µm | 2 µm | 3 µm | 4 µm | 6 µm | 8 µm |
|---|---|---|---|---|---|---|---|
| Latérales : U-Net / classique | 0,10 / 0 | **0,98 / 0** | **1,00 / 0** | 1 / 1 | 1 / 1 | 1 / 1 | — |
| Axiales : U-Net / classique | 0,02 / 0 | — | 0 / 0 | 0,02 / 0,35 | 0,58 / **0,90** | 0,90 / 1 | 0,94 / 1 |

Figures : [detection_sans_fond_200um.png](RAPPORT_UNET3D_figures/detection_sans_fond_200um.png),
[paires_sans_fond.png](RAPPORT_UNET3D_figures/paires_sans_fond.png).

### 5.2 Film réel (images 85–99)

La seule vérité disponible est formée des points de trajectoire d'HOLOTRACKER : 49 par image sur 0–100 µm. Elle
est incomplète, et biaisée en faveur du classique, qui a servi à la produire.

| Détections par image de l'étage 1 (seuil de validation 0,4) | Nombre |
|---|---|
| correspondent à une trajectoire (**rappel 0,934**) | 46,0 |
| au-delà de 100 µm, hors de la plage reconstruite par HOLOTRACKER | 14,6 |
| sur un objet non suivi d'HOLOTRACKER, de la taille d'une bactérie (1452 voxels en médiane) | 19,8 |
| doublons à moins de 2 µm d'une détection juste | 6,3 |
| inexpliquées | 13,4 |
| **total** | 100,1 |

Le classique trouve 192 maxima par image. Image annotée :
[film_reel_sans_fond.jpg](RAPPORT_UNET3D_figures/film_reel_sans_fond.jpg).

| Même réseau (CenterNet) entraîné sur… | Rappel au seuil de validation |
|---|---|
| fonds réels hybrides | 0,34 |
| fond simulé plat (100 µm, ancienne géométrie) | 0,66 |
| **fond plat, géométrie corrigée, 200 µm, bactéries rapprochées (modèle retenu)** | **0,93** |

Sur fonds hybrides, les vraies bactéries obtenaient des scores plus bas que les simulées. Hypothèse la plus
probable : les restes de vraies bactéries mal nettoyées dans ces fonds étaient appris comme du « fond ».
**Entraîner sans fond règle l'essentiel du problème.** Les doublons suggèrent que les vraies bactéries sont un peu
plus grandes ou plus contrastées que les simulées.

### 5.3 Segmentation d'instance (chaîne complète, test simulé)

| | Rappel | Précision | F1 | PQ | Bactéries IoU > 0,5 | IoU des appariées |
|---|---|---|---|---|---|---|
| Étage 1 seul | 0,890 | 0,992 | 0,938 | — | — | — |
| Chaîne, 1 tour | 0,889 | 0,993 | 0,938 | 0,361 | 0,525 | 0,534 |
| **Chaîne, 2 tours** | **0,911** | 0,990 | **0,949** | 0,360 | 0,531 | 0,531 |
| Étage 2 sur les vraies positions | 0,980 | 0,997 | 0,989 | 0,376 | 0,574 | 0,535 |

- **Position** : le barycentre du masque donne 0,12 µm en XY et 0,21 µm en Z, contre 0,15 et 0,31 µm pour
  l'étage 1.
- **Axe** : 11° en médiane.
- **Taille** : erreur médiane de 0,17 µm sur la longueur et 0,07 µm sur l'épaisseur. Mais la corrélation avec la
  vraie longueur n'est que de 0,24, contre 0,81 pour le même estimateur sur les masques vrais : le réseau dessine
  une bactérie type.

| Voisine la plus proche | < 1 µm | 1–2 µm | 2–4 µm | 4–8 µm | > 8 µm |
|---|---|---|---|---|---|
| Segmentées (IoU > 0,5) | 0,09 | 0,35 | 0,57 | 0,58 | 0,65 |

| Paires segmentées (2 IoU > 0,5) | 1 µm | 1,5 µm | 2 µm | 3 µm | 4 µm | 6 µm | 8 µm |
|---|---|---|---|---|---|---|---|
| Latérales | 0,19 | 0,94 | 0,90 | 1,00 | 0,98 | 0,96 | — |
| Axiales | 0 | — | 0 | 0,19 | 0,13 | 0,35 | 0,44 |

Une IoU de 0,55 à 0,65 correspond déjà à un recouvrement visuellement très bon : une bactérie compte environ 60
voxels, et un voxel d'écart sur le contour divise l'IoU par deux. Coupes XY, XZ, YZ :
[segmentation_exemples_3_plans.png](RAPPORT_UNET3D_figures/segmentation_exemples_3_plans.png). Autres figures :
[segmentation_strates_sans_fond.png](RAPPORT_UNET3D_figures/segmentation_strates_sans_fond.png),
[segmentation_paires_sans_fond.png](RAPPORT_UNET3D_figures/segmentation_paires_sans_fond.png),
[segmentation_forme_sans_fond.png](RAPPORT_UNET3D_figures/segmentation_forme_sans_fond.png).

### 5.4 Temps (hologramme 1024 × 1024, fp16, GPU libre, médianes)

| | Reconstruction | Réseau / focus | Post-traitement | Total |
|---|---|---|---|---|
| Étage 1, 200 µm (208 plans) | 31 ms | 1377 ms | 39 ms | 1447 ms |
| Classique, 200 µm (400 plans) | 195 ms | 510 ms | 1906 ms | 2611 ms |

| Étage 2 (deux tours) | N = 50 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| Temps | 77 ms | 146 ms | 281 ms | 403 ms | 542 ms |

À 100 µm (mesures de la section 6.1, ancienne géométrie) : étage 1 678 ms au pas Z de 1 µm, **294 ms au pas de
2 µm**, 170 ms avec un réseau 4× plus petit ; classique 953 ms. fp32 inutilisable : la mémoire déborde, et un
hologramme prend 15 à 77 s.

## 6. Expériences préliminaires (ancienne géométrie, données et modèles supprimés ; scripts dans `experiences/`)

### 6.1 Variantes de l'étage 1 (100 µm, fonds hybrides, N ≤ 200)

| Variante | F1 | Couchées | σxy | σz | Inférence |
|---|---|---|---|---|---|
| **CenterNet** | **0,993** | 0,989 | **0,061** | 0,169 | 678 ms |
| MSE pondérée | 0,993 | 0,990 | 0,097 | 0,218 | 680 ms |
| BCE + Dice | 0,990 | 0,989 | 0,078 | 0,185 | 663 ms |
| Focale + Tversky | 0,994 | 0,988 | 0,079 | 0,165 | 666 ms |
| + MAD local | 0,991 | 0,979 | 0,059 | 0,159 | 697 ms |
| + tête d'orientation (axe : 15° en médiane) | 0,993 | 0,989 | 0,062 | 0,173 | 720 ms |
| Pas Z de 2 µm | 0,992 | 0,986 | 0,067 | 0,283 | 294 ms |
| Pas Z de 2 µm, 4× plus petit | 0,989 | 0,981 | 0,074 | 0,309 | 170 ms |
| Classique | 0,801 | 0,993 | 0,112 | 0,194 | 953 ms |

Les quatre loss se valent ; CenterNet est retenue pour sa précision en XY. Le MAD local n'apporte rien. Figures :
[detection_hybride_100um.png](RAPPORT_UNET3D_figures/detection_hybride_100um.png),
[paires_etage1.png](RAPPORT_UNET3D_figures/paires_etage1.png).

### 6.2 Segmentation entraînée sur fonds hybrides

Mêmes conclusions qu'en 5.3, avec un étage 1 plus faible (rappel 0,83) : le second tour apporte +7 points de
rappel, et l'IoU plafonne à 0,53.

## 7. Dynamique des bactéries et films simulés

Analyse ([stats/stat_report_trajectories.py](../stats/stat_report_trajectories.py), [TRAJECTORY_REPORT/](../TRAJECTORY_REPORT/))
des 57 trajectoires HOLOTRACKER du film (5108 points). Les estimateurs sont robustes au bruit de localisation :
MSD par composante, vitesse lissée, persistance de direction. Unité : l'image. Cadence conventionnelle : 100 Hz,
la même pour les films simulés, donc sans importance.

| | Nageuses : 23 (40 %) | Non nageuses : 34 (60 %) |
|---|---|---|
| Vitesse | 0,86 µm/image (de 0,50 à 1,18) | aucune dérive |
| Diffusion | — | 0,009 µm²/image en XY |
| Direction | persistance cos θ = 0,73 après 30 images | sans objet |
| Inclinaison de la nage | surtout horizontale (environ 70° de l'axe Z) | sans objet |
| Rectitude (déplacement net / chemin) | 0,76 | 0,15 |

Les virages de plus de 120° (0,012 par image) sont incompatibles avec cette persistance : ce sont des artefacts
de localisation, pas des demi-tours. Le film est peu dense : la voisine suivie la plus proche est en médiane à
21 µm. Figure : [dynamique_film_reel.png](RAPPORT_UNET3D_figures/dynamique_film_reel.png).

`generate_sequences.py` reproduit ce modèle :
- **nageuses** : vitesse et inclinaison tirées parmi les vraies, diffusion de rotation 0,0052 rad²/image ;
- **non nageuses** : diffusion de translation et de rotation ;
- **bords** : une bactérie qui sort du champ en x, y rentre de l'autre côté sous une nouvelle identité ; parois
  réfléchissantes en z.

Validation (même analyse, bruit de localisation d'HOLOTRACKER ajouté) :

| | Réel | Simulé |
|---|---|---|
| Part de nageuses | 0,40 | 0,42 |
| Vitesse en XY des nageuses (µm/image) | 0,65 | 0,62 |
| Rectitude des nageuses | 0,76 | 0,75 |
| Persistance à 5 images | 0,38 | 0,38 |
| Vitesse en Z des nageuses (µm/image) | 0,24 | 0,35 |
| Persistance à 30 images | 0,40 | 0,26 |

`data/seq` : 14 films (480 images), fond plat + bruit, masques d'instance et identités, trajectoires vraies au
format d'HOLOTRACKER.

## 8. Limites et suite

1. **Écart avec le réel résiduel** : doublons, et 13 détections inexpliquées par image. Pistes : élargir la
   variabilité simulée (taille, Δn, flou) ; caler ou affiner le modèle sur le film réel, avec les trajectoires
   d'HOLOTRACKER comme pseudo-étiquettes.
2. **Paires axiales** : la carte des centres fusionne jusqu'à 4 µm. Pistes : une cible de centre plus étroite en Z,
   ou un pas Z plus fin à l'étage 1.
3. **A priori temporel** : `data/seq` permet maintenant de l'évaluer. La version simple (centres de l'image
   précédente ajoutés aux candidats) est dans `GUI_inference.py` ; la version CenterTrack (carte précédente en
   entrée, déplacement appris) donnerait aussi le suivi.
4. **Taille des bactéries** : non mesurable à ± 10 % à cette résolution.
5. **Temps** : l'étage 1 à 200 µm (1,23 s) domine. Le pas Z de 2 µm le diviserait environ par 2,3, au prix de la
   précision en Z.
