# Détection des bactéries en holographie inline : orientation, seuil et critère de focus

Rapport du 2026-10-12, **refait le 2026-10-13 avec des bactéries de taille réaliste**. La première
version utilisait des bactéries de 3–4 µm de long et 1–2 µm d'épaisseur, deux fois trop grosses.
Les mécanismes décrits restent vrais, mais leur ampleur change beaucoup, et certaines conclusions
s'inversent (§ 3).

Toutes les mesures sont faites sur des hologrammes simulés par le simulateur BPM du projet. Le
bruit est uniquement celui du capteur : gaussien, ajouté à l'intensité après propagation, avec un
écart-type de 1,5 % du fond, mesuré sur `REAL HOLO FILM`. La chaîne de détection est celle
d'HOLOTRACKER (propagation par spectre angulaire, critère de focus, seuil, composantes connexes),
sauf mention contraire.

Géométrie des benchmarks : 1024 × 1024 px, pixel caméra de 7 µm, grossissement ×40 (0,175 µm
par pixel), 200 plans de 0,5 µm, premier plan à 10 µm de l'hologramme, filtre spectral 15–250 px,
nettoyage par division. Les bactéries mesurent **1,5 à 2 µm de long et 0,8 µm d'épaisseur**, avec
n = 1,335 dans l'eau (n = 1,33). Les orientations sont aléatoires : φ, l'angle avec l'axe optique,
uniforme dans [0°, 90°] ; θ, l'azimut, uniforme dans [0°, 360°). Une détection est appariée à une
bactérie simulée si elle tombe à moins de 3 µm en XY et 5 µm en Z. Les benchmarks denses utilisent
les hologrammes de `FOCUS_REPORT/DENSE/holograms` (5, 10, 50, 100 et 200 bactéries, 3 hologrammes
par densité) ; pour chaque méthode, K est celui qui maximise F1. Les écarts de 2 à 3 points ne
sont pas significatifs.

---

## 1. L'orientation, et le seuil global

### 1.1 Une bactérie couchée donne moins de signal

Le déphasage vaut 2π·Δn·(trajet dans l'objet)/λ. Alignée sur l'axe optique, une bactérie de
1,75 × 0,8 µm présente toute sa longueur ; couchée, seulement son épaisseur. C'est un effet
projectif : un montage hors-axe n'y changerait rien. Mesure du 2026-10-13
(`CLAUDE_HOLOTRACKER_TO_DO.md` § 1.1) : **×1,47 sur le pic de focus TENEGRAD** entre une bactérie
debout et une bactérie couchée (×3,6 avec l'ancienne bactérie de 3 × 1 µm).

Avec un seuil figé, cet écart devient ×5,1 sur le nombre de voxels et ×5,3 sur la plus grosse
composante connexe (§ 1.2 du même document). **Le nombre de voxels (`nb_pix`) mesure donc
l'orientation autant que l'objet**, et un filtre sur la taille (`CCL_filter`) élimine
préférentiellement les bactéries couchées. Surtout, une bactérie couchée isolée n'est qu'à
**1,26× un seuil de moyenne + 10σ** : elle est déjà à la limite de détection, sans voisine.

### 1.2 Le seuil global dépend du contenu du volume

Le seuil d'HOLOTRACKER vaut moyenne + K·σ du volume de focus entier, objets compris.

- **Il dépend de la densité, modérément.** Pour une bactérie témoin couchée, toujours au même
  endroit, le seuil est multiplié par 1,09 entre 1 et 41 bactéries et par 1,37 jusqu'à 201. Le
  témoin est perdu à partir d'une centaine de bactéries (`CLAUDE_HOLOTRACKER_TO_DO.md` § 1.3).
  Avec l'ancienne bactérie, le seuil était multiplié par 7,3 dès 41 bactéries.
- **Il dépend de la position, même avec un seul objet.** Une sphère seule (r = 1 µm), balayée en
  Z dans un volume de 10 à 210 µm, donne un nombre de voxels segmentés en U : 7 939 à Z = 30 µm,
  3 952 à 110 µm, 5 980 à 200 µm. Le pic de focus de la sphère est pourtant constant (4,02 à
  4,13). C'est le seuil qui varie (0,84 à 1,16) : la réponse défocalisée de la sphère est
  entièrement contenue dans le volume quand elle est au centre, et en partie coupée près des bords.

### 1.3 Type de seuil global

Trois façons de calculer le seuil global, testées sur la même sphère (Z = 30, 110 et 200 µm) :

| Type de seuil | Définition | Variation du seuil | Nombre de voxels |
|---|---|---|---|
| `STD_VAR` | moyenne + K·σ | ±16 % | 3 952 à 7 939 (facteur 2,0) |
| `MAD` | médiane + K·1,4826·MAD | ±9 % | 2 824 à 4 421 (facteur 1,6) |
| `MAX_FOCUS` | K × voxel max | ±1 % | 3 925 à 4 346 (±5 %) |

- **`MAD`** résiste aux voxels très forts des objets, mais pas à leur réponse défocalisée. Avec la
  bactérie témoin réaliste, médiane + 15·MAD est plus haut que moyenne + 10σ, et rate le témoin
  même seul (`CLAUDE_HOLOTRACKER_TO_DO.md` § 1.4) : il faudrait un K propre à cet estimateur, non
  testé.
- **`MAX_FOCUS`** est stable pour un objet seul. En milieu dense, en revanche, il dépend de la
  bactérie la plus brillante du volume, en général une bactérie alignée sur l'axe : il remonte
  alors le seuil pour toutes les autres.

Ces trois types sont disponibles dans les rapports `stats/` (`threshold_type`) et dans
`libs/CCL3D.py` (`threshold_center_scale`).

**Conclusion 1 : le seuil global est biaisé par l'orientation et par la densité, mais avec des
bactéries réalistes l'effet de densité est modéré.** Le fond monte réellement, de 10 à 40 % entre
5 et 200 bactéries selon le critère (`FOCUS_REPORT/DENSE/RAPPORT_DENSE.md` § 3.2). En benchmark
dense, le seuil global ne perd que 4 points de rappel à N = 200 face au seuil local (§ 3). La
limite principale est le faible signal des bactéries couchées, pas le couplage à la densité.

---

## 2. Choix du critère de focus

Les résultats de la première version (bactérie isolée de 3,5 × 1,5 µm ; benchmark dense avec des
bactéries de 3–4 × 1–2 µm, où TENEGRAD, SUM_OF_VARIANCE et SUM_OF_GRADIENT se valaient) sont
retirés. Ils sont remplacés par les deux rapports suivants, faits avec des bactéries réalistes.

### 2.1 Bactérie isolée : erreur de position, bruit et orientation

Détail complet dans **`FOCUS_REPORT/1_bacteria/RAPPORT_FOCUS.md`**. Une bactérie isolée de
1,6 × 0,8 µm à Z = 50 µm, φ de 0° à 90°, 10 tirages de bruit par angle, les 8 critères appliqués
aux mêmes hologrammes, avec une segmentation à mi-hauteur du pic au-dessus du fond
(`PEAK_ABOVE_BG` 0,5) :

| | SUM_OF_VARIANCE | TENEGRAD |
|---|---|---|
| \|err Z\| à φ = 90° | 0,11 µm | 0,29 µm |
| Dispersion Z due au bruit à φ = 90° | ±0,13 µm | ±0,36 µm |
| Contraste de la bactérie couchée | 62σ | 19σ |
| Pic couchée / pic alignée | ÷ 2,9 | ÷ 1,7 |

- **Sensibilité au bruit** : c'est le défaut principal de TENEGRAD. Le pic d'une bactérie couchée
  n'est qu'à 19σ du bruit, et sa position Z varie 3 à 5 fois plus que celle donnée par
  SUM_OF_VARIANCE.
- **Orientation et seuil** : aucun seuil fixe ne garde les bactéries couchées sans faire grossir
  les bactéries alignées. Les critères quadratiques résistent mieux au bruit, mais l'écart entre
  orientations y est plus grand (un carré élève aussi ce rapport au carré).
- **Laplacien** : non détecté au-delà de φ = 40°, son seuil étant fixé par des pics de bruit.
- La première version de cette mesure concluait à un biais de −0,7 µm de TENEGRAD et à un effet
  de la direction de couchage, X ou Y. Ces deux effets n'existent pas : c'étaient des fluctuations
  dues au bruit. Le code d'orientation a été vérifié, et il est symétrique en X et Y.

### 2.2 Milieu dense : la comparaison qui compte

Détail complet dans **`FOCUS_REPORT/DENSE/RAPPORT_DENSE.md`** : 8 critères, seuil local, K au F1
maximal pour chaque critère, 5 à 200 bactéries.

| Critère | Rappel N = 10 / 50 / 200 | Précision N = 200 | Couchées retrouvées, N = 50 / 200 |
|---|---|---:|---|
| SUM_OF_VARIANCE | 0,97 / 0,94 / **0,92** | 0,94 | 0,92 / **0,85** |
| SUM_OF_INTENSITY | 0,97 / 0,95 / **0,93** | 0,92 | 0,92 / **0,90** |
| SUM_OF_GRADIENT | 0,87 / 0,90 / 0,86 | 0,95 | 0,83 / 0,75 |
| TENEGRAD | 0,93 / 0,93 / **0,83** | 0,96 | 0,90 / **0,70** |
| VAR_IMAG_OVER_REAL | 0,27 / 0,67 / 0,55 | 0,59 | 0,58 / 0,48 |
| SUM_OF_LAPLACIAN | 0,37 / 0,45 / 0,48 | 0,74 | 0,26 / 0,25 |

**Conclusion 2 : SUM_OF_VARIANCE pour localiser et pour détecter.** Avec des bactéries réalistes,
le halo des voisines reste faible, et l'avantage de contraste des critères au carré se maintient
en milieu dense. TENEGRAD, le critère actuel d'HOLOTRACKER, est le moins bon des critères
utilisables à forte densité : la bactérie couchée typique est à peine au-dessus du seuil.

### 2.3 Critère sur le champ complexe : var(Im) / var(Re) (ajouté le 2026-10-13)

`VAR_IMAG_OVER_REAL` (`libs/focus.py`) : rapport de la variance locale de la partie imaginaire
à celle de la partie réelle du champ reconstruit, sur une fenêtre de `sum_size` px. Il exige le
champ complexe, avec la porteuse exp(2iπz/λ) retirée plan par plan. Il est calculé plan par plan
pendant la propagation (`propagation.volume_propag_angular_spectrum_to_focus_complex`), sans
stocker le volume complexe. Le principe : au focus d'un objet de phase, t ≈ 1 + iφ, donc la
perturbation est imaginaire ; hors focus, Re et Im se mélangent.

- **Bactérie isolée** (`FOCUS_REPORT/1_bacteria`) : très précis en Z à φ = 0° et 80–90°
  (≤ 0,03 µm), moins bon entre les deux (jusqu'à 0,34 µm à 40°).
- **Milieu dense** (`FOCUS_REPORT/DENSE`) : inutilisable pour la détection. À N = 200, rappel
  0,55 pour une précision de 0,59, avec 75 faux positifs par hologramme.

Le critère reste disponible (`focus_type: VAR_IMAG_OVER_REAL` dans les YAML de `stats/`,
`FOCUS_TYPE` dans les pipelines à seuil local et à maxima locaux). Il pourrait servir à affiner
la position en Z d'objets déjà détectés, pas à détecter.

---

## 3. Seuil global, seuil local, maxima locaux

### 3.1 Les méthodes comparées

- **Seuil global** (`pipeline_holotracker_locate_simple.py`) : moyenne + K·σ du volume entier,
  puis composantes connexes.
- **Seuil local** (`pipeline_holotracker_locate_local_threshold.py`) :
  T(x) = fond local + K·max(σ local, σ plancher). Le fond et σ sont estimés par mailles de
  25 plans × 64 × 64 px (12,5 × 11,2 × 11,2 µm), en écartant les voxels à plus de ±3σ sur
  3 passes, pour que les objets ne fassent pas monter leur propre seuil (méthode de SExtractor).
  Le seuil est ensuite interpolé en trilinéaire, puis les voxels au-dessus sont regroupés en
  composantes connexes.
- **Maxima locaux** (`pipeline_holotracker_locate_local_maxima.py`) : un point par objet,
  là où le focus est le maximum d'une boîte et dépasse le seuil local. La boîte fait
  3,85 µm × 3,85 µm × 12 µm, la largeur de réponse mesurée en 2026-09 avec une bactérie de
  3 × 1 µm ; la réponse d'une bactérie réaliste fait 2,1 × 2,2 × 5,3 µm. La boîte n'a pas été
  réduite ni testée plus petite. La position est affinée sous le voxel, et chaque objet est
  décrit par son pic et son rapport signal sur bruit (RSB) local plutôt que par un nombre de voxels.

### 3.2 Résultats, K au F1 maximal

Critère TENEGRAD (celui d'HOLOTRACKER). K retenu : 10 pour le seuil global (15 à N ≤ 10) et le
seuil local, 8 pour les maxima locaux (10 à N = 5) :

| N | Rappel global / local / maxima | Précision global / local / maxima | Couchées (φ > 60°) retrouvées global / local / maxima |
|---:|---:|---:|---:|
| 10 | 0,80 / 0,93 / 0,97 | 1,00 / 0,88 / 0,97 | 0,67 / 0,87 / 0,93 |
| 50 | 0,93 / 0,93 / 0,96 | 0,87 / 0,90 / 0,96 | 0,88 / 0,90 / 0,92 |
| 100 | 0,89 / 0,89 / 0,93 | 0,95 / 0,94 / 0,97 | 0,79 / 0,79 / 0,86 |
| 200 | 0,79 / 0,83 / **0,89** | 0,98 / 0,96 / **0,99** | 0,65 / 0,70 / **0,81** |

Critère SUM_OF_VARIANCE. K retenu : 10 à 20 selon la méthode et la densité :

| N | Rappel global / local / maxima | Précision global / local / maxima | Couchées retrouvées global / local / maxima |
|---:|---:|---:|---:|
| 10 | 0,93 / 0,97 / 0,97 | 1,00 / 0,91 / 1,00 | 0,87 / 0,93 / 0,93 |
| 50 | 0,93 / 0,94 / 0,97 | 0,95 / 0,93 / 0,99 | 0,88 / 0,92 / 0,94 |
| 100 | 0,90 / 0,91 / 0,94 | 0,97 / 0,98 / 1,00 | 0,79 / 0,82 / 0,89 |
| 200 | 0,90 / 0,92 / 0,90 | 0,92 / 0,94 / **1,00** | 0,81 / 0,85 / 0,83 |

(À N = 5, il n'y a qu'une bactérie couchée sur 15 ; la ligne est omise.)

- **Le seuil global ne s'effondre plus.** Avec des bactéries réalistes, il ne perd que 4 points
  face au seuil local à N = 200 avec TENEGRAD, 2 avec SUM_OF_VARIANCE. La première version
  (bactéries de 3–4 × 1–2 µm) donnait 38 % de rappel au seuil global contre 62 % au seuil local.
- **Les maxima locaux sont désormais la meilleure méthode avec TENEGRAD** : +6 points de rappel
  et +11 points sur les couchées à N = 200, avec 99 % de précision. Avec SUM_OF_VARIANCE, ils
  font jeu égal en rappel et sont les seuls à 100 % de précision à N ≥ 100.
- **Le meilleur couple est SUM_OF_VARIANCE, quelle que soit la méthode** (rappel 0,90 à 0,92 à
  N = 200), ou TENEGRAD + maxima locaux (0,89).

### 3.3 Choix de K

Balayage de K avec TENEGRAD, à N = 200 :

| K | Rappel global / local / maxima | Précision global / local / maxima |
|---:|---:|---:|
| 4 | 0,97 / 0,98 / 0,98 | 0,04 / 0,03 / 0,07 |
| 6 | 0,95 / 0,96 / 0,96 | 0,42 / 0,38 / 0,77 |
| 8 | 0,87 / 0,90 / 0,89 | 0,87 / 0,85 / **0,99** |
| 10 | 0,79 / 0,83 / 0,82 | 0,98 / 0,96 / 1,00 |
| 15 | 0,41 / 0,48 / 0,48 | 1,00 / 0,99 / 1,00 |

Et à K = 8 (valeur par défaut actuelle des deux scripts), selon la densité :

| N | Précision global / local / maxima | Rappel global / local / maxima |
|---:|---:|---:|
| 5 | 0,23 / 0,44 / 0,74 | 0,93 / 0,93 / 0,93 |
| 10 | 0,29 / 0,48 / 0,97 | 0,97 / 0,97 / 0,97 |
| 50 | 0,56 / 0,75 / 0,96 | 0,97 / 0,97 / 0,96 |
| 100 | 0,69 / 0,70 / 0,97 | 0,93 / 0,94 / 0,93 |
| 200 | 0,87 / 0,85 / 0,99 | 0,87 / 0,90 / 0,89 |

- **Avec des bactéries réalistes, le seuil local + CCL à K = 8 a une précision insuffisante**
  (44 à 85 %), surtout à faible densité. Une bactérie à peine au-dessus du seuil se fragmente en
  plusieurs composantes, comptées comme autant de détections. Les maxima locaux regroupent ces
  fragments, d'où leur précision de 96 à 99 % dès N = 10 au même K.
- **K = 8 reste le bon choix pour les maxima locaux.** Pour le seuil local + CCL avec TENEGRAD,
  le F1 maximal est à K = 10 à toutes les densités. Avec SUM_OF_VARIANCE, il faut K = 15 à 20 :
  le contraste étant plus grand, K = 8 donne 30 % de précision seulement à N = 200. **Le K par
  défaut dépend donc du critère** ; les scripts sont restés à K = 8 (non modifiés).
- **Coût du seuil et de la détection** (mesures de la première version, qui ne dépendent pas de la
  taille des bactéries) : 0,14 s pour le seuil global, 0,35 s pour le seuil local, 0,73 s pour les
  maxima locaux (dont 0,35 s pour le seul filtre max).

### 3.4 Taille des mailles (portée du seuil local)

Balayage de la maille en XY (32 à 256 px, soit 5,6 à 45 µm) et en Z (25 à 100 plans, soit 12,5 à
50 µm). Seuil local + composantes connexes, N = 200, K au F1 maximal ; chaque case donne le
rappel, avec la précision entre parenthèses :

| Maille Z × XY | TENEGRAD | SUM_OF_VARIANCE |
|---|---|---|
| 25 plans × 32 px | 0,84 (0,96) | 0,87 (0,98) |
| **25 plans × 64 px** | **0,83 (0,96)** | **0,92 (0,94)** |
| 25 plans × 128 px | 0,84 (0,94) | 0,86 (0,98) |
| 25 plans × 256 px | 0,85 (0,94) | 0,87 (0,96) |
| 50 plans × 64 px | 0,83 (0,96) | 0,92 (0,93) |
| 50 plans × 128 px | 0,85 (0,94) | 0,86 (0,98) |
| 100 plans × 64 px | 0,83 (0,95) | 0,92 (0,92) |

- **La taille des mailles compte peu** avec des bactéries réalistes : 2 points d'écart avec
  TENEGRAD, 6 avec SUM_OF_VARIANCE, et des écarts de même ordre à N = 50 et 100.
- **L'auto-masquage des petites mailles a disparu.** Avec les grosses bactéries, une maille de
  32 px faisait tomber le rappel à 0,37 : l'objet (3,85 µm de réponse) pesait dans son propre σ.
  Une bactérie réaliste (2,1 µm de réponse) est assez petite devant 5,6 µm pour que l'écrêtage à
  ±3σ l'écarte.
- **64 px × 25 plans reste un bon choix par défaut**, sans raison forte d'en changer.

**Conclusion 3 : avec TENEGRAD, les maxima locaux au-dessus du seuil local sont la meilleure
méthode** (89 % de rappel, 99 % de précision à N = 200, K = 8). Avec SUM_OF_VARIANCE, les trois
méthodes se valent en rappel (90 à 92 %), et les maxima locaux donnent la meilleure précision.
L'écart entre seuil global et seuil local est faible ; c'est le passage à SUM_OF_VARIANCE, ou aux
maxima locaux, qui compte.

---

## 4. Ce qui reste à faire

1. **Les bactéries couchées en milieu dense restent la limite** : 81 à 85 % retrouvées au mieux à
   N = 200. Il faut mesurer leur contraste par rapport au fond local, σ estimé sans les objets
   (`FOCUS_REPORT/DENSE/RAPPORT_DENSE.md` § 3.2 donne la médiane : 31σ avec SUM_OF_VARIANCE,
   12σ avec TENEGRAD), puis voir si un détecteur qui exploite la forme 3D (filtre adapté, CLEAN
   ou U-Net 3D) retrouve les manquantes.
2. Ajuster le K par défaut au critère dans les deux scripts (10 pour seuil local + CCL avec
   TENEGRAD, 15 à 20 avec SUM_OF_VARIANCE), et réduire la boîte des maxima locaux à la réponse
   réaliste (environ 2 × 2 × 5 µm), après test.
3. Tester les maxima locaux à K bas (4–6), avec un tri des candidats par RSB local.
4. Calibrer σ plancher sur un hologramme réel sans objet, au lieu du percentile 10 des mailles.
5. Ramener le seuil local sous les 400 ms : estimation des mailles sur un sous-échantillon XY, et
   pas de 2 µm en Z (§ 3 de `CLAUDE_HOLOTRACKER_TO_DO.md`).
6. Augmenter le nombre d'hologrammes par densité, pour donner des barres d'erreur.

Les scripts de benchmark sont restés hors du dépôt, dans le scratchpad de la session :
`remesure_faits_1_2.py`, `remesure_fait_3.py`, `remesure_addition.py` et
`remesure_dense_methodes.py` (2026-10-13). Les hologrammes denses sont dans
`FOCUS_REPORT/DENSE/holograms`.
