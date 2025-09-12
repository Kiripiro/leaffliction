# Filtres de Transformation d'Images - LeafFliction

Ce dossier contient l'ensemble des filtres de transformation d'images utilisés dans le pipeline d'analyse de maladies des feuilles. Chaque filtre effectue une analyse spécifique sur les images de feuilles pour extraire des caractéristiques visuelles et diagnostiques.

## Vue d'ensemble du pipeline de transformation

Le système de transformation d'images de LeafFliction applique une série de filtres séquentiels pour analyser les feuilles et détecter les maladies. Le processus suit cette architecture :

```
Image d'entrée → Masque → ROI → Analyse → Landmarks → Détection maladies → Sortie
```

## Filtres disponibles

### 1. **Mask** (`mask.py`)
**Objectif :** Segmentation automatique de la feuille du fond

- Utilise plusieurs approches : HSV, LAB, K-means clustering
- Suppression avancée des ombres et artefacts
- Raffinement par GrabCut pour les contours précis
- Extension du masque pour inclure les régions malades

**Algorithmes utilisés :**
- Seuillage adaptatif HSV/LAB
- Classification K-means pour la séparation fond/objet
- Morphologie mathématique pour le nettoyage
- PlantCV pour les opérations botaniques

### 2. **ROI** (`roi.py`)
**Objectif :** Extraction de la région d'intérêt standardisée

- Extraction de la bounding box de la feuille
- Redimensionnement avec préservation du ratio d'aspect
- Padding automatique pour format standard
- Visualisation du cadre de sélection

### 3. **Analyze** (`analyze.py`)
**Objectif :** Analyse morphologique complète de la feuille

- Calcul des métriques de forme via PlantCV
- Détection des points extrêmes (gauche, droite, haut, bas)
- Analyse par composantes principales (PCA) pour axes majeur/mineur
- Extraction de l'enveloppe convexe
- Détection des contours et veines via Canny
- Visualisation des caractéristiques géométriques

### 4. **Landmarks** (`landmarks.py`)
**Objectif :** Placement de points de repère anatomiques

- **Points de bordure :** Échantillonnage uniforme du contour
- **Points de veines :** Détection via gradients et Harris corners
- **Points de maladie :** Localisation des zones pathologiques
- Répartition intelligente selon la densité des caractéristiques
- Amélioration du contraste pour la détection des veines

### 5. **Brown** (`brown.py`)
**Objectif :** Détection spécifique des maladies brunes/nécrotiques avec exclusion intelligente des ombres

- **Détection multi-colorspace :** HSV et LAB des régions brunâtres
- **Exclusion des ombres :** Détection automatique des zones d'ombre pour éviter les faux positifs
- **Analyse de texture :** Utilisation de la variance locale et détection des contours pour distinguer les vraies lésions des ombres
- **Filtrage morphologique :** Élimination du bruit et nettoyage des régions détectées
- **Analyse des composantes connexes :** Filtrage par taille minimale des lésions
- **Métriques quantitatives :** Calcul du pourcentage de surface affectée et nombre de lésions

**Algorithmes avancés :**
- **Détection d'ombre :** Analyse HSV avec seuils adaptatifs (percentile/Otsu)
- **Analyse de texture :** Variance locale pour différencier spots de maladie et ombres lisses
- **Détection d'arêtes :** Canny edge detection pour identifier les contours des lésions
- **Morphologie multi-kernel :** Nettoyage séparé pour ombres et zones brunes

**Seuils configurables :**
- Plage de teinte HSV pour les tons bruns
- Seuils de saturation et luminosité
- Paramètres de détection d'ombre (shadow_s_max, shadow_v_percentile)
- Taille minimale des lésions
- Mode de détection avancée (use_enhanced_brown_detection)

### 6. **Blur** (`blur.py`)
**Objectif :** Création d'une carte de saillance visuelle

- Cartographie des zones d'intérêt en niveaux de gris
- Pondération des caractéristiques : contours, texture, maladie
- Détection des variations de couleur locales
- Seuillage adaptatif pour les régions importantes
- Lissage gaussien pour transitions fluides

### 7. **Hist** (`hist.py`)
**Objectif :** Analyse statistique des couleurs

- Histogrammes HSV séparés (Teinte, Saturation, Valeur)
- Rendu graphique avec matplotlib
- Export en image RGB pour intégration
- Analyse de la distribution colorimétrique

## Usage et commandes

### Transformation d'une image unique

```bash
# Depuis la racine du projet
python3 srcs/cli/Transformation.py --image images/Apple/Apple_healthy/image\ \(1000\).JPG

# Avec types spécifiques
python3 srcs/cli/Transformation.py --image images/Apple/Apple_healthy/image\ \(1000\).JPG --types Mask,Brown,Landmarks

# Avec configuration personnalisée
python3 srcs/cli/Transformation.py --image images/Apple/Apple_healthy/image\ \(1000\).JPG --config transform/config.yaml
```

### Traitement par lots

```bash
# Traitement d'un dossier complet
python3 srcs/cli/Transformation.py --src images/Apple/Apple_rust/ --dst artifacts/batch_output/

# Avec parallélisation
python3 srcs/cli/Transformation.py --src images/Apple/ --dst artifacts/all_apple/ --workers 4

# Ignorer les fichiers existants
python3 srcs/cli/Transformation.py --src images/Apple/ --dst artifacts/output/ --skip-existing
```

### Options avancées

```bash
# Liste de tous les types disponibles
python3 srcs/cli/Transformation.py --list-types

# Aide complète
python3 srcs/cli/Transformation.py --help

# Mode verbose avec logs détaillés
python3 srcs/cli/Transformation.py --image images/Apple/Apple_scab/image\ \(1050\).JPG --verbose
```

## Structure des sorties

Pour chaque image traitée, le système génère :

```
artifacts/transformations/{numero_image}/
├── image_T_Mask.jpg      # Masque de segmentation
├── image_T_ROI.jpg       # Région d'intérêt extraite
├── image_T_Analyze.jpg   # Analyse morphologique
├── image_T_Landmarks.jpg # Points de repère anatomiques
├── image_T_Brown.jpg     # Détection des maladies brunes
├── image_T_Blur.jpg      # Carte de saillance
└── image_T_Hist.jpg      # Histogrammes colorimétriques
```
