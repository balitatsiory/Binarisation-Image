# BinarOCR Studio

Interface web de binarisation d'images pour pipeline OCR.  
Prend en entrée des scans de documents (image seule ou archive ZIP avec sous-dossiers), produit des TIFF 1-bit en compression CCITT Fax Group 4 à 300 DPI, prêts à être consommés par Tesseract, EasyOCR ou tout autre moteur OCR.

Conversation avec claude : [text](https://claude.ai/share/b01a283d-8d01-404b-bd01-6895a0910125)

---

## Pourquoi ce projet

Un moteur OCR n'analyse pas l'image couleur brute — il travaille sur une image binarisée (pixels strictement noirs ou blancs). La qualité de cette étape conditionne directement le taux de reconnaissance : une binarisation mal calibrée sur un document photographié avec un éclairage inégal peut faire chuter la précision de 95 % à 60 %.

Les outils disponibles (ImageMagick, GIMP, scripts Tesseract) offrent peu de contrôle visuel et aucune gestion de lot avec suivi par dossier. BinarOCR Studio comble ce manque avec :

- un prétraitement complet et paramétrable (contraste, débruitage, correction d'angle)
- plusieurs algorithmes de seuillage dont Sauvola, référence académique pour l'OCR
- un export TIFF CCITT G4 avec fallback automatique de DPI pour rester sous 300 Ko
- un traitement asynchrone de grandes archives (testé jusqu'à 15 Go) avec progression en temps réel et téléchargement automatique dossier par dossier

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- pip

```bash
# Vérifier la version Python
python --version
```

### Installation des dépendances

```bash
git clone https://github.com/<ton-repo>/binarocr-studio.git
cd binarocr-studio

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Lancement

```bash
python app.py
```

Ouvrir [http://localhost:5000](http://localhost:5000)

---

## Utilisation

### Interface web — image unique

1. Cliquer sur **Image** dans la sidebar
2. Déposer ou sélectionner une image (PNG, JPG, TIFF, BMP, WebP)
3. Choisir un préset ou ajuster les paramètres manuellement
4. Cliquer **Binariser**
5. Comparer avant / après (mode côte à côte ou curseur glissant)
6. Télécharger le TIFF via le bouton **⬇ TIFF**

### Interface web — traitement de lot

1. Cliquer sur **ZIP / Dossiers** dans la sidebar
2. Déposer un fichier `.zip` (peut contenir des sous-dossiers, jusqu'à 15 Go)
3. Choisir les paramètres
4. Cliquer **Binariser**
5. Le modal de progression s'ouvre :
   - Suivi image par image en temps réel
   - Dès qu'un dossier est terminé → son ZIP se télécharge automatiquement
   - Cliquer sur un dossier pour voir le détail de chaque image
6. En cas d'interruption, le bouton ⬇ ZIP reste disponible dans le modal

---

## Fonctionnalités

### Mode image unique
- Upload par clic ou glisser-déposer
- Comparaison avant / après côte à côte ou avec curseur glissant
- Téléchargement TIFF immédiat

### Mode lot (ZIP)
- Archive ZIP pouvant contenir des sous-dossiers
- Traitement asynchrone : l'interface reste réactive pendant le traitement
- Modal de progression en temps réel (SSE) :
  - nombre total d'images, traitées, erreurs
  - barre de progression globale avec ETA dynamique
  - une section par dossier avec sa propre barre et la liste des images
- Dès qu'un dossier est terminé : son ZIP est disponible et **téléchargé automatiquement**


### Traitement en ligne de commande
Script `process_folder.py` pour traiter un dossier local sans passer par le navigateur.
```bash
# Traitement d'un dossier local, préset équilibré (défaut)
python process_folder.py ./scans

# Spécifier un dossier de sortie et un préset
python process_folder.py ./scans ./resultats --preset quality

# Personnaliser la méthode
python process_folder.py ./scans --method otsu --no-deskew --no-morph

# Ne pas descendre dans les sous-dossiers
python process_folder.py ./scans --no-recursive

# Paramètres Sauvola personnalisés
python process_folder.py ./scans --sauvola-window 35 --sauvola-k 0.4
```

Les images de sortie conservent la structure de sous-dossiers de la source.
### API REST

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/api/process` | Binariser une image (multipart: `image`) |
| `POST` | `/api/process-batch` | Démarrer un job batch (multipart: `archive`) |
| `GET` | `/api/job/<sid>` | État JSON d'un job |
| `GET` | `/api/job/<sid>/stream` | SSE — événements temps réel |
| `GET` | `/api/download/<filename>` | Télécharger un fichier produit |
| `GET` | `/api/presets` | Retourne les présets disponibles |
| `GET` | `/api/debug` | Versions des dépendances, état des jobs |

Tous les paramètres du pipeline sont passés en form-data avec la requête `POST`.

---
## Configuration

Les constantes suivantes sont dans `app.py` :

```python
MAX_CONTENT_LENGTH = 20 * 1024 * 1024 * 1024  # Taille max upload : 20 Go
LARGE_FILE_BYTES   = 5 * 1024 * 1024           # Au-delà : preview désactivée
MAX_TIFF_KB        = 300                        # Taille max TIFF en sortie (Ko)
DPI_FALLBACK_STEPS = [300, 250, 200, 150]       # Paliers DPI si dépassement
```

### Configuration nginx (si proxy inverse)

```nginx
client_max_body_size 20G;
proxy_read_timeout   3600s;
proxy_buffering      off;    # indispensable pour les SSE
```

---


## Installation

```bash
# 1. Cloner / décompresser le projet
cd ocr-binarizer

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
python app.py
```

Ouvrir http://localhost:5000 dans le navigateur.

---

## Fonctionnalités

### Upload
- **Image unique** : PNG, JPG, TIFF, BMP, WebP
- **Dossier (ZIP)** : archive contenant plusieurs images

### Visualisation
- **Côte à côte** : comparaison avant/après en parallèle
- **Curseur glissant** : superposition interactive

### Paramètres
#### Présets
| Préset | Usage |
|--------|-------|
| ⚡ Rapide | Otsu simple, sans corrections — idéal pour tester |
| ⚖️ Équilibré | Sauvola + CLAHE + Deskew — bon compromis qualité/vitesse |
| ✨ Qualité | Sauvola agressif + toutes corrections — pour documents complexes |

#### Méthodes de seuillage
- **Sauvola** — meilleur choix pour OCR, adaptatif local
- **Otsu** — rapide, global, bon sur fond uniforme
- **Adaptive Gaussien/Moyen** — pour variations d'éclairage
- **Manuel** — seuil fixe pour cas particuliers

#### Pré-traitement
- CLAHE — amélioration du contraste local
- Débruitage — filtre gaussien ou médian
- Deskew — correction automatique de l'angle d'inclinaison

#### Post-traitement
- Morphologie — Opening (supprime bruit), Closing (comble trous), ou les deux

### Export
- Téléchargement image individuelle (PNG)
- Téléchargement ZIP pour le traitement en lot
- Galerie des résultats batch

---

## Structure du projet

```
ocr-binarizer/
├── app.py              # Backend Flask + pipeline de binarisation
├── requirements.txt
├── templates/
│   └── index.html      # UI complète
├── uploads/            # Dossier temporaire (auto-nettoyé)
└── outputs/            # Images traitées
```

## Tips OCR

- Viser **300 DPI minimum** avant de binariser
- Utiliser **Sauvola** avec `k=0.3` et fenêtre `25px` comme point de départ
- Si le texte est fin ou cassé → réduire `k` ou augmenter la fenêtre
- Si du bruit persiste → augmenter le noyau morphologique (Opening)
- Pour documents photographiés → activer CLAHE avec clip > 2.0


## Conseils pour de bons résultats OCR

**Choisir la bonne méthode de seuillage**

- Tester interface pour bien choisir le type
- Document scanné proprement, fond blanc : **Otsu** suffit
- Document photographié, éclairage inégal : **Sauvola** obligatoire
- Très vieux document, fond texturé : **Sauvola** avec fenêtre large (35–51 px) et k élevé (0.4–0.5)
- Texte imprimé sur fond coloré : **Adaptive Gaussien**

**Régler Sauvola**

- `k` trop bas (< 0.2) → texte qui disparaît sur les parties sombres
- `k` trop haut (> 0.5) → bruit qui envahit le fond
- Fenêtre trop petite → sensible au bruit ; trop grande → perd les variations locales
- Point de départ recommandé : `k=0.3`, `window=25`

**Morphologie**

- Ne pas activer si le texte est très fin (police < 8pt) — risque d'effacer des traits
- Opening seul suffit dans la majorité des cas (supprime le bruit de fond)
- Closing utile si les caractères sont cassés (encre insuffisante, scan dégradé)

**DPI**

- En dessous de 200 DPI physiques, le texte est trop pixelisé pour un OCR fiable
- Le rééchantillonnage Monte le DPI mais ne crée pas d'information — scanner à la bonne résolution reste préférable
- 300 DPI est la cible standard ; 400–600 DPI pour du texte manuscrit ou très petit

---

## Dépendances

```
flask>=3.0.0
opencv-python-headless>=4.8.0
scikit-image>=0.22.0
numpy>=1.24.0
Pillow>=10.0.0
werkzeug>=3.0.0
```

## Meilleurs conf :

