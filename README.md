# BinarOCR — Image Binarization Studio

Interface Flask pour binariser des images en vue d'un OCR.

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
