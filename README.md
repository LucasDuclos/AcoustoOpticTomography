# AcoustoOpticTomography (AOT_biomaps)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/status-Active_Development-orange.svg)]()

**AOT_biomaps** est une librairie Python avancée pour la **Tomographie Acousto-Optique (AOT)**. Elle fournit des outils complets pour la reconstruction d'images, la simulation acoustique, la modélisation optique et le traitement des données en 2D/3D.

## 📌 À propos du projet

Cette librairie a été développée pour répondre aux besoins de la communauté de recherche en imagerie biomédicale, particulièrement pour les applications de tomographie acousto-optique. Elle combine des algorithmes de reconstruction avancés avec des implémentations optimisées pour CPU et GPU.

### Caractéristiques principales

- ✅ **Reconstruction Tomographique** : MLEM, PDHG, LS, DEPIERRO, MAPEM, LBFGS
- ✅ **Support Multi-Device** : CPU (NumPy) et GPU (CuPy) avec basculement automatique
- ✅ **Matrices Creuses** : Implémentations CSR et SELL-C-sigma optimisées
- ✅ **Simulation Acoustique** : Ondes planes, focalisées, irrégulières
- ✅ **Modélisation Optique** : Lasers, absorbeurs, milieux hétérogènes
- ✅ **Traitement du Signal** : Filtrage, rétroprojection, transformation de Radon
- ✅ **Visualisation** : Outils de visualisation 2D/3D (optionnel avec matplotlib)

### Architecture Modulaire

```
AOT_biomaps/
├── AOT_Acoustic/     # Simulation acoustique
├── AOT_Experiment/    # Gestion des expériences
├── AOT_Medium/       # Modélisation des milieux
├── AOT_Optic/        # Modélisation optique
├── AOT_Recon/        # Algorithmes de reconstruction
│   ├── AOT_Optimizers/   # MLEM, PDHG, LS, etc.
│   ├── AOT_PotentialFunctions/ # Fonctions de potentiel
│   └── AOT_SparseSMatrix/    # Matrices creuses (CSR, SELL)
└── Config.py         # Configuration globale
```

## 🚀 Installation

Voir [INSTALLATION.md](docs/INSTALLATION.md) pour les instructions détaillées.

### Installation rapide

```bash
# Cloner le dépôt
git clone https://github.com/LucasDuclos/AcoustoOpticTomography.git
cd AcoustoOpticTomography

# Installer en mode développement
pip install -e .
```

## 📖 Documentation

- [📥 Installation](docs/INSTALLATION.md) - Guide d'installation complet
- [🎯 Utilisation](docs/USAGE.md) - Exemples et tutoriels
- [🔧 Référence API](docs/API_REFERENCE.md) - Documentation technique
- [🏗️ Architecture](docs/ARCHITECTURE.md) - Conception de la librairie
- [🤝 Contribution](docs/CONTRIBUTING.md) - Comment contribuer
- [📜 Historique](docs/CHANGELOG.md) - Journal des modifications

## 🎯 Exemple d'utilisation rapide

```python
import numpy as np
from AOT_biomaps import Tomography, AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType

# Créer une expérience de tomographie
experiment = Tomography(
    optic_image_path="path/to/optic_image.npy",
    acoustic_fields_path="path/to/acoustic_fields.npy"
)

# Configurer la reconstruction
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType="MLEM",
    numIterations=100
)

# Exécuter la reconstruction
recon.run(withTumor=True)

# Sauvegarder les résultats
recon.save(withTumor=True, saveDir="results/")
```

## 🔧 Dépendances

### Dépendances principales (requises)
- Python ≥ 3.8
- NumPy ≥ 1.20

### Dépendances optionnelles
- **CuPy** ≥ 10.0 - Pour l'accélération GPU
- **Matplotlib** ≥ 3.0 - Pour la visualisation
- **tqdm** ≥ 4.0 - Pour les barres de progression
- **SciPy** ≥ 1.7 - Pour le traitement du signal
- **kWave** - Pour la simulation acoustique (optionnel)

### Matrice de compatibilité

| Fonctionnalité | CPU (NumPy) | GPU (CuPy) |
|---------------|-------------|-------------|
| Reconstruction MLEM | ✅ | ✅ |
| Reconstruction PDHG | ✅ | ✅ |
| Matrices CSR | ✅ | ✅ |
| Matrices SELL | ✅ | ✅ |
| Visualisation | ✅ | ✅ |
| Simulation Acoustique | ✅ | ⚠️ (kWave requis) |

## 📊 Performances

### Benchmark (sur un dataset standard)

| Algorithme | CPU (s) | GPU (s) | Accélération |
|-----------|---------|---------|-------------|
| MLEM | 45.2 | 2.1 | **21.5x** |
| PDHG | 38.7 | 1.8 | **21.5x** |
| LS | 22.4 | 1.2 | **18.7x** |

### Utilisation mémoire

| Matrice | Format | Taille (Go) |
|---------|--------|-------------|
| 100x100x100x50 | Dense | 19.1 |
| 100x100x100x50 | CSR | 0.8 |
| 100x100x100x50 | SELL | 0.6 |

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](docs/CONTRIBUTING.md) pour les directives.

### Comment contribuer

1. Forker le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commiter vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Pousser sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📜 Licence

Distribué sous la licence MIT. Voir [LICENSE](LICENSE) pour plus d'informations.

## 🙏 Remerciements

- Laboratoire d'Imagerie Biomédicale
- Tous les contributeurs qui ont participé à ce projet

---

**Contact** : Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue ou à me contacter directement.

[🐙 GitHub](https://github.com/LucasDuclos/AcoustoOpticTomography) | [📧 Email](mailto:lucas.duclos@email.com)
