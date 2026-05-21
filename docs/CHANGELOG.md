# Changelog - AOT_biomaps

Toutes les modifications notables de la librairie AOT_biomaps seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/), et ce projet respecte [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### ✨ Ajouté

- **Nouveaux fichiers de documentation** :
  - `docs/INSTALLATION.md` - Guide d'installation complet
  - `docs/USAGE.md` - Guide d'utilisation détaillé avec exemples
  - `docs/API_REFERENCE.md` - Référence API complète
  - `docs/ARCHITECTURE.md` - Documentation de l'architecture
  - `docs/CONTRIBUTING.md` - Guide de contribution
  - `docs/CHANGELOG.md` - Historique des modifications

- **Nouveaux modules** :
  - `AOT_biomaps/AOT_Recon/AOT_Kernels.py` - Centralisation des noyaux CPU/GPU
  - `AOT_biomaps/AOT_Recon/SparseMatrixWrapper.py` - Wrapper unifié pour matrices creuses
  - `AOT_biomaps/AOT_Recon/test_kernels.py` - Tests des noyaux et matrices
  - `AOT_biomaps/AOT_Recon/AOT_SparseSMatrix/` - Package pour matrices creuses (CSR, SELL)
  - `AOT_biomaps/AOT_Medium/` - Package pour modélisation des milieux
  - `AOT_biomaps/AOT_Experiment/ExperimentTools.py` - Outils pour expériences

- **Nouveaux algorithmes** :
  - Support complet de MLEM avec matrices creuses
  - Implémentation de PDHG (Primal-Dual Hybrid Gradient)
  - Implémentation de LS (Least Squares)
  - Implémentation de MAPEM (Maximum A Posteriori)
  - Implémentation de DEPIERRO
  - Implémentation de LBFGS

- **Fonctionnalités** :
  - Support des matrices creuses CSR et SELL-C-sigma
  - Basculement automatique CPU/GPU
  - Gestion des dépendances optionnelles (CuPy, matplotlib)
  - Noyaux CUDA centralisés dans `AOT_biomaps_kernels.cu`

### 🔧 Modifié

- **Gestion des dépendances** :
  - Toutes les dépendances optionnelles (CuPy, matplotlib, kWave) sont maintenant gérées avec try/except
  - La librairie fonctionne sans CuPy en utilisant uniquement NumPy
  - Les imports de matplotlib sont protégés

- **Type hints** :
  - Correction des type hints utilisant `cp.ndarray` pour éviter les erreurs quand CuPy n'est pas installé
  - Utilisation de `TYPE_CHECKING` et de string literals pour les types CuPy

- **AOT_Kernels.py** :
  - Ajout de fonctions helpers (`_is_cupy_array`, `_as_cupy_array`, `_as_numpy_array`)
  - Toutes les fonctions utilisent maintenant les helpers pour la gestion des arrays

- **SparseMatrix files** :
  - `SparseSMatrix_CSR.py` : Ajout de TYPE_CHECKING et gestion de cp=None
  - `SparseSMatrix_SELL.py` : Ajout de TYPE_CHECKING et gestion de cp=None
  - `SparseMatrixWrapper.py` : Ajout de TYPE_CHECKING

- **Optimizers** :
  - `MLEM.py` : Support des matrices creuses via SparseMatrixWrapper
  - `PDHG.py` : Made CuPy import optional
  - `LS.py` : Made CuPy import optional
  - `MAPEM.py` : Made CuPy import optional
  - `DEPIERRO.py` : Made CuPy import optional
  - `LBFGS.py` : Made CuPy import optional

- **Recon files** :
  - `AlgebraicRecon.py` : Made matplotlib import optional
  - `AnalyticRecon.py` : Made CuPy import optional
  - `_mainRecon.py` : Made matplotlib import optional

- **Acoustic files** :
  - `_mainAcoustic.py` : Made kWave import optional with warning

- **README.md** : Mise à jour avec badges, structure améliorée, exemples

### 🐛 Corrigé

- **Import errors** : Correction des erreurs d'import quand CuPy ou matplotlib ne sont pas installés
- **Type hints errors** : Correction des erreurs de type hints avec `cp.ndarray` quand CuPy n'est pas disponible
- **test_kernels.py** : Correction des caractères Unicode pour la compatibilité Windows
- **MLEM test data** : Correction de la forme des données de test (y doit être (T, N) pas (T, Z, X))

### 🗑️ Supprimé

- Aucune suppression majeure dans cette version

---

## [1.0.0] - 2025-11-24

### ✨ Ajouté

- **Version initiale** : Première version stable de AOT_biomaps
- **Fonctionnalités de base** :
  - Reconstruction tomographique (MLEM, Analytic)
  - Simulation acoustique (PlaneWave, FocusedWave)
  - Modélisation optique (Laser)
  - Gestion des expériences (Tomography)

---

## 📜 Format du Changelog

### Types de modifications

- `✨ Ajouté` : Pour les nouvelles fonctionnalités
- `🔧 Modifié` : Pour les modifications de fonctionnalités existantes
- `🐛 Corrigé` : Pour les correctifs de bugs
- `🗑️ Supprimé` : Pour les suppressions de fonctionnalités
- `📚 Documentation` : Pour les modifications de documentation
- `🔒 Sécurité` : Pour les correctifs de sécurité
- `⚡ Performance` : Pour les améliorations de performance

### Conventions de versionnage

Cette librairie utilise le [Semantic Versioning](https://semver.org/) :

- **MAJOR** : Modifications incompatibles avec les versions précédentes
- **MINOR** : Ajout de fonctionnalités compatibles avec les versions précédentes
- **PATCH** : Correctifs de bugs compatibles avec les versions précédentes

---

## 🎯 Roadmap

### Prochaines versions

- **1.1.0** (Prévu) :
  - Amélioration des performances GPU
  - Ajout de nouveaux algorithmes de reconstruction
  - Meilleure documentation et exemples

- **2.0.0** (Futur) :
  - Refactorisation majeure de l'architecture
  - Support de nouveaux types de données
  - Intégration avec d'autres librairies d'imagerie

---

## 🙏 Remerciements

Merci à tous les contributeurs qui ont rendu cette librairie possible !

---

**Retour à la [documentation principale](README.md)**
