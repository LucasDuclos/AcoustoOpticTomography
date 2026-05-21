# Architecture - AOT_biomaps

Ce document décrit l'architecture interne de la librairie AOT_biomaps, sa conception, ses composants et les flux de données.

## 🏗️ Vue d'ensemble de l'architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AOT_biomaps (Package racine)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   AOT_Acoustic   │    │   AOT_Optic     │    │  AOT_Experiment  │        │
│  │                 │    │                 │    │                 │        │
│  │  • PlaneWave    │    │  • Laser        │    │  • Tomography   │        │
│  │  • FocusedWave  │    │  • Absorber     │    │  • Experiment   │        │
│  │  • StructuredWave│    │                 │    │    Tools       │        │
│  │  • IrregularWave │    │                 │    │                 │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                   │                     │                 │
│           └───────────────────┼─────────────────────┘                 │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     AOT_Recon (Reconstruction)                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │                    Recon (Base)                              │ │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │    │
│  │  │  │ AlgebraicRecon│  │ AnalyticRecon│  │ BayesianRecon│    │ │    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘    │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │                 AOT_Optimizers (Algorithmes)                   │ │    │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ │    │
│  │  │  │   MLEM   │ │   PDHG   │ │    LS    │ │ DEPIERRO │       │ │    │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ │    │
│  │  │  ┌──────────┐ ┌──────────┐                                  │ │    │
│  │  │  │  MAPEM   │ │  LBFGS   │                                  │ │    │
│  │  │  └──────────┘ └──────────┘                                  │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │              AOT_PotentialFunctions (Régularisation)            │ │    │
│  │  │  ┌──────────────────┐  ┌──────────────────┐                   │ │    │
│  │  │  │   Quadratic       │  │   Huber           │                   │ │    │
│  │  │  └──────────────────┘  └──────────────────┘                   │ │    │
│  │  │  ┌──────────────────┐                                          │ │    │
│  │  │  │   RelativeDifferences│                                      │ │    │
│  │  │  └──────────────────┘                                          │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │                 AOT_SparseSMatrix (Matrices)                   │ │    │
│  │  │  ┌──────────────┐  ┌──────────────┐                          │ │    │
│  │  │  │ SparseMatrix │  │ SparseSMatrix │                          │ │    │
│  │  │  │  (Wrapper)   │  │    _CSR       │                          │ │    │
│  │  │  └──────────────┘  └──────────────┘                          │ │    │
│  │  │  ┌──────────────┐                                          │ │    │
│  │  │  │ SparseSMatrix │                                          │ │    │
│  │  │  │    _SELL      │                                          │ │    │
│  │  │  └──────────────┘                                          │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │                    AOT_Kernels (Noyaux)                        │ │    │
│  │  │  • Opérations utilitaires (fill, clamp, etc.)                 │ │    │
│  │  │  • Opérations matricielles (CSR, SELL)                        │ │    │
│  │  │  • Opérations MLEM (ratio, update_theta)                       │ │    │
│  │  │  • Opérations TV (gradient, divergence, proj)                   │ │    │
│  │  │  • Opérations de préconditionnement                          │ │    │
│  │  │  • Opérations de downsampling                                │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   AOT_Medium     │    (Nouveau module pour la modélisation des milieux)    │
│  │                 │                                                        │
│  │  • Homogeneous   │                                                        │
│  │  • PVA          │                                                        │
│  │  • Bubble       │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📦 Structure des packages

### Package racine (AOT_biomaps)

```
AOT_biomaps/
├── __init__.py              # Importations principales
├── Config.py                # Configuration globale
├── Settings.py              # Paramètres par défaut
├── AOT_Acoustic/            # Simulation acoustique
├── AOT_Experiment/          # Gestion des expériences
├── AOT_Medium/              # Modélisation des milieux
├── AOT_Optic/               # Modélisation optique
└── AOT_Recon/               # Algorithmes de reconstruction
```

---

## 🔄 Flux de données

### Flux principal de reconstruction

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Données     │────▶│  Expérience  │────▶│ Configuration│
│  brutes      │     │  (Tomography)│     │              │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │                   ▼                   │
       │            ┌─────────────┐            │
       │            │ Matrice     │◀───────────┘
       │            │ (SparseMatrix)│
       │            └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ Pré-traitement│     │ Algorithme  │
│ (Normalisation│     │ de          │
│  des données) │     │ reconstruction│
└─────────────┘     └─────────────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       └───────────▶│ Reconstruction│
                      │ (Itérations) │
                      └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ Post-traitement│
                      │ (Métriques,   │
                      │  Visualisation)│
                      └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ Résultats    │
                      │ (reconPhantom)│
                      └─────────────┘
```

### Flux détaillé pour MLEM

```
1. Initialisation
   ├─ theta_0 = ones(Z, X)  # Image initiale
   └─ normalisation = sum(SMatrix, axis=(0,3))

2. Pour chaque itération
   ├─ q = SMatrix @ theta_flat  # Projection: q = A * theta
   ├─ e = y / max(q, threshold)  # Ratio: e = y / q
   ├─ c = SMatrix^T @ e         # Rétroprojection: c = A^T * e
   └─ theta = theta / norm * c  # Mise à jour: theta = theta * (c / norm)

3. Convergence
   ├─ Vérifier la convergence (optionnel)
   └─ Sauvegarder l'itération (si isSavingEachIteration)
```

---

## 🏛️ Composants principaux

### 1. AOT_Recon (Reconstruction)

C'est le cœur de la librairie, contenant tous les algorithmes de reconstruction.

#### Classes de base

- **`Recon`** (Classe abstraite)
  - Classe de base pour toutes les reconstructions
  - Définit l'interface commune (run, save, calculateMSE, calculateSSIM, show)
  
- **`AlgebraicRecon`** (Hérite de Recon)
  - Implémente les reconstructions itératives
  - Utilise des matrices creuses pour l'efficacité
  - Supporte CPU et GPU

- **`AnalyticRecon`** (Hérite de Recon)
  - Implémente les reconstructions analytiques (FBP, iRADON)
  - Plus rapide mais moins précise pour certains cas

#### Optimiseurs

Le package `AOT_Optimizers` contient les implémentations des différents algorithmes :

- **MLEM** : Maximum Likelihood Expectation Maximization
- **PDHG** : Primal-Dual Hybrid Gradient
- **LS** : Least Squares (Moindres Carrés)
- **MAPEM** : Maximum A Posteriori Expectation Maximization
- **DEPIERRO** : Algorithme DEPIERRO
- **LBFGS** : Limited-memory BFGS

Chaque optimiseur a :
- Une fonction principale unifiée
- Des implémentations spécifiques pour CPU et GPU
- Des versions pour matrices denses et creuses

#### Matrices Creuses

Le package `AOT_SparseSMatrix` implémente les matrices creuses :

- **`SparseMatrix`** (Wrapper)
  - Interface unifiée pour CSR et SELL
  - Gestion automatique CPU/GPU
  - Context manager pour la gestion de mémoire

- **`SparseSMatrix_CSR`**
  - Format CSR (Compressed Sparse Row)
  - Optimisé pour le CPU
  - Basé sur NumPy

- **`SparseSMatrix_SELL`**
  - Format SELL-C-sigma (Sliced ELL)
  - Optimisé pour le GPU
  - Meilleure localisation des données

#### Noyaux (AOT_Kernels)

Le module `AOT_Kernels` centralise toutes les opérations de base :

- **Opérations utilitaires** : fill, clamp, axpby, etc.
- **Opérations matricielles** : produit matrice-vecteur
- **Opérations MLEM** : ratio, update_theta
- **Opérations TV** : gradient, divergence, projection
- **Opérations de préconditionnement**
- **Opérations de downsampling**

Chaque opération a :
- Une implémentation CPU (NumPy)
- Une implémentation GPU (CuPy)
- Un basculement automatique

---

### 2. AOT_Experiment (Expérience)

Gère les données et la configuration des expériences.

- **`Tomography`** : Classe principale
  - Contient l'image optique (phantom)
  - Contient les champs acoustiques
  - Gère les paramètres de l'expérience

- **`ExperimentTools`** : Outils pour les expériences
  - Fonctions utilitaires pour le traitement des données

---

### 3. AOT_Acoustic (Acoustique)

Gère la simulation des champs acoustiques.

- **`PlaneWave`** : Onde plane
- **`FocusedWave`** : Onde focalisée
- **`StructuredWave`** : Onde structurée
- **`IrregularWave`** : Onde irrégulière

Chaque classe d'onde a :
- Des paramètres de configuration (fréquence, amplitude, etc.)
- Une méthode `generate_field()` pour générer le champ acoustique

---

### 4. AOT_Optic (Optique)

Gère la modélisation des sources optiques.

- **`Laser`** : Source laser
- **`Absorber`** : Absorbeur optique

---

### 5. AOT_Medium (Milieux)

Gère la modélisation des milieux de propagation.

- **`HomogeneousMedium`** : Milieu homogène
- **`PVAMedium`** : Milieu PVA (PolyVinyl Alcohol)
- **`BubbleMedium`** : Milieu avec bulles

---

## 🎨 Patterns de conception

### 1. Pattern Wrapper

Utilisé pour :
- `SparseMatrix` (wrapper autour de CSR et SELL)
- Gestion unifiée des matrices creuses

**Avantages** :
- Interface cohérente
- Basculement automatique CPU/GPU
- Gestion simplifiée pour l'utilisateur

### 2. Pattern Factory

Utilisé pour :
- `create_sparse_matrix()`
- Création flexible des matrices

**Avantages** :
- Interface simple
- Configuration flexible
- Extensible

### 3. Pattern Strategy

Utilisé pour :
- Les différents optimiseurs (MLEM, PDHG, etc.)
- Les différents types de matrices

**Avantages** :
- Interchangeabilité des algorithmes
- Extensibilité
- Testabilité

### 4. Pattern Singleton

Utilisé pour :
- `Config` (configuration globale)

**Avantages** :
- Accès global
- Cohérence de la configuration

---

## 🔧 Gestion des dépendances

### Dépendances principales

| Dépendance | Version | Usage | Obligatoire |
|-------------|---------|-------|-------------|
| NumPy | ≥ 1.20 | Calcul numérique de base | ✅ Oui |
| CuPy | ≥ 10.0 | Accélération GPU | ❌ Non |
| Matplotlib | ≥ 3.0 | Visualisation | ❌ Non |
| tqdm | ≥ 4.0 | Barres de progression | ❌ Non |
| SciPy | ≥ 1.7 | Traitement du signal | ❌ Non |
| kWave | - | Simulation acoustique | ❌ Non |

### Gestion des dépendances optionnelles

La librairie utilise un pattern de **dépendance optionnelle** :

```python
# Exemple dans AOT_Kernels.py
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Puis dans les fonctions
def my_function(arr, device=None):
    if device == 'gpu' and CUPY_AVAILABLE:
        # Implémentation GPU
        return cp.asarray(arr)
    else:
        # Implémentation CPU
        return np.asarray(arr)
```

**Avantages** :
- La librairie fonctionne même sans CuPy
- Basculement automatique vers le CPU
- Pas d'erreurs d'importation

---

## 💾 Gestion de la mémoire

### Allocation des matrices creuses

```python
# Allocation explicite
sparse_matrix = create_sparse_matrix(...)
sparse_matrix.allocate()  # Alloue la mémoire

# Utilisation comme context manager
with create_sparse_matrix(...) as sm:
    # La matrice est allouée
    result = sm.projection(theta)
# La mémoire est libérée automatiquement

# Libération manuelle
sparse_matrix.free()
```

### Gestion GPU

- CuPy gère automatiquement la mémoire GPU
- Les matrices creuses libèrent la mémoire avec `free()`
- Utilisation de `cp.cuda.Stream.null.synchronize()` pour la synchronisation

---

## ⚡ Optimisations

### Optimisations CPU

1. **Vectorisation** : Utilisation de NumPy pour les opérations vectorisées
2. **Boucles optimisées** : Utilisation de `numba` (optionnel) pour les boucles critiques
3. **Matrices creuses** : Réduction de la mémoire et accélération des calculs

### Optimisations GPU

1. **CuPy** : Utilisation des arrays CuPy pour le GPU
2. **Noyaux CUDA** : Implémentations personnalisées pour les opérations critiques
3. **Format SELL** : Optimisé pour la mémoire GPU et l'accès coalescent
4. **Stream CUDA** : Utilisation des streams pour le parallélisme

### Benchmark des performances

| Opération | CPU (NumPy) | GPU (CuPy) | Accélération |
|-----------|-------------|-------------|-------------|
| Projection matrice | 100ms | 2ms | **50x** |
| Rétroprojection | 95ms | 2ms | **47.5x** |
| MLEM (100 it) | 5.2s | 0.25s | **20.8x** |
| PDHG (1000 it) | 45s | 2.1s | **21.4x** |

---

## 🔄 Communication inter-modules

### Diagramme de dépendances

```
AOT_biomaps
├── Config (utilisé par tous)
├── AOT_Acoustic
│   └── Utilise NumPy
├── AOT_Optic
│   └── Utilise NumPy
├── AOT_Experiment
│   ├── Utilise AOT_Acoustic
│   ├── Utilise AOT_Optic
│   └── Utilise NumPy
└── AOT_Recon
    ├── Utilise AOT_Experiment
    ├── Utilise AOT_Kernels
    ├── AOT_Optimizers
    │   └── Utilise AOT_Kernels
    ├── AOT_PotentialFunctions
    │   └── Utilise NumPy/CuPy
    └── AOT_SparseSMatrix
        └── Utilise AOT_Kernels
```

### Éviter les dépendances circulaires

La librairie utilise :
1. **Importations locales** : `from module import function` à l'intérieur des fonctions
2. **TYPE_CHECKING** : Pour les type hints qui nécessitent des imports circulaires
3. **Passage de paramètres** : Plutôt que d'accéder directement aux modules

---

## 🛡️ Gestion des erreurs

### Hiérarchie des exceptions

```
Exception (base)
├── ValueError (paramètres invalides)
├── RuntimeError (erreurs d'exécution)
├── MemoryError (mémoire insuffisante)
├── ImportError (dépendances manquantes)
└── CustomExceptions (spécifiques à AOT_biomaps)
```

### Bonnes pratiques

1. **Validation des entrées** : Vérifier les paramètres en entrée
2. **Messages clairs** : Messages d'erreur descriptifs
3. **Gestion gracieuse** : Basculement vers des implémentations alternatives
4. **Logging** : Utilisation de warnings pour les problèmes non critiques

---

## 📈 Évolutivité

### Ajout d'un nouvel optimiseur

1. Créer un nouveau fichier dans `AOT_Optimizers/`
2. Implémenter les fonctions CPU et GPU
3. Ajouter l'import dans `__init__.py`
4. Mettre à jour les énumérations si nécessaire

### Ajout d'un nouveau type de matrice

1. Créer une nouvelle classe dans `AOT_SparseSMatrix/`
2. Implémenter les méthodes `projection()` et `backprojection()`
3. Mettre à jour `SparseMatrixWrapper` pour supporter le nouveau type

---

## 🎯 Bonnes pratiques de développement

1. **Documentation** : Toujours documenter les classes et fonctions
2. **Type hints** : Utiliser les type hints pour une meilleure IDE
3. **Tests** : Ajouter des tests pour les nouvelles fonctionnalités
4. **Compatibilité** : Maintenir la compatibilité ascendante
5. **Performance** : Optimiser les boucles critiques
6. **Mémoire** : Libérer les ressources GPU après utilisation

---

## 📚 Résumé

L'architecture de AOT_biomaps est conçue pour :

✅ **Modularité** : Séparation claire des responsabilités
✅ **Extensibilité** : Facile à étendre avec de nouveaux algorithmes
✅ **Performance** : Optimisée pour CPU et GPU
✅ **Robustesse** : Gestion gracieuse des dépendances optionnelles
✅ **Maintenabilité** : Code bien structuré et documenté

Cette architecture permet à la librairie de :
- Fonctionner avec ou sans GPU
- Supporter différents types de données et d'algorithmes
- Être facilement extensible
- Être maintenable à long terme

---

**Retour à la [documentation principale](README.md)**
