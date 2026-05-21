# Guide d'Utilisation - AOT_biomaps

Ce guide vous explique comment utiliser la librairie AOT_biomaps pour effectuer des reconstructions tomographiques acousto-optiques.

## 📚 Table des matières

- [Concepts de base](#-concepts-de-base)
- [Premiers pas](#-premiers-pas)
- [Reconstruction Tomographique](#-reconstruction-tomographique)
  - [Reconstruction Algébrique (MLEM)](#reconstruction-algébrique-mlem)
  - [Reconstruction Analytique](#reconstruction-analytique)
  - [Reconstruction Bayésienne](#reconstruction-bayésienne)
  - [Reconstruction par Moindres Carrés](#reconstruction-par-moindres-carrés)
  - [Reconstruction Primal-Dual (PDHG)](#reconstruction-primal-dual-pdhg)
- [Utilisation des Matrices Creuses](#-utilisation-des-matrices-creuses)
- [Simulation Acoustique](#-simulation-acoustique)
- [Visualisation des Résultats](#-visualisation-des-résultats)
- [Exemples Complets](#-exemples-complets)
- [Bonnes Pratiques](#-bonnes-pratiques)

## 🎯 Concepts de base

### Tomographie Acousto-Optique (AOT)

La Tomographie Acousto-Optique combine les avantages de l'imagerie optique (haute résolution) et de l'imagerie ultrasonore (profondeur de pénétration). Le principe est le suivant :

1. Un faisceau laser illumine le tissu
2. Une onde ultrasonore module la lumière
3. La lumière modulée est détectée et utilisée pour reconstruire une image

### Architecture de la librairie

```
AOT_biomaps/
├── AOT_Acoustic/     # Simulation des champs acoustiques
├── AOT_Experiment/    # Gestion des expériences et données
├── AOT_Medium/       # Modélisation des milieux de propagation
├── AOT_Optic/        # Modélisation des sources optiques
└── AOT_Recon/        # Algorithmes de reconstruction
    ├── AOT_Optimizers/   # MLEM, PDHG, LS, DEPIERRO, MAPEM, LBFGS
    ├── AOT_PotentialFunctions/ # Fonctions de régularisation
    └── AOT_SparseSMatrix/    # Matrices creuses (CSR, SELL)
```

### Workflow typique

```
1. Charger les données optiques et acoustiques
   ↓
2. Configurer l'expérience (Tomography)
   ↓
3. Choisir un algorithme de reconstruction
   ↓
4. Exécuter la reconstruction
   ↓
5. Visualiser et sauvegarder les résultats
```

## 🚀 Premiers pas

### Importation de la librairie

```python
# Importation de base
import numpy as np
from AOT_biomaps import Tomography, AlgebraicRecon, AnalyticRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType

# Vérifier la disponibilité du GPU
from AOT_biomaps.AOT_Recon.AOT_Kernels import check_cuda_available
print(f"CUDA disponible: {check_cuda_available()}")
```

### Configuration de base

```python
from AOT_biomaps.Config import config

# Définir le device par défaut (cpu ou gpu)
config.set_process('gpu')  # Utilise le GPU si disponible

# Activer le mode multi-CPU pour les grands datasets
config.set_multi_cpu(True)

# Définir le niveau de verbosité
config.set_verbose(True)
```

## 🔬 Reconstruction Tomographique

### Préparation des données

Avant de commencer la reconstruction, vous devez avoir :
1. Une image optique (phantom) - l'image de référence
2. Des champs acoustiques - les données de mesure

```python
import numpy as np

# Exemple: Charger des données de test
# (Dans la pratique, chargez vos propres données)
T, Z, X, N = 10, 50, 50, 20

# Générer une image optique de test (phantom)
optic_image = np.random.rand(Z, X).astype(np.float32)

# Générer des champs acoustiques de test
acoustic_fields = [type('AcousticField', (), {
    'field': np.random.rand(T, Z, X).astype(np.float32)
})() for _ in range(N)]
```

### Reconstruction Algébrique (MLEM)

**MLEM (Maximum Likelihood Expectation Maximization)** est l'algorithme le plus couramment utilisé pour la reconstruction tomographique.

#### Exemple de base

```python
from AOT_biomaps import Tomography, AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType

# Créer une expérience de tomographie
experiment = Tomography(
    optic_image=optic_image,
    acoustic_fields=acoustic_fields
)

# Configurer la reconstruction MLEM
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType=OptimizerType.MLEM,
    numIterations=100,
    isGPU=False  # Utiliser le CPU
)

# Exécuter la reconstruction
recon.run(withTumor=True)

# Récupérer les résultats
reconstructed_image = recon.reconPhantom[-1]  # Dernière itération
print(f"Forme de l'image reconstruite: {reconstructed_image.shape}")
```

#### Avec matrice creuse (recommandé pour les grands datasets)

```python
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix

# Créer une matrice creuse (CSR ou SELL)
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',  # ou 'CSR'
    device='gpu'  # ou 'cpu'
)

# Allouer la matrice
sparse_matrix.allocate()

# Utiliser la matrice creuse dans MLEM
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import MLEM_sparse

result, indices = MLEM_sparse(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,  # Données de mesure
    numIterations=50,
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

#### Paramètres avancés MLEM

```python
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType=OptimizerType.MLEM,
    numIterations=200,           # Nombre d'itérations
    isSavingEachIteration=True, # Sauvegarder chaque itération
    max_saves=100,              # Nombre max de sauvegardes
    denominator_threshold=1e-6, # Seuil pour éviter division par zéro
    show_logs=True,             # Afficher la barre de progression
    isGPU=True                  # Utiliser le GPU
)
```

### Reconstruction Analytique

La reconstruction analytique utilise des méthodes de rétroprojection filtrée pour une reconstruction rapide.

```python
from AOT_biomaps import AnalyticRecon
from AOT_biomaps.AOT_Recon.ReconEnums import AnalyticType

# Créer une reconstruction analytique
analytic_recon = AnalyticRecon(
    experiment=experiment,
    analyticType=AnalyticType.iRADON,  # ou FBP (Filtered Back Projection)
    Lc=0.01,  # Longueur de cohérence (pour iRADON)
    isGPU=True
)

# Exécuter la reconstruction
analytic_recon.run(withTumor=True)

# Récupérer le résultat
reconstructed_image = analytic_recon.reconPhantom
```

### Reconstruction Bayésienne (MAPEM)

**MAPEM (Maximum A Posteriori Expectation Maximization)** ajoute des informations a priori pour améliorer la reconstruction.

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MAPEM import MAPEM
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic import Omega_QUADRATIC

# Définir la fonction de potentiel (régularisation)
Omega = Omega_QUADRATIC

# Exécuter MAPEM
result, indices = MAPEM(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,
    Omega=Omega_QUADRATIC,
    beta=0.1,  # Paramètre de régularisation
    numIterations=100,
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

### Reconstruction par Moindres Carrés (LS)

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.LS import LS

result, indices = LS(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,
    numIterations=100,
    alpha=0.01,  # Paramètre de régularisation
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

### Reconstruction Primal-Dual (PDHG)

**PDHG (Primal-Dual Hybrid Gradient)** est efficace pour les problèmes avec régularisation TV (Total Variation).

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.PDHG import CP_TV

result, indices = CP_TV(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,
    alpha=0.01,    # Paramètre de régularisation L1
    beta=1e-4,     # Paramètre de régularisation TV
    theta=1.0,     # Paramètre de relaxation
    numIterations=1000,
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

## 🗃️ Utilisation des Matrices Creuses

### Pourquoi utiliser des matrices creuses ?

Les matrices creuses permettent de :
- Réduire la consommation mémoire (jusqu'à 95% de réduction)
- Accélérer les calculs (surtout sur GPU)
- Traiter des datasets plus grands

### Comparaison des formats

| Format | Avantages | Inconvénients | Meilleur pour |
|--------|-----------|---------------|---------------|
| Dense | Simple à implémenter | Très gourmand en mémoire | Petits datasets |
| CSR | Économique en mémoire | Accès aléatoire lent | CPU, datasets moyens |
| SELL | Optimisé pour GPU | Construction plus complexe | GPU, grands datasets |

### Création d'une matrice creuse

```python
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix, SparseMatrix

# Méthode 1: Utiliser la factory function
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',  # ou 'CSR'
    device='gpu',
    block_rows=64,
    relative_threshold=0.3
)

# Méthode 2: Créer directement
from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL import SparseSMatrix_SELL

sell_matrix = SparseSMatrix_SELL(
    manip=experiment.AcousticFields,
    device='gpu',
    slice_height=32
)

# Allouer et construire la matrice
sparse_matrix.allocate()

# Utiliser la matrice
projection = sparse_matrix.projection(theta)  # Projection: q = A * theta
backprojection = sparse_matrix.backprojection(e)  # Rétroprojection: c = A^T * e
```

### Paramètres de la matrice creuse

```python
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',      # 'CSR' ou 'SELL'
    device='gpu',           # 'cpu' ou 'gpu'
    block_rows=64,          # Taille des blocs pour le traitement
    relative_threshold=0.3, # Seuil pour considérer une valeur comme non-nulle
    slice_height=32         # Hauteur des slices (pour SELL uniquement)
)
```

### Gestion de la mémoire

```python
# Libérer la mémoire GPU
sparse_matrix.free()

# Utiliser un context manager pour la gestion automatique
with create_sparse_matrix(manip, matrix_type='SELL', device='gpu') as sm:
    # La matrice est automatiquement allouée
    result = sm.projection(theta)
    # La mémoire est libérée automatiquement à la sortie du bloc
```

## 🔊 Simulation Acoustique

### Types d'ondes acoustiques

La librairie supporte plusieurs types d'ondes :
- **PlaneWave**: Onde plane
- **FocusedWave**: Onde focalisée
- **StructuredWave**: Onde structurée
- **IrregularWave**: Onde irrégulière

### Création d'une onde plane

```python
from AOT_biomaps.AOT_Acoustic.PlaneWave import PlaneWave

# Créer une onde plane
plane_wave = PlaneWave(
    frequency=1e6,      # Fréquence en Hz
    direction=[1, 0],   # Direction de propagation
    amplitude=1.0,      # Amplitude
    phase=0.0,          # Phase initiale
    sampling_rate=1e7   # Fréquence d'échantillonnage
)

# Générer le champ acoustique
field = plane_wave.generate_field(
    size=(100, 100, 100),  # Taille du champ (T, Z, X)
    speed_of_sound=1500   # Vitesse du son en m/s
)
```

### Création d'une onde focalisée

```python
from AOT_biomaps.AOT_Acoustic.FocusedWave import FocusedWave

# Créer une onde focalisée
focused_wave = FocusedWave(
    frequency=1e6,
    focal_point=[50, 50, 50],  # Point focal (Z, X, Y)
    amplitude=1.0,
    radius=20,                 # Rayon du transducteur
    sampling_rate=1e7
)

# Générer le champ
field = focused_wave.generate_field(
    size=(100, 100, 100),
    speed_of_sound=1500
)
```

## 📊 Visualisation des Résultats

### Visualisation de base avec Matplotlib

```python
import matplotlib.pyplot as plt

# Visualiser l'image optique originale
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(experiment.OpticImage.phantom, cmap='hot')
plt.title('Image Optique (Phantom)')
plt.colorbar()

# Visualiser l'image reconstruite
plt.subplot(1, 2, 2)
plt.imshow(recon.reconPhantom[-1], cmap='hot')
plt.title('Image Reconstruite')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### Visualisation avec la méthode show()

```python
# La classe Recon a une méthode show() intégrée
recon.show(withTumor=True, savePath='results/')
```

### Visualisation des itérations

```python
# Visualiser plusieurs itérations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    iteration = i * 10  # Afficher toutes les 10 itérations
    if iteration < len(recon.reconPhantom):
        ax.imshow(recon.reconPhantom[iteration], cmap='hot')
        ax.set_title(f'Itération {iteration}')
        ax.axis('off')
plt.tight_layout()
plt.show()
```

### Calcul des métriques

```python
# Calculer le MSE (Mean Squared Error)
recon.calculateMSE(withTumor=True)
print(f"MSE: {recon.MSE}")

# Calculer le SSIM (Structural Similarity Index)
recon.calculateSSIM(withTumor=True)
print(f"SSIM: {recon.SSIM}")

# Calculer le CRC (Contrast Recovery Coefficient)
recon.calculateCRC(use_ROI=True)
print(f"CRC: {recon.CRC}")
```

## 📝 Exemples Complets

### Exemple 1: Reconstruction MLEM complète

```python
import numpy as np
from AOT_biomaps import Tomography, AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType
from AOT_biomaps.Config import config

# Configuration
config.set_process('gpu')
config.set_verbose(True)

# Générer des données de test
T, Z, X, N = 10, 64, 64, 32
optic_image = np.random.rand(Z, X).astype(np.float32)
acoustic_fields = [type('AcousticField', (), {
    'field': np.random.rand(T, Z, X).astype(np.float32)
})() for _ in range(N)]

# Créer l'expérience
experiment = Tomography(
    optic_image=optic_image,
    acoustic_fields=acoustic_fields
)

# Configurer et exécuter MLEM
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType=OptimizerType.MLEM,
    numIterations=50,
    isSavingEachIteration=True,
    show_logs=True
)

recon.run(withTumor=True)

# Afficher les résultats
print(f"Nombre d'itérations: {len(recon.reconPhantom)}")
print(f"Forme des images: {recon.reconPhantom[0].shape}")

# Calculer les métriques
recon.calculateMSE(withTumor=True)
recon.calculateSSIM(withTumor=True)
print(f"MSE final: {recon.MSE[-1] if isinstance(recon.MSE, list) else recon.MSE}")
print(f"SSIM final: {recon.SSIM[-1] if isinstance(recon.SSIM, list) else recon.SSIM}")

# Sauvegarder les résultats
recon.save(withTumor=True, saveDir='results/')
```

### Exemple 2: Comparaison CPU vs GPU

```python
import time
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import MLEM_sparse

# Créer une matrice creuse
sparse_matrix_cpu = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',
    device='cpu'
)
sparse_matrix_cpu.allocate()

sparse_matrix_gpu = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',
    device='gpu'
)
sparse_matrix_gpu.allocate()

# Mesurer le temps CPU
start = time.time()
result_cpu, _ = MLEM_sparse(
    SMatrix=sparse_matrix_cpu,
    y=experiment.AcousticFields[0].field,
    numIterations=20,
    show_logs=False,
    device='cpu'
)
cpu_time = time.time() - start

# Mesurer le temps GPU
start = time.time()
result_gpu, _ = MLEM_sparse(
    SMatrix=sparse_matrix_gpu,
    y=experiment.AcousticFields[0].field,
    numIterations=20,
    show_logs=False,
    device='gpu'
)
gpu_time = time.time() - start

print(f"Temps CPU: {cpu_time:.3f}s")
print(f"Temps GPU: {gpu_time:.3f}s")
print(f"Accélération: {cpu_time/gpu_time:.1f}x")
```

### Exemple 3: Reconstruction avec différentes méthodes

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers import MLEM, LS, PDHG
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix

# Créer une matrice creuse
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',
    device='gpu'
)
sparse_matrix.allocate()

# Tester différentes méthodes
methods = {
    'MLEM': MLEM.MLEM_sparse,
    'LS': LS.LS,
    'PDHG': PDHG.CP_TV
}

results = {}
for name, method in methods.items():
    try:
        if name == 'PDHG':
            result, _ = method(
                SMatrix=sparse_matrix,
                y=experiment.AcousticFields[0].field,
                alpha=0.01,
                beta=1e-4,
                numIterations=50,
                show_logs=False
            )
        else:
            result, _ = method(
                SMatrix=sparse_matrix,
                y=experiment.AcousticFields[0].field,
                numIterations=50,
                show_logs=False
            )
        results[name] = result
        print(f"{name}: Succès")
    except Exception as e:
        print(f"{name}: Échec - {e}")

# Comparer les résultats
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
for (name, result), ax in zip(results.items(), axes):
    ax.imshow(result, cmap='hot')
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## ✅ Bonnes Pratiques

### 1. Gestion de la mémoire

```python
# Toujours libérer la mémoire GPU après utilisation
sparse_matrix.free()

# Utiliser des context managers pour une gestion automatique
with create_sparse_matrix(...) as sm:
    # Travailler avec la matrice
    result = sm.projection(theta)
# La mémoire est libérée automatiquement
```

### 2. Choix du device

```python
# Vérifier la disponibilité du GPU
from AOT_biomaps.AOT_Recon.AOT_Kernels import check_cuda_available

if check_cuda_available():
    device = 'gpu'
    print("Utilisation du GPU")
else:
    device = 'cpu'
    print("Utilisation du CPU")
```

### 3. Sauvegarde des résultats

```python
# Sauvegarder régulièrement pendant les longues reconstructions
recon = AlgebraicRecon(
    experiment=experiment,
    numIterations=1000,
    isSavingEachIteration=True,
    max_saves=100  # Sauvegarder 100 itérations
)

# Sauvegarder dans un dossier dédié
recon.save(withTumor=True, saveDir='results/experience_001/')
```

### 4. Reproductibilité

```python
# Fixer la graine aléatoire pour la reproductibilité
import numpy as np
np.random.seed(42)

# Utiliser la même graine pour toutes les opérations
from AOT_biomaps.Config import config
config.set_seed(42)
```

### 5. Gestion des erreurs

```python
try:
    recon.run(withTumor=True)
    recon.calculateMSE()
    recon.calculateSSIM()
except Exception as e:
    print(f"Erreur lors de la reconstruction: {e}")
    # Sauvegarder l'état avant l'erreur
    if hasattr(recon, 'reconPhantom'):
        np.save('recon_error.npy', recon.reconPhantom)
```

## 📚 Ressources supplémentaires

- [Installation](INSTALLATION.md) - Guide d'installation
- [Référence API](API_REFERENCE.md) - Documentation technique complète
- [Architecture](ARCHITECTURE.md) - Conception de la librairie
- [Contribution](CONTRIBUTING.md) - Comment contribuer au projet
