# Référence API - AOT_biomaps

Cette documentation technique décrit toutes les classes, fonctions et modules disponibles dans la librairie AOT_biomaps.

## 📚 Table des matières

- [Modules Principaux](#-modules-principaux)
- [AOT_Recon](#-aot_recon)
  - [Classes de Reconstruction](#classes-de-reconstruction)
  - [Optimiseurs](#optimiseurs)
  - [Matrices Creuses](#matrices-creuses)
  - [Noyaux et Fonctions Utilitaires](#noyaux-et-fonctions-utilitaires)
- [AOT_Acoustic](#-aot_acoustic)
- [AOT_Optic](#-aot_optic)
- [AOT_Experiment](#-aot_experiment)
- [AOT_Medium](#-aot_medium)
- [Énumérations](#énumérations)
- [Configuration](#configuration)

---

## 📦 Modules Principaux

### AOT_biomaps (Package racine)

```python
from AOT_biomaps import (
    Tomography,           # Classe principale pour les expériences
    AlgebraicRecon,      # Reconstruction algébrique
    AnalyticRecon,        # Reconstruction analytique
    BayesianRecon,        # Reconstruction bayésienne
    DeepLearningRecon,    # Reconstruction par deep learning
    PrimalDualRecon,      # Reconstruction primal-dual
    config,               # Configuration globale
    Settings             # Paramètres par défaut
)
```

---

## 🔬 AOT_Recon

### Classes de Reconstruction

#### AlgebraicRecon

**Classe** : `AOT_biomaps.AOT_Recon.AlgebraicRecon.AlgebraicRecon`

Reconstruction tomographique utilisant des méthodes algébriques (itératives).

**Héritage** : `Recon` → `AlgebraicRecon`

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `reconPhantom` | list or ndarray | Images reconstruites avec tumeur |
| `reconLaser` | list or ndarray | Images reconstruites sans tumeur |
| `experiment` | Tomography | Expérience associée |
| `reconType` | ReconType | Type de reconstruction |
| `MSE` | float or list | Mean Squared Error |
| `SSIM` | float or list | Structural Similarity Index |
| `CRC` | float or list | Contrast Recovery Coefficient |

**Méthodes** :

```python
# Constructeur
AlgebraicRecon(
    experiment: Tomography,
    saveDir: str = None,
    isGPU: bool = config.get_process() == 'gpu',
    isMultiCPU: bool = True
)

# Exécuter la reconstruction
run(withTumor: bool = True)

# Sauvegarder les résultats
save(
    withTumor: bool = True,
    overwrite: bool = False,
    date: str = None,
    show_logs: bool = True
)

# Calculer les métriques
calculateMSE(withTumor: bool = True)
calculateSSIM(withTumor: bool = True, show_log: bool = False)
calculateCRC(use_ROI: bool = True)

# Visualiser les résultats
show(withTumor: bool = True, savePath: str = None, scale: str = 'same')
```

**Exemple** :
```python
from AOT_biomaps import Tomography, AlgebraicRecon

experiment = Tomography(...)
recon = AlgebraicRecon(experiment=experiment, isGPU=True)
recon.run(withTumor=True)
recon.calculateMSE()
recon.show()
```

---

#### AnalyticRecon

**Classe** : `AOT_biomaps.AOT_Recon.AnalyticRecon.AnalyticRecon`

Reconstruction tomographique utilisant des méthodes analytiques (rétroprojection filtrée).

**Héritage** : `Recon` → `AnalyticRecon`

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `analyticType` | AnalyticType | Type de reconstruction analytique |
| `Lc` | float | Longueur de cohérence (pour iRADON) |
| `AOsignal_demoldulated` | ndarray | Signal AOT démodulé |

**Méthodes** :

```python
# Constructeur
AnalyticRecon(
    analyticType: AnalyticType,
    Lc: float = None,
    **kwargs
)

# Exécuter la reconstruction
run(withTumor: bool = True)
```

**Exemple** :
```python
from AOT_biomaps import AnalyticRecon
from AOT_biomaps.AOT_Recon.ReconEnums import AnalyticType

recon = AnalyticRecon(
    experiment=experiment,
    analyticType=AnalyticType.iRADON,
    Lc=0.01
)
recon.run(withTumor=True)
```

---

### Optimiseurs

#### Module MLEM

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM`

```python
# Fonction principale MLEM (unifiée)
MLEM(
    SMatrix: Union[SparseMatrix, ndarray],
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    withTumor: bool = True,
    device: Optional[str] = None,
    denominator_threshold: float = 1e-6,
    max_saves: int = 5000,
    show_logs: bool = True,
    smatrixType: SMatrixType = SMatrixType.SELL
) -> Tuple[Union[np.ndarray, list], Optional[list]]

# MLEM pour matrices denses
MLEM_dense(
    SMatrix: ndarray,
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    tumor_str: str = "WITH",
    max_saves: int = 5000,
    denominator_threshold: float = 1e-6,
    show_logs: bool = True,
    device: Optional[str] = None
) -> Tuple[Union[np.ndarray, list], Optional[list]]

# MLEM pour matrices creuses
MLEM_sparse(
    SMatrix: SparseMatrix,
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    tumor_str: str = "WITH",
    max_saves: int = 5000,
    denominator_threshold: float = 1e-6,
    show_logs: bool = True,
    device: Optional[str] = None
) -> Tuple[Union[np.ndarray, list], Optional[list]]
```

**Exemple** :
```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import MLEM, MLEM_sparse

# Avec matrice dense
result, indices = MLEM_dense(SMatrix, y, numIterations=50)

# Avec matrice creuse
result, indices = MLEM_sparse(sparse_matrix, y, numIterations=50)
```

---

#### Module PDHG

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_Optimizers.PDHG`

```python
# PDHG avec régularisation TV
CP_TV(
    SMatrix: Union[SparseMatrix, ndarray],
    y: Union[np.ndarray, cp.ndarray],
    alpha: Optional[float] = None,
    beta: float = 1e-4,
    theta: float = 1.0,
    numIterations: int = 5000,
    isSavingEachIteration: bool = True,
    L: Optional[float] = None,
    noiseType: NoiseType = NoiseType.Gaussian,
    smatrixType: SMatrixType = SMatrixType.SELL,
    show_logs: bool = True,
    device: Optional[str] = None
) -> Tuple[Union[np.ndarray, list], Optional[list]]

# PDHG pour matrices denses avec TV
CP_TV_dense_cupy(
    SMatrix: ndarray,
    y: Union[np.ndarray, cp.ndarray],
    alpha: float,
    beta: float,
    theta: float,
    numIterations: int,
    isSavingEachIteration: bool,
    tumor_str: str,
    max_saves: int,
    show_logs: bool
)

# PDHG avec régularisation KL
CP_KL_dense_cupy(
    SMatrix: ndarray,
    y: Union[np.ndarray, cp.ndarray],
    alpha: float,
    theta: float,
    numIterations: int,
    isSavingEachIteration: bool,
    tumor_str: str,
    max_saves: int,
    show_logs: bool
)
```

**Exemple** :
```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.PDHG import CP_TV

result, indices = CP_TV(
    SMatrix=sparse_matrix,
    y=y,
    alpha=0.01,
    beta=1e-4,
    numIterations=1000
)
```

---

#### Module LS (Moindres Carrés)

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_Optimizers.LS`

```python
# LS principal
LS(
    SMatrix: Union[SparseMatrix, ndarray],
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int = 100,
    alpha: float = 0.01,
    isSavingEachIteration: bool = True,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
    smatrixType: SMatrixType = SMatrixType.SELL,
    device: Optional[str] = None
) -> Tuple[Union[np.ndarray, list], Optional[list]]

# LS avec matrice CSR
LS_CG_sparseCSR_cupy(
    SMatrix: SparseMatrix,
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int,
    alpha: float,
    isSavingEachIteration: bool,
    tumor_str: str,
    max_saves: int,
    show_logs: bool
)

# LS avec matrice SELL
LS_CG_sparseSELL_cupy(
    SMatrix: SparseMatrix,
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int,
    alpha: float,
    isSavingEachIteration: bool,
    tumor_str: str,
    max_saves: int,
    show_logs: bool
)
```

---

#### Module MAPEM

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_Optimizers.MAPEM`

```python
MAPEM(
    SMatrix: Union[SparseMatrix, ndarray],
    y: Union[np.ndarray, cp.ndarray],
    Omega: callable,
    beta: float,
    delta: Optional[float] = None,
    gamma: Optional[float] = None,
    sigma: Optional[float] = None,
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
    smatrixType: SMatrixType = SMatrixType.SELL,
    device: Optional[str] = None
) -> Tuple[Union[np.ndarray, list], Optional[list]]
```

---

#### Module DEPIERRO

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_Optimizers.DEPIERRO`

```python
DEPIERRO(
    SMatrix: Union[SparseMatrix, ndarray],
    y: Union[np.ndarray, cp.ndarray],
    numIterations: int,
    beta: float,
    sigma: float,
    isSavingEachIteration: bool,
    withTumor: bool,
    max_saves: int,
    show_logs: bool
) -> Tuple[Union[np.ndarray, list], Optional[list]]
```

---

#### Module LBFGS

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_Optimizers.LBFGS`

```python
lbfgs_aniso_tv(
    SMatrix: SparseMatrix,
    y: ndarray,
    n_epochs: int = 100,
    window: int = 5,
    tol: float = 1e-5,
    alpha_x: float = 0.01,
    alpha_z: float = 0.05,
    eps: float = 1e-4
) -> Tuple[ndarray, list]
```

---

### Matrices Creuses

#### SparseMatrix (Wrapper)

**Classe** : `AOT_biomaps.AOT_Recon.SparseMatrixWrapper.SparseMatrix`

Wrapper unifié pour les matrices creuses (CSR et SELL).

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `matrix_type` | str | Type de matrice ('CSR' ou 'SELL') |
| `device` | str | Device ('cpu' ou 'gpu') |
| `N` | int | Nombre d'angles |
| `T` | int | Nombre de pas de temps |
| `Z` | int | Taille en Z |
| `X` | int | Taille en X |
| `total_nnz` | int | Nombre total d'éléments non-nuls |

**Méthodes** :

```python
# Constructeur
SparseMatrix(
    manip: Any,
    matrix_type: str = 'CSR',
    device: Optional[str] = None,
    **kwargs
)

# Allouer la matrice
allocate()

# Opérations matricielles
projection(theta: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
backprojection(e: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
apply_normalization(x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]

# Propriétés
get_norm_factor_inv() -> Union[np.ndarray, cp.ndarray]
get_matrix_size() -> dict
compute_density() -> float

# Gestion de la mémoire
free()

# Utilisation comme context manager
with SparseMatrix(...) as sm:
    # La matrice est automatiquement allouée
    result = sm.projection(theta)
# La mémoire est libérée automatiquement
```

**Exemple** :
```python
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import SparseMatrix

sparse_matrix = SparseMatrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',
    device='gpu'
)
sparse_matrix.allocate()

# Projection
q = sparse_matrix.projection(theta)

# Rétroprojection
c = sparse_matrix.backprojection(e)

# Libérer la mémoire
sparse_matrix.free()
```

---

#### SparseSMatrix_CSR

**Classe** : `AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_CSR`

Matrice creuse au format CSR (Compressed Sparse Row).

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `row_ptr` | ndarray | Pointeurs de ligne |
| `h_col_ind` | ndarray | Indices de colonne |
| `h_values` | ndarray | Valeurs non-nulles |
| `norm_factor_inv` | ndarray | Facteurs de normalisation |
| `total_nnz` | int | Nombre total d'éléments non-nuls |

**Méthodes** :

```python
# Constructeur
SparseSMatrix_CSR(
    manip: Any,
    block_rows: int = 64,
    relative_threshold: float = 0.3,
    device: Optional[str] = None
)

# Allouer la matrice
allocate()

# Opérations
projection(theta: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
backprojection(e: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]

# Propriétés
getMatrixSize() -> dict
compute_density() -> float
free()
flipAngle()
```

---

#### SparseSMatrix_SELL

**Classe** : `AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL`

Matrice creuse au format SELL-C-sigma (Sliced ELL avec padding).

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `sell_values` | ndarray | Valeurs non-nulles |
| `sell_colinds` | ndarray | Indices de colonne |
| `slice_ptr` | ndarray | Pointeurs de slice |
| `slice_len` | ndarray | Longueurs de slice |
| `total_storage` | int | Taille totale de stockage |

**Méthodes** :

```python
# Constructeur
SparseSMatrix_SELL(
    manip: Any,
    block_rows: int = 64,
    relative_threshold: float = 0.3,
    device: Optional[str] = None,
    slice_height: int = 32
)

# Allouer la matrice
allocate()

# Opérations
projection(theta: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
backprojection(e: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
apply_apodization(window_vector: Union[np.ndarray, cp.ndarray])

# Propriétés
getMatrixSize() -> dict
compute_density() -> float
free()
flipAngle()
```

---

### Noyaux et Fonctions Utilitaires

#### AOT_Kernels

**Module** : `AOT_biomaps.AOT_Recon.AOT_Kernels`

Fonctions utilitaires pour les opérations CPU/GPU.

**Fonctions de vérification** :

```python
check_cuda_available() -> bool
check_pycuda_available() -> bool
get_device_memory_info(device: Optional[str] = None) -> Tuple[int, int]
```

**Fonctions utilitaires** :

```python
# Remplissage de tableaux
fill_array_value(arr: Union[np.ndarray, cp.ndarray], value: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
fill_array_zero(arr: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Opérations arithmétiques
clamp_positive(arr: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
vector_axpby(z: Union[np.ndarray, cp.ndarray], x: Union[np.ndarray, cp.ndarray], y: Union[np.ndarray, cp.ndarray], alpha: float, beta: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
vector_minus_axpy(r: Union[np.ndarray, cp.ndarray], z: Union[np.ndarray, cp.ndarray], alpha: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
invert_vector(vec: Union[np.ndarray, cp.ndarray], clip_min: float = 1e-12, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Opérations sur matrices creuses
sparse_matrix_vector_product_csr(data: ndarray, indices: ndarray, indptr: ndarray, x: ndarray, num_rows: int, device: Optional[str] = None) -> ndarray

# Opérations MLEM
ratio_kernel(y: Union[np.ndarray, cp.ndarray], q: Union[np.ndarray, cp.ndarray], threshold: float = 1e-12, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
update_theta(theta: Union[np.ndarray, cp.ndarray], c: Union[np.ndarray, cp.ndarray], norm_factor_inv: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Opérations TV (Total Variation)
gradient_2d(x: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
divergence_2d(p: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
proj_tv(p: Union[np.ndarray, cp.ndarray], alpha: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Opérations de préconditionnement
update_dual_data_precond(q: Union[np.ndarray, cp.ndarray], Ax: Union[np.ndarray, cp.ndarray], y: Union[np.ndarray, cp.ndarray], sigma_vec: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
update_primal_precond(x: Union[np.ndarray, cp.ndarray], gradient_combined: Union[np.ndarray, cp.ndarray], tau_vec: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Opérations de downsampling
downsample_3d(field: Union[np.ndarray, cp.ndarray], mode: str = 'avg', device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Calcul d'enveloppe
calculate_envelope_squared(field: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
```

---

## 🔊 AOT_Acoustic

### Classes d'ondes acoustiques

#### PlaneWave

**Classe** : `AOT_biomaps.AOT_Acoustic.PlaneWave.PlaneWave`

Génère des ondes planes pour la simulation acoustique.

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `frequency` | float | Fréquence en Hz |
| `direction` | list | Direction de propagation [dz, dx] |
| `amplitude` | float | Amplitude de l'onde |
| `phase` | float | Phase initiale |
| `sampling_rate` | float | Fréquence d'échantillonnage en Hz |

**Méthodes** :

```python
# Constructeur
PlaneWave(
    frequency: float,
    direction: list,
    amplitude: float = 1.0,
    phase: float = 0.0,
    sampling_rate: float = 1e7
)

# Générer le champ acoustique
generate_field(
    size: tuple,
    speed_of_sound: float = 1500.0
) -> ndarray
```

---

#### FocusedWave

**Classe** : `AOT_biomaps.AOT_Acoustic.FocusedWave.FocusedWave`

Génère des ondes acoustiques focalisées.

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `frequency` | float | Fréquence en Hz |
| `focal_point` | list | Point focal [z, x, y] |
| `amplitude` | float | Amplitude |
| `radius` | float | Rayon du transducteur |
| `sampling_rate` | float | Fréquence d'échantillonnage |

**Méthodes** :

```python
# Constructeur
FocusedWave(
    frequency: float,
    focal_point: list,
    amplitude: float = 1.0,
    radius: float = 20.0,
    sampling_rate: float = 1e7
)

# Générer le champ
generate_field(
    size: tuple,
    speed_of_sound: float = 1500.0
) -> ndarray
```

---

#### StructuredWave

**Classe** : `AOT_biomaps.AOT_Acoustic.StructuredWave.StructuredWave`

Génère des ondes structurées.

---

#### IrregularWave

**Classe** : `AOT_biomaps.AOT_Acoustic.IrregularWave.IrregularWave`

Génère des ondes irrégulières.

---

## 🎯 AOT_Optic

### Classes optiques

#### Laser

**Classe** : `AOT_biomaps.AOT_Optic.Laser.Laser`

Modélise une source laser pour l'imagerie acousto-optique.

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `wavelength` | float | Longueur d'onde en nm |
| `intensity` | ndarray | Distribution d'intensité |
| `beam_width` | float | Largeur du faisceau |

---

## 🧪 AOT_Experiment

### Tomography

**Classe** : `AOT_biomaps.AOT_Experiment.Tomography.Tomography`

Classe principale pour gérer les expériences de tomographie acousto-optique.

**Attributs** :

| Attribut | Type | Description |
|----------|------|-------------|
| `OpticImage` | OpticImage | Image optique associée |
| `AcousticFields` | list | Liste des champs acoustiques |
| `params` | dict | Paramètres de l'expérience |

**Méthodes** :

```python
# Constructeur
Tomography(
    optic_image: Optional[ndarray] = None,
    acoustic_fields: Optional[list] = None,
    params: Optional[dict] = None
)

# Charger les données
load_optic_image(path: str)
load_acoustic_fields(path: str)

# Sauvegarder les données
save_optic_image(path: str)
save_acoustic_fields(path: str)
```

---

## 🌍 AOT_Medium

### Classes de milieux

#### HomogeneousMedium

**Classe** : `AOT_biomaps.AOT_Medium.HomogeneousMedium.HomogeneousMedium`

Milieu homogène avec propriétés constantes.

#### PVAMedium

**Classe** : `AOT_biomaps.AOT_Medium.PVAMedium.PVAMedium`

Milieu avec propriétés variables (PVA - PolyVinyl Alcohol).

#### BubbleMedium

**Classe** : `AOT_biomaps.AOT_Medium.BubbleMedium.BubbleMedium`

Milieu avec bulles.

---

## 📜 Énumérations

### ReconEnums

**Module** : `AOT_biomaps.AOT_Recon.ReconEnums`

```python
from AOT_biomaps.AOT_Recon.ReconEnums import (
    ReconType,       # Type de reconstruction
    OptimizerType,   # Type d'optimiseur
    AnalyticType,    # Type de reconstruction analytique
    SMatrixType,     # Type de matrice creuse
    ProcessType,     # Type de traitement
    NoiseType        # Type de bruit
)

# Valeurs possibles
ReconType.Algebraic      # Reconstruction algébrique
ReconType.Analytic       # Reconstruction analytique
ReconType.Bayesian       # Reconstruction bayésienne
ReconType.DeepLearning   # Reconstruction par deep learning
ReconType.Convex         # Reconstruction convexe

OptimizerType.MLEM
OptimizerType.PDHG
OptimizerType.LS
OptimizerType.DEPIERRO
OptimizerType.MAPEM
OptimizerType.LBFGS

SMatrixType.DENSE
SMatrixType.CSR
SMatrixType.SELL

AnalyticType.iRADON
AnalyticType.FBP

ProcessType.CPU
ProcessType.GPU
ProcessType.MultiCPU

NoiseType.Gaussian
NoiseType.Poisson
```

---

### AcousticEnums

**Module** : `AOT_biomaps.AOT_Acoustic.AcousticEnums`

```python
from AOT_biomaps.AOT_Acoustic.AcousticEnums import (
    WaveType,
    BoundaryCondition
)
```

---

### OpticEnums

**Module** : `AOT_biomaps.AOT_Optic.OpticEnums`

---

### MediumEnums

**Module** : `AOT_biomaps.AOT_Medium.MediumEnums`

---

## ⚙️ Configuration

### Config

**Module** : `AOT_biomaps.Config`

Gère la configuration globale de la librairie.

**Fonctions** :

```python
from AOT_biomaps.Config import config

# Obtenir la configuration actuelle
config.get_process()  # Retourne 'cpu' ou 'gpu'
config.get_multi_cpu()  # Retourne True ou False
config.get_verbose()  # Retourne True ou False

# Définir la configuration
config.set_process(device: str)  # 'cpu' ou 'gpu'
config.set_multi_cpu(enabled: bool)
config.set_verbose(enabled: bool)
config.set_seed(seed: int)
```

---

### Settings

**Module** : `AOT_biomaps.Settings`

Contient les paramètres par défaut de la librairie.

---

## 📊 Fonctions de calcul de métriques

### ReconTools

**Module** : `AOT_biomaps.AOT_Recon.ReconTools`

```python
from AOT_biomaps.AOT_Recon.ReconTools import (
    mse,                    # Mean Squared Error
    ssim,                   # Structural Similarity Index
    fourierz_gpu,           # Transformée de Fourier en z (GPU)
    ifourierx_gpu,          # Transformée de Fourier inverse en x (GPU)
    rotate_theta_gpu,       # Rotation theta (GPU)
    filter_radon_gpu,       # Filtrage Radon (GPU)
    ifourierz_gpu,          # Transformée de Fourier inverse en z (GPU)
    EvalDelayLawOS_center,   # Évaluation de la loi de retard
    calculate_memory_requirement,  # Calcul des besoins mémoire
    check_gpu_memory,        # Vérification de la mémoire GPU
    _build_adjacency_sparse  # Construction de la matrice d'adjacence
)

# Exemple
mse_value = mse(image1, image2)
ssim_value = ssim(image1, image2, data_range=1.0)
```

---

## 🎨 Fonctions de potentiel

### Module Quadratic

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic`

```python
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic import (
    _Omega_QUADRATIC_CPU,
    _Omega_QUADRATIC_GPU
)
```

### Module Huber

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Huber`

```python
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Huber import (
    _Omega_HUBER_PIECEWISE_CPU,
    _Omega_HUBER_PIECEWISE_GPU
)
```

### Module RelativeDifferences

**Fonctions** : `AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.RelativeDifferences`

```python
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.RelativeDifferences import (
    _Omega_RELATIVE_DIFFERENCE_CPU,
    _Omega_RELATIVE_DIFFERENCE_GPU
)
```

---

## 📝 Index des classes et fonctions

### Classes principales

| Classe | Module | Description |
|--------|--------|-------------|
| `Tomography` | AOT_Experiment | Expérience de tomographie |
| `AlgebraicRecon` | AOT_Recon | Reconstruction algébrique |
| `AnalyticRecon` | AOT_Recon | Reconstruction analytique |
| `SparseMatrix` | AOT_Recon.SparseMatrixWrapper | Wrapper de matrice creuse |
| `SparseSMatrix_CSR` | AOT_Recon.AOT_SparseSMatrix | Matrice CSR |
| `SparseSMatrix_SELL` | AOT_Recon.AOT_SparseSMatrix | Matrice SELL |
| `PlaneWave` | AOT_Acoustic | Onde plane |
| `FocusedWave` | AOT_Acoustic | Onde focalisée |
| `Laser` | AOT_Optic | Source laser |

### Fonctions principales

| Fonction | Module | Description |
|----------|--------|-------------|
| `MLEM` | AOT_Optimizers.MLEM | Algorithme MLEM |
| `PDHG` / `CP_TV` | AOT_Optimizers.PDHG | Algorithme PDHG |
| `LS` | AOT_Optimizers.LS | Moindres carrés |
| `MAPEM` | AOT_Optimizers.MAPEM | Algorithme MAPEM |
| `DEPIERRO` | AOT_Optimizers.DEPIERRO | Algorithme DEPIERRO |
| `create_sparse_matrix` | SparseMatrixWrapper | Créer une matrice creuse |
| `check_cuda_available` | AOT_Kernels | Vérifier CUDA |
| `mse` | ReconTools | Calculer MSE |
| `ssim` | ReconTools | Calculer SSIM |

---

## 🙏 Support

Pour toute question ou problème, veuillez :
1. Consulter la [documentation](https://github.com/LucasDuclos/AcoustoOpticTomography/tree/main/docs)
2. Vérifier les [issues existantes](https://github.com/LucasDuclos/AcoustoOpticTomography/issues)
3. Ouvrir une nouvelle [issue](https://github.com/LucasDuclos/AcoustoOpticTomography/issues/new)

---

**Retour à la [documentation principale](README.md)**
