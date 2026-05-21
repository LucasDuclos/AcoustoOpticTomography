# API Reference - AOT_biomaps

This technical documentation describes all classes, functions, and modules available in the AOT_biomaps library.

## 📚 Table of Contents

- [Main Modules](#-main-modules)
- [AOT_Recon](#-aot_recon)
  - [Reconstruction Classes](#reconstruction-classes)
  - [Optimizers](#optimizers)
  - [Sparse Matrices](#sparse-matrices)
  - [Kernels and Utility Functions](#kernels-and-utility-functions)
- [AOT_Acoustic](#-aot_acoustic)
- [AOT_Optic](#-aot_optic)
- [AOT_Experiment](#-aot_experiment)
- [AOT_Medium](#-aot_medium)
- [Enumerations](#enumerations)
- [Configuration](#configuration)

---

## 📦 Main Modules

### AOT_biomaps (Root Package)

```python
from AOT_biomaps import (
    Tomography,           # Main class for experiments
    AlgebraicRecon,      # Algebraic reconstruction
    AnalyticRecon,        # Analytic reconstruction
    BayesianRecon,        # Bayesian reconstruction
    DeepLearningRecon,    # Deep learning reconstruction
    PrimalDualRecon,      # Primal-dual reconstruction
    config,               # Global configuration
    Settings             # Default settings
)
```

---

## 🔬 AOT_Recon

### Reconstruction Classes

#### AlgebraicRecon

**Class**: `AOT_biomaps.AOT_Recon.AlgebraicRecon.AlgebraicRecon`

Algebraic tomographic reconstruction using iterative methods.

**Inheritance**: `Recon` → `AlgebraicRecon`

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `reconPhantom` | list or ndarray | Reconstructed images with tumor |
| `reconLaser` | list or ndarray | Reconstructed images without tumor |
| `experiment` | Tomography | Associated experiment |
| `reconType` | ReconType | Reconstruction type |
| `MSE` | float or list | Mean Squared Error |
| `SSIM` | float or list | Structural Similarity Index |
| `CRC` | float or list | Contrast Recovery Coefficient |

**Methods**:

```python
# Constructor
AlgebraicRecon(
    experiment: Tomography,
    saveDir: str = None,
    isGPU: bool = config.get_process() == 'gpu',
    isMultiCPU: bool = True
)

# Run reconstruction
run(withTumor: bool = True)

# Save results
save(
    withTumor: bool = True,
    overwrite: bool = False,
    date: str = None,
    show_logs: bool = True
)

# Calculate metrics
calculateMSE(withTumor: bool = True)
calculateSSIM(withTumor: bool = True, show_log: bool = False)
calculateCRC(use_ROI: bool = True)

# Visualize results
show(withTumor: bool = True, savePath: str = None, scale: str = 'same')
```

**Example**:
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

**Class**: `AOT_biomaps.AOT_Recon.AnalyticRecon.AnalyticRecon`

Tomographic reconstruction using analytic methods (filtered backprojection).

**Inheritance**: `Recon` → `AnalyticRecon`

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `analyticType` | AnalyticType | Analytic reconstruction type |
| `Lc` | float | Coherence length (for iRADON) |
| `AOsignal_demoldulated` | ndarray | Demodulated AOT signal |

**Methods**:

```python
# Constructor
AnalyticRecon(
    analyticType: AnalyticType,
    Lc: float = None,
    **kwargs
)

# Run reconstruction
run(withTumor: bool = True)
```

**Example**:
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

### Optimizers

#### MLEM Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM`

```python
# Main MLEM function (unified)
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

# MLEM for dense matrices
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

# MLEM for sparse matrices
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

**Example**:
```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import MLEM, MLEM_sparse

# With dense matrix
result, indices = MLEM_dense(SMatrix, y, numIterations=50)

# With sparse matrix
result, indices = MLEM_sparse(sparse_matrix, y, numIterations=50)
```

---

#### PDHG Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_Optimizers.PDHG`

```python
# PDHG with TV regularization
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

# PDHG for dense matrices with TV
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

# PDHG with KL regularization
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

**Example**:
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

#### LS Module (Least Squares)

**Functions**: `AOT_biomaps.AOT_Recon.AOT_Optimizers.LS`

```python
# Main LS function
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

# LS with CSR matrix
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

# LS with SELL matrix
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

#### MAPEM Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_Optimizers.MAPEM`

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

#### DEPIERRO Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_Optimizers.DEPIERRO`

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

#### LBFGS Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_Optimizers.LBFGS`

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

### Sparse Matrices

#### SparseMatrix (Wrapper)

**Class**: `AOT_biomaps.AOT_Recon.SparseMatrixWrapper.SparseMatrix`

Unified wrapper for sparse matrices (CSR and SELL).

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `matrix_type` | str | Matrix type ('CSR' or 'SELL') |
| `device` | str | Device ('cpu' or 'gpu') |
| `N` | int | Number of angles |
| `T` | int | Number of time steps |
| `Z` | int | Size in Z |
| `X` | int | Size in X |
| `total_nnz` | int | Total number of non-zero elements |

**Methods**:

```python
# Constructor
SparseMatrix(
    manip: Any,
    matrix_type: str = 'CSR',
    device: Optional[str] = None,
    **kwargs
)

# Allocate the matrix
allocate()

# Matrix operations
projection(theta: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
backprojection(e: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
apply_normalization(x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]

# Properties
get_norm_factor_inv() -> Union[np.ndarray, cp.ndarray]
get_matrix_size() -> dict
compute_density() -> float

# Memory management
free()

# Context manager usage
with SparseMatrix(...) as sm:
    # Matrix is automatically allocated
    result = sm.projection(theta)
# Memory is automatically freed
```

**Example**:
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

# Backprojection
c = sparse_matrix.backprojection(e)

# Free memory
sparse_matrix.free()
```

---

#### SparseSMatrix_CSR

**Class**: `AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_CSR`

Sparse matrix in CSR (Compressed Sparse Row) format.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `row_ptr` | ndarray | Row pointers |
| `h_col_ind` | ndarray | Column indices |
| `h_values` | ndarray | Non-zero values |
| `norm_factor_inv` | ndarray | Normalization factors |
| `total_nnz` | int | Total number of non-zero elements |

**Methods**:

```python
# Constructor
SparseSMatrix_CSR(
    manip: Any,
    block_rows: int = 64,
    relative_threshold: float = 0.3,
    device: Optional[str] = None
)

# Allocate the matrix
allocate()

# Operations
projection(theta: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
backprojection(e: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]

# Properties
getMatrixSize() -> dict
compute_density() -> float
free()
flipAngle()
```

---

#### SparseSMatrix_SELL

**Class**: `AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL`

Sparse matrix in SELL-C-sigma (Sliced ELL) format.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `sell_values` | ndarray | Non-zero values |
| `sell_colinds` | ndarray | Column indices |
| `slice_ptr` | ndarray | Slice pointers |
| `slice_len` | ndarray | Slice lengths |
| `total_storage` | int | Total storage size |

**Methods**:

```python
# Constructor
SparseSMatrix_SELL(
    manip: Any,
    block_rows: int = 64,
    relative_threshold: float = 0.3,
    device: Optional[str] = None,
    slice_height: int = 32
)

# Allocate the matrix
allocate()

# Operations
projection(theta: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
backprojection(e: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]
apply_apodization(window_vector: Union[np.ndarray, cp.ndarray])

# Properties
getMatrixSize() -> dict
compute_density() -> float
free()
flipAngle()
```

---

### Kernels and Utility Functions

#### AOT_Kernels

**Module**: `AOT_biomaps.AOT_Recon.AOT_Kernels`

Utility functions for CPU/GPU operations.

**Verification Functions**:

```python
check_cuda_available() -> bool
check_pycuda_available() -> bool
get_device_memory_info(device: Optional[str] = None) -> Tuple[int, int]
```

**Utility Functions**:

```python
# Array filling
fill_array_value(arr: Union[np.ndarray, cp.ndarray], value: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
fill_array_zero(arr: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Arithmetic operations
clamp_positive(arr: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
vector_axpby(z: Union[np.ndarray, cp.ndarray], x: Union[np.ndarray, cp.ndarray], y: Union[np.ndarray, cp.ndarray], alpha: float, beta: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
vector_minus_axpy(r: Union[np.ndarray, cp.ndarray], z: Union[np.ndarray, cp.ndarray], alpha: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
invert_vector(vec: Union[np.ndarray, cp.ndarray], clip_min: float = 1e-12, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Sparse matrix operations
sparse_matrix_vector_product_csr(data: ndarray, indices: ndarray, indptr: ndarray, x: ndarray, num_rows: int, device: Optional[str] = None) -> ndarray

# MLEM operations
ratio_kernel(y: Union[np.ndarray, cp.ndarray], q: Union[np.ndarray, cp.ndarray], threshold: float = 1e-12, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
update_theta(theta: Union[np.ndarray, cp.ndarray], c: Union[np.ndarray, cp.ndarray], norm_factor_inv: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# TV (Total Variation) operations
gradient_2d(x: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
divergence_2d(p: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
proj_tv(p: Union[np.ndarray, cp.ndarray], alpha: float, device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Preconditioning operations
update_dual_data_precond(q: Union[np.ndarray, cp.ndarray], Ax: Union[np.ndarray, cp.ndarray], y: Union[np.ndarray, cp.ndarray], sigma_vec: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
update_primal_precond(x: Union[np.ndarray, cp.ndarray], gradient_combined: Union[np.ndarray, cp.ndarray], tau_vec: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Downsampling operations
downsample_3d(field: Union[np.ndarray, cp.ndarray], mode: str = 'avg', device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]

# Envelope calculation
calculate_envelope_squared(field: Union[np.ndarray, cp.ndarray], device: Optional[str] = None) -> Union[np.ndarray, cp.ndarray]
```

---

## 🔊 AOT_Acoustic

### Acoustic Wave Classes

#### PlaneWave

**Class**: `AOT_biomaps.AOT_Acoustic.PlaneWave.PlaneWave`

Generates plane waves for acoustic simulation.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `frequency` | float | Frequency in Hz |
| `direction` | list | Propagation direction [dz, dx] |
| `amplitude` | float | Wave amplitude |
| `phase` | float | Initial phase |
| `sampling_rate` | float | Sampling rate in Hz |

**Methods**:

```python
# Constructor
PlaneWave(
    frequency: float,
    direction: list,
    amplitude: float = 1.0,
    phase: float = 0.0,
    sampling_rate: float = 1e7
)

# Generate acoustic field
generate_field(
    size: tuple,
    speed_of_sound: float = 1500.0
) -> ndarray
```

---

#### FocusedWave

**Class**: `AOT_biomaps.AOT_Acoustic.FocusedWave.FocusedWave`

Generates focused acoustic waves.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `frequency` | float | Frequency in Hz |
| `focal_point` | list | Focal point [z, x, y] |
| `amplitude` | float | Amplitude |
| `radius` | float | Transducer radius |
| `sampling_rate` | float | Sampling rate |

**Methods**:

```python
# Constructor
FocusedWave(
    frequency: float,
    focal_point: list,
    amplitude: float = 1.0,
    radius: float = 20.0,
    sampling_rate: float = 1e7
)

# Generate field
generate_field(
    size: tuple,
    speed_of_sound: float = 1500.0
) -> ndarray
```

---

#### StructuredWave

**Class**: `AOT_biomaps.AOT_Acoustic.StructuredWave.StructuredWave`

Generates structured waves.

---

#### IrregularWave

**Class**: `AOT_biomaps.AOT_Acoustic.IrregularWave.IrregularWave`

Generates irregular waves.

---

## 🎯 AOT_Optic

### Optical Classes

#### Laser

**Class**: `AOT_biomaps.AOT_Optic.Laser.Laser`

Models a laser source for acousto-optic imaging.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `wavelength` | float | Wavelength in nm |
| `intensity` | ndarray | Intensity distribution |
| `beam_width` | float | Beam width |

---

## 🧪 AOT_Experiment

### Tomography

**Class**: `AOT_biomaps.AOT_Experiment.Tomography.Tomography`

Main class for managing acousto-optic tomography experiments.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `OpticImage` | OpticImage | Associated optical image |
| `AcousticFields` | list | List of acoustic fields |
| `params` | dict | Experiment parameters |

**Methods**:

```python
# Constructor
Tomography(
    optic_image: Optional[ndarray] = None,
    acoustic_fields: Optional[list] = None,
    params: Optional[dict] = None
)

# Load data
load_optic_image(path: str)
load_acoustic_fields(path: str)

# Save data
save_optic_image(path: str)
save_acoustic_fields(path: str)
```

---

## 🌍 AOT_Medium

### Medium Classes

#### HomogeneousMedium

**Class**: `AOT_biomaps.AOT_Medium.HomogeneousMedium.HomogeneousMedium`

Homogeneous medium with constant properties.

#### PVAMedium

**Class**: `AOT_biomaps.AOT_Medium.PVAMedium.PVAMedium`

Medium with variable properties (PVA - PolyVinyl Alcohol).

#### BubbleMedium

**Class**: `AOT_biomaps.AOT_Medium.BubbleMedium.BubbleMedium`

Medium with bubbles.

---

## 📜 Enumerations

### ReconEnums

**Module**: `AOT_biomaps.AOT_Recon.ReconEnums`

```python
from AOT_biomaps.AOT_Recon.ReconEnums import (
    ReconType,       # Reconstruction type
    OptimizerType,   # Optimizer type
    AnalyticType,    # Analytic reconstruction type
    SMatrixType,     # Sparse matrix type
    ProcessType,     # Process type
    NoiseType        # Noise type
)

# Possible values
ReconType.Algebraic      # Algebraic reconstruction
ReconType.Analytic       # Analytic reconstruction
ReconType.Bayesian       # Bayesian reconstruction
ReconType.DeepLearning   # Deep learning reconstruction
ReconType.Convex         # Convex reconstruction

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

**Module**: `AOT_biomaps.AOT_Acoustic.AcousticEnums`

```python
from AOT_biomaps.AOT_Acoustic.AcousticEnums import (
    WaveType,
    BoundaryCondition
)
```

---

### OpticEnums

**Module**: `AOT_biomaps.AOT_Optic.OpticEnums`

---

### MediumEnums

**Module**: `AOT_biomaps.AOT_Medium.MediumEnums`

---

## ⚙️ Configuration

### Config

**Module**: `AOT_biomaps.Config`

Manages global library configuration.

**Functions**:

```python
from AOT_biomaps.Config import config

# Get current configuration
config.get_process()  # Returns 'cpu' or 'gpu'
config.get_multi_cpu()  # Returns True or False
config.get_verbose()  # Returns True or False

# Set configuration
config.set_process(device: str)  # 'cpu' or 'gpu'
config.set_multi_cpu(enabled: bool)
config.set_verbose(enabled: bool)
config.set_seed(seed: int)
```

---

### Settings

**Module**: `AOT_biomaps.Settings`

Contains default library settings.

---

## 📊 Metric Calculation Functions

### ReconTools

**Module**: `AOT_biomaps.AOT_Recon.ReconTools`

```python
from AOT_biomaps.AOT_Recon.ReconTools import (
    mse,                    # Mean Squared Error
    ssim,                   # Structural Similarity Index
    fourierz_gpu,           # Fourier transform in z (GPU)
    ifourierx_gpu,          # Inverse Fourier transform in x (GPU)
    rotate_theta_gpu,       # Theta rotation (GPU)
    filter_radon_gpu,       # Radon filtering (GPU)
    ifourierz_gpu,          # Inverse Fourier transform in z (GPU)
    EvalDelayLawOS_center,   # Delay law evaluation
    calculate_memory_requirement,  # Memory requirement calculation
    check_gpu_memory,        # GPU memory check
    _build_adjacency_sparse  # Adjacency matrix construction
)

# Example
mse_value = mse(image1, image2)
ssim_value = ssim(image1, image2, data_range=1.0)
```

---

## 🎨 Potential Functions

### Quadratic Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic`

```python
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic import (
    _Omega_QUADRATIC_CPU,
    _Omega_QUADRATIC_GPU
)
```

### Huber Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Huber`

```python
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Huber import (
    _Omega_HUBER_PIECEWISE_CPU,
    _Omega_HUBER_PIECEWISE_GPU
)
```

### RelativeDifferences Module

**Functions**: `AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.RelativeDifferences`

```python
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.RelativeDifferences import (
    _Omega_RELATIVE_DIFFERENCE_CPU,
    _Omega_RELATIVE_DIFFERENCE_GPU
)
```

---

## 📝 Class and Function Index

### Main Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Tomography` | AOT_Experiment | Tomography experiment |
| `AlgebraicRecon` | AOT_Recon | Algebraic reconstruction |
| `AnalyticRecon` | AOT_Recon | Analytic reconstruction |
| `SparseMatrix` | AOT_Recon.SparseMatrixWrapper | Sparse matrix wrapper |
| `SparseSMatrix_CSR` | AOT_Recon.AOT_SparseSMatrix | CSR matrix |
| `SparseSMatrix_SELL` | AOT_Recon.AOT_SparseSMatrix | SELL matrix |
| `PlaneWave` | AOT_Acoustic | Plane wave |
| `FocusedWave` | AOT_Acoustic | Focused wave |
| `Laser` | AOT_Optic | Laser source |

### Main Functions

| Function | Module | Description |
|----------|--------|-------------|
| `MLEM` | AOT_Optimizers.MLEM | MLEM algorithm |
| `PDHG` / `CP_TV` | AOT_Optimizers.PDHG | PDHG algorithm |
| `LS` | AOT_Optimizers.LS | Least squares |
| `MAPEM` | AOT_Optimizers.MAPEM | MAPEM algorithm |
| `DEPIERRO` | AOT_Optimizers.DEPIERRO | DEPIERRO algorithm |
| `create_sparse_matrix` | SparseMatrixWrapper | Create sparse matrix |
| `check_cuda_available` | AOT_Kernels | Check CUDA |
| `mse` | ReconTools | Calculate MSE |
| `ssim` | ReconTools | Calculate SSIM |

---

## 🙏 Support

For any questions or issues, please:
1. Check the [documentation](https://github.com/LucasDuclos/AcoustoOpticTomography/tree/main/docs)
2. Check existing [issues](https://github.com/LucasDuclos/AcoustoOpticTomography/issues)
3. Open a new [issue](https://github.com/LucasDuclos/AcoustoOpticTomography/issues/new)

---

**Back to [main documentation](README.md)**
