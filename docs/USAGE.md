# Usage Guide - AOT_biomaps

This guide explains how to use the AOT_biomaps library for Acousto-Optic Tomography reconstruction.

## 📚 Table of Contents

- [Basic Concepts](#-basic-concepts)
- [Getting Started](#-getting-started)
- [Tomographic Reconstruction](#-tomographic-reconstruction)
  - [Algebraic Reconstruction (MLEM)](#algebraic-reconstruction-mlem)
  - [Analytic Reconstruction](#analytic-reconstruction)
  - [Bayesian Reconstruction](#bayesian-reconstruction)
  - [Least Squares Reconstruction](#least-squares-reconstruction)
  - [Primal-Dual Reconstruction (PDHG)](#primal-dual-reconstruction-pdhg)
- [Sparse Matrix Usage](#-sparse-matrix-usage)
- [Acoustic Simulation](#-acoustic-simulation)
- [Result Visualization](#-result-visualization)
- [Complete Examples](#-complete-examples)
- [Best Practices](#-best-practices)

## 🎯 Basic Concepts

### Acousto-Optic Tomography (AOT)

Acousto-Optic Tomography combines the advantages of optical imaging (high resolution) and ultrasound imaging (penetration depth). The principle is as follows:

1. A laser beam illuminates the tissue
2. An ultrasound wave modulates the light
3. The modulated light is detected and used to reconstruct an image

### Library Architecture

```
AOT_biomaps/
├── AOT_Acoustic/     # Acoustic simulation
├── AOT_Experiment/    # Experiment management
├── AOT_Medium/       # Medium modeling
├── AOT_Optic/        # Optical modeling
└── AOT_Recon/        # Reconstruction algorithms
    ├── AOT_Optimizers/   # MLEM, PDHG, LS, etc.
    ├── AOT_PotentialFunctions/ # Potential functions
    └── AOT_SparseSMatrix/    # Sparse matrices (CSR, SELL)
```

### Typical Workflow

```
1. Load optical and acoustic data
   ↓
2. Configure the experiment (Tomography)
   ↓
3. Choose a reconstruction algorithm
   ↓
4. Run the reconstruction
   ↓
5. Visualize and save results
```

## 🚀 Getting Started

### Basic Import

```python
# Basic imports
import numpy as np
from AOT_biomaps import Tomography, AlgebraicRecon, AnalyticRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType

# Check GPU availability
from AOT_biomaps.AOT_Recon.AOT_Kernels import check_cuda_available
print(f"CUDA available: {check_cuda_available()}")
```

### Basic Configuration

```python
from AOT_biomaps.Config import config

# Set default device (cpu or gpu)
config.set_process('gpu')  # Use GPU if available

# Enable multi-CPU mode for large datasets
config.set_multi_cpu(True)

# Set verbosity level
config.set_verbose(True)
```

## 🔬 Tomographic Reconstruction

### Data Preparation

Before starting reconstruction, you need:
1. An optical image (phantom) - the reference image
2. Acoustic fields - the measurement data

```python
import numpy as np

# Example: Load test data
# (In practice, load your own data)
T, Z, X, N = 10, 50, 50, 20

# Generate a test optical image (phantom)
optic_image = np.random.rand(Z, X).astype(np.float32)

# Generate test acoustic fields
acoustic_fields = [type('AcousticField', (), {
    'field': np.random.rand(T, Z, X).astype(np.float32)
})() for _ in range(N)]
```

### Algebraic Reconstruction (MLEM)

**MLEM (Maximum Likelihood Expectation Maximization)** is the most commonly used algorithm for tomographic reconstruction.

#### Basic Example

```python
from AOT_biomaps import Tomography, AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType

# Create a tomography experiment
experiment = Tomography(
    optic_image=optic_image,
    acoustic_fields=acoustic_fields
)

# Setup MLEM reconstruction
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType=OptimizerType.MLEM,
    numIterations=100,
    isGPU=False  # Use CPU
)

# Run reconstruction
recon.run(withTumor=True)

# Get results
reconstructed_image = recon.reconPhantom[-1]  # Last iteration
print(f"Reconstructed image shape: {reconstructed_image.shape}")
```

#### With Sparse Matrix (Recommended for Large Datasets)

```python
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix

# Create a sparse matrix (CSR or SELL)
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',  # or 'CSR'
    device='gpu'  # or 'cpu'
)

# Allocate the matrix
sparse_matrix.allocate()

# Use sparse matrix in MLEM
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import MLEM_sparse

result, indices = MLEM_sparse(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,  # Measurement data
    numIterations=50,
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

#### Advanced MLEM Parameters

```python
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType=OptimizerType.MLEM,
    numIterations=200,           # Number of iterations
    isSavingEachIteration=True, # Save each iteration
    max_saves=100,              # Max number of saves
    denominator_threshold=1e-6, # Threshold to avoid division by zero
    show_logs=True,             # Show progress bar
    isGPU=True                  # Use GPU
)
```

### Analytic Reconstruction

Analytic reconstruction uses filtered backprojection methods for fast reconstruction.

```python
from AOT_biomaps import AnalyticRecon
from AOT_biomaps.AOT_Recon.ReconEnums import AnalyticType

# Create analytic reconstruction
analytic_recon = AnalyticRecon(
    experiment=experiment,
    analyticType=AnalyticType.iRADON,  # or FBP (Filtered Back Projection)
    Lc=0.01,  # Coherence length (for iRADON)
    isGPU=True
)

# Run reconstruction
analytic_recon.run(withTumor=True)

# Get result
reconstructed_image = analytic_recon.reconPhantom
```

### Bayesian Reconstruction (MAPEM)

**MAPEM (Maximum A Posteriori Expectation Maximization)** adds prior information to improve reconstruction.

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MAPEM import MAPEM
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic import Omega_QUADRATIC

# Define potential function (regularization)
Omega = Omega_QUADRATIC

# Run MAPEM
result, indices = MAPEM(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,
    Omega=Omega_QUADRATIC,
    beta=0.1,  # Regularization parameter
    numIterations=100,
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

### Least Squares Reconstruction

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.LS import LS

result, indices = LS(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,
    numIterations=100,
    alpha=0.01,  # Regularization parameter
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

### Primal-Dual Reconstruction (PDHG)

**PDHG (Primal-Dual Hybrid Gradient)** is effective for problems with TV (Total Variation) regularization.

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers.PDHG import CP_TV

result, indices = CP_TV(
    SMatrix=sparse_matrix,
    y=experiment.AcousticFields[0].field,
    alpha=0.01,    # L1 regularization parameter
    beta=1e-4,     # TV regularization parameter
    theta=1.0,     # Relaxation parameter
    numIterations=1000,
    isSavingEachIteration=True,
    withTumor=True,
    device='gpu'
)
```

## 🗃️ Sparse Matrix Usage

### Why Use Sparse Matrices?

Sparse matrices allow:
- Reduced memory consumption (up to 95% reduction)
- Faster computations (especially on GPU)
- Handling larger datasets

### Format Comparison

| Format | Advantages | Disadvantages | Best For |
|--------|-----------|---------------|----------|
| Dense | Simple to implement | Very memory intensive | Small datasets |
| CSR | Memory efficient | Slow random access | CPU, medium datasets |
| SELL | GPU optimized | More complex construction | GPU, large datasets |

### Creating a Sparse Matrix

```python
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix, SparseMatrix

# Method 1: Use factory function
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',  # or 'CSR'
    device='gpu',
    block_rows=64,
    relative_threshold=0.3
)

# Method 2: Create directly
from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL import SparseSMatrix_SELL

sell_matrix = SparseSMatrix_SELL(
    manip=experiment.AcousticFields,
    device='gpu',
    slice_height=32
)

# Allocate and build the matrix
sparse_matrix.allocate()

# Use the matrix
projection = sparse_matrix.projection(theta)  # Projection: q = A * theta
backprojection = sparse_matrix.backprojection(e)  # Backprojection: c = A^T * e
```

### Sparse Matrix Parameters

```python
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',      # 'CSR' or 'SELL'
    device='gpu',           # 'cpu' or 'gpu'
    block_rows=64,          # Block size for processing
    relative_threshold=0.3, # Threshold for considering a value as non-zero
    slice_height=32         # Slice height (for SELL only)
)
```

### Memory Management

```python
# Free GPU memory
sparse_matrix.free()

# Use context manager for automatic management
with create_sparse_matrix(manip, matrix_type='SELL', device='gpu') as sm:
    # Matrix is automatically allocated
    result = sm.projection(theta)
    # Memory is automatically freed on exit
```

## 🔊 Acoustic Simulation

### Acoustic Wave Types

The library supports several wave types:
- **PlaneWave**: Plane wave
- **FocusedWave**: Focused wave
- **StructuredWave**: Structured wave
- **IrregularWave**: Irregular wave

### Creating a Plane Wave

```python
from AOT_biomaps.AOT_Acoustic.PlaneWave import PlaneWave

# Create a plane wave
plane_wave = PlaneWave(
    frequency=1e6,      # Frequency in Hz
    direction=[1, 0],   # Propagation direction [dz, dx]
    amplitude=1.0,      # Amplitude
    phase=0.0,          # Initial phase
    sampling_rate=1e7   # Sampling rate in Hz
)

# Generate the acoustic field
field = plane_wave.generate_field(
    size=(100, 100, 100),  # Field size (T, Z, X)
    speed_of_sound=1500   # Speed of sound in m/s
)
```

### Creating a Focused Wave

```python
from AOT_biomaps.AOT_Acoustic.FocusedWave import FocusedWave

# Create a focused wave
focused_wave = FocusedWave(
    frequency=1e6,
    focal_point=[50, 50, 50],  # Focal point [Z, X, Y]
    amplitude=1.0,
    radius=20,                 # Transducer radius
    sampling_rate=1e7
)

# Generate the field
field = focused_wave.generate_field(
    size=(100, 100, 100),
    speed_of_sound=1500
)
```

## 📊 Result Visualization

### Basic Visualization with Matplotlib

```python
import matplotlib.pyplot as plt

# Visualize the original optical image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(experiment.OpticImage.phantom, cmap='hot')
plt.title('Optical Image (Phantom)')
plt.colorbar()

# Visualize the reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(recon.reconPhantom[-1], cmap='hot')
plt.title('Reconstructed Image')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### Visualization with show() Method

```python
# The Recon class has a built-in show() method
recon.show(withTumor=True, savePath='results/')
```

### Iteration Visualization

```python
# Visualize multiple iterations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    iteration = i * 10  # Show every 10th iteration
    if iteration < len(recon.reconPhantom):
        ax.imshow(recon.reconPhantom[iteration], cmap='hot')
        ax.set_title(f'Iteration {iteration}')
        ax.axis('off')
plt.tight_layout()
plt.show()
```

### Metrics Calculation

```python
# Calculate MSE (Mean Squared Error)
recon.calculateMSE(withTumor=True)
print(f"MSE: {recon.MSE}")

# Calculate SSIM (Structural Similarity Index)
recon.calculateSSIM(withTumor=True)
print(f"SSIM: {recon.SSIM}")

# Calculate CRC (Contrast Recovery Coefficient)
recon.calculateCRC(use_ROI=True)
print(f"CRC: {recon.CRC}")
```

## 📝 Complete Examples

### Example 1: Full MLEM Reconstruction

```python
import numpy as np
from AOT_biomaps import Tomography, AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType
from AOT_biomaps.Config import config

# Configuration
config.set_process('gpu')
config.set_verbose(True)

# Generate test data
T, Z, X, N = 10, 64, 64, 32
optic_image = np.random.rand(Z, X).astype(np.float32)
acoustic_fields = [type('AcousticField', (), {
    'field': np.random.rand(T, Z, X).astype(np.float32)
})() for _ in range(N)]

# Create experiment
experiment = Tomography(
    optic_image=optic_image,
    acoustic_fields=acoustic_fields
)

# Setup and run MLEM
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType=OptimizerType.MLEM,
    numIterations=50,
    isSavingEachIteration=True,
    show_logs=True
)

recon.run(withTumor=True)

# Display results
print(f"Number of iterations: {len(recon.reconPhantom)}")
print(f"Image shape: {recon.reconPhantom[0].shape}")

# Calculate metrics
recon.calculateMSE(withTumor=True)
recon.calculateSSIM(withTumor=True)
print(f"Final MSE: {recon.MSE[-1] if isinstance(recon.MSE, list) else recon.MSE}")
print(f"Final SSIM: {recon.SSIM[-1] if isinstance(recon.SSIM, list) else recon.SSIM}")

# Save results
recon.save(withTumor=True, saveDir='results/')
```

### Example 2: CPU vs GPU Comparison

```python
import time
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix
from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import MLEM_sparse

# Create sparse matrices
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

# Measure CPU time
start = time.time()
result_cpu, _ = MLEM_sparse(
    SMatrix=sparse_matrix_cpu,
    y=experiment.AcousticFields[0].field,
    numIterations=20,
    show_logs=False,
    device='cpu'
)
cpu_time = time.time() - start

# Measure GPU time
start = time.time()
result_gpu, _ = MLEM_sparse(
    SMatrix=sparse_matrix_gpu,
    y=experiment.AcousticFields[0].field,
    numIterations=20,
    show_logs=False,
    device='gpu'
)
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.3f}s")
print(f"GPU time: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

### Example 3: Reconstruction with Different Methods

```python
from AOT_biomaps.AOT_Recon.AOT_Optimizers import MLEM, LS, PDHG
from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix

# Create sparse matrix
sparse_matrix = create_sparse_matrix(
    manip=experiment.AcousticFields,
    matrix_type='SELL',
    device='gpu'
)
sparse_matrix.allocate()

# Test different methods
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
        print(f"{name}: Success")
    except Exception as e:
        print(f"{name}: Failed - {e}")

# Compare results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
for (name, result), ax in zip(results.items(), axes):
    ax.imshow(result, cmap='hot')
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## ✅ Best Practices

### 1. Memory Management

```python
# Always free GPU memory after use
sparse_matrix.free()

# Use context managers for automatic management
with create_sparse_matrix(...) as sm:
    # Work with the matrix
    result = sm.projection(theta)
# Memory is automatically freed
```

### 2. Device Selection

```python
# Check GPU availability
from AOT_biomaps.AOT_Recon.AOT_Kernels import check_cuda_available

if check_cuda_available():
    device = 'gpu'
    print("Using GPU")
else:
    device = 'cpu'
    print("Using CPU")
```

### 3. Saving Results

```python
# Save regularly during long reconstructions
recon = AlgebraicRecon(
    experiment=experiment,
    numIterations=1000,
    isSavingEachIteration=True,
    max_saves=100  # Save 100 iterations
)

# Save to a dedicated folder
recon.save(withTumor=True, saveDir='results/experiment_001/')
```

### 4. Reproducibility

```python
# Set random seed for reproducibility
import numpy as np
np.random.seed(42)

# Use the same seed for all operations
from AOT_biomaps.Config import config
config.set_seed(42)
```

### 5. Error Handling

```python
try:
    recon.run(withTumor=True)
    recon.calculateMSE()
    recon.calculateSSIM()
except Exception as e:
    print(f"Error during reconstruction: {e}")
    # Save state before error
    if hasattr(recon, 'reconPhantom'):
        np.save('recon_error.npy', recon.reconPhantom)
```

## 📚 Additional Resources

- [Installation](INSTALLATION.md) - Installation guide
- [API Reference](API_REFERENCE.md) - Technical API documentation
- [Architecture](ARCHITECTURE.md) - Library design
- [Contributing](CONTRIBUTING.md) - How to contribute
