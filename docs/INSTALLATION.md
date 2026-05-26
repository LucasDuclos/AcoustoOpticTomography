# Installation Guide - AOT_biomaps

This guide explains how to install and configure the AOT_biomaps library for Acousto-Optic Tomography.

## 📋 Prerequisites

### Operating System
- Windows 10/11 (recommended)
- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (10.15+)

### Python
- **Required version**: Python ≥ 3.8
- **Recommended**: Python 3.10 or 3.11

Check your Python version:
```bash
python --version
# or
python3 --version
```

## 🎯 Installation

### Method 1: Development Installation (Recommended)

This method is ideal if you want to contribute or modify the code.

```bash
# 1. Clone the repository
git clone https://github.com/LucasDuclos/AcoustoOpticTomography.git
cd AcoustoOpticTomography

# 2. Create a virtual environment (optional but recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the library in development mode
pip install -e .
```

### Method 2: pip Installation (Coming Soon)

```bash
pip install aot-biomaps
```

### Method 3: Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/LucasDuclos/AcoustoOpticTomography.git
cd AcoustoOpticTomography

# 2. Install core dependencies
pip install numpy

# 3. Add to PYTHONPATH
# On Windows (PowerShell):
$env:PYTHONPATH = ".;$env:PYTHONPATH"

# On Linux/macOS:
export PYTHONPATH=".:$PYTHONPATH"
```

## 📦 Dependencies

### Core Dependencies (Required)

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ≥ 1.26.4 | Core numerical computing |
| scipy | ≥ 1.13.1 | Signal processing |
| tqdm | ≥ 4.60.0 | Progress bars |
| matplotlib | ≥ 3.9.2 | Visualization |

### Default Dependencies (Included)

By default, the following packages are automatically included:

| Package | Version | Description |
|---------|---------|-------------|
| cupy | ≥ 12.0 | GPU acceleration (auto-detected) |
| kwave | ≥ 0.3.5 | Acoustic simulation |

### CuPy Installation (Automatic)

✅ **CuPy is automatically included** if CUDA is detected on your system. The installer will:
1. Check if CUDA is available (via `nvcc`)
2. Detect your CUDA version
3. Install the appropriate CuPy package (`cupy-cuda11x`, `cupy-cuda12x`, etc.)

**Manual installation (if automatic detection fails):**
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 13.x
pip install cupy-cuda13x

# For CUDA 14.x (latest)
pip install cupy-cuda14x
```

Verify installation:
```python
import cupy as cp
print(cp.cuda.runtime.getVersion())  # Should print CUDA version
print(cp.__version__)  # Should print CuPy version
```

### CPU-Only Installation

To install without GPU support:
```bash
pip install aot-biomaps --cpu
# or
AOT_BIOMAPS_CPU_ONLY=true pip install aot-biomaps
```

### Without Acoustic Simulation

To install without kWave:
```bash
pip install aot-biomaps --without-acoustic
# or
AOT_BIOMAPS_WITHOUT_ACOUSTIC=true pip install aot-biomaps
```

### Python Version Compatibility

| Python Version | CuPy Support | Status |
|----------------|--------------|--------|
| 3.8 | CuPy 10.x - 12.x | ✅ Supported (limited in CuPy 13+) |
| 3.9 | CuPy 11.x - 14.x | ✅ Fully supported |
| 3.10 | CuPy 12.x - 14.x | ✅ Fully supported |
| 3.11 | CuPy 12.x - 14.x | ✅ Fully supported |
| 3.12 | CuPy 13.x - 14.x | ✅ Supported |
| 3.13+ | - | ⚠️ Not yet supported by CuPy |

**Note**: If you use Python 3.13+, you will get a warning on import. The library will still work in CPU mode.

### kWave Installation (Optional)

kWave is used for acoustic simulation. It requires MATLAB or is available as a Python version:

```bash
# Python version (experimental)
pip install kwave

# Or use MATLAB with k-Wave toolbox
# See: https://www.k-wave.org/
```

## ⚙️ Configuration

### Configuration File

The library uses a `Config.py` file for global configuration. You can modify default parameters:

```python
from AOT_biomaps.Config import config

# Set default device
config.set_process('gpu')  # or 'cpu'

# Enable/disable multi-CPU mode
config.set_multi_cpu(True)
```

### Environment Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `AOT_DEVICE` | Default device (cpu/gpu) | cpu |
| `AOT_MULTI_CPU` | Enable multi-CPU | False |
| `AOT_VERBOSE` | Verbose mode | False |

Example:
```bash
# On Windows
export AOT_DEVICE=gpu

# On Linux/macOS
export AOT_DEVICE=gpu
```

## ⚡ Installation Verification

Run the test script to verify everything works:

```bash
python AOT_biomaps/AOT_Recon/test_kernels.py
```

You should see:
```
Testing AOT_biomaps kernel and sparse matrix implementations...
======================================================================

[Test 1] Importing AOT_Kernels module...
[OK] AOT_Kernels module imported successfully
  - CUDA available: True/False
  - PyCUDA available: True/False

[Test 2] Testing CPU implementations...
[OK] All CPU tests passed!

[Test 3] Testing SparseMatrix wrapper...
[OK] SparseMatrix wrapper imported successfully

[Test 4] Testing MLEM module...
[OK] MLEM module imported successfully

[Test 5] Testing sparse matrix imports...
[OK] Sparse matrix classes imported successfully

All tests completed successfully!
```

## 🛠️ Troubleshooting

### Error: ModuleNotFoundError: No module named 'cupy'

**Solution**: Install CuPy or disable GPU acceleration:
```bash
pip install cupy-cuda11x  # or appropriate version
```

Or use CPU only:
```python
from AOT_biomaps.Config import config
config.set_process('cpu')
```

### Error: CUDA not available

**Solution**: Verify CUDA is properly installed:
```bash
nvcc --version  # Check CUDA
nvidia-smi     # Check NVIDIA drivers
```

### Error: kWave is not available

**Solution**: Install kWave or disable acoustic features:
```bash
pip install kwave
```

Or simply ignore the warning - reconstruction features will work without kWave.

### Performance Issues

If performance is slow:
1. Verify CuPy is using GPU:
   ```python
   import cupy as cp
   print(cp.cuda.runtime.getDeviceCount())  # Should be ≥ 1
   ```
2. Ensure sparse matrix is being used:
   ```python
   from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix
   # Use matrix_type='SELL' for better GPU performance
   ```

## 📚 Next Steps

Once installation is complete, check out:
- [USAGE.md](USAGE.md) - Complete usage guide
- [API_REFERENCE.md](API_REFERENCE.md) - Technical API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Library architecture
