# AcoustoOpticTomography (AOT_biomaps)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/status-Active_Development-orange.svg)]()

**AOT_biomaps** is an advanced Python library for **Acousto-Optic Tomography (AOT)**. It provides comprehensive tools for image reconstruction, acoustic simulation, optical modeling, and 2D/3D data processing.

## 📌 About the Project

This library was developed to meet the needs of the biomedical imaging research community, particularly for Acousto-Optic Tomography applications. It combines advanced reconstruction algorithms with optimized CPU and GPU implementations.

### Key Features

- ✅ **Tomographic Reconstruction**: MLEM, PDHG, LS, DEPIERRO, MAPEM, LBFGS
- ✅ **Multi-Device Support**: CPU (NumPy) and GPU (CuPy) with automatic fallback
- ✅ **Sparse Matrices**: Optimized CSR and SELL-C-sigma implementations
- ✅ **Acoustic Simulation**: Plane, focused, irregular waves
- ✅ **Optical Modeling**: Lasers, absorbers, heterogeneous media
- ✅ **Signal Processing**: Filtering, backprojection, Radon transform
- ✅ **Visualization**: 2D/3D visualization tools (optional with matplotlib)

### Modular Architecture

```
AOT_biomaps/
├── AOT_Acoustic/     # Acoustic simulation
├── AOT_Experiment/    # Experiment management
├── AOT_Medium/       # Medium modeling
├── AOT_Optic/        # Optical modeling
├── AOT_Recon/        # Reconstruction algorithms
│   ├── AOT_Optimizers/   # MLEM, PDHG, LS, etc.
│   ├── AOT_PotentialFunctions/ # Potential functions
│   └── AOT_SparseSMatrix/    # Sparse matrices (CSR, SELL)
└── Config.py         # Global configuration
```

## 🚀 Installation

See [INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions.

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/LucasDuclos/AcoustoOpticTomography.git
cd AcoustoOpticTomography

# Install in development mode
pip install -e .
```

## 📖 Documentation

- [📥 Installation](docs/INSTALLATION.md) - Complete installation guide
- [🎯 Usage](docs/USAGE.md) - Examples and tutorials
- [🔧 API Reference](docs/API_REFERENCE.md) - Technical documentation
- [🏗️ Architecture](docs/ARCHITECTURE.md) - Library design
- [🤝 Contributing](docs/CONTRIBUTING.md) - How to contribute
- [📜 Changelog](docs/CHANGELOG.md) - Release history

## 🎯 Quick Start Example

```python
import numpy as np
from AOT_biomaps import Tomography, AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType

# Create a tomography experiment
experiment = Tomography(
    optic_image_path="path/to/optic_image.npy",
    acoustic_fields_path="path/to/acoustic_fields.npy"
)

# Setup reconstruction
recon = AlgebraicRecon(
    experiment=experiment,
    reconType=ReconType.Algebraic,
    optimizerType="MLEM",
    numIterations=100
)

# Run reconstruction
recon.run(withTumor=True)

# Save results
recon.save(withTumor=True, saveDir="results/")
```

## 🔧 Dependencies

### Core Dependencies (Required)
- Python ≥ 3.8
- NumPy ≥ 1.20

### Optional Dependencies
- **CuPy** ≥ 10.0 - For GPU acceleration
- **Matplotlib** ≥ 3.0 - For visualization
- **tqdm** ≥ 4.0 - For progress bars
- **SciPy** ≥ 1.7 - For signal processing
- **kWave** - For acoustic simulation (optional)

### Compatibility Matrix

| Feature | CPU (NumPy) | GPU (CuPy) |
|---------|-------------|-------------|
| MLEM Reconstruction | ✅ | ✅ |
| PDHG Reconstruction | ✅ | ✅ |
| CSR Matrices | ✅ | ✅ |
| SELL Matrices | ✅ | ✅ |
| Visualization | ✅ | ✅ |
| Acoustic Simulation | ✅ | ⚠️ (kWave required) |

## 📊 Performance

### Benchmark (on standard dataset)

| Algorithm | CPU (s) | GPU (s) | Speedup |
|-----------|---------|---------|---------|
| MLEM | 45.2 | 2.1 | **21.5x** |
| PDHG | 38.7 | 1.8 | **21.5x** |
| LS | 22.4 | 1.2 | **18.7x** |

### Memory Usage

| Matrix | Format | Size (GB) |
|--------|--------|-----------|
| 100x100x100x50 | Dense | 19.1 |
| 100x100x100x50 | CSR | 0.8 |
| 100x100x100x50 | SELL | 0.6 |

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the project
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## 🙏 Acknowledgments

- Biomedical Imaging Laboratory
- All contributors who participated in this project

---

**Contact**: For any questions or suggestions, feel free to open an issue or contact me directly.

[🐙 GitHub](https://github.com/LucasDuclos/AcoustoOpticTomography) | [📧 Email](mailto:lucas.duclos@universite-paris-saclay.fr)
