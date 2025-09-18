# AOT-BioMaps
**Tomographic Reconstruction for Acousto-Optic Imaging**

---

## Overview
AOT-BioMaps is a Python library designed for tomographic reconstruction in acousto-optic imaging. It supports both simulation and experimental data processing, offering a range of reconstruction algorithms (analytical, algebraic, and Bayesian) optimized for CPU and GPU environments.

---

## Installation

### CPU Installation

```bash
pip install --upgrade aot-biomaps
```

### GPU Installation
```bash
pip install --upgrade aot-biomaps
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-\$(python -c "import torch; print(torch.__version__)")+\$(python -c "import torch; print(''.join(torch.__version__.split('+')[1:]))").html
```
**Note:** Ensure CUDA is available on your machine.

### Verify Installation
```python
import AOT_biomaps
print(AOT_biomaps.__version__)
print(AOT_biomaps.__process__)
```
The `AOT_biomaps.__process__` variable returns the type of process (CPU or GPU) used for computations.

---

## Library Usage

### Setting Up Parameters
```python
fieldDir = "/path/to/folder/Fieldfolder"
paramPath = "/path/to/folder/parameters.yaml"
systemPath = "/path/to/folder/System_matrixParams.txt"
param = AOT_biomaps.Settings.Params(paramPath)
```
The `param` object contains the following sections: `general`, `acoustic`, `optic`, and `reconstruction`. For details on the structure and definition of each parameter, refer to the example files: [`ExampleParameters.yaml`](ExampleParameters.yaml) and [`ExampleSystem_matrixParams.txt`](ExampleSystem_matrixParams.txt).

To access a specific parameter:
```python
param.acoustic['f_US']
```

---

## AOT_Experiment Class
The `AOT_Experiment` class manages acousto-optic imaging experiments, integrating:
- Optical images
- Acoustic fields
- Acousto-optic signals

The library supports two modes:
- **Simulation**: Generates optical images and acousto-optic signals.
- **Experimental**: Loads acousto-optic signals from experimental data.

### Simulation Mode
```python
manip = AOT_biomaps.AOT_Experiment.Tomography(params=param)
manip.generatePhantom()
manip.generateAcousticFields(fieldDataPath, systemPath, show_log=False)
manip.generateAOsignal(withTumor=True)
```

### Experimental Mode
```python
manip = AOT_biomaps.AOT_Experiment.Tomography(params=param)
manip.generateAcousticFields(fieldDataPath, systemPath, show_log=False)
manip.loadAOsignal(withTumor=True)
```
**Note:** Simulating acoustic fields may introduce artifacts at the edges of the simulation grid. Truncate the fields if necessary:
```python
manip.cutAcousticFields(min_t=0, max_t=2.5e-5, saveFields=True)
```
- `min_t` and `max_t` are in seconds.
- If `saveFields=True`, truncated fields are saved to the directory.

---

## Reconstruction Algorithms

### Analytical Reconstruction
*(Details to be added)*

### Algebraic Reconstruction
The default optimizer is **Maximum Likelihood Estimation Method (ML-EM)**. For more information, see the [documentation](#).

```python
optimizer = AOT_biomaps.AOT_Reconstruction.OptimizerType.MLEM
recon = AOT_biomaps.AOT_Reconstruction.AlgebraicRecon(
    experiment=manip,
    opti=optimizer,
    numIterations=200,
    saveDir="/home/duclos/AOT/SetMixte/{set}/recon",
    isGPU=False
)
recon.run()
```

### Bayesian Reconstruction
Supported optimizers:
- Preconditioned Conjugate Gradient Maximum A Posteriori Expectation Maximization (**PCG MAP-EM**)
- PCG MAP-EM with stopping condition (**PCG MAP-EM stop**)
- De Pierro MAP-EM (**Pierro MAP-EM**)

Supported potential functions:
- Huber (`AOT_biomaps.AOT_Reconstruction.PotentialType.HUBER_PIECEWISE`)
- Quadratic (`AOT_biomaps.AOT_Reconstruction.PotentialType.QUADRATIC`)
- Relative Difference (`AOT_biomaps.AOT_Reconstruction.PotentialType.RELATIVE_DIFFERENCE`)

```python
optimizer = AOT_biomaps.AOT_Reconstruction.OptimizerType.PGC
potentialFunction = AOT_biomaps.AOT_Reconstruction.PotentialType.HUBER_PIECEWISE
recon = AOT_biomaps.AOT_Reconstruction.BayesianRecon(
    experiment=manip,
    opti=optimizer,
    potentialFunction=potentialFunction,
    numIterations=200,
    saveDir="/home/duclos/AOT/SetMixte/{set}/recon",
    isGPU=False
)
recon.run()
```

---

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


## Contact
For questions or feedback, please open an issue or contact the maintainers.
