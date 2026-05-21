# Installation Guide - AOT_biomaps

Ce guide vous explique comment installer et configurer la librairie AOT_biomaps pour la Tomographie Acousto-Optique.

## 📋 Prérequis

### Système d'exploitation
- Windows 10/11 (recommandé)
- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (10.15+)

### Python
- **Version requise**: Python ≥ 3.8
- **Recommandé**: Python 3.10 ou 3.11

Vérifiez votre version de Python :
```bash
python --version
# ou
python3 --version
```

## 🎯 Installation

### Méthode 1: Installation en mode développement (recommandé)

Cette méthode est idéale si vous souhaitez contribuer ou modifier le code.

```bash
# 1. Cloner le dépôt
git clone https://github.com/LucasDuclos/AcoustoOpticTomography.git
cd AcoustoOpticTomography

# 2. Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv

# Sur Windows:
venv\Scripts\activate

# Sur Linux/macOS:
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer la librairie en mode développement
pip install -e .
```

### Méthode 2: Installation via pip (bientôt disponible)

```bash
pip install aot-biomaps
```

### Méthode 3: Installation manuelle

```bash
# 1. Cloner le dépôt
git clone https://github.com/LucasDuclos/AcoustoOpticTomography.git
cd AcoustoOpticTomography

# 2. Installer les dépendances de base
pip install numpy

# 3. Ajouter le chemin au PYTHONPATH
# Sur Windows (PowerShell):
$env:PYTHONPATH = ".;$env:PYTHONPATH"

# Sur Linux/macOS:
export PYTHONPATH=".:$PYTHONPATH"
```

## 📦 Dépendances

### Dépendances principales (requises)

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ≥ 1.20 | Calcul numérique de base |

### Dépendances optionnelles (recommandées)

| Package | Version | Description | Installation |
|---------|---------|-------------|-------------|
| cupy | ≥ 10.0 | Accélération GPU | `pip install cupy-cuda11x` |
| matplotlib | ≥ 3.0 | Visualisation | `pip install matplotlib` |
| tqdm | ≥ 4.0 | Barres de progression | `pip install tqdm` |
| scipy | ≥ 1.7 | Traitement du signal | `pip install scipy` |
| kwave | - | Simulation acoustique | Voir ci-dessous |

### Installation de CuPy

CuPy nécessite CUDA et cuDNN. Choisissez la version appropriée pour votre GPU :

```bash
# Pour CUDA 11.x
pip install cupy-cuda11x

# Pour CUDA 12.x
pip install cupy-cuda12x

# Version CPU-only (pour le développement)
pip install cupy
```

Vérifiez l'installation :
```python
import cupy as cp
print(cp.cuda.runtime.getVersion())  # Affiche la version CUDA
```

### Installation de kWave (optionnel)

kWave est utilisé pour la simulation acoustique. Il nécessite MATLAB ou est disponible en version Python :

```bash
# Version Python (expérimentale)
pip install kwave

# Ou utiliser MATLAB avec le toolbox k-Wave
# Voir: https://www.k-wave.org/
```

## 🔧 Configuration

### Fichier de configuration

La librairie utilise un fichier `Config.py` pour la configuration globale. Vous pouvez modifier les paramètres par défaut :

```python
from AOT_biomaps.Config import config

# Définir le device par défaut
config.set_process('gpu')  # ou 'cpu'

# Activer/désactiver le mode multi-CPU
config.set_multi_cpu(True)
```

### Variables d'environnement

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `AOT_DEVICE` | Device par défaut (cpu/gpu) | cpu |
| `AOT_MULTI_CPU` | Activer le multi-CPU | False |
| `AOT_VERBOSE` | Mode verbeux | False |

Exemple :
```bash
# Sur Windows
export AOT_DEVICE=gpu

# Sur Linux/macOS
export AOT_DEVICE=gpu
```

## ⚡ Vérification de l'installation

Exécutez le script de test pour vérifier que tout fonctionne :

```bash
python AOT_biomaps/AOT_Recon/test_kernels.py
```

Vous devriez voir :
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

## 🛠️ Dépannage

### Erreur: ModuleNotFoundError: No module named 'cupy'

**Solution**: Installez CuPy ou désactivez l'accélération GPU :
```bash
pip install cupy-cuda11x  # ou la version appropriée
```

Ou utilisez uniquement le CPU :
```python
from AOT_biomaps.Config import config
config.set_process('cpu')
```

### Erreur: CUDA not available

**Solution**: Vérifiez que CUDA est correctement installé :
```bash
nvcc --version  # Vérifie CUDA
nvidia-smi     # Vérifie les pilotes NVIDIA
```

### Erreur: kWave is not available

**Solution**: Installez kWave ou désactivez les fonctionnalités acoustiques :
```bash
pip install kwave
```

Ou ignorez simplement l'avertissement - les fonctionnalités de reconstruction fonctionneront sans kWave.

### Problèmes de performance

Si les performances sont lentes :
1. Vérifiez que CuPy utilise bien le GPU :
   ```python
   import cupy as cp
   print(cp.cuda.runtime.getDeviceCount())  # Doit afficher ≥ 1
   ```
2. Assurez-vous que la matrice creuse est utilisée :
   ```python
   from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import create_sparse_matrix
   # Utilisez matrix_type='SELL' pour de meilleures performances GPU
   ```

## 📚 Prochaines étapes

Une fois l'installation terminée, consultez :
- [USAGE.md](USAGE.md) - Guide d'utilisation complet
- [API_REFERENCE.md](API_REFERENCE.md) - Référence de l'API
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture de la librairie
