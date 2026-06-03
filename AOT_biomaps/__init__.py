import os
import sys
import warnings
import numpy as np

# --- 1. JIT DEPENDENCY CHECK ---
# Must be executed before k-wave or anything requiring C libraries
from .system_deps import ensure_system_dependencies
ensure_system_dependencies()

# --- 2. Check Python version compatibility with CuPy ---
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
        f"AOT_biomaps requires Python 3.8 or later for CuPy compatibility."
    )
elif sys.version_info > (3, 12):
    warnings.warn(
        f"Python {sys.version_info.major}.{sys.version_info.minor} may not be fully supported by CuPy. "
        f"Consider using Python 3.8-3.12 for best compatibility.",
        UserWarning
    )

# --- 3. Load CuPy BEFORE k-wave imports ---
try:
    import cupy as cp
    if not cp.cuda.is_available():
        os.environ['KWAVE_CPU_ONLY'] = '1'
except ImportError:
    warnings.warn("CuPy not available. Falling back to CPU.", UserWarning)
    os.environ['KWAVE_CPU_ONLY'] = '1'
except Exception:
    # Silently fall back to CPU - detailed error will be shown by Config._init_gpu() if needed
    os.environ['KWAVE_CPU_ONLY'] = '1'

# --- Then import modules (k-wave will load AFTER CuPy and szip) ---
from .AOT_Medium._mainMedium import *
from .AOT_Medium.HomogeneousMedium import *
from .AOT_Medium.PVAMedium import *
from .AOT_Medium.MediumEnums import *
# ACOUSTIC
from .AOT_Acoustic._mainAcoustic import *
from .AOT_Acoustic.AcousticEnums import *
from .AOT_Acoustic.AcousticTools import *
from .AOT_Acoustic.FocusedWave import *
from .AOT_Acoustic.IrregularWave import *
from .AOT_Acoustic.PlaneWave import *
from .AOT_Acoustic.StructuredWave import *
# EXPERIMENT
from .AOT_Experiment._mainExperiment import *
from .AOT_Experiment.Focus import *
from .AOT_Experiment.Tomography import *
# OPTIC
from .AOT_Optic._mainOptic import *
from .AOT_Optic.Absorber import *
from .AOT_Optic.Laser import *
from .AOT_Optic.OpticEnums import *
# RECONSTRUCTION
from .AOT_Recon._mainRecon import *
from .AOT_Recon.AlgebraicRecon import *
from .AOT_Recon.AnalyticRecon import *
from .AOT_Recon.DeepLearningRecon import *
from .AOT_Recon.ReconEnums import *
from .AOT_Recon.ReconTools import *
# OPTIMIZERS
from .AOT_Recon.AOT_Optimizers.DEPIERRO import *
from .AOT_Recon.AOT_Optimizers.MAPEM import *
from .AOT_Recon.AOT_Optimizers.MLEM import *
from .AOT_Recon.AOT_Optimizers.PDHG import *
from .AOT_Recon.AOT_Optimizers.LS import *
from .AOT_Recon.AOT_Optimizers.LBFGS import *
from .AOT_Recon.AOT_Optimizers.PPGMLEM import *
from .AOT_Recon.AOT_Optimizers.PGC import *
# SPARSE S-MATRIX
from .AOT_Recon.AOT_SMatrix.SMatrix_CSR import *
from .AOT_Recon.AOT_SMatrix.SMatrix_SELL import *
from .AOT_Recon.AOT_SMatrix.SMatrix_DENSE import *
from .AOT_Recon.AOT_SMatrix._mainSMatrix import *
# CONFIG AND SETTINGS
from .Config import config
from .Settings import *

__version__ = '2.9.713'
__process__ = config.get_process()

# Reference to the config object
__config__ = config

def initialize(process=None, gpu_id=None):
    """
    Initialize or modify the compute backend (GPU/CPU) and GPU selection.

    Args:
        process (str, optional): 'gpu' to force GPU, 'cpu' to force CPU.
        gpu_id (int, optional): Specific GPU device ID to use. If None and process='gpu',
                               auto-selects the best GPU.

    Raises:
        ValueError: If `process` is not 'cpu' or 'gpu'.
        ValueError: If `gpu_id` is invalid or out of range.
        RuntimeError: If no GPUs are available but process='gpu'.
    
    Returns:
        str: The current process ('cpu' or 'gpu')
        
    Examples:
        >>> import AOT_biomaps as aot
        >>> aot.initialize('gpu')  # Auto-select best GPU
        >>> aot.initialize('gpu', gpu_id=0)  # Use first GPU
        >>> aot.initialize('cpu')  # Force CPU mode
    """
    global __process__, __config__
    
    if process is not None:
        if process not in ['cpu', 'gpu']:
            raise ValueError("process must be 'cpu' or 'gpu'")
        
        config.set_process(process)
        __process__ = process
        
        # Handle GPU selection
        if process == 'gpu':
            try:
                import cupy as cp
                if not cp.cuda.is_available():
                    config.set_process('cpu')
                    __process__ = 'cpu'
                elif gpu_id is not None:
                    # Select specific GPU
                    __config__.select_gpu(gpu_id)
            except Exception:
                config.set_process('cpu')
                __process__ = 'cpu'
        else:
            # CPU mode - no GPU selection needed
            config.set_process('cpu')
            __process__ = 'cpu'
    elif gpu_id is not None:
        # Only gpu_id specified, keep current process but change GPU
        if __process__ == 'gpu':
            __config__.select_gpu(gpu_id)
    
    # Update config
    __config__._update()
    
    return __process__










































































































































































