import os
import warnings
import numpy as np

# ---
# MEDIUM
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
from .AOT_Recon.BayesianRecon import *
from .AOT_Recon.DeepLearningRecon import *
from .AOT_Recon.PrimalDualRecon import *
from .AOT_Recon.ReconEnums import *
from .AOT_Recon.ReconTools import *
# OPTIMIZERS
from .AOT_Recon.AOT_Optimizers.DEPIERRO import *
from .AOT_Recon.AOT_Optimizers.MAPEM import *
from .AOT_Recon.AOT_Optimizers.MLEM import *
from .AOT_Recon.AOT_Optimizers.PDHG import *
# SPARSE S-MATRIX
from .AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_CSR import *
from .AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL import *
# POTENTIAL FUNCTIONS
from .AOT_Recon.AOT_PotentialFunctions.Huber import *
from .AOT_Recon.AOT_PotentialFunctions.Quadratic import *
from .AOT_Recon.AOT_PotentialFunctions.RelativeDifferences import *
# CONFIG AND SETTINGS
from .Config import config
from .Settings import *

__version__ = '2.9.522'
__process__ = config.get_process()

def initialize(process=None):
    """
    Initialize or modify the compute backend (GPU/CPU).

    Args:
        process (str, optional): 'gpu' to force GPU, 'cpu' to force CPU.

    Raises:
        ValueError: If `process` is not 'cpu' or 'gpu'.
    """
    global __process__
    if process is not None:
        if process not in ['cpu', 'gpu']:
            raise ValueError("process must be 'cpu' or 'gpu'")
        config.set_process(process)
        __process__ = process

    if __process__ == 'gpu':
        try:
            import cupy as cp
            if not cp.cuda.is_available():
                warnings.warn("GPU requested but CuPy cannot access it. Falling back to CPU.", UserWarning)
                config.set_process('cpu')
                __process__ = 'cpu'
        except ImportError:
            warnings.warn("CuPy not available. Falling back to CPU.", UserWarning)
            config.set_process('cpu')
            __process__ = 'cpu'
        except Exception as e:
            warnings.warn(f"CuPy GPU check failed: {e}. Falling back to CPU.", UserWarning)
            config.set_process('cpu')
            __process__ = 'cpu'

    return __process__

