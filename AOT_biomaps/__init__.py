import warnings
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
# POTENTIAL FUNCTIONS
from .AOT_Recon.AOT_PotentialFunctions.Huber import *
from .AOT_Recon.AOT_PotentialFunctions.Quadratic import *
from .AOT_Recon.AOT_PotentialFunctions.RelativeDifferences import *
# CONFIG AND SETTINGS
from .Config import config
from .Settings import *

__version__ = '2.9.29'
__process__ = config.get_process()  # Initialise avec la valeur actuelle de config

def initialize(process=None):
    """
    Initialise ou modifie le backend de calcul (GPU/CPU).

    Args:
        process (str, optional): 'gpu' pour forcer le GPU, 'cpu' pour forcer le CPU.
                                 Si None, utilise la configuration actuelle.

    Raises:
        ValueError: Si `process` n'est pas 'cpu' ou 'gpu'.
    """
    global __process__

    if process is not None:
        if process not in ['cpu', 'gpu']:
            raise ValueError("process must be 'cpu' or 'gpu'")
        config.set_process(process)
        __process__ = process

    # Vérifications et warnings si nécessaire
    if __process__ == 'gpu':
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.warn("GPU requested but PyTorch cannot access it. Falling back to CPU.", UserWarning)
                config.set_process('cpu')
                __process__ = 'cpu'
        except ImportError:
            warnings.warn("PyTorch not installed. Falling back to CPU.", UserWarning)
            config.set_process('cpu')
            __process__ = 'cpu'

    return __process__

# Initialisation automatique (silencieuse)
initialize()





