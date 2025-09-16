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

__version__ = '2.9.9'

if config.get_process() == 'gpu':
    __process__ = 'gpu'
else:
    __process__ = 'cpu'

def initialize(process='cpu'):
    config.set_process(process)
    if process == 'gpu':
        config.select_best_gpu()
        print(f"Initialized with process: {config.get_process()} using GPU: {config.bestGPU}")
    else:
        print(f"Initialized with process: {config.get_process()}")
