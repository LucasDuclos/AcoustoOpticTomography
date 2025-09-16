from AOT_biomaps.AOT_Recon.AlgebraicRecon import AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, OptimizerType, PotentialType, ProcessType
from .ReconTools import check_gpu_memory, calculate_memory_requirement
from .AOT_Optimizers import MAPEM, DEPIERRO
from AOT_biomaps.Config import config

import warnings
import numpy as np

class BayesianRecon(AlgebraicRecon):
    """
    This class implements the Bayesian reconstruction process.
    It currently does not perform any operations but serves as a template for future implementations.
    """
    def __init__(self, 
                opti = OptimizerType.PGC,
                potentialFunction = PotentialType.HUBER_PIECEWISE,  
                beta=None, 
                delta=None, 
                gamma=None, 
                sigma=None,
                corner = (0.5-np.sqrt(2)/4)/np.sqrt(2),
                face = 0.5-np.sqrt(2)/4, 
                **kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.Bayesian
        self.potentialFunction = potentialFunction
        self.optimizer = opti
        self.beta = beta           
        self.delta = delta          # typical value is 0.1
        self.gamma = gamma          # typical value is 0.01
        self.sigma = sigma          # typical value is 1.0
        self.corner = corner        # typical value is (0.5-np.sqrt(2)/4)/np.sqrt(2)
        self.face = face            # typical value is 0.5-np.sqrt(2)/4 

        if not isinstance(self.potentialFunction, PotentialType):
            raise TypeError(f"Potential functions must be of type PotentialType, got {type(self.potentialFunction)}")  

    def run(self, processType=ProcessType.PYTHON, withTumor=True):
        """
        This method is a placeholder for the Bayesian reconstruction process.
        It currently does not perform any operations but serves as a template for future implementations.
        """
        if(processType == ProcessType.CASToR):
            self._bayesianReconCASToR(withTumor)
        elif(processType == ProcessType.PYTHON):
            self._bayesianReconPython(withTumor)
        else:
            raise ValueError(f"Unknown Bayesian reconstruction type: {processType}")
        
    def _bayesianReconCASToR(self, withTumor):
        raise NotImplementedError("CASToR Bayesian reconstruction is not implemented yet.")

    def _bayesianReconPython(self, withTumor):

        if withTumor:
            if self.experiment.AOsignal_withTumor is None:
                raise ValueError("AO signal with tumor is not available. Please generate AO signal with tumor the experiment first in the experiment object.")
            if self.optimizer.value ==  OptimizerType.PPGMLEM.value:
                self.reconPhantom = self._MAPEM_STOP(SMatrix=self.SMatrix, y=self.experiment.AOsignal_withTumor, withTumor=withTumor)
            elif self.optimizer.value == OptimizerType.PGC.value:
                self.reconPhantom = self._MAPEM(SMatrix=self.SMatrix, y=self.experiment.AOsignal_withTumor, withTumor=withTumor)
            elif self.optimizer.value == OptimizerType.DEPIERRO95.value:
                self.reconPhantom = self._DEPIERRO(SMatrix=self.SMatrix, y=self.experiment.AOsignal_withTumor, withTumor=withTumor)
            else:
                raise ValueError(f"Unknown optimizer type: {self.optimizer.value}")
        else:
            if self.experiment.AOsignal_withoutTumor is None:
                raise ValueError("AO signal without tumor is not available. Please generate AO signal without tumor the experiment first in the experiment object.")
            if self.optimizer.value ==  OptimizerType.PPGMLEM.value:
                self.reconLaser = self._MAPEM_STOP(SMatrix=self.SMatrix, y=self.experiment.AOsignal_withoutTumor, withTumor=withTumor)
            elif self.optimizer.value == OptimizerType.PGC.value:
                self.reconLaser = self._MAPEM(SMatrix=self.SMatrix, y=self.experiment.AOsignal_withoutTumor, withTumor=withTumor)
            elif self.optimizer.value == OptimizerType.DEPIERRO95.value:
                self.reconLaser = self._DEPIERRO(SMatrix=self.SMatrix, y=self.experiment.AOsignal_withoutTumor, withTumor=withTumor)
            else:
                raise ValueError(f"Unknown optimizer type: {self.optimizer.value}")

    def _MAPEM_STOP(self, SMatrix, y, withTumor):
        """
        This method implements the MAPEM_STOP algorithm using either CPU or single-GPU PyTorch acceleration.
        Multi-GPU and Multi-CPU modes are not implemented for this algorithm.
        """
        result = None
        required_memory = calculate_memory_requirement(SMatrix, y)

        if self.isGPU:
            if check_gpu_memory(config.select_best_gpu(), required_memory):
                try:
                    result = MAPEM._MAPEM_GPU_STOP(SMatrix=SMatrix, y=y, Omega=self.potentialFunction, numIterations=self.numIterations, beta=self.beta, delta=self.delta, gamma=self.gamma, sigma=self.sigma, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                except Exception as e:
                    warnings.warn(f"Falling back to CPU implementation due to an error in GPU implementation: {e}")
            else:
                warnings.warn("Insufficient GPU memory for single GPU MAPEM_STOP. Falling back to CPU.")

        if result is None:
            try:
                result = MAPEM._MAPEM_CPU_STOP(SMatrix=SMatrix, y=y, Omega=self.potentialFunction, numIterations=self.numIterations, beta=self.beta, delta=self.delta, gamma=self.gamma, sigma=self.sigma, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
            except Exception as e:
                warnings.warn(f"An error occurred in CPU implementation: {e}")
                result = None

        return result

    def _MAPEM(self, SMatrix, y, withTumor):
        """
        This method implements the MAPEM algorithm using either CPU or single-GPU PyTorch acceleration.
        Multi-GPU and Multi-CPU modes are not implemented for this algorithm.
        """
        result = None
        required_memory = calculate_memory_requirement(SMatrix, y)

        if self.isGPU:
            if check_gpu_memory(config.select_best_gpu(), required_memory):
                try:
                    result = MAPEM._MAPEM_GPU(SMatrix=SMatrix, y=y, Omega=self.potentialFunction, numIterations=self.numIterations, beta=self.beta, delta=self.delta, gamma=self.gamma, sigma=self.sigma, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                except Exception as e:
                    warnings.warn(f"Falling back to CPU implementation due to an error in GPU implementation: {e}")
            else:
                warnings.warn("Insufficient GPU memory for single GPU MAPEM. Falling back to CPU.")

        if result is None:
            try:
                result = MAPEM._MAPEM_CPU(SMatrix=SMatrix, y=y, Omega=self.potentialFunction, numIterations=self.numIterations, beta=self.beta, delta=self.delta, gamma=self.gamma, sigma=self.sigma, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
            except Exception as e:
                warnings.warn(f"An error occurred in CPU implementation: {e}")
                result = None

        return result

    def _DEPIERRO(self, SMatrix, y, withTumor):
        """
        This method implements the DEPIERRO algorithm using either CPU or single-GPU PyTorch acceleration.
        Multi-GPU and Multi-CPU modes are not implemented for this algorithm.
        """
        result = None
        required_memory = calculate_memory_requirement(SMatrix, y)

        if self.isGPU:
            if check_gpu_memory(config.select_best_gpu(), required_memory):
                try:
                    result = DEPIERRO._DEPIERRO_GPU(SMatrix=SMatrix, y=y, Omega=self.potentialFunction, numIterations=self.numIterations, beta=self.beta, sigma=self.sigma, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                except Exception as e:
                    warnings.warn(f"Falling back to CPU implementation due to an error in GPU implementation: {e}")
            else:
                warnings.warn("Insufficient GPU memory for single GPU DEPIERRO. Falling back to CPU.")

        if result is None:
            try:
                result = DEPIERRO._DEPIERRO_CPU(SMatrix=SMatrix, y=y, Omega=self.potentialFunction, numIterations=self.numIterations, beta=self.beta, sigma=self.sigma, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
            except Exception as e:
                warnings.warn(f"An error occurred in CPU implementation: {e}")
                result = None

        return result

