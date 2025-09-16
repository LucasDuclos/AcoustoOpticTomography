from AOT_biomaps.AOT_Recon.AlgebraicRecon import AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, ProcessType, NoiseType
from AOT_biomaps.AOT_Recon.AOT_Optimizers.PDHG import chambolle_pock_TV_cpu, chambolle_pock_TV_gpu, chambolle_pock_KL_cpu, chambolle_pock_KL_gpu

class PrimalDualRecon(AlgebraicRecon):
    """
    This class implements the convex reconstruction process.
    It currently does not perform any operations but serves as a template for future implementations.
    """
    def __init__(self, alpha, theta=1.0, L=None, noiseModel = NoiseType.GAUSSIAN,**kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.Convex
        self.alpha = alpha # regularization parameter
        self.theta = theta # relaxation parameter (between 1 and 2)
        self.L = L # norme spectrale de l'opérateur linéaire défini par les matrices P et P^T
        self.noiseModel = noiseModel

    def run(self, processType=ProcessType.PYTHON, withTumor=True):
        """
        This method is a placeholder for the convex reconstruction process.
        It currently does not perform any operations but serves as a template for future implementations.
        """
        if(processType == ProcessType.CASToR):
            self._convexReconCASToR(withTumor)
        elif(processType == ProcessType.PYTHON):
            self._convexReconPython(withTumor)
        else:
            raise ValueError(f"Unknown convex reconstruction type: {processType}")

    def _convexReconCASToR(self, withTumor):
        pass

    def _convexReconPython(self, withTumor):
        if withTumor:
            self.reconPhantom = self._chambolle_pock(self.SMatrix, y=self.experiment.AOsignal_withTumor, withTumor=withTumor)
        else:
            self.reconLaser = self._chambolle_pock(self.SMatrix, y=self.experiment.AOsignal_withoutTumor, withTumor=withTumor)
    
    def _chambolle_pock(self, SMatrix, y, withTumor):
        if self.isGPU:
            try:
                if self.noiseModel == NoiseType.GAUSSIAN:
                    return chambolle_pock_TV_gpu(SMatrix=SMatrix, y=y, alpha=self.alpha, theta=self.theta, numIterations=self.numIterations, L=self.L, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                elif self.noiseModel == NoiseType.POISSON:
                    return chambolle_pock_KL_gpu(SMatrix=SMatrix, y=y, alpha=self.alpha, theta=self.theta, numIterations=self.numIterations, L=self.L, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                else:
                    raise ValueError(f"Noise model must be either GAUSSIAN or POISSON, got {self.noiseModel}")
            except:
                if self.noiseModel == NoiseType.GAUSSIAN:
                    return chambolle_pock_TV_cpu(SMatrix=SMatrix, y=y, alpha=self.alpha, theta=self.theta, numIterations=self.numIterations, L=self.L, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                elif self.noiseModel == NoiseType.POISSON:
                    return chambolle_pock_KL_cpu(SMatrix=SMatrix, y=y, alpha=self.alpha, theta=self.theta, numIterations=self.numIterations, L=self.L, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
                else:
                    raise ValueError(f"Noise model must be either GAUSSIAN or POISSON, got {self.noiseModel}")
        else:
            if self.noiseModel == NoiseType.GAUSSIAN:
                return chambolle_pock_TV_cpu(SMatrix=SMatrix, y=y, alpha=self.alpha, theta=self.theta, numIterations=self.numIterations, L=self.L, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
            elif self.noiseModel == NoiseType.POISSON:
                return chambolle_pock_KL_cpu(SMatrix=SMatrix, y=y, alpha=self.alpha, theta=self.theta, numIterations=self.numIterations, L=self.L, isSavingEachIteration=self.isSavingEachIteration, withTumor=withTumor)
            else:
                raise ValueError(f"Noise model must be either GAUSSIAN or POISSON, got {self.noiseModel}")
            



   
