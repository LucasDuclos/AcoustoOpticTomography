from AOT_biomaps.AOT_Recon._mainRecon import Recon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, ProcessType


class DeepLearningRecon(Recon):
    """
    This class implements the deep learning reconstruction process.
    It currently does not perform any operations but serves as a template for future implementations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.DeepLearning
        self.model = None  # Placeholder for the deep learning model
        self.theta_matrix = []

    def run(self, processType=ProcessType.PYTHON):
        """
        This method is a placeholder for the deep learning reconstruction process.
        It currently does not perform any operations but serves as a template for future implementations.
        """
        if(processType == ProcessType.CASToR):
            self._deepLearningReconCASToR()
        elif(processType == ProcessType.PYTHON):
            self._deepLearningReconPython()
        else:
            raise ValueError(f"Unknown deep learning reconstruction type: {processType}")

    def _deepLearningReconCASToR(self):
        pass

    def _deepLearningReconPython(self):
        pass