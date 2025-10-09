from AOT_biomaps.AOT_Recon.AlgebraicRecon import AlgebraicRecon
from AOT_biomaps.AOT_Recon.ReconEnums import ReconType, ProcessType
from AOT_biomaps.AOT_Recon.AOT_Optimizers import CP_KL, CP_TV
from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType

import os
from datetime import datetime
import numpy as np

class PrimalDualRecon(AlgebraicRecon):
    """
    This class implements the convex reconstruction process.
    It currently does not perform any operations but serves as a template for future implementations.
    """
    def __init__(self, alpha, theta=1.0, L=None, **kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.Convex
        self.alpha = alpha # regularization parameter
        self.theta = theta # relaxation parameter (between 1 and 2)
        self.L = L # norme spectrale de l'opérateur linéaire défini par les matrices P et P^T

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
        raise NotImplementedError("CASToR convex reconstruction is not implemented yet.")


    def checkExistingFile(self, date = None, withTumor=True):
        """
        Check if the file already exists, based on current instance parameters.
        Returns:
            tuple: (bool: whether to save, str: the filepath)
        """
        date = datetime.now().strftime("%d%m")
        results_dir = os.path.join(
            self.saveDir,
            f'results_{date}_{self.optimizer.value}_Alpha_{self.alpha}_Theta_{self.theta}_L_{self.L}'
        )
        os.makedirs(results_dir, exist_ok=True)

        filename = 'reconPhantom.npy' if withTumor else 'reconLaser.npy'
        filepath = os.path.join(results_dir, filename)

        if os.path.exists(filepath):
            return (True, filepath)

        return (False, filepath)



    def load(self, withTumor=True, results_date=None, optimizer=None, alpha=None, theta=None, L=None, filePath=None):
        """
        Load the reconstruction results and indices and store them in self.
        Args:
            withTumor (bool): If True, loads the reconstruction with tumor; otherwise, loads the reconstruction without tumor.
            results_date (str): Date string (format "ddmm") to specify which results to load. If None, uses the most recent date in saveDir.
            optimizer (OptimizerType): Optimizer type to filter results. If None, uses the current optimizer of the instance.
            alpha (float): Alpha parameter to match the saved directory. If None, uses the current alpha of the instance.
            theta (float): Theta parameter to match the saved directory. If None, uses the current theta of the instance.
            L (float): L parameter to match the saved directory. If None, uses the current L of the instance.
            filePath (str): Optional. If provided, loads directly from this path (overrides saveDir and results_date).
        """
        if filePath is not None:
            # Mode chargement direct depuis un fichier
            recon_key = 'reconPhantom' if withTumor else 'reconLaser'
            recon_path = filePath
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"No reconstruction file found at {recon_path}.")

            if withTumor:
                self.reconPhantom = np.load(recon_path, allow_pickle=True)
            else:
                self.reconLaser = np.load(recon_path, allow_pickle=True)

            # Essayer de charger les indices (fichier avec suffixe "_indices.npy" ou "reconIndices.npy")
            base_dir, file_name = os.path.split(recon_path)
            file_base, _ = os.path.splitext(file_name)
            indices_path = os.path.join(base_dir, f"{file_base}_indices.npy")
            if not os.path.exists(indices_path):
                indices_path = os.path.join(base_dir, 'reconIndices.npy')  # Alternative

            if os.path.exists(indices_path):
                self.indices = np.load(indices_path, allow_pickle=True)
            else:
                self.indices = None

            print(f"Loaded reconstruction results and indices from {recon_path}")
        else:
            # Mode chargement depuis le répertoire de résultats
            if self.saveDir is None:
                raise ValueError("Save directory is not specified. Please set saveDir before loading.")

            # Use current optimizer if not provided
            opt_name = optimizer.value if optimizer is not None else self.optimizer.value

            # Build the directory path
            if results_date is None:
                dir_pattern = f'results_*_{opt_name}_Alpha_{alpha if alpha is not None else self.alpha}_Theta_{theta if theta is not None else self.theta}_L_{L if L is not None else self.L}'
                dirs = [d for d in os.listdir(self.saveDir) if os.path.isdir(os.path.join(self.saveDir, d)) and dir_pattern in d]
                if not dirs:
                    raise FileNotFoundError(f"No matching results directory found for pattern '{dir_pattern}' in {self.saveDir}.")
                dirs.sort(reverse=True)  # Most recent first
                results_dir = os.path.join(self.saveDir, dirs[0])
            else:
                results_dir = os.path.join(self.saveDir, f'results_{results_date}_{opt_name}_Alpha_{alpha if alpha is not None else self.alpha}_Theta_{theta if theta is not None else self.theta}_L_{L if L is not None else self.L}')
                if not os.path.exists(results_dir):
                    raise FileNotFoundError(f"Directory {results_dir} does not exist.")

            # Load reconstruction results
            recon_key = 'reconPhantom' if withTumor else 'reconLaser'
            recon_path = os.path.join(results_dir, f'{recon_key}.npy')
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"No reconstruction file found at {recon_path}.")

            if withTumor:
                self.reconPhantom = np.load(recon_path, allow_pickle=True)
            else:
                self.reconLaser = np.load(recon_path, allow_pickle=True)

            # Load saved indices
            indices_path = os.path.join(results_dir, 'reconIndices.npy')
            if not os.path.exists(indices_path):
                raise FileNotFoundError(f"No indices file found at {indices_path}.")

            self.indices = np.load(indices_path, allow_pickle=True)

            print(f"Loaded reconstruction results and indices from {results_dir}")

    def _convexReconPython(self, withTumor):
        if withTumor:
            y=self.experiment.AOsignal_withTumor

        else:
            y=self.experiment.AOsignal_withoutTumor

        if self.optimizer == OptimizerType.CP_TV:
            if withTumor:
                self.reconPhantom, self.indices = CP_TV(
                    self.SMatrix, 
                    y=self.experiment.AOsignal_withTumor, 
                    alpha=self.alpha, 
                    theta=self.theta, 
                    numIterations=self.numIterations, 
                    isSavingEachIteration=self.isSavingEachIteration,
                    L=self.L, 
                    withTumor=withTumor,
                    device=None
                    )
            else:
                self.reconLaser, self.indices = CP_TV(
                    self.SMatrix, 
                    y=self.experiment.AOsignal_withoutTumor, 
                    alpha=self.alpha, 
                    theta=self.theta, 
                    numIterations=self.numIterations, 
                    isSavingEachIteration=self.isSavingEachIteration,
                    L=self.L, 
                    withTumor=withTumor,
                    device=None
                    )
        elif self.optimizer == OptimizerType.CP_KL:
            if withTumor:
                self.reconPhantom, self.indices = CP_KL(
                    self.SMatrix, 
                    y=self.experiment.AOsignal_withTumor, 
                    alpha=self.alpha, 
                    theta=self.theta, 
                    numIterations=self.numIterations, 
                    isSavingEachIteration=self.isSavingEachIteration,
                    L=self.L, 
                    withTumor=withTumor,
                    device=None
                )
            else:
                self.reconLaser, self.indices = CP_KL(
                    self.SMatrix, 
                    y=self.experiment.AOsignal_withoutTumor, 
                    alpha=self.alpha, 
                    theta=self.theta, 
                    numIterations=self.numIterations, 
                    isSavingEachIteration=self.isSavingEachIteration,
                    L=self.L, 
                    withTumor=withTumor,
                    device=None
                )
        else:
            raise ValueError(f"Optimizer value must be CP_TV or CP_KL, got {self.optimizer}")

            



   
