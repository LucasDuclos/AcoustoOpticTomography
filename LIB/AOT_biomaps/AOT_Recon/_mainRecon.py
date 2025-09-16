from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Experiment.Tomography import Tomography

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from .ReconEnums import ReconType
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

class Recon(ABC):
    def __init__(self, experiment, saveDir = None, isGPU = config.get_process() == 'gpu',  isMultiGPU =  True if config.numGPUs > 1 else False, isMultiCPU = True):
        self.reconPhantom = None
        self.reconLaser = None
        self.experiment = experiment
        self.reconType = None
        self.saveDir = saveDir
        self.MSE = None
        self.SSIM = None

        self.isGPU = isGPU
        self.isMultiGPU = isMultiGPU
        self.isMultiCPU = isMultiCPU

        if str(type(self.experiment)) != str(Tomography):
            raise TypeError(f"Experiment must be of type {Tomography}")

    @abstractmethod
    def run(self,withTumor = True):
        pass

    def calculateCRC(self,iteration,ROI_mask = None):
        """
        Computes the Contrast Recovery Coefficient (CRC) for a given ROI.
        """
        if self.reconType is ReconType.Analytic:
            raise TypeError(f"Impossible to calculate CRC with analytical reconstruction")
        elif self.reconType is None:
            raise ValueError("Run reconstruction first")
        
        if self.reconLaser is None or self.reconLaser == []:
            raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
        if isinstance(self.Laser,list) and len(self.Laser) == 1:
            raise ValueError("Reconstructed Image without tumor is a single frame. Run reconstruction with isSavingEachIteration=True to get a sequence of frames.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
        if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
            raise ValueError("Reconstructed Image with tumor is a single frame. Run reconstruction with isSavingEachIteration=True to get a sequence of frames.")
        
        if self.reconLaser is None or self.reconLaser == []:
            print("Reconstructed laser is empty. Running reconstruction without tumor...")
            self.run(withTumor = False, isSavingEachIteration=True)
        if ROI_mask is not None:
            recon_ratio = np.mean(self.reconPhantom[iteration][ROI_mask]) / np.mean(self.reconLaser[iteration][ROI_mask])
            lambda_ratio = np.mean(self.experiment.OpticImage.phantom[ROI_mask]) / np.mean(self.experiment.OpticImage.laser[ROI_mask]) 
        else:
            recon_ratio = np.mean(self.reconPhantom[iteration]) / np.mean(self.reconLaser[iteration])
            lambda_ratio = np.mean(self.experiment.OpticImage.phantom) / np.mean(self.experiment.OpticImage.laser)
        
        # Compute CRC
        CRC = (recon_ratio - 1) / (lambda_ratio - 1)
        return CRC
    
    def calculateMSE(self):
        """
        Calculate the Mean Squared Error (MSE) of the reconstruction.

        Returns:
            mse: float or list of floats, Mean Squared Error of the reconstruction
        """
                
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")

        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            self.MSE = mean_squared_error(self.experiment.OpticImage.phantom, self.reconPhantom)

        elif self.reconType in (ReconType.Algebraic, ReconType.Bayesian):
            self.MSE = []
            for theta in self.reconPhantom:
                self.MSE.append(mean_squared_error(self.experiment.OpticImage.phantom, theta))
  
    def calculateSSIM(self):
        """
        Calculate the Structural Similarity Index (SSIM) of the reconstruction.

        Returns:
            ssim: float or list of floats, Structural Similarity Index of the reconstruction
        """

        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
    
        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            data_range = self.reconPhantom.max() - self.reconPhantom.min()
            self.SSIM = ssim(self.experiment.OpticImage.phantom, self.reconPhantom, data_range=data_range)

        elif self.reconType in (ReconType.Algebraic, ReconType.Bayesian):
            self.SSIM = []
            for theta in self.reconPhantom:
                data_range = theta.max() - theta.min()
                ssim_value = ssim(self.experiment.OpticImage.phantom, theta, data_range=data_range)
                self.SSIM.append(ssim_value)
 
    def show(self, withTumor=True, savePath=None):
        if withTumor:
            if self.reconPhantom is None or self.reconPhantom == []:
                raise ValueError("Reconstructed phantom with tumor is empty. Run reconstruction first.")
            if isinstance(self.reconPhantom, list):
                image = self.reconPhantom[-1]
            else:
                image = self.reconPhantom
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(self.experiment.OpticImage.phantom, cmap='hot', vmin=0, vmax=np.max(self.experiment.OpticImage.phantom), extent=(self.experiment.params.general['Xrange'][0],self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]))
            plt.title("Phantom with tumor")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(image, cmap='hot', vmin=0, vmax=np.max(image), extent=(self.experiment.params.general['Xrange'][0],self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]))
            plt.title("Reconstructed phantom with tumor")
            plt.colorbar()
            plt.show()
        else:
            if self.reconLaser is None or self.reconLaser == []:
                raise ValueError("Reconstructed laser without tumor is empty. Run reconstruction first.")
            if isinstance(self.reconLaser, list):
                image = self.reconLaser[-1]
            else:
                image = self.reconLaser
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(self.experiment.OpticImage.laser, cmap='hot', vmin=0, vmax=np.max(self.experiment.OpticImage.laser), extent=(self.experiment.params.general['Xrange'][0],self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]))
            plt.title("Laser without tumor")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(image, cmap='hot', vmin=0, vmax=np.max(image), extent=(self.experiment.params.general['Xrange'][0],self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]))
            plt.title("Reconstructed laser without tumor")
            plt.colorbar()
            plt.show()
        
        if savePath is not None:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            if withTumor:
                plt.savefig(os.path.join(savePath, 'recon_with_tumor.png'))
            else:
                plt.savefig(os.path.join(savePath, 'recon_without_tumor.png'))

