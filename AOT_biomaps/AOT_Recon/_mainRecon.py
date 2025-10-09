from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Experiment.Tomography import Tomography
from .ReconEnums import ReconType
from .ReconTools import mse, ssim

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Recon(ABC):
    def __init__(self, experiment, saveDir = None, isGPU = config.get_process() == 'gpu',  isMultiGPU =  True if config.numGPUs > 1 else False, isMultiCPU = True):
        self.reconPhantom = None
        self.reconLaser = None
        self.experiment = experiment
        self.reconType = None
        self.saveDir = saveDir
        self.MSE = None
        self.SSIM = None
        self.CRC = None

        self.isGPU = isGPU
        self.isMultiGPU = isMultiGPU
        self.isMultiCPU = isMultiCPU

        if str(type(self.experiment)) != str(Tomography):
            raise TypeError(f"Experiment must be of type {Tomography}")

    @abstractmethod
    def run(self,withTumor = True):
        pass

    def save(self, withTumor=True, overwrite=False):
        """
        Save the reconstruction results (reconPhantom is with tumor, reconLaser is without tumor) and indices of the saved recon results, in numpy format.

        Args:
            withTumor (bool): If True, saves reconPhantom. If False, saves reconLaser. Default is True.
            overwrite (bool): If False, does not save if the file already exists. Default is False.

        Warnings:
            reconPhantom and reconLaser are lists of 2D numpy arrays, each array corresponding to one iteration.
        """
        should_save, filepath = self.checkExistingFile(withTumor, overwrite)
        if not should_save:
            return

        if withTumor:
            if not self.reconPhantom or len(self.reconPhantom) == 0:
                raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
            np.save(filepath, np.array(self.reconPhantom))
        else:
            if not self.reconLaser or len(self.reconLaser) == 0:
                raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
            np.save(filepath, np.array(self.reconLaser))

        print(f"Reconstruction results saved to {os.path.dirname(filepath)}")

    @abstractmethod
    def checkExistingFile(self, withTumor=True, overwrite=False):
        pass

    def calculateCRC(self, use_ROI=True):
        """
        Computes the Contrast Recovery Coefficient (CRC) for all ROIs combined or globally.
        For analytic reconstruction: returns a single CRC value.
        For iterative reconstruction: returns a list of CRC values (one per iteration).
        If iteration is specified, returns CRC for that specific iteration only.

        :param iteration: Specific iteration index (optional). If None, computes for all iterations.
        :param use_ROI: If True, computes CRC for all ROIs combined. If False, computes global CRC.
        :return: CRC value or list of CRC values.
        """
        if self.reconType is None:
            raise ValueError("Run reconstruction first")

        if self.reconLaser is None or self.reconLaser == []:
            raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")

        # Handle empty reconstructions
        if self.reconLaser is None or self.reconLaser == []:
            print("Reconstructed laser is empty. Running reconstruction without tumor...")
            self.run(withTumor=False, isSavingEachIteration=True)

        # Get the ROI mask(s) from the phantom if needed
        if use_ROI:
            self.experiment.OpticImage.find_ROI()
            global_mask = np.logical_or.reduce(self.experiment.OpticImage.maskList)

        # Analytic reconstruction case
        if self.reconType is ReconType.Analytic:
            if use_ROI:
                recon_ratio = np.mean(self.reconPhantom[global_mask]) / np.mean(self.reconLaser[global_mask])
                lambda_ratio = np.mean(self.experiment.OpticImage.phantom[global_mask]) / np.mean(self.experiment.OpticImage.laser.intensity[global_mask])
            else:
                recon_ratio = np.mean(self.reconPhantom) / np.mean(self.reconLaser)
                lambda_ratio = np.mean(self.experiment.OpticImage.phantom) / np.mean(self.experiment.OpticImage.laser.intensity)

            self.CRC =(recon_ratio - 1) / (lambda_ratio - 1)

        # Iterative reconstruction case
        else:
            iterations = range(len(self.reconPhantom))

            crc_list = []
            for it in iterations:
                if use_ROI:
                    recon_ratio = np.mean(self.reconPhantom[it][global_mask]) / np.mean(self.reconLaser[it][global_mask])
                    lambda_ratio = np.mean(self.experiment.OpticImage.phantom[global_mask]) / np.mean(self.experiment.OpticImage.laser.intensity[global_mask])
                else:
                    recon_ratio = np.mean(self.reconPhantom[it]) / np.mean(self.reconLaser[it])
                    lambda_ratio = np.mean(self.experiment.OpticImage.phantom) / np.mean(self.experiment.OpticImage.laser.intensity)

                crc_list.append((recon_ratio - 1) / (lambda_ratio - 1))

            self.CRC = crc_list

    def calculateMSE(self):
        """
        Calculate the Mean Squared Error (MSE) of the reconstruction.

        Returns:
            mse: float or list of floats, Mean Squared Error of the reconstruction
        """
                
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")

        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            self.MSE = mse(self.experiment.OpticImage.phantom, self.reconPhantom)

        elif self.reconType in (ReconType.Algebraic, ReconType.Bayesian):
            self.MSE = []
            for theta in self.reconPhantom:
                self.MSE.append(mse(self.experiment.OpticImage.phantom, theta))
  
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
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        if withTumor:
            if self.reconPhantom is None or self.reconPhantom == []:
                raise ValueError("Reconstructed phantom with tumor is empty. Run reconstruction first.")
            if isinstance(self.reconPhantom, list):
                image = self.reconPhantom[-1]
            else:
                image = self.reconPhantom
            # Phantom original
            im0 = axs[0].imshow(
                self.experiment.OpticImage.phantom,
                cmap='hot',
                vmin=0,
                vmax=1,
                extent=(
                    self.experiment.params.general['Xrange'][0],
                    self.experiment.params.general['Xrange'][1],
                    self.experiment.params.general['Zrange'][1],
                    self.experiment.params.general['Zrange'][0]
                ),
                aspect='equal'  
            )
            axs[0].set_title("Phantom with tumor")
            axs[0].set_xlabel("x (mm)", fontsize=12)
            axs[0].set_ylabel("z (mm)", fontsize=12)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            # Phantom reconstruit
            im1 = axs[1].imshow(
                image,
                cmap='hot',
                vmin=0,
                vmax=1,
                extent=(
                    self.experiment.params.general['Xrange'][0],
                    self.experiment.params.general['Xrange'][1],
                    self.experiment.params.general['Zrange'][1],
                    self.experiment.params.general['Zrange'][0]
                ),
                aspect='equal'  
            )
            axs[1].set_title("Reconstructed phantom with tumor")
            axs[1].set_xlabel("x (mm)", fontsize=12)
            axs[1].set_ylabel("z (mm)", fontsize=12)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        else:
            if self.reconLaser is None or self.reconLaser == []:
                raise ValueError("Reconstructed laser without tumor is empty. Run reconstruction first.")
            if isinstance(self.reconLaser, list):
                image = self.reconLaser[-1]
            else:
                image = self.reconLaser
            # Laser original
            im0 = axs[0].imshow(
                self.experiment.OpticImage.laser.intensity,
                cmap='hot',
                vmin=0,
                vmax=np.max(self.experiment.OpticImage.laser.intensity),
                extent=(
                    self.experiment.params.general['Xrange'][0],
                    self.experiment.params.general['Xrange'][1],
                    self.experiment.params.general['Zrange'][1],
                    self.experiment.params.general['Zrange'][0]
                ),
                aspect='equal'  
            )
            axs[0].set_title("Laser without tumor")
            axs[0].set_xlabel("x (mm)", fontsize=12)
            axs[0].set_ylabel("z (mm)", fontsize=12)
            axs[0].tick_params(axis='both', which='major', labelsize=8)
            # Laser reconstruit
            im1 = axs[1].imshow(
                image,
                cmap='hot',
                vmin=0,
                vmax=np.max(self.experiment.OpticImage.laser.intensity),
                extent=(
                    self.experiment.params.general['Xrange'][0],
                    self.experiment.params.general['Xrange'][1],
                    self.experiment.params.general['Zrange'][1],
                    self.experiment.params.general['Zrange'][0]
                ),
                aspect='equal'
            )
            axs[1].set_title("Reconstructed laser without tumor")
            axs[1].set_xlabel("x (mm)", fontsize=12)
            axs[1].set_ylabel("z (mm)", fontsize=12)
            axs[1].tick_params(axis='both', which='major', labelsize=8)
            axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Colorbar commune
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Normalized Intensity', fontsize=12)
        cbar.ax.tick_params(labelsize=8)

        plt.subplots_adjust(wspace=0.3)

        if savePath is not None:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            if withTumor:
                plt.savefig(os.path.join(savePath, 'recon_with_tumor.png'), dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(savePath, 'recon_without_tumor.png'), dpi=300, bbox_inches='tight')

        plt.show()


