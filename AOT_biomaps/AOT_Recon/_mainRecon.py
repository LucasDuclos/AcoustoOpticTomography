from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Experiment.Tomography import Tomography
from .ReconEnums import ReconType
from .ReconTools import mse
from skimage.metrics import structural_similarity as ssim

import os
import numpy as np
from abc import ABC, abstractmethod
from tqdm import trange
import matplotlib.pyplot as plt

    


class Recon(ABC):
    def __init__(self, experiment, saveDir = None, isGPU = config.get_process() == 'gpu', isMultiCPU = True):
        self.reconPhantom = None
        self.reconLaser = None
        self.experiment = experiment
        self.reconType = None
        self.saveDir = saveDir
        self.MSE = None
        self.SSIM = None
        self.CRC = None

        self.isGPU = isGPU
        self.isMultiCPU = isMultiCPU

        if str(type(self.experiment)) != str(Tomography):
            raise TypeError(f"[AOT-biomaps] Experiment must be of type {Tomography}")

    @abstractmethod
    def run(self,withTumor = True):
        pass

    def save(self, withTumor=True, overwrite=False, date=None, show_logs=True):
        """
        Save the reconstruction results (reconPhantom is with tumor, reconLaser is without tumor) and indices of the saved recon results, in numpy format.

        Args:
            withTumor (bool): If True, saves reconPhantom. If False, saves reconLaser. Default is True.
            overwrite (bool): If False, does not save if the file already exists. Default is False.

        Warnings:
            reconPhantom and reconLaser are lists of 2D numpy arrays, each array corresponding to one iteration.
        """
        isExisting, filepath = self.check_existing_file(date=date, withTumor=withTumor)
        if isExisting and not overwrite:
            return
        
        filename = 'reconPhantom.npy' if withTumor else 'reconLaser.npy'
        filepathRecon = os.path.join(filepath, filename)

        if withTumor:
            if not self.reconPhantom or len(self.reconPhantom) == 0:
                raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")
            np.save(filepathRecon, np.array(self.reconPhantom))
        else:
            if not self.reconLaser or len(self.reconLaser) == 0:
                raise ValueError("[AOT-biomaps] Reconstructed laser is empty. Run reconstruction first.")
            np.save(filepathRecon, np.array(self.reconLaser))

        if self.indices is not None and len(self.indices) > 0:
            filepathIndices = os.path.join(filepath, f"indices_{'withTumor' if withTumor else 'withoutTumor'}.npy")
            np.save(filepathIndices, np.array(self.indices))

        if show_logs:
            print(f"[AOT-biomaps] Reconstruction results saved to {os.path.dirname(filepath)}")

    @abstractmethod
    def check_existing_file(self, date=None, withTumor=True):
        pass

    def calculate_CRC(self, use_ROI=True):
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
            raise ValueError("[AOT-biomaps] Run reconstruction first")

        if self.reconLaser is None or self.reconLaser == []:
            raise ValueError("[AOT-biomaps] Reconstructed laser is empty. Run reconstruction first.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")

        # Handle empty reconstructions
        if self.reconLaser is None or self.reconLaser == []:
            print("[AOT-biomaps] Reconstructed laser is empty. Running reconstruction without tumor...")
            self.run(withTumor=False, isSavingEachIteration=True)

        # Get the ROI mask(s) from the phantom if needed
        if use_ROI:
            self.experiment.OpticImage.find_ROI()
            global_mask = np.logical_or.reduce(self.experiment.OpticImage.maskList)
        if len(global_mask) == 0:
            print("[AOT-biomaps] No ROIs found in the phantom. Computing global CRC instead.")
            use_ROI = False

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
            iterations = range(np.min([len(self.reconPhantom), len(self.reconLaser)]))

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

    def calculate_MSE(self,withTumor=True):
        """
        Calculate the Mean Squared Error (MSE) of the reconstruction.

        Returns:
            mse: float or list of floats, Mean Squared Error of the reconstruction
        """
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")

        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            self.MSE = mse(None, self.experiment.OpticImage.phantom, self.reconPhantom)

        elif self.reconType in (ReconType.Algebraic, ReconType.Bayesian, ReconType.Convex):
            self.MSE = []
            if withTumor:
                for theta in self.reconPhantom:
                    self.MSE.append(mse(None, self.experiment.OpticImage.phantom, theta))
            else:
                for theta in self.reconLaser:
                    self.MSE.append(mse(None, self.experiment.OpticImage.laser.intensity, theta))

    def calculate_SSIM(self, withTumor=True, show_log=False):
        """
        Calculate SSIM without normalizing images, using original data_range.
        """
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")

        # Select reference image
        if withTumor:
            ref_img = self.experiment.OpticImage.phantom
        else:
            ref_img = self.experiment.OpticImage.laser.intensity

        # Get data_range for reference image
        ref_min, ref_max = ref_img.min(), ref_img.max()
        data_range = ref_max - ref_min

        # Process reconstructions
        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            # Single reconstruction case
            recon = self.reconPhantom
            self.SSIM = ssim(ref_img, recon, data_range=data_range)

        else:  # Algebraic/Bayesian (multiple reconstructions)
            self.SSIM = []
            recon_list = self.reconPhantom if withTumor else self.reconLaser

            # Use trange if show_log is True, otherwise range
            iteration = trange(len(recon_list), desc=f"Calculating SSIM {'with' if withTumor else 'without'} tumor") if show_log else range(len(recon_list))

            for i in iteration:
                theta = recon_list[i]
                # Calculate data_range for each reconstruction (if different from reference)
                theta_min, theta_max = theta.min(), theta.max()
                current_data_range = max(data_range, theta_max - theta_min)  # Use the larger range
                self.SSIM.append(ssim(ref_img, theta, data_range=current_data_range))