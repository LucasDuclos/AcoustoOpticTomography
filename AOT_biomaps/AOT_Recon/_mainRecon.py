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

    def _gt(self, withTumor=True):
        """Ground truth cropped to the effective SMatrix geometry (handles virtual/physical truncation)."""
        optic = self.experiment.OpticImage
        if optic is None:
            return None
        gt = optic.phantom if withTumor else optic.laser.intensity
        S = getattr(self, 'SMatrix', None)
        if S is not None and hasattr(S, 'crop_to_effective'):
            return S.crop_to_effective(gt)
        return gt

    def _crop_mask(self, mask):
        """Crop a full-size (Z, X) mask/label array to the effective SMatrix geometry."""
        S = getattr(self, 'SMatrix', None)
        if S is not None and hasattr(S, 'crop_to_effective'):
            return S.crop_to_effective(mask)
        return mask
    
    def _eff_extent(self, scale=1e3):
        """imshow extent [x0, x1, z1, z0] in mm, matching the effective SMatrix geometry."""
        g = self.experiment.params.general
        Xrange = [g['Xrange'][0] * scale, g['Xrange'][1] * scale]
        Zrange = [g['Zrange'][0] * scale, g['Zrange'][1] * scale]
        S = getattr(self, 'SMatrix', None)
        if S is not None and hasattr(S, 'effective_extent'):
            return S.effective_extent(Xrange, Zrange)
        return [Xrange[0], Xrange[1], Zrange[1], Zrange[0]]

    def calculate_CRC(self, use_ROI=True):
        """
        Computes the Contrast Recovery Coefficient (CRC) for all ROIs combined or globally.
        Ground truths and ROI masks are cropped to the effective SMatrix geometry
        (virtual or physical truncation aware).
        """
        if self.reconType is None:
            raise ValueError("[AOT-biomaps] Run reconstruction first")

        if self.reconLaser is None or self.reconLaser == []:
            raise ValueError("[AOT-biomaps] Reconstructed laser is empty. Run reconstruction first.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")

        # Effective ground truths (cropped to the reconstruction geometry)
        gt_phantom = self._gt(withTumor=True)
        gt_laser = self._gt(withTumor=False)

        # Get the ROI mask(s) from the phantom if needed (cropped to effective geometry)
        global_mask = None
        if use_ROI:
            self.experiment.OpticImage.find_ROI()
            masks = [self._crop_mask(m) for m in self.experiment.OpticImage.maskList]
            global_mask = np.logical_or.reduce(masks)
        if global_mask is not None and len(global_mask) == 0:
            print("[AOT-biomaps] No ROIs found in the phantom. Computing global CRC instead.")
            use_ROI = False

        # Analytic reconstruction case
        if self.reconType is ReconType.Analytic:
            if use_ROI:
                recon_ratio = np.mean(self.reconPhantom[global_mask]) / np.mean(self.reconLaser[global_mask])
                lambda_ratio = np.mean(gt_phantom[global_mask]) / np.mean(gt_laser[global_mask])
            else:
                recon_ratio = np.mean(self.reconPhantom) / np.mean(self.reconLaser)
                lambda_ratio = np.mean(gt_phantom) / np.mean(gt_laser)

            self.CRC = (recon_ratio - 1) / (lambda_ratio - 1)

        # Iterative reconstruction case
        else:
            iterations = range(np.min([len(self.reconPhantom), len(self.reconLaser)]))

            crc_list = []
            for it in iterations:
                if use_ROI:
                    recon_ratio = np.mean(self.reconPhantom[it][global_mask]) / np.mean(self.reconLaser[it][global_mask])
                    lambda_ratio = np.mean(gt_phantom[global_mask]) / np.mean(gt_laser[global_mask])
                else:
                    recon_ratio = np.mean(self.reconPhantom[it]) / np.mean(self.reconLaser[it])
                    lambda_ratio = np.mean(gt_phantom) / np.mean(gt_laser)

                crc_list.append((recon_ratio - 1) / (lambda_ratio - 1))

            self.CRC = crc_list

    def calculate_MSE(self, withTumor=True):
        """
        Calculate the Mean Squared Error (MSE) of the reconstruction,
        against the ground truth cropped to the effective geometry.
        """
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")

        gt = self._gt(withTumor=withTumor)

        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            self.MSE = mse(None, gt, self.reconPhantom)

        elif self.reconType in (ReconType.Algebraic, ReconType.Bayesian, ReconType.Convex):
            self.MSE = []
            if withTumor:
                for theta in self.reconPhantom:
                    self.MSE.append(mse(None, gt, theta))
            else:
                for theta in self.reconLaser:
                    self.MSE.append(mse(None, gt, theta))

    def calculate_SSIM(self, withTumor=True, show_log=False):
        """
        Calculate SSIM without normalizing images, using original data_range.
        Reference image is cropped to the effective geometry.
        """
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("[AOT-biomaps] Reconstructed phantom is empty. Run reconstruction first.")

        ref_img = self._gt(withTumor=withTumor)

        ref_min, ref_max = ref_img.min(), ref_img.max()
        data_range = ref_max - ref_min

        if self.reconType in (ReconType.Analytic, ReconType.DeepLearning):
            recon = self.reconPhantom
            self.SSIM = ssim(ref_img, recon, data_range=data_range)

        else:
            self.SSIM = []
            recon_list = self.reconPhantom if withTumor else self.reconLaser

            iteration = trange(len(recon_list), desc=f"Calculating SSIM {'with' if withTumor else 'without'} tumor") if show_log else range(len(recon_list))

            for i in iteration:
                theta = recon_list[i]
                theta_min, theta_max = theta.min(), theta.max()
                current_data_range = max(data_range, theta_max - theta_min)
                self.SSIM.append(ssim(ref_img, theta, data_range=current_data_range))