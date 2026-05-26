import concurrent
from AOT_biomaps.Config import config

from ._mainRecon import Recon
from .ReconEnums import ReconType, OptimizerType, ProcessType, SMatrixType
from .AOT_Optimizers import MLEM, LS
from .AOT_SMatrix import SMatrix_CSR, SMatrix_SELL

import os
import subprocess
import numpy as np
from datetime import datetime
from tempfile import gettempdir
import math

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    animation = None
    MATPLOTLIB_AVAILABLE = False

class AlgebraicRecon(Recon):
    """
    This class implements the Algebraic reconstruction process.
    It currently does not perform any operations but serves as a template for future implementations.
    """
    def __init__(self, opti = OptimizerType.MLEM, numIterations = 10000, numSubsets = 1, isSavingEachIteration=True, maxSaves = 5000, denominatorThreshold = 1e-6, smatrixType = SMatrixType.SELL, sparseThreshold=0.1, isComplexeRecon = False, device = None, **kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.Algebraic
        self.optimizer = opti
        self.reconPhantom = []
        self.reconLaser = []
        self.indices = []
        self.numIterations = numIterations
        self.numSubsets = numSubsets
        self.isSavingEachIteration = isSavingEachIteration
        self.maxSaves = maxSaves
        self.denominatorThreshold = denominatorThreshold
        self.isComplexeRecon = isComplexeRecon
        self.device = device if device is not None else config.select_best_gpu()
        self.SMatrix = None  # system matrix
        self.smatrixType = smatrixType # SMatrixType.DENSE if no sparsing, else SMatrixType.SELL or SMatrixType.CSR or SMatrixType.COO
        # Sparse matrix attributes

        self.sparseThreshold = sparseThreshold

        if self.numIterations <= 0:
            raise ValueError("Number of iterations must be greater than 0.")
        if self.numSubsets <= 0:
            raise ValueError("Number of subsets must be greater than 0.")
        if type(self.numIterations) is not int:
            raise TypeError("Number of iterations must be an integer.")
        if type(self.numSubsets) is not int:
            raise TypeError("Number of subsets must be an integer.")
        
        print("Generating system matrix (processing acoustic fields)...")
        if self.isComplexeRecon:
            if self.experiment.AOsignal_withTumor is not None:
                self.experiment.AOsignal_withTumor_demodulated = self.experiment.parse_and_demodulate(withTumor=True)
            elif self.experiment.AOsignal_withoutTumor is not None:
                self.experiment.AOsignal_withoutTumor_demodulated = self.experiment.parse_and_demodulate(withTumor=False)
            else:
                raise ValueError("No AO signal available for demodulation. Please provide at least one signal, with or without tumor.")
            self.experiment.AcousticFields_demodulated = self.experiment.demodulate_acoustic_fields()

        if self.smatrixType == SMatrixType.DENSE:
            self.SMatrix = self._fillDenseSMatrix()
        else:
            self.SMatrix = self._fillSparseSMatrix(isShowLogs=True)

    # PUBLIC METHODS

    def run(self, processType = ProcessType.PYTHON, withTumor= True, show_logs=True):
        """
        This method is a placeholder for the Algebraic reconstruction process.
        It currently does not perform any operations but serves as a template for future implementations.
        """
        if(processType == ProcessType.CASToR):
            self._AlgebraicReconCASToR(withTumor=withTumor, show_logs=show_logs)
        elif(processType == ProcessType.PYTHON):
            self._AlgebraicReconPython(withTumor=withTumor, show_logs=show_logs)
        else:
            raise ValueError(f"Unknown Algebraic reconstruction type: {processType}")
    
    def plot_MSE(self, isSaving=True, log_scale_x=False, log_scale_y=False, figSize=(4,3), show_logs=True):
        """
        Plot the Mean Squared Error (MSE) of the reconstruction.

        Parameters:
            isSaving: bool, whether to save the plot.
            log_scale_x: bool, if True, use logarithmic scale for the x-axis.
            log_scale_y: bool, if True, use logarithmic scale for the y-axis.
        Returns:
            None
        """
        if not self.MSE:
            raise ValueError("MSE is empty. Please calculate MSE first.")

        best_idx = self.indices[np.argmin(self.MSE)]
        if show_logs:
            print(f"Lowest MSE = {np.min(self.MSE):.4f} at iteration {best_idx+1}")
        # Plot MSE curve
        plt.figure(figsize=figSize)
        plt.plot(self.indices, self.MSE, 'r-', label="MSE curve")
        # Add blue dashed lines
        plt.axhline(np.min(self.MSE), color='blue', linestyle='--', label=f"Min MSE = {np.min(self.MSE):.4f}")
        plt.axvline(best_idx, color='blue', linestyle='--', label=f"Iteration = {best_idx+1}")
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE vs. Iteration")
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        if isSaving and self.saveDir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            scale_str = ""
            if log_scale_x and log_scale_y:
                scale_str = "_loglog"
            elif log_scale_x:
                scale_str = "_logx"
            elif log_scale_y:
                scale_str = "_logy"
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_MSE_plot_{self.optimizer.name}_{scale_str}{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"MSE plot saved to {SavingFolder}")

        plt.show()

    def show_MSE_bestRecon(self, isSaving=True, show_logs=True, figSize=(15, 5)):
        if not self.MSE:
            raise ValueError("MSE is empty. Please calculate MSE first.")

        best_idx = np.argmin(self.MSE)
        best_recon = self.reconPhantom[best_idx]

        # Crée la figure et les axes
        fig, axs = plt.subplots(1, 3, figsize=figSize)

        # Left: Best reconstructed image (normalized)
        im0 = axs[0].imshow(best_recon,
                            extent=(self.experiment.params.general['Xrange'][0]*1000, self.experiment.params.general['Xrange'][1]*1000,
                                    self.experiment.params.general['Zrange'][1]*1000, self.experiment.params.general['Zrange'][0]*1000),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[0].set_title(f"Min MSE Reconstruction\nIter {self.indices[best_idx]}, MSE={np.min(self.MSE):.4f}")
        axs[0].set_xlabel("x (mm)", fontsize=12)
        axs[0].set_ylabel("z (mm)", fontsize=12)
        axs[0].tick_params(axis='both', which='major', labelsize=8)

        # Middle: Ground truth (normalized)
        im1 = axs[1].imshow(self.experiment.OpticImage.phantom,
                            extent=(self.experiment.params.general['Xrange'][0]*1000, self.experiment.params.general['Xrange'][1]*1000,
                                    self.experiment.params.general['Zrange'][1]*1000, self.experiment.params.general['Zrange'][0]*1000),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[1].set_title(r"Ground Truth ($\lambda$)")
        axs[1].set_xlabel("x (mm)", fontsize=12)
        axs[1].set_ylabel("z (mm)", fontsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Right: Reconstruction at last iteration
        lastRecon = self.reconPhantom[-1]
        if self.experiment.OpticImage.phantom.shape != lastRecon.shape:
            lastRecon = lastRecon.T
        im2 = axs[2].imshow(lastRecon,
                            extent=(self.experiment.params.general['Xrange'][0]*1000, self.experiment.params.general['Xrange'][1]*1000,
                                    self.experiment.params.general['Zrange'][1]*1000, self.experiment.params.general['Zrange'][0]*1000),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[2].set_title(f"Last Reconstruction\nIter {self.numIterations * self.numSubsets}, MSE={np.mean((self.experiment.OpticImage.phantom - lastRecon) ** 2):.4f}")
        axs[2].set_xlabel("x (mm)", fontsize=12)
        axs[2].set_ylabel("z (mm)", fontsize=12)
        axs[2].tick_params(axis='both', which='major', labelsize=8)

        # Ajoute une colorbar horizontale centrée en dessous des trois plots
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Normalized Intensity', fontsize=12)
        cbar.ax.tick_params(labelsize=8)

        plt.subplots_adjust(wspace=0.3)

        if isSaving and self.saveDir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            savePath = os.path.join(self.saveDir, 'results')
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_comparison_MSE_BestANDLastRecon_{self.optimizer.name}_{date_str}.png')
            plt.savefig(SavingFolder, dpi=300, bbox_inches='tight')
            if show_logs:
                print(f"MSE plot saved to {SavingFolder}")

        plt.show()

    def show_theta_animation(self, vmin=None, vmax=None, total_duration_ms=3000, save_path=None, max_frames=1000, figSize=(4, 4), isPropMSE=True, show_logs=True):
        """
        Show theta iteration animation with speed proportional to MSE acceleration.
        In "propMSE" mode: slow down when MSE changes rapidly, speed up when MSE stagnates.

        Parameters:
            vmin, vmax: color limits (optional)
            total_duration_ms: total duration of the animation in milliseconds
            save_path: path to save animation (e.g., 'theta.gif')
            max_frames: maximum number of frames to include (default: 1000)
            isPropMSE: if True, use adaptive speed based on MSE (default: True)
        """
        import matplotlib as mpl
        mpl.rcParams['animation.embed_limit'] = 200

        if len(self.reconPhantom) == 0 or len(self.reconPhantom) < 2:
            raise ValueError("Not enough theta matrices available for animation.")

        if isPropMSE and (self.MSE is None or len(self.MSE) == 0):
            raise ValueError("MSE is empty or not calculated. Please calculate MSE first.")

        frames = np.array(self.reconPhantom)
        mse = np.array(self.MSE)

        # Sous-échantillonnage initial
        step = max(1, len(frames) // max_frames)
        frames_subset = frames[::step]
        indices_subset = self.indices[::step]
        mse_subset = mse[::step]

        if vmin is None:
            vmin = np.min(frames_subset)
        if vmax is None:
            vmax = np.max(frames_subset)

        fig, ax = plt.subplots(figsize=figSize, dpi=100)
        im = ax.imshow(
            frames_subset[0],
            extent=(
                self.experiment.params.general['Xrange'][0],
                self.experiment.params.general['Xrange'][1],
                self.experiment.params.general['Zrange'][1],
                self.experiment.params.general['Zrange'][0]
            ),
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            cmap='hot'
        )
        title = ax.set_title(f"Iteration {indices_subset[0]}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        plt.tight_layout()

        if isPropMSE:
            # Calcule la dérivée première (variation du MSE)
            mse_diff = np.gradient(mse_subset)
            # Calcule la dérivée seconde (accélération du MSE)
            mse_accel = np.gradient(mse_diff)
            # Normalise l'accélération entre 0 et 1 (en valeur absolue)
            mse_accel_normalized = np.abs(mse_accel)
            mse_accel_normalized /= (np.max(mse_accel_normalized) + 1e-10)

            # Prépare les frames pour le mode "propMSE"
            all_frames = []
            all_indices = []

            for i in range(len(frames_subset)):
                # Nombre de duplications inversement proportionnel à l'accélération (pour ralentir quand MSE change vite)
                # Plus l'accélération est élevée, plus on duplique (pour ralentir)
                num_duplicates = max(1, int(1 + 9 * mse_accel_normalized[i]))
                all_frames.extend([frames_subset[i]] * num_duplicates)
                all_indices.extend([indices_subset[i]] * num_duplicates)

            # Ajuste le nombre total de frames pour respecter la durée
            target_frames = int(total_duration_ms / 10)  # 10 ms par frame
            if len(all_frames) > target_frames:
                step_prop = len(all_frames) // target_frames
                all_frames = all_frames[::step_prop]
                all_indices = all_indices[::step_prop]

        else:  # Mode "linéaire"
            all_frames = frames_subset
            all_indices = indices_subset

        def update(frame_idx):
            im.set_array(all_frames[frame_idx])
            title.set_text(f"Iteration {all_indices[frame_idx]}")
            return [im, title]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(all_frames),
            interval=10,  # 10 ms par frame
            blit=False,
        )

        if save_path:
            if save_path.endswith(".gif"):
                ani.save(save_path, writer=animation.PillowWriter(fps=100))
            elif save_path.endswith(".mp4"):
                ani.save(save_path, writer="ffmpeg", fps=30)
            if show_logs:
                print(f"Animation saved to {save_path}")

        plt.close(fig)
        return HTML(ani.to_jshtml())

    def plot_SSIM(self, isSaving=True, log_scale_x=False, log_scale_y=False, figSize=(4,3), show_logs=True):
        if not self.SSIM:
            raise ValueError("SSIM is empty. Please calculate SSIM first.")

        best_idx = self.indices[np.argmax(self.SSIM)]
        if show_logs:
            print(f"Highest SSIM = {np.max(self.SSIM):.4f} at iteration {best_idx+1}")
        # Plot SSIM curve
        plt.figure(figsize=figSize)
        plt.plot(self.indices, self.SSIM, 'r-', label="SSIM curve")
        # Add blue dashed lines
        plt.axhline(np.max(self.SSIM), color='blue', linestyle='--', label=f"Max SSIM = {np.max(self.SSIM):.4f}")
        plt.axvline(best_idx, color='blue', linestyle='--', label=f"Iteration = {best_idx}")
        plt.xlabel("Iteration")
        plt.ylabel("SSIM")
        plt.title("SSIM vs. Iteration")
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        if isSaving and self.saveDir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            scale_str = ""
            if log_scale_x and log_scale_y:
                scale_str = "_loglog"
            elif log_scale_x:
                scale_str = "_logx"
            elif log_scale_y:
                scale_str = "_logy"
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_SSIM_plot_{self.optimizer.name}_{scale_str}{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"SSIM plot saved to {SavingFolder}")

        plt.show()

    def show_SSIM_bestRecon(self, isSaving=True, figSize=(15, 5), show_logs=True):
        
        if not self.SSIM:
            raise ValueError("SSIM is empty. Please calculate SSIM first.")

        best_idx = np.argmax(self.SSIM)
        best_recon = self.reconPhantom[best_idx]

        # ----------------- Plotting -----------------
        _, axs = plt.subplots(1, 3, figsize=figSize)  # 1 row, 3 columns

        # Left: Best reconstructed image (normalized)
        im0 = axs[0].imshow(best_recon, 
                            extent=(self.experiment.params.general['Xrange'][0], self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[0].set_title(f"Max SSIM Reconstruction\nIter {self.indices[best_idx]}, SSIM={np.min(self.MSE):.4f}")
        axs[0].set_xlabel("x (mm)")
        axs[0].set_ylabel("z (mm)")
        plt.colorbar(im0, ax=axs[0])

        # Middle: Ground truth (normalized)
        im1 = axs[1].imshow(self.experiment.OpticImage.laser.intensity, 
                            extent=(self.experiment.params.general['Xrange'][0], self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[1].set_title(r"Ground Truth ($\lambda$)")
        axs[1].set_xlabel("x (mm)")
        axs[1].set_ylabel("z (mm)")
        plt.colorbar(im1, ax=axs[1])

        # Right: Reconstruction at iter 350
        lastRecon = self.reconPhantom[-1] 
        im2 = axs[2].imshow(lastRecon,
                            extent=(self.experiment.params.general['Xrange'][0], self.experiment.params.general['Xrange'][1], self.experiment.params.general['Zrange'][1], self.experiment.params.general['Zrange'][0]),
                            cmap='hot', aspect='equal', vmin=0, vmax=1)
        axs[2].set_title(f"Last Reconstruction\nIter {self.numIterations * self.numSubsets}, SSIM={self.SSIM[-1]:.4f}")
        axs[2].set_xlabel("x (mm)")
        axs[2].set_ylabel("z (mm)")
        plt.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        if isSaving:
            now = datetime.now()    
            date_str = now.strftime("%Y_%d_%m_%y")
            SavingFolder = os.path.join(self.saveDir, f'{len(self.experiment.AcousticFields)}_SCANS_comparison_SSIM_BestANDLastRecon_{self.optimizer.name}_{date_str}.png')
            plt.savefig(SavingFolder, dpi=300)
            if show_logs:
                print(f"SSIM plot saved to {SavingFolder}")
        plt.show()

    def plot_CRC_vs_Noise(self, use_ROI=True, fin=None, min_distance=0.01, figSize = (4,3),
                     log_scale_x=False, log_scale_y=False, isSaving=False, show_logs=True):
        """
        Plot CRC vs Noise with min_distance always calculated in linear space.

        Args:
            min_distance: Minimum linear distance between points (always calculated in linear space)
            log_scale_x: Display X axis in logarithmic scale (but distance calculation remains linear)
            log_scale_y: Display Y axis in logarithmic scale
        """
        # Vérifications initiales
        if self.reconLaser is None or self.reconLaser == []:
            raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
        if isinstance(self.reconLaser, list) and len(self.reconLaser) == 1:
            raise ValueError("Reconstructed Image without tumor is a single frame. Run with isSavingEachIteration=True.")
        if self.reconPhantom is None or self.reconPhantom == []:
            raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
        if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
            raise ValueError("Reconstructed Image with tumor is a single frame. Run with isSavingEachIteration=True.")

        if fin is None:
            fin = len(self.reconPhantom) - 1
        iter_range = self.indices[:fin+1]

        if self.CRC is None:
            self.calculateCRC(use_ROI=use_ROI)

        # Calcul des valeurs de bruit
        noise_values = []
        for i in range(len(iter_range)):
            recon_without_tumor = self.reconLaser[i]
            noise = np.mean(np.abs(recon_without_tumor - self.experiment.OpticImage.laser.intensity))
            noise_values.append(max(noise, 1e-10))  # Évite les valeurs nulles

        # Sous-échantillonnage TOUJOURS basé sur la distance linéaire
        sampled_indices = [0]
        for i in range(1, len(noise_values)):
            last_noise = noise_values[sampled_indices[-1]]
            current_noise = noise_values[i]

            # Calcul de la distance EN LINÉAIRE (peu importe l'échelle d'affichage)
            distance = abs(current_noise - last_noise)

            if distance > min_distance:
                sampled_indices.append(i)

        sampled_indices.append(len(noise_values) - 1)

        # Vérification si l'avant-dernier point est trop proche du dernier
        if len(sampled_indices) > 1:
            last_idx = sampled_indices[-1]
            prev_idx = sampled_indices[-2]
            last_noise = noise_values[last_idx]
            prev_noise = noise_values[prev_idx]

            if abs(last_noise - prev_noise) <= min_distance:
                sampled_indices = sampled_indices[:-1]  # Supprime l'avant-dernier

        # Extraction des données
        sampled_noise = [noise_values[i] for i in sampled_indices]
        sampled_CRC = [self.CRC[i] for i in sampled_indices]
        sampled_iter = [iter_range[i] for i in sampled_indices]

        # Création de la figure
        plt.figure(figsize=figSize)
        plt.plot(sampled_noise, sampled_CRC, 'o-', color='blue', label=f'{self.optimizer.name}')

        # Positionnement des labels
        for x, y, it in zip(sampled_noise, sampled_CRC, sampled_iter):
            plt.text(x * 1.01, y * 1.01, str(it),
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # Configuration des axes
        plt.xlabel("Noise (mean absolute error)")
        plt.ylabel("CRC (Contrast Recovery Coefficient)")
        plt.title(f"CRC vs Noise (linear min_distance={min_distance})")

        # Application des échelles (uniquement pour l'affichage)
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')

        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        # Sauvegarde
        if isSaving:
            if self.saveDir is None:
                print("Warning: saveDir is None. Configure saving path to save the figure.")
            else:
                os.makedirs(self.saveDir, exist_ok=True)
                now = datetime.now()
                date_str = now.strftime("%Y_%m_%d_%H%M")
                scale_desc = "logX" if log_scale_x else "linX"
                scale_desc += "_logY" if log_scale_y else "_linY"
                filename = f"CRCvsNOISE_{len(self.experiment.AcousticFields)}scans_{self.optimizer.name}_{scale_desc}_{date_str}.png"
                save_path = os.path.join(self.saveDir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if show_logs:
                    print(f"Plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()

    def show_reconstruction_progress(self, start=0, fin=None, save_path=None, with_tumor=True, show_logs=True):
        """
        Show the reconstruction progress for either with or without tumor.
        If isPropMSE is True, the frame selection is adapted to MSE changes.
        Otherwise, indices are evenly spaced between start and fin.

        Parameters:
            start: int, starting iteration index
            fin: int, ending iteration index (inclusive)
            duration: int, duration of the animation in milliseconds
            save_path: str, path to save the figure (optional)
            with_tumor: bool, if True, show reconstruction with tumor; else without (default: True)
            isPropMSE: bool, if True, use adaptive speed based on MSE (default: True)
        """
        import matplotlib as mpl
        mpl.rcParams['animation.embed_limit'] = 200

        if fin is None:
            fin = len(self.reconPhantom) - 1 if with_tumor else len(self.reconLaser) - 1

        # Check data availability
        if with_tumor:
            if self.reconPhantom is None or self.reconPhantom == []:
                raise ValueError("Reconstructed phantom is empty. Run reconstruction first.")
            if isinstance(self.reconPhantom, list) and len(self.reconPhantom) == 1:
                raise ValueError("Reconstructed Image with tumor is a single frame. Run reconstruction with isSavingEachIteration=True.")
            recon_list = self.reconPhantom
            ground_truth = self.experiment.OpticImage.phantom
            title_suffix = "with_tumor"
        else:
            if self.reconLaser is None or self.reconLaser == []:
                raise ValueError("Reconstructed laser is empty. Run reconstruction first.")
            if isinstance(self.reconLaser, list) and len(self.reconLaser) == 1:
                raise ValueError("Reconstructed Image without tumor is a single frame. Run reconstruction with isSavingEachIteration=True.")
            recon_list = self.reconLaser
            ground_truth = self.experiment.OpticImage.laser.intensity
            title_suffix = "without_tumor"

        # Collect data for all iterations
        recon_list_data = []
        diff_abs_list = []
        mse_list = []
        noise_list = []

        for i in range(start, fin + 1):
            recon = recon_list[i]
            diff_abs = np.abs(recon - ground_truth)
            mse = np.mean((ground_truth.flatten() - recon.flatten())**2)
            noise = np.mean(np.abs(recon - ground_truth))

            recon_list_data.append(recon)
            diff_abs_list.append(diff_abs)
            mse_list.append(mse)
            noise_list.append(noise)

        # Calculate global min/max for difference images
        global_min_diff = np.min([d.min() for d in diff_abs_list[1:]])
        global_max_diff = np.max([d.max() for d in diff_abs_list[1:]])

        # Evenly spaced indices
        num_frames = min(5, fin - start + 1)
        all_indices = np.linspace(start, fin, num_frames, dtype=int).tolist()

        # Plot
        nrows = min(5, len(all_indices))
        ncols = 3  # Recon, |Recon - GT|, Ground Truth
        vmin, vmax = 0, 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))

        for i, iter_idx in enumerate(all_indices[:nrows]):
            idx_in_list = iter_idx - start  # Index in the collected data lists
            recon = recon_list_data[idx_in_list]
            diff_abs = diff_abs_list[idx_in_list]
            mse_val = mse_list[idx_in_list]
            noise = noise_list[idx_in_list]

            im0 = axs[i, 0].imshow(recon, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
            axs[i, 0].set_title(f"Reconstruction\nIter {self.indices[iter_idx]}, MSE={mse_val:.2e}", fontsize=10)
            axs[i, 0].axis('off')
            plt.colorbar(im0, ax=axs[i, 0])

            im1 = axs[i, 1].imshow(diff_abs, cmap='viridis',
                                vmin=global_min_diff,
                                vmax=global_max_diff,
                                aspect='equal')
            axs[i, 1].set_title(f"|Recon - Ground Truth|\nNoise={noise:.2e}", fontsize=10)
            axs[i, 1].axis('off')
            plt.colorbar(im1, ax=axs[i, 1])

            im2 = axs[i, 2].imshow(ground_truth, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
            axs[i, 2].set_title(r"Ground Truth", fontsize=10)
            axs[i, 2].axis('off')
            plt.colorbar(im2, ax=axs[i, 2])

        plt.tight_layout()

        if save_path:
            # Add suffix to filename based on with_tumor parameter
            if '.' in save_path:
                name, ext = save_path.rsplit('.', 1)
                save_path = f"{name}_{title_suffix}.{ext}"
            else:
                save_path = f"{save_path}_{title_suffix}"
            plt.savefig(save_path, dpi=300)
            if show_logs:
                print(f"Figure saved to: {save_path}")

        plt.show()

    def checkExistingFile(self, date=None, withTumor=True):
        """
        Check if the reconstruction file already exists, based on current instance parameters.

        Args:
            date (str, optional): Date string in format "ddmm". If None, uses current date.
            withTumor (bool): If True, checks reconPhantom.npy; otherwise, checks reconLaser.npy.
            overwrite (bool): If True, ignores existing files and returns True for saving.

        Returns:
            tuple: (bool: whether to save, str: the filepath)
        """
        if self.saveDir is None:
            raise ValueError("Save directory is not specified.")
        if date is None:
            date = datetime.now().strftime("%d%m")
        results_dir = os.path.join(self.saveDir, f'results_{date}_{self.optimizer.value}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Détermine le nom du fichier en fonction de withTumor
        indices_file = os.path.join(results_dir, f"indices_{'withTumor' if withTumor else 'withoutTumor'}.npy")

        # Si le fichier existe retourne True
        if os.path.exists(indices_file):
            return (True, results_dir)

        # Sinon, retourne False, indiquant qu'on peut sauvegarder
        return (False, results_dir)

    def load(self, withTumor=True, results_date=None, optimizer=None, filePath=None, show_logs=True):
        """
        Load the reconstruction results (reconPhantom or reconLaser) and indices as lists of 2D np arrays into self.
        If the loaded file is a 3D array, it is split into a list of 2D arrays.
        Args:
            withTumor: If True, loads reconPhantom (with tumor), else reconLaser (without tumor).
            results_date: Date string (format "ddmm") to specify which results to load. If None, uses the most recent date in saveDir.
            optimizer: Optimizer name (as string or enum) to filter results. If None, uses the current optimizer of the instance.
            filePath: Optional. If provided, loads directly from this path (overrides saveDir and results_date).
        """
        if filePath is not None:
            # Mode chargement direct depuis un fichier
            recon_key = 'reconPhantom' if withTumor else 'reconLaser'
            recon_path = filePath
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"No reconstruction file found at {recon_path}.")
            # Charge le fichier (3D ou liste de 2D)
            data = np.load(recon_path, allow_pickle=True)
            # Découpe en liste de 2D si c'est un tableau 3D
            if isinstance(data, np.ndarray) and data.ndim == 3:
                if withTumor:
                    self.reconPhantom = [data[i, :, :] for i in range(data.shape[0])]
                else:
                    self.reconLaser = [data[i, :, :] for i in range(data.shape[0])]
            else:
                # Sinon, suppose que c'est déjà une liste de 2D
                if withTumor:
                    self.reconPhantom = data
                else:
                    self.reconLaser = data
            # Essayer de charger les indices
            base_dir, _ = os.path.split(recon_path)
            indices_path = os.path.join(base_dir, 'indices.npy')
            if os.path.exists(indices_path):
                indices_data = np.load(indices_path, allow_pickle=True)
                if isinstance(indices_data, np.ndarray) and indices_data.ndim == 3:
                    self.indices = [indices_data[i, :, :] for i in range(indices_data.shape[0])]
                else:
                    self.indices = indices_data
            else:
                self.indices = None
                
            if show_logs:
                print(f"Loaded reconstruction results and indices from {recon_path}")
        else:
            # Mode chargement depuis le répertoire de résultats
            if self.saveDir is None:
                raise ValueError("Save directory is not specified. Please set saveDir before loading.")
            # Use current optimizer and potential function if not provided
            opt_name = optimizer.value if optimizer is not None else self.optimizer.value
            # Build the base directory pattern
            dir_pattern = f'results_*_{opt_name}'
            # Add parameters to the pattern based on the optimizer
            if optimizer is None:
                optimizer = self.optimizer
            if optimizer == OptimizerType.PPGMLEM:
                beta_str = f'_Beta_{self.beta}'
                delta_str = f'_Delta_{self.delta}'
                gamma_str = f'_Gamma_{self.gamma}'
                sigma_str = f'_Sigma_{self.sigma}'
                dir_pattern += f'{beta_str}{delta_str}{gamma_str}{sigma_str}'
            elif optimizer in (OptimizerType.PGC, OptimizerType.DEPIERRO95):
                beta_str = f'_Beta_{self.beta}'
                sigma_str = f'_Sigma_{self.sigma}'
                dir_pattern += f'{beta_str}{sigma_str}'
            # Find the most recent results directory if no date is specified
            if results_date is None:
                dirs = [d for d in os.listdir(self.saveDir) if os.path.isdir(os.path.join(self.saveDir, d)) and dir_pattern in d]
                if not dirs:
                    raise FileNotFoundError(f"No matching results directory found for pattern '{dir_pattern}' in {self.saveDir}.")
                dirs.sort(reverse=True)  # Most recent first
                results_dir = os.path.join(self.saveDir, dirs[0])
            else:
                results_dir = os.path.join(self.saveDir, f'results_{results_date}_{opt_name}')
                if optimizer == OptimizerType.MLEM:
                    pass
                elif optimizer == OptimizerType.LS:
                    results_dir += f'_Alpha_{self.alpha}'
                if not os.path.exists(results_dir):
                    raise FileNotFoundError(f"Directory {results_dir} does not exist.")
            # Load reconstruction results
            recon_key = 'reconPhantom' if withTumor else 'reconLaser'
            recon_path = os.path.join(results_dir, f'{recon_key}.npy')
            if not os.path.exists(recon_path):
                raise FileNotFoundError(f"No reconstruction file found at {recon_path}.")
            data = np.load(recon_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.ndim == 3:
                if withTumor:
                    self.reconPhantom = [data[i, :, :] for i in range(data.shape[0])]
                else:
                    self.reconLaser = [data[i, :, :] for i in range(data.shape[0])]
            else:
                if withTumor:
                    self.reconPhantom = data
                else:
                    self.reconLaser = data
            # Load saved indices as list of 2D arrays
            indices_path = os.path.join(results_dir, 'indices.npy')
            if not os.path.exists(indices_path):
                raise FileNotFoundError(f"No indices file found at {indices_path}.")
            indices_data = np.load(indices_path, allow_pickle=True)
            if isinstance(indices_data, np.ndarray) and indices_data.ndim == 3:
                self.indices = [indices_data[i, :, :] for i in range(indices_data.shape[0])]
            else:
                self.indices = indices_data
            if show_logs:
                print(f"Loaded reconstruction results and indices from {results_dir}")
        
    def normalizeSMatrix(self):
        self.SMatrix = self.SMatrix / (float(self.experiment.params.acoustic['emission']['voltage'])*float(self.experiment.params.acoustic['emission']['sensitivity']))  

    # PRIVATE METHODS

    def _fillDenseSMatrix(self):
        """
        Construit une matrice dense en mémoire.
        """
        T, Z, X = self.experiment.AcousticFields[0].field.shape
        N = len(self.experiment.AcousticFields)
        S = np.empty((T, Z, X, N), dtype=np.float32)
        def copy_block(i):
            np.copyto(S[..., i], self.experiment.AcousticFields[i].field)
        with concurrent.futures.ThreadPoolExecutor() as ex:
            ex.map(copy_block, range(N))
        return S
           
    def _fillSparseSMatrix(self, isShowLogs=True):
        if self.smatrixType == SMatrixType.CSR:
            return self._fillSparseSMatrix_CSR(isShowLogs=isShowLogs)
        if self.smatrixType == SMatrixType.COO:
            raise NotImplementedError("COO sparse matrix not implemented yet.")
        if self.smatrixType == SMatrixType.SELL:
            return self._fillSparseSMatrix_SELL(isShowLogs=isShowLogs)
    
    def _fillSparseSMatrix_CSR(self, isShowLogs=True):
        """
        Construit une matrice sparse CSR par morceaux sans concaténation intermédiaire.
        Libère toute la mémoire temporaire à chaque étape.
        """
        sparse_matrix = SparseSMatrix_CSR(self.experiment,relative_threshold=self.sparseThreshold,device=self.device)
        sparse_matrix.allocate()
        if isShowLogs:
            print(f" Sparse matrix size: {sparse_matrix.getMatrixSize()} GB")
            print(f"Sparse matrix density: {sparse_matrix.compute_density()}")
        return sparse_matrix
    
    def _fillSparseSMatrix_SELL(self, isShowLogs=True):
        """
        Construit une matrice sparse SELL par morceaux sans concaténation intermédiaire.
        Libère toute la mémoire temporaire à chaque étape.
        """
        sparse_matrix = SparseSMatrix_SELL(self.experiment,relative_threshold=self.sparseThreshold,device=self.device)
        sparse_matrix.allocate()
        # fenetre_gpu = get_apodization_vector_gpu(sparse_matrix)
        # sparse_matrix.apply_apodization_gpu(fenetre_gpu)
        if isShowLogs:
            print(f" Sparse matrix size: {sparse_matrix.getMatrixSize()} GB")
            print(f"Sparse matrix density: {sparse_matrix.compute_density()}")
        return sparse_matrix
        
    def _AlgebraicReconPython(self,withTumor, show_logs):
    
        if withTumor:
            if self.experiment.AOsignal_withTumor is None:
                raise ValueError("AO signal with tumor is not available. Please generate AO signal with tumor the experiment first in the experiment object.")
        else:
            if self.experiment.AOsignal_withoutTumor is None:
                raise ValueError("AO signal without tumor is not available. Please generate AO signal without tumor the experiment first in the experiment object.")

        if self.optimizer.value == OptimizerType.MLEM.value:
            if withTumor:
                self.reconPhantom, self.indices = MLEM(SMatrix=self.SMatrix, 
                                                        y=self.experiment.AOsignal_withTumor,
                                                        numIterations=self.numIterations,
                                                        isSavingEachIteration=self.isSavingEachIteration,
                                                        withTumor=withTumor,
                                                        device=self.device,
                                                        use_numba=self.isMultiCPU,
                                                        denominator_threshold=self.denominatorThreshold,
                                                        max_saves=self.maxSaves,
                                                        show_logs=show_logs,
                                                        smatrixType=self.smatrixType,
                                                        )
            else:
                self.reconLaser, self.indices = MLEM(SMatrix=self.SMatrix, 
                                                        y=self.experiment.AOsignal_withoutTumor,
                                                        numIterations=self.numIterations,
                                                        isSavingEachIteration=self.isSavingEachIteration,
                                                        withTumor=withTumor,
                                                        device=self.device,
                                                        use_numba=self.isMultiCPU,
                                                        denominator_threshold=self.denominatorThreshold,
                                                        max_saves=self.maxSaves,
                                                        show_logs=show_logs,
                                                        smatrixType=self.smatrixType,
                                                        )
        elif self.optimizer.value == OptimizerType.LS.value:
            if withTumor:
                self.reconPhantom, self.indices = LS(SMatrix=self.SMatrix, 
                                                        y=self.experiment.AOsignal_withTumor,
                                                        numIterations=self.numIterations,
                                                        isSavingEachIteration=self.isSavingEachIteration,
                                                        withTumor=withTumor,
                                                        device=self.device,
                                                        max_saves=self.maxSaves,
                                                        show_logs=show_logs,
                                                        smatrixType=self.smatrixType
                                                )
            else:
                self.reconLaser, self.indices = LS(SMatrix=self.SMatrix, 
                                                        y=self.experiment.AOsignal_withoutTumor,
                                                        numIterations=self.numIterations,
                                                        isSavingEachIteration=self.isSavingEachIteration,
                                                        withTumor=withTumor,
                                                        device=self.device,
                                                        max_saves=self.maxSaves,
                                                        show_logs=show_logs,
                                                        smatrixType=self.smatrixType
                                                )
        else:
            raise ValueError(f"Only MLEM and LS are supported for simple algebraic reconstruction. {self.optimizer.value} need Bayesian reconstruction")

    def _AlgebraicReconCASToR(self,withTumor, show_logs):
        # Définir les chemins
        smatrix = os.path.join(self.saveDir, "system_matrix")
        if withTumor:
            fileName = 'AOSignals_withTumor.cdh'
        else:
            fileName = 'AOSignals_withoutTumor.cdh'

        # Vérifier et générer les fichiers d'entrée si nécessaire
        if not os.path.isfile(os.path.join(self.saveDir, fileName)):
            if show_logs:
                print(f"Fichier .cdh manquant. Génération de {fileName}...")
            self.experiment.saveAOsignals_Castor(self.saveDir)

        # Vérifier/générer la matrice système
        if not os.path.isdir(smatrix):
            os.makedirs(smatrix, exist_ok=True)
        if not os.listdir(smatrix):
            if show_logs:
                print("Matrice système manquante. Génération...")
            self.experiment.saveAcousticFields(self.saveDir)

        # Vérifier que le fichier .cdh existe (redondant mais sûr)
        if not os.path.isfile(os.path.join(self.saveDir, fileName)):
            raise FileNotFoundError(f"Le fichier .cdh n'existe toujours pas : {fileName}")

        # Créer le dossier de sortie
        os.makedirs(os.path.join(self.saveDir, 'results', 'recon'), exist_ok=True)

        # Configuration de l'environnement pour CASToR
        env = os.environ.copy()
        env.update({
            "CASTOR_DIR": self.experiment.params.reconstruction['castor_executable'],
            "CASTOR_CONFIG": os.path.join(self.experiment.params.reconstruction['castor_executable'], "config"),
            "CASTOR_64bits": "1",
            "CASTOR_OMP": "1",
            "CASTOR_SIMD": "1",
            "CASTOR_ROOT": "1",
        })

        # Construire la commande
        cmd = [
            os.path.join(self.experiment.params.reconstruction['castor_executable'], "bin", "castor-recon"),
            "-df", os.path.join(self.saveDir, fileName),
            "-opti", self.optimizer.value,
            "-it", f"{self.numIterations}:{self.numSubsets}",
            "-proj", "matrix",
            "-dout", os.path.join(self.saveDir, 'results', 'recon'),
            "-th", str(os.cpu_count()),
            "-vb", "5",
            "-proj-comp", "1",
            "-ignore-scanner",
            "-data-type", "AOT",
            "-ignore-corr", "cali,fdur",
            "-system-matrix", smatrix,
        ]

        # Afficher la commande (pour débogage)
        if show_logs:
            print("Commande CASToR :")
            print(" ".join(cmd))

        # Chemin du script temporaire
        recon_script_path = os.path.join(gettempdir(), 'recon.sh')

        # Écrire le script bash
        with open(recon_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"export PATH={env['CASTOR_DIR']}/bin:$PATH\n")  # Ajoute le chemin de CASToR au PATH
            f.write(f"export LD_LIBRARY_PATH={env['CASTOR_DIR']}/lib:$LD_LIBRARY_PATH\n")  # Ajoute les bibliothèques si nécessaire
            f.write(" ".join(cmd) + "\n")

        # Rendre le script exécutable et l'exécuter
        subprocess.run(["chmod", "+x", recon_script_path], check=True)
        if show_logs:
            print(f"Exécution de la reconstruction avec CASToR...")
        result = subprocess.run(recon_script_path, env=env, check=True, capture_output=True, text=True)

        # Afficher la sortie de CASToR (pour débogage)
        if show_logs:
            print("Sortie CASToR :")
            print(result.stdout)
            if result.stderr:
                print("Erreurs :")
                print(result.stderr)

        if show_logs:
            print("Reconstruction terminée avec succès.")
        self.load_reconCASToR(withTumor=withTumor)
    
    def flipAngle(self):
        if self.smatrixType == SMatrixType.CSR:
            self.SMatrix.flip_angle()
    # STATIC METHODS
    @staticmethod
    def plot_mse_comparison(recon_list, figSize=(4.5, 3.5), labels=None):
        """
        Affiche les courbes de MSE pour chaque reconstruction dans recon_list.

        Args:
            recon_list (list): Liste d'objets recon (doivent avoir les attributs 'indices' et 'MSE').
            figSize (tuple, optional): Taille de la figure.
            labels (list, optional): Liste des labels pour chaque courbe. Si None, utilise "Recon i".
        """
        if labels is None:
            labels = [f"Recon {i+1}" for i in range(len(recon_list))]

        plt.figure(figsize=figSize)
        colors = ['red', 'green', 'blue', 'orange', 'purple']  # Ajoute d'autres couleurs si nécessaire

        for i, recon in enumerate(recon_list):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f"Recon {i+1}"

            # Trouve l'index et la valeur minimale du MSE
            best_idx = recon.indices[np.argmin(recon.MSE)]
            min_mse = np.min(recon.MSE)

            # Trace la courbe de MSE
            plt.plot(recon.indices, recon.MSE, f'{color}-', label=label)
            # Ligne horizontale pour le min MSE
            plt.axhline(min_mse, color=color, linestyle='--', alpha=0.5)
            # Ligne verticale pour l'itération du min MSE
            plt.axvline(best_idx, color=color, linestyle='--', alpha=0.5)

        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("MSE vs. Iteration (Comparison)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")

        # Légende personnalisée
        handles = []
        for i, recon in enumerate(recon_list):
            color = colors[i % len(colors)]
            best_idx = recon.indices[np.argmin(recon.MSE)]
            min_mse = np.min(recon.MSE)
            handles.append(
                plt.Line2D([0], [0], color=color,
                        label=f"{labels[i] if labels and i < len(labels) else f'Recon {i+1}'} (min={min_mse:.4f} @ it.{best_idx+1})")
            )

        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        plt.show()

