from ._mainRecon import Recon
from .ReconEnums import ReconType, AnalyticType, ProcessType
from AOT_biomaps.AOT_Experiment.ExperimentTools import add_sincos_cpu
from .ReconTools import fourierz_gpu, EvalDelayLawOS_center, ifourierx_gpu, rotate_theta_gpu, filter_radon_gpu, ifourierz_gpu  

import numpy as np
from tqdm import trange
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

class AnalyticRecon(Recon):
    def __init__(self, analyticType, Lc = None,**kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.Analytic
        self.analyticType = analyticType
        if self.analyticType == AnalyticType.iRADON and Lc is None:
            raise ValueError("[AOT-biomaps] Lc parameter must be provided for iRADON analytic reconstruction.")
        self.Lc = Lc # in meters
        self.AOsignal_demoldulated = None

    def run(self, processType = ProcessType.PYTHON, withTumor= True):
        """
        This method is a placeholder for the analytic reconstruction process.
        It currently does not perform any operations but serves as a template for future implementations.
        """
        if(processType == ProcessType.CASToR):
            raise NotImplementedError("[AOT-biomaps] CASToR analytic reconstruction is not implemented yet.")
        elif(processType == ProcessType.PYTHON):
            self._analyticReconPython(withTumor)
        else:
            raise ValueError(f"[AOT-biomaps] Unknown analytic reconstruction type: {processType}")
        
    def check_existing_file(self, date=None, withTumor=True):
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
            raise ValueError("[AOT-biomaps] Save directory is not specified.")
        if date is None:
            date = datetime.now().strftime("%d%m")
        results_dir = os.path.join(self.saveDir, f'results_{date}_{self.analyticType.name}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Détermine le nom du fichier en fonction de withTumor
        indices_file = os.path.join(results_dir, f"indices_{'withTumor' if withTumor else 'withoutTumor'}.npy")
        # Si le fichier existe retourne True
        if os.path.exists(indices_file):
            return (True, results_dir)

        # Sinon, retourne False
        return (False, results_dir)

    def _analyticReconPython(self,withTumor):
        """
        This method is a placeholder for the analytic reconstruction process in Python.
        It currently does not perform any operations but serves as a template for future implementations.
        
        Parameters:
            analyticType: The type of analytic reconstruction to perform (default is iFOURIER).
        """

        if withTumor:
            AOsignal = self.experiment.AOsignal_withTumor
        else:
            AOsignal = self.experiment.AOsignal_withoutTumor

        SampleRate = self.experiment.expParams['SampleRate'] if self.experiment.expParams['SampleRate'] is not None else self.experiment.params.acoustic['f_saving']
        d_t = 1 / float(SampleRate)
        t_array = np.arange(0, AOsignal.shape[0])*d_t
        Z = t_array * self.experiment.params.acoustic['medium']['c0']
        X_m = np.arange(0, self.experiment.params.acoustic['probe']['num_elements'])* self.experiment.params.acoustic['probe']['element_width']
        dfX = 1 / (X_m[1] - X_m[0]) / len(X_m)
        self.experiment.expParams['Xrange'] = [X_m[0], X_m[-1]]
        self.experiment.expParams['Zrange'] = [Z[0], Z[-1]]
        if withTumor:
            # self.AOsignal_demoldulated = self.experiment.demodulate_AOsignal(withTumor=True)
            if self.analyticType == AnalyticType.iFOURIER:
                self.reconPhantom = self._iFourierRecon(
                    R = AOsignal,
                    z = Z,    
                    X_m=X_m,
                    theta=self.experiment.theta,
                    decimation=self.experiment.decimations,
                    c=self.experiment.params.acoustic['medium']['c0'],
                    DelayLAWS=self.experiment.DelayLaw,
                    ActiveLIST=self.experiment.ActiveList,
                    withTumor=True,
                )
                    
            elif self.analyticType == AnalyticType.iRADON:
                self.reconPhantom = self._iRadonRecon(
                    R=AOsignal,
                    z=Z,
                    X_m=X_m,
                    theta=self.experiment.theta,
                    decimation=self.experiment.decimations,
                    df0x=dfX,
                    Lc =self.Lc,
                    c=self.experiment.params.acoustic['medium']['c0'],
                    DelayLAWS=self.experiment.DelayLaw,
                    ActiveLIST=self.experiment.ActiveList,
                    withTumor=True)
            else:            
                raise ValueError(f"[AOT-biomaps] Unknown analytic type: {self.analyticType}")
        else:
            # self.AOsignal_demoldulated = self.experiment.demodulate_AOsignal(withTumor=False)
            if self.analyticType == AnalyticType.iFOURIER:
                self.reconLaser = self._iFourierRecon(
                    R = AOsignal    ,
                    z = Z,    
                    X_m=X_m,
                    theta=self.experiment.theta,
                    decimation=self.experiment.decimations,
                    c=self.experiment.params.acoustic['medium']['c0'],
                    DelayLAWS=self.experiment.DelayLaw,
                    ActiveLIST=self.experiment.ActiveList,
                    withTumor=False,
                )
            elif self.analyticType == AnalyticType.iRADON:
                self.reconLaser = self._iRadonRecon(
                    R=AOsignal  ,
                    z=Z,
                    X_m=X_m,
                    theta=self.experiment.theta,
                    decimation=self.experiment.decimations,
                    df0x=dfX,
                    Lc = self.Lc,
                    c=self.experiment.params.acoustic['medium']['c0'],
                    DelayLAWS=self.experiment.DelayLaw,
                    ActiveLIST=self.experiment.ActiveList,
                    withTumor=False)
            else:            
                raise ValueError(f"[AOT-biomaps] Unknown analytic type: {self.analyticType}")
    
    def _iFourierRecon(
        self,
        R,
        z, 
        X_m, 
        theta, 
        decimation,  
        c, 
        DelayLAWS, 
        ActiveLIST,
        withTumor,
    ):
        """
        Image reconstruction using the iFourier method (GPU).
        Physical normalization included.
        """
        R = cp.asarray(R)
        z = cp.asarray(z)
        X_m = cp.asarray(X_m)
        theta = cp.asarray(theta)
        decimation = cp.asarray(decimation)
        DelayLAWS = cp.asarray(DelayLAWS)
        ActiveLIST = cp.asarray(ActiveLIST)

        DelayLAWS_s = cp.where(cp.max(DelayLAWS) > 1e-3, DelayLAWS / 1000.0, DelayLAWS)
        ScanParam_cpu = cp.asnumpy(cp.stack([decimation, cp.round(theta, 4)], axis=1))
        _, ia_cpu, ib_cpu = np.unique(ScanParam_cpu, axis=0, return_index=True, return_inverse=True)
        ia = cp.asarray(ia_cpu)
        ib = cp.asarray(ib_cpu)

        F_complex_cpu, theta_u_cpu, decim_u_cpu = add_sincos_cpu(
            cp.asnumpy(R),
            cp.asnumpy(decimation),
            np.radians(cp.asnumpy(theta))
        )

        # Rotation center M0 calculation (CPU)
        M0 = EvalDelayLawOS_center(
            X_m,
            theta_u_cpu,
            DelayLAWS_s.T[:, ia],
            ActiveLIST.T[:, ia],
            c
        )

        F_complex = cp.asarray(F_complex_cpu)
        theta_u = cp.asarray(theta_u_cpu)
        decim_u = cp.asarray(decim_u_cpu)
        M0_gpu = cp.asarray(M0)

        Nz = z.size
        Nx = X_m.size
        dx = X_m[1] - X_m[0]
        X_grid, Z_grid = cp.meshgrid(X_m, z)
        idx0_x = Nx // 2

        # Angles uniques
        angles_group, ia_u, ib_u = cp.unique(theta_u, return_index=True, return_inverse=True)
        Ntheta = angles_group.size

        # Initialisation reconstruction
        I_final = cp.zeros((Nz, Nx), dtype=cp.complex64)

        # Inverse Fourier X for each unique angle (GPU)
        for i_ang in trange(
            Ntheta,
            desc=f"[AOT-biomaps] iFourier ({'with tumor' if withTumor else 'without tumor'}) -- GPU",
            unit="angle"
        ):
            F_fx_z = cp.zeros((Nz, Nx), dtype=cp.complex64)
            indices = cp.where(ib_u == i_ang)[0]

            for idx in indices:
                n = int(decim_u[idx])
                trace_z = F_complex[:, idx]
                ip = idx0_x + n
                if 0 <= ip < Nx:
                    F_fx_z[:, ip] = trace_z
                if n != 0:
                    im = idx0_x - n
                    if 0 <= im < Nx:
                        col_conj = cp.zeros(Nz, dtype=cp.complex64)
                        col_conj[1:] = cp.conj(trace_z[:-1])
                        F_fx_z[:, im] = col_conj

            # DC correction
            F_fx_z[:, idx0_x] *= 0.5

            I_spatial = ifourierx_gpu(F_fx_z, dx) * Nx
            I_rot = rotate_theta_gpu(
                X_grid,
                Z_grid,
                I_spatial,
                -angles_group[i_ang],
                M0_gpu[i_ang, :]
            )

            I_final += I_rot

        Ntheta_total = len(theta_u)
        Ntirs_complex = (R.shape[1] - Ntheta_total) / 4.0  # 4 phases for each unique decimation and angle

        I_final /= (Ntheta_total * Ntirs_complex)
        I_final *= dx 

        return cp.real(I_final).get()

    def _iRadonRecon(
        self,
        R, 
        z, 
        X_m,
        theta, 
        decimation,
        df0x,
        Lc,
        c,
        DelayLAWS,
        ActiveLIST,
        withTumor,
    ):
        """
        Image reconstruction using the iRadon method (GPU).
        Physical normalization included (phases, angles, dz).
        """

        theta = np.radians(theta)
        F_ct_kx, theta_u, decim_u = add_sincos_cpu(R, decimation, theta)

        ScanParam = np.stack([decimation, theta], axis=1)
        _, ia, _ = np.unique(ScanParam, axis=0, return_index=True, return_inverse=True)

        ActiveLIST = np.asarray(ActiveLIST).T
        DelayLAWS = np.asarray(DelayLAWS).T
        ActiveLIST_unique = ActiveLIST[:, ia]

        z_gpu = cp.asarray(z)
        Fin = fourierz_gpu(z, F_ct_kx)

        dz = float(z[1] - z[0]) 
        fz = cp.fft.fftshift(cp.fft.fftfreq(len(z), d=dz))

        Nz, Nk = Fin.shape

        decim_gpu = cp.asarray(decim_u)
        I0 = decim_gpu == 0
        F0 = Fin * I0[None, :]

        DEC, FZ = cp.meshgrid(decim_gpu, fz)

        Hinf = cp.abs(FZ) < cp.abs(DEC) * df0x
        Hsup = FZ >= 0

        Fc = 1 / Lc
        FILTER = filter_radon_gpu(fz, Fc)[:, None]

        Finf = F0 * FILTER[:, :F0.shape[1]] * Hinf[:, :F0.shape[1]]
        Fsup = Fin * FILTER * Hsup

        Finf = ifourierz_gpu(z, Finf)
        Fsup = ifourierz_gpu(z, Fsup)

        X_gpu = cp.asarray(X_m)
        X, Z = cp.meshgrid(X_gpu, z_gpu)
        Xc = float(np.mean(X_m))

        # Compute rotation center M0 for each unique angle and decimation (CPU)
        M0 = EvalDelayLawOS_center(X_m, theta, DelayLAWS[:, ia], ActiveLIST_unique, c)
        M0_gpu = cp.asarray(M0)

        # backprojection
        Irec = cp.zeros_like(X, dtype=cp.complex64)

        for i in trange(
            len(theta_u),
            desc=f"[AOT-biomaps] iRadon ({'with tumor' if withTumor else 'without tumor'}) -- GPU",
            unit="angle"
        ):
            th = float(theta_u[i])

            T = (X - M0_gpu[i, 0]) * cp.sin(th) + (Z - M0_gpu[i, 1]) * cp.cos(th) + M0_gpu[i, 1]
            S = (X - Xc) * cp.cos(th) - (Z - M0_gpu[i, 1]) * cp.sin(th)
            h0 = cp.exp(1j * 2 * cp.pi * decim_u[i] * df0x * S)

            Tind = (T - z_gpu[0]) / dz
            i0 = cp.floor(Tind).astype(cp.int32)
            i1 = i0 + 1
            i0 = cp.clip(i0, 0, Nz - 1)
            i1 = cp.clip(i1, 0, Nz - 1)
            w = Tind - i0

            proj_sup = (1 - w) * Fsup[i0, i] + w * Fsup[i1, i]
            proj_inf = (1 - w) * Finf[i0, i] + w * Finf[i1, i]

            Irec += 2 * h0 * proj_sup + proj_inf


        Ntheta = len(theta_u)
        Ntirs_complex = (R.shape[1] - Ntheta) / 4.0 # 4 phases for each unique decimation and angle

        Irec /= (Ntheta * Ntirs_complex)
        print(f"[AOT-biomaps] dz normalization: {dz}")
        Irec *= dz

        return cp.real(Irec).get()

    def show(self, withTumor=True, savePath=None, scale='same', figsize=(8, 4)):
        """
        Display the reconstructed images with a properly positioned colorbar.
        Args:
            withTumor (bool): If True, displays reconPhantom. If False, displays reconLaser. Default is True.
            savePath (str): Path to save the figure. If None, the figure is not saved. Default is None.
            scale (str): Scale for the aspect ratio of the plots. Default is 'same'. Options are 'same' or 'auto'.
            figsize (tuple): Figure size (width, height). Default is (8, 4).

        Note:
            Requires matplotlib to be installed. If matplotlib is not available, this method will raise an ImportError.
        """
        extent = [self.experiment.params.general['Xrange'][0] * 1e3, self.experiment.params.general['Xrange'][1] * 1e3, self.experiment.params.general['Zrange'][1] * 1e3, self.experiment.params.general['Zrange'][0] * 1e3] if self.experiment.expParams['Xrange'] is None and self.experiment.expParams['Zrange'] is None else [self.experiment.expParams['Xrange'][0] * 1e3, self.experiment.expParams['Xrange'][1] * 1e3, self.experiment.expParams['Zrange'][1] * 1e3, self.experiment.expParams['Zrange'][0] * 1e3]

        # Determine the image to display
        if withTumor:
            if self.reconPhantom is None:
                raise ValueError("[AOT-biomaps] Reconstructed phantom with tumor is empty. Run reconstruction first.")
            image = self.reconPhantom
            ground_truth = self.experiment.OpticImage.phantom if self.experiment.OpticImage else None
            title_recon = "Reconstructed phantom with tumor"
            title_gt = "Phantom with tumor"
        else:
            if self.reconLaser is None:
                raise ValueError("[AOT-biomaps] Reconstructed laser without tumor is empty. Run reconstruction first.")
            image = self.reconLaser
            ground_truth = self.experiment.OpticImage.laser.intensity if self.experiment.OpticImage else None
            title_recon = "Reconstructed laser without tumor"
            title_gt = "Laser without tumor"

        # Gestion propre des sous-graphes avec squeeze=False pour garantir un tableau 2D
        n_cols = 2 if ground_truth is not None else 1
        fig, axs = plt.subplots(1, n_cols, figsize=figsize if n_cols == 2 else (figsize[0]/2, figsize[1]), squeeze=False)

        if ground_truth is not None:
            vmin, vmax = (0, 1) if scale == 'same' else (np.min(image), np.max(image))
        else:
            vmin, vmax = (0, np.max(image))

        im0 = axs[0, 0].imshow(image, cmap='hot', vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
        axs[0, 0].set_title(title_recon)
        axs[0, 0].set_xlabel("X (mm)")
        axs[0, 0].set_ylabel("Z (mm)")
        axs[0, 0].tick_params(axis='both', which='major')

        # Plot ground truth if available
        if ground_truth is not None:
            gt_vmin, gt_vmax = (0, 1) if scale == 'same' else (np.min(ground_truth), np.max(ground_truth))

            im1 = axs[0, 1].imshow(ground_truth, cmap='hot', vmin=gt_vmin, vmax=gt_vmax, extent=extent, aspect='equal')
            axs[0, 1].set_title(title_gt)
            axs[0, 1].set_xlabel("X (mm)")
            axs[0, 1].set_ylabel("Z (mm)")
            axs[0, 1].tick_params(axis='both', which='major')

        plt.subplots_adjust(bottom=0.15, wspace=0.3)

        # Calculate colorbar position dynamically based on figsize
        cbar_width = 0.05 * figsize[0] / figsize[1]  # Relative to figure height
        cbar_height = 0.05
        cbar_x = 0.25  # Centered horizontally
        cbar_y = -0.06 # Positioned at the bottom

        # Add colorbar
        cbar_ax = fig.add_axes([cbar_x, cbar_y, 0.5, cbar_height])
        cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
        if ground_truth is not None and scale == 'same':
            cbar.set_label('Normalized Intensity') 
        else:
            cbar.set_label('Intensity')
        cbar.ax.tick_params(labelsize=8)

        # Save figure if path is provided
        if savePath is not None:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            filename = 'recon_with_tumor.png' if withTumor else 'recon_without_tumor.png'
            plt.savefig(os.path.join(savePath, filename), dpi=300, bbox_inches='tight')

        plt.show()