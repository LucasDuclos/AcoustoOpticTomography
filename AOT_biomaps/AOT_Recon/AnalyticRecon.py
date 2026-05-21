from ._mainRecon import Recon
from .ReconEnums import ReconType, AnalyticType, ProcessType
from AOT_biomaps.AOT_Experiment.ExperimentTools import add_sincos_cpu
from .ReconTools import fourierz_gpu, EvalDelayLawOS_center, ifourierx_gpu, rotate_theta_gpu, filter_radon_gpu, ifourierz_gpu  

import numpy as np
from tqdm import trange
import os
from datetime import datetime

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
            raise ValueError("Lc parameter must be provided for iRADON analytic reconstruction.")
        self.Lc = Lc # in meters
        self.AOsignal_demoldulated = None

    def run(self, processType = ProcessType.PYTHON, withTumor= True):
        """
        This method is a placeholder for the analytic reconstruction process.
        It currently does not perform any operations but serves as a template for future implementations.
        """
        if(processType == ProcessType.CASToR):
            raise NotImplementedError("CASToR analytic reconstruction is not implemented yet.")
        elif(processType == ProcessType.PYTHON):
            self._analyticReconPython(withTumor)
        else:
            raise ValueError(f"Unknown analytic reconstruction type: {processType}")
        
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

        d_t = 1 / float(self.experiment.params.acoustic['f_saving'])
        t_array = np.arange(0, AOsignal.shape[0])*d_t
        Z = t_array * self.experiment.params.acoustic['medium']['c0']
        X_m = np.arange(0, self.experiment.params.acoustic['probe']['num_elements'])* self.experiment.params.general['dx']
        dfX = 1 / (X_m[1] - X_m[0]) / len(X_m)
        if withTumor:
            self.AOsignal_demoldulated = self.experiment.parse_and_demodulate(withTumor=True)
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
                raise ValueError(f"Unknown analytic type: {self.analyticType}")
        else:
            self.AOsignal_demoldulated = self.experiment.parse_and_demodulate(withTumor=False)
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
                raise ValueError(f"Unknown analytic type: {self.analyticType}")
    
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
    Reconstruction d'image utilisant la méthode iFourier (GPU).
        Normalisation physique complète incluse.
        """

        # ======================================================
        # 1. Préparation GPU
        # ======================================================
        R = cp.asarray(R)
        z = cp.asarray(z)
        X_m = cp.asarray(X_m)
        theta = cp.asarray(theta)
        decimation = cp.asarray(decimation)
        DelayLAWS = cp.asarray(DelayLAWS)
        ActiveLIST = cp.asarray(ActiveLIST)

        # Normalisation DelayLAWS (ms -> s si nécessaire)
        DelayLAWS_s = cp.where(cp.max(DelayLAWS) > 1e-3, DelayLAWS / 1000.0, DelayLAWS)

        # Regroupement tirs (CPU pour np.unique plus rapide)
        ScanParam_cpu = cp.asnumpy(cp.stack([decimation, cp.round(theta, 4)], axis=1))
        _, ia_cpu, ib_cpu = np.unique(ScanParam_cpu, axis=0, return_index=True, return_inverse=True)
        ia = cp.asarray(ia_cpu)
        ib = cp.asarray(ib_cpu)

        # ======================================================
        # 2. Structuration complexe
        # ======================================================
        F_complex_cpu, theta_u_cpu, decim_u_cpu = add_sincos_cpu(
            cp.asnumpy(R),
            cp.asnumpy(decimation),
            np.radians(cp.asnumpy(theta))
        )

        # Calcul des centres de rotation
        M0 = EvalDelayLawOS_center(
            X_m,
            theta_u_cpu,
            DelayLAWS_s.T[:, ia],
            ActiveLIST.T[:, ia],
            c
        )

        # Transfert GPU
        F_complex = cp.asarray(F_complex_cpu)
        theta_u = cp.asarray(theta_u_cpu)
        decim_u = cp.asarray(decim_u_cpu)
        M0_gpu = cp.asarray(M0)

        # ======================================================
        # 3. Paramètres de la grille
        # ======================================================
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

        # ======================================================
        # 4. Boucle Inverse Fourier X
        # ======================================================
        for i_ang in trange(
            Ntheta,
            desc=f"AOT-BioMaps -- iFourier ({'with tumor' if withTumor else 'without tumor'}) -- GPU",
            unit="angle"
        ):

            # Grille Fourier locale (z, fx)
            F_fx_z = cp.zeros((Nz, Nx), dtype=cp.complex64)

            # Indices correspondant à cet angle
            indices = cp.where(ib_u == i_ang)[0]

            for idx in indices:
                n = int(decim_u[idx])
                trace_z = F_complex[:, idx]

                # Mapping positif
                ip = idx0_x + n
                if 0 <= ip < Nx:
                    F_fx_z[:, ip] = trace_z

                # Mapping négatif (symétrie hermitienne MATLAB)
                if n != 0:
                    im = idx0_x - n
                    if 0 <= im < Nx:
                        col_conj = cp.zeros(Nz, dtype=cp.complex64)
                        col_conj[1:] = cp.conj(trace_z[:-1])
                        F_fx_z[:, im] = col_conj

            # Correction DC
            F_fx_z[:, idx0_x] *= 0.5

            # Inverse Fourier X (GPU) + facteur Nx pour correspondance MATLAB
            I_spatial = ifourierx_gpu(F_fx_z, dx) * Nx

            # Rotation spatiale autour du centre M0
            I_rot = rotate_theta_gpu(
                X_grid,
                Z_grid,
                I_spatial,
                -angles_group[i_ang],
                M0_gpu[i_ang, :]
            )

            # Somme incohérente
            I_final += I_rot

        # ======================================================
        # 5. Normalisation physique finale
        # ======================================================
        Ntheta_total = len(theta_u)
        Ntirs_complex = (R.shape[1] - Ntheta_total) / 4.0  # 4 phases par tir

        I_final /= (Ntheta_total * Ntirs_complex)
        I_final *= dx  # normalisation physique sur l’axe x

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
    Reconstruction d'image utilisant la méthode iRadon.
        Normalisation physique correcte (phases, angles, dz).
        """

        # ======================================================
        # 1. AddSinCos (structuration) — CPU volontairement
        # ======================================================
        theta = np.radians(theta)
        F_ct_kx, theta_u, decim_u = add_sincos_cpu(R, decimation, theta)

        ScanParam = np.stack([decimation, theta], axis=1)
        _, ia, _ = np.unique(ScanParam, axis=0, return_index=True, return_inverse=True)

        ActiveLIST = np.asarray(ActiveLIST).T
        DelayLAWS = np.asarray(DelayLAWS).T
        ActiveLIST_unique = ActiveLIST[:, ia]

        # ======================================================
        # 2. FFT z
        # ======================================================
        z_gpu = cp.asarray(z)
        Fin = fourierz_gpu(z, F_ct_kx)

        dz = float(z[1] - z[0])                 # <<< Δz PHYSIQUE
        fz = cp.fft.fftshift(cp.fft.fftfreq(len(z), d=dz))

        Nz, Nk = Fin.shape

        # ======================================================
        # 3. Filtrage OS exact
        # ======================================================
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

        # ======================================================
        # 4. Retour espace z
        # ======================================================
        Finf = ifourierz_gpu(z, Finf)
        Fsup = ifourierz_gpu(z, Fsup)

        # ======================================================
        # 5. Grille image
        # ======================================================
        X_gpu = cp.asarray(X_m)
        X, Z = cp.meshgrid(X_gpu, z_gpu)
        Xc = float(np.mean(X_m))

        # ======================================================
        # 6. Centre de rotation M0
        # ======================================================
        M0 = EvalDelayLawOS_center(X_m, theta, DelayLAWS[:, ia], ActiveLIST_unique, c)
        M0_gpu = cp.asarray(M0)

        # ======================================================
        # 7. Rétroprojection
        # ======================================================
        Irec = cp.zeros_like(X, dtype=cp.complex64)

        for i in trange(
            len(theta_u),
            desc=f"AOT-BioMaps -- iRadon ({'with tumor' if withTumor else 'without tumor'}) -- GPU",
            unit="angle"
        ):
            th = float(theta_u[i])

            T = (X - M0_gpu[i, 0]) * cp.sin(th) + (Z - M0_gpu[i, 1]) * cp.cos(th) + M0_gpu[i, 1]
            S = (X - Xc) * cp.cos(th) - (Z - M0_gpu[i, 1]) * cp.sin(th)
            h0 = cp.exp(1j * 2 * cp.pi * decim_u[i] * df0x * S)

            # interpolation linéaire en z
            Tind = (T - z_gpu[0]) / dz
            i0 = cp.floor(Tind).astype(cp.int32)
            i1 = i0 + 1
            i0 = cp.clip(i0, 0, Nz - 1)
            i1 = cp.clip(i1, 0, Nz - 1)
            w = Tind - i0

            proj_sup = (1 - w) * Fsup[i0, i] + w * Fsup[i1, i]
            proj_inf = (1 - w) * Finf[i0, i] + w * Finf[i1, i]

            # >>> SOMME BRUTE (correcte)
            Irec += 2 * h0 * proj_sup + proj_inf

        # ======================================================
        # 8. NORMALISATION PHYSIQUE GLOBALE
        # ======================================================

        Ntheta = len(theta_u)

        # nombre de tirs complexes indépendants (4 phases)
        Ntirs_complex = (R.shape[1] - Ntheta) / 4.0

        # normalisation finale
        Irec /= (Ntheta * Ntirs_complex)
        print(f"dz normalization: {dz}")
        Irec *= dz

        return cp.real(Irec).get()
