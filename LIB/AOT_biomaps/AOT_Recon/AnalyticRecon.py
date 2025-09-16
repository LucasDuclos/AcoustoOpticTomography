from ._mainRecon import Recon
from .ReconEnums import ReconType, AnalyticType, ProcessType

import numpy as np
from tqdm import trange

class AnalyticRecon(Recon):
    def __init__(self, analyticType, **kwargs):
        super().__init__(**kwargs)
        self.reconType = ReconType.Analytic
        self.analyticType = analyticType

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

    def _analyticReconPython(self,withTumor):
        """
        This method is a placeholder for the analytic reconstruction process in Python.
        It currently does not perform any operations but serves as a template for future implementations.
        
        Parameters:
            analyticType: The type of analytic reconstruction to perform (default is iFOURIER).
        """
        if withTumor:
            if self.analyticType == AnalyticType.iFOURIER:
                self.reconPhantom = self._iFourierRecon(self.experiment.AOsignal_withTumor)
            elif self.analyticType == AnalyticType.iRADON:
                self.reconPhantom = self._iRadonRecon(self.experiment.AOsignal_withTumor)
            else:            
                raise ValueError(f"Unknown analytic type: {self.analyticType}")
        else:
            if self.analyticType == AnalyticType.iFOURIER:
                self.reconLaser = self._iFourierRecon(self.experiment.AOsignal_withoutTumor)
            elif self.analyticType == AnalyticType.iRADON:
                self.reconLaser = self._iRadonRecon(self.experiment.AOsignal_withoutTumor)
            else:            
                raise ValueError(f"Unknown analytic type: {self.analyticType}")
    
    def _iFourierRecon(self, AOsignal):
        """
        Reconstruction d'image utilisant la transformation de Fourier inverse.

        :param AOsignal: Signal dans le domaine temporel.
        :return: Image reconstruite dans le domaine spatial.
        """
        # Signal dans le domaine fréquentiel (FFT sur l'axe temporel)
        s_tilde = np.fft.fft(AOsignal, axis=0)

        theta = np.array([af.angle for af in self.experiment.AcousticFields])  # angles (N_theta,)
        f_s = np.array([af.f_s for af in self.experiment.AcousticFields])  # spatial freqs (N_theta,)
        f_t = np.fft.fftfreq(AOsignal.shape[0], d=self.experiment.dt)  # temporal freqs

        x = self.experiment.OpticImage.laser.x
        z = self.experiment.OpticImage.laser.z
        X, Z = np.meshgrid(x, z, indexing='ij')  # shape (Nx, Nz)

        N_theta = len(theta)
        I_rec = np.zeros((len(x), len(z)), dtype=complex)

        for i, th in enumerate(trange(N_theta, desc="AOT-BioMaps -- Analytic Recontruction Tomography : iFourier (Processing projection) ---- processing on single CPU ----")):
            fs = f_s[i]

            # Projection des coordonnées dans le repère tourné
            x_prime = X * np.cos(th) + Z * np.sin(th)
            z_prime = -X * np.sin(th) + Z * np.cos(th)

            # Signal spectral pour cet angle (1D pour chaque f_t)
            s_angle = s_tilde[:, i]  # shape (len(f_t),)

            # Grille 2D des fréquences
            F_t, F_s = np.meshgrid(f_t, [fs], indexing='ij')  # F_t: (len(f_t), 1), F_s: (1, 1)

            # Phase : exp(2iπ(x' f_s + z' f_t)) = (x_prime * f_s + z_prime * f_t)
            phase = 2j * np.pi * (x_prime[:, :, None] * fs + z_prime[:, :, None] * f_t[None, None, :])

            # reshape s_angle to (len(f_t), 1, 1)
            s_angle = s_angle[:, None, None]

            # Contribution de cet angle
            integrand = s_angle * np.exp(phase)

            # Intégration sur f_t (somme discrète)
            I_theta = np.sum(integrand, axis=0)

            # Ajout à la reconstruction
            I_rec += I_theta

        I_rec /= N_theta

        return np.abs(I_rec)

    def _iRadonRecon(self, AOsignal):
        """
        Reconstruction d'image utilisant la méthode iRadon.

        :return: Image reconstruite.
        """
        @staticmethod
        def trapz(y, x):
            """Compute the trapezoidal rule for integration."""
            return np.sum((y[:-1] + y[1:]) * (x[1:] - x[:-1]) / 2)

        # Initialisation de l'image reconstruite
        I_rec = np.zeros((len(self.experiment.OpticImage.laser.x), len(self.experiment.OpticImage.laser.z)), dtype=complex)

        # Transformation de Fourier du signal
        s_tilde = np.fft.fft(AOsignal, axis=0)

        # Extraction des angles et des fréquences spatiales
        theta = [acoustic_field.angle for acoustic_field in self.experiment.AcousticFields]
        f_s = [acoustic_field.f_s for acoustic_field in self.experiment.AcousticFields]

        # Calcul des coordonnées transformées et intégrales
        with trange(len(theta) * 2, desc="AOT-BioMaps -- Analytic Reconstruction Tomography: iRadon") as pbar:
            for i in range(len(theta)):
                pbar.set_description("AOT-BioMaps -- Analytic Reconstruction Tomography: iRadon (Processing frequency contributions)  ---- processing on single CPU ----")
                th = theta[i]
                x_prime = self.experiment.OpticImage.x[:, np.newaxis] * np.cos(th) - self.experiment.OpticImage.z[np.newaxis, :] * np.sin(th)
                z_prime = self.experiment.OpticImage.z[np.newaxis, :] * np.cos(th) + self.experiment.OpticImage.x[:, np.newaxis] * np.sin(th)

                # Première intégrale : partie réelle
                for j in range(len(f_s)):
                    fs = f_s[j]
                    integrand = s_tilde[i, j] * np.exp(2j * np.pi * (x_prime * fs + z_prime * fs))
                    integral = self.trapz(integrand * fs, fs)
                    I_rec += 2 * np.real(integral)
                pbar.update(1)

            for i in range(len(theta)):
                pbar.set_description("AOT-BioMaps -- Analytic Reconstruction Tomography: iRadon (Processing central contributions)  ---- processing on single CPU ----")
                th = theta[i]
                x_prime = self.experiment.OpticImage.x[:, np.newaxis] * np.cos(th) - self.experiment.OpticImage.z[np.newaxis, :] * np.sin(th)
                z_prime = self.experiment.OpticImage.z[np.newaxis, :] * np.cos(th) + self.experiment.OpticImage.x[:, np.newaxis] * np.sin(th)

                # Filtrer les fréquences spatiales pour ne garder que celles inférieures ou égales à f_s_max
                filtered_f_s = np.array([fs for fs in f_s if fs <= self.f_s_max])
                integrand = s_tilde[i, np.where(np.array(f_s) == 0)[0][0]] * np.exp(2j * np.pi * z_prime * filtered_f_s)
                integral = self.trapz(integrand * filtered_f_s, filtered_f_s)
                I_rec += integral
                pbar.update(1)

        return np.abs(I_rec)
