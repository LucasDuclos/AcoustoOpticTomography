from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType
from AOT_biomaps.AOT_Acoustic.FocusedWave import FocusedWave
from ._mainExperiment import Experiment

from tqdm import trange
import os
import numpy as np
from scipy.ndimage import zoom, gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Focus(Experiment):
    """
    Experiment class for focused acoustic wave imaging.
    Inherits from Experiment and provides methods for generating acoustic fields
    and reconstructing focused images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check(self):
        """
        Check if the experiment is correctly initialized.
        """
        if self.TypeAcoustic is None or self.TypeAcoustic.value != WaveType.FocusedWave.value:
           raise ValueError("[AOT-biomaps] acousticType must be provided and must be FocusedWave for Focus experiment")
        if self.AcousticFields is None:
           raise ValueError("[AOT-biomaps] AcousticFields is not initialized. Please generate the system matrix first.")
        if self.AOsignal_withTumor is None:
            raise ValueError("[AOT-biomaps] AOsignal with tumor is not initialized. Please generate the AO signal with tumor first.")
        if self.AOsignal_withoutTumor is None:
            raise ValueError("[AOT-biomaps] AOsignal without tumor is not initialized. Please generate the AO signal without tumor first.")
        if self.OpticImage is None:
            raise ValueError("[AOT-biomaps] OpticImage is not initialized. Please generate the optic image first.")
        if self.AOsignal_withoutTumor.shape != self.AOsignal_withTumor.shape:
            raise ValueError("[AOT-biomaps] AOsignal with and without tumor must have the same shape.")
        for field in self.AcousticFields:
            if field.field.shape[0] != self.AOsignal_withTumor.shape[0]:
                raise ValueError(f"[AOT-biomaps] Field {field.get_name_field()} has an invalid Time shape: {field.field.shape[0]}. Expected time shape to be {self.AOsignal_withTumor.shape[0]}.")
        if not all(field.field.shape == self.AcousticFields[0].field.shape for field in self.AcousticFields):
            raise ValueError("[AOT-biomaps] All AcousticFields must have the same shape.")
        if self.OpticImage is None:
            raise ValueError("[AOT-biomaps] OpticImage is not initialized. Please generate the optic image first.")
        if self.OpticImage.phantom is None:
            raise ValueError("[AOT-biomaps] OpticImage phantom is not initialized. Please generate the phantom first.")
        if self.OpticImage.laser is None:
            raise ValueError("[AOT-biomaps] OpticImage laser is not initialized. Please generate the laser first.")
        if self.OpticImage.laser.shape != self.OpticImage.phantom.shape:
            raise ValueError("[AOT-biomaps] OpticImage laser and phantom must have the same shape.")
        if self.OpticImage.phantom.shape[0] != self.AcousticFields[0].field.shape[1] or self.OpticImage.phantom.shape[1] != self.AcousticFields[0].field.shape[2]:
            raise ValueError(f"[AOT-biomaps] OpticImage phantom shape {self.OpticImage.phantom.shape} does not match AcousticFields shape {self.AcousticFields[0].field.shape[1:]}.")

        print("[AOT-biomaps] Experiment is correctly initialized.")

    def generate_acoustic_fields(self, fieldDataPath=None, isGPU=None, GPUdevice=None, tempFieldName="Kwave", nameBlock=None, generation_type="envelope_squarred", show_log=False):
        """
        Generate a list of focused acoustic fields for each probe element.
        Uses trange to display a progress bar and manages memory.

        Parameters:
            fieldDataPath (str): Path to save generated fields.
            isGPU (bool): Whether to use GPU for field generation. (Default is None, which uses CPU.)
            GPUdevice (int): The GPU device to use for field generation. (Default is None, which uses the default GPU.)
            tempFieldName (str): Name for the temporary field files.
            nameBlock (str): Optional name for the h5 file format.
            generation_type (str): The type of field generation to perform.
            show_log (bool): If True, displays progress logs.

        Returns:
            list: List of generated FocusedWave objects.
        """
        listAcousticFields = []
        num_elements = self.params.acoustic['probe']['num_elements']
        element_width = self.params.acoustic['probe']['element_width']

        progress_bar = trange(num_elements, desc="[AOT-biomaps] Generating focused acoustic fields")

        for k in progress_bar:
            x_k = (k - (num_elements - 1) / 2) * element_width

            field_name = f"field_focused_X{x_k*1000:.2f}_Z{self.params.acoustic['emission']['Foc']*1000:.2f}"

            if fieldDataPath is not None:
                pathField = os.path.join(fieldDataPath, field_name + ".hdr")
            else:
                pathField = None

            if pathField is not None and os.path.exists(pathField):
                progress_bar.set_postfix_str(f"Loading field - {field_name}")
                try:
                    focused_wave = FocusedWave(params=self.params, focal_line=x_k, medium=self.medium)
                    focused_wave.load_field(fieldDataPath, self.FormatSave, nameBlock)
                except Exception as e:
                    progress_bar.set_postfix_str(f"Error loading field -> Generating field - {field_name}")
                    focused_wave = FocusedWave(params=self.params, focal_line=x_k, medium=self.medium)
                    focused_wave.generate_field(isGPU=isGPU, GPUdevice=GPUdevice, tempFieldName=tempFieldName, generation_type=generation_type, show_log=show_log)
                    if not os.path.exists(pathField):
                        os.makedirs(os.path.dirname(pathField), exist_ok=True)
                        focused_wave.save_field(fieldDataPath)
            else:
                progress_bar.set_postfix_str(f"Generating field - {field_name}")
                focused_wave = FocusedWave(params=self.params, focal_line=x_k, medium=self.medium)
                focused_wave.generate_field(isGPU=isGPU, GPUdevice=GPUdevice, tempFieldName=tempFieldName, generation_type=generation_type, show_log=show_log)
                if pathField is not None:
                    os.makedirs(os.path.dirname(pathField), exist_ok=True)
                    focused_wave.save_field(fieldDataPath)

            listAcousticFields.append(focused_wave)
            progress_bar.set_postfix_str("")

        self.AcousticFields = listAcousticFields

    def recon_focus(self, withTumor=True, signals_AO=None, isFiltered=True):
        """
        Reconstruct the focused zone image from AO signals.
        AO signals are already in the form of a NumPy array with dimensions (times, N).
        Signals are limited to Nt samples, then resized to obtain an image (X, Z).

        Parameters:
            withTumor (bool): If True, uses signals with tumor. If False, uses signals without tumor.
            signals_AO (numpy.ndarray): Optional AO signals to use. If None, uses the experiment's signals.
            isFiltered (bool): If True, applies filtering to the reconstructed image.
        Returns:
            numpy.ndarray: Reconstructed image with dimensions (X, Z).
        """
        if signals_AO is None:
            if withTumor:
                signals_AO = self.AOsignal_withTumor
            else:
                signals_AO = self.AOsignal_withoutTumor

        Z = self.params.general['Nz']
        dx = self.params.general['dx']
        c0 = self.params.acoustic['medium']['c0']
        f_saving = self.params.acoustic['f_saving']
        Nt = int(np.ceil(Z * dx / c0 * f_saving))

        signals_AO_truncated = signals_AO[5:Nt, :]

        X = self.params.general['Nx']
        Z = self.params.general['Nz']

        zoom_factor_z = Z / signals_AO_truncated.shape[0]
        zoom_factor_x = X / signals_AO_truncated.shape[1]

        raw_recon = zoom(signals_AO_truncated, (zoom_factor_z, zoom_factor_x), order=1)

        # raw_recon = (raw_recon - np.min(raw_recon)) / (np.max(raw_recon) - np.min(raw_recon) + 1e-9)

        if not isFiltered:
            print("[AOT-biomaps] Returning raw reconstruction without filtering.")
            return raw_recon
        print("[AOT-biomaps] Applying filtering to the reconstructed image.")

        raw_recon = raw_recon - np.min(raw_recon,axis=0,keepdims=True) # used as a DC and really low frequencies filter
        
        c0 = np.min(self.medium.kmedium.sound_speed)
        
        f_US = self.params.acoustic['f_US']
        lambda_us = c0 / f_US

        FWHM_axial = self.params.acoustic['emission']['num_cycles'] * lambda_us
        sigma_axial = FWHM_axial / (2 * np.sqrt(2 * np.log(2)))

        FWHM_foc = 1.028 * lambda_us * (self.params.acoustic['emission']['Foc'] / (self.params.acoustic['emission']['N_piezoFocal'] * self.params.acoustic['probe']['element_width']))
        w0 = FWHM_foc / (2 * np.sqrt(2 * np.log(2)))  

        z_R = (np.pi * (w0**2)) / lambda_us
        dz = self.params.general['dz']
        dx = self.params.general['dx']

        Nz = raw_recon.shape[0] 
        Nx = raw_recon.shape[1]

        z_array = np.arange(Nz) * dz
        sigma_lateral_z = w0 * np.sqrt(1 + ((z_array - self.params.acoustic['emission']['Foc']) / z_R)**2)

        filtered_recon = np.zeros_like(raw_recon)
        sigma_axial_px = sigma_axial / dz
        AO_axial_filtered = gaussian_filter1d(raw_recon, sigma=sigma_axial_px, axis=0)
        for iz in range(Nz):
            sigma_lat_px = sigma_lateral_z[iz] / dx
            ligne_laterale = AO_axial_filtered[iz, :]
            filtered_recon[iz, :] = gaussian_filter1d(ligne_laterale, sigma=sigma_lat_px)

        filtered_recon = (filtered_recon - np.min(filtered_recon)) / (np.max(filtered_recon) - np.min(filtered_recon) + 1e-9)
        return filtered_recon

    def show_recon(self, withTumor=True, save_dir=None, wave_name=None, figsize=(12, 5)):
        """
        Show the reconstructed focused image.

        Parameters:
            withTumor (bool): If True, shows image with tumor. If False, shows image without tumor.
            save_dir (str): Optional directory to save the figure.
            wave_name (str): Optional name for the figure file.
            figsize (tuple): Size of the figure.

        Returns:
            Reconstructed image.
        """
        recon_image = self.recon_focus(withTumor=withTumor)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(recon_image.T, extent=[self.params.general['Xrange'][0]*1e3, self.params.general['Xrange'][1]*1e3,
                                              self.params.general['Zrange'][1]*1e3, self.params.general['Zrange'][0]*1e3],
                       cmap='hot', aspect='auto')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('Reconstructed Focused Image' + (' with Tumor' if withTumor else ' without Tumor'))
        fig.colorbar(im, ax=ax)

        if save_dir is not None and wave_name is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, f"recon_focus_{wave_name}_{'withTumor' if withTumor else 'withoutTumor'}.png")
            plt.savefig(fig_path)
            print(f"[AOT-biomaps] Reconstructed image saved at {fig_path}")