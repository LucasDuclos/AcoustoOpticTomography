from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType
from AOT_biomaps.AOT_Acoustic.FocusedWave import FocusedWave
from ._mainExperiment import Experiment

from tqdm import trange
import os
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


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

        Returns:
            tuple: (bool, str) - (True, "success message") if valid, (False, "error message") otherwise
        """
        if self.TypeAcoustic is None or self.TypeAcoustic.value != WaveType.FocusedWave.value:
           return False, "acousticType must be provided and must be FocusedWave for Focus experiment"
        if self.AcousticFields is None:
           return False, "AcousticFields is not initialized. Please generate the system matrix first."
        if self.AOsignal_withTumor is None:
            return False, "AOsignal with tumor is not initialized. Please generate the AO signal with tumor first."
        if self.AOsignal_withoutTumor is None:
            return False, "AOsignal without tumor is not initialized. Please generate the AO signal without tumor first."
        if self.OpticImage is None:
            return False, "OpticImage is not initialized. Please generate the optic image first."
        if self.AOsignal_withoutTumor.shape != self.AOsignal_withTumor.shape:
            return False, "AOsignal with and without tumor must have the same shape."
        for field in self.AcousticFields:
            if field.field.shape[0] != self.AOsignal_withTumor.shape[0]:
                return False, f"Field {field.getName_field()} has an invalid Time shape: {field.field.shape[0]}. Expected time shape to be {self.AOsignal_withTumor.shape[0]}."
        if not all(field.field.shape == self.AcousticFields[0].field.shape for field in self.AcousticFields):
            return False, "All AcousticFields must have the same shape."
        if self.OpticImage is None:
            return False, "OpticImage is not initialized. Please generate the optic image first."
        if self.OpticImage.phantom is None:
            return False, "OpticImage phantom is not initialized. Please generate the phantom first."
        if self.OpticImage.laser is None:
            return False, "OpticImage laser is not initialized. Please generate the laser first."
        if self.OpticImage.laser.shape != self.OpticImage.phantom.shape:
            return False, "OpticImage laser and phantom must have the same shape."
        if self.OpticImage.phantom.shape[0] != self.AcousticFields[0].field.shape[1] or self.OpticImage.phantom.shape[1] != self.AcousticFields[0].field.shape[2]:
            return False, f"OpticImage phantom shape {self.OpticImage.phantom.shape} does not match AcousticFields shape {self.AcousticFields[0].field.shape[1:]}."

        return True, "Experiment is correctly initialized."

    def generate_acoustic_fields(self, fieldDataPath=None, show_log=False, nameBlock=None):
        """
        Generate a list of focused acoustic fields for each probe element.
        Uses trange to display a progress bar and manages memory.

        Parameters:
            fieldDataPath (str): Path to save generated fields.
            show_log (bool): If True, displays progress logs.
            nameBlock (str): Optional name for the block when saving.

        Returns:
            list: List of generated FocusedWave objects.
        """
        listAcousticFields = []
        num_elements = self.params.acoustic['probe']['num_elements']
        element_width = self.params.acoustic['probe']['element_width']

        progress_bar = trange(num_elements, desc="Generating focused acoustic fields")

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
                    focused_wave.load_field(fieldDataPath)
                except Exception as e:
                    progress_bar.set_postfix_str(f"Error loading field -> Generating field - {field_name}")
                    focused_wave = FocusedWave(params=self.params, focal_line=x_k, medium=self.medium)
                    focused_wave.generate_field(show_log=show_log)
                    if not os.path.exists(pathField):
                        os.makedirs(os.path.dirname(pathField), exist_ok=True)
                        focused_wave.save_field(fieldDataPath)
            else:
                progress_bar.set_postfix_str(f"Generating field - {field_name}")
                focused_wave = FocusedWave(params=self.params, focal_line=x_k, medium=self.medium)
                focused_wave.generate_field(show_log=show_log)
                if pathField is not None:
                    os.makedirs(os.path.dirname(pathField), exist_ok=True)
                    focused_wave.save_field(fieldDataPath)

            listAcousticFields.append(focused_wave)
            progress_bar.set_postfix_str("")

        self.AcousticFields = listAcousticFields

    def recon_focus(self, withTumor=True, signals_AO=None):
        """
        Reconstruct the focused zone image from AO signals.
        AO signals are already in the form of a NumPy array with dimensions (times, N).
        Signals are limited to Nt samples, then resized to obtain an image (X, Z).

        Parameters:
            withTumor (bool): If True, uses signals with tumor. If False, uses signals without tumor.
            signals_AO (numpy.ndarray): Optional AO signals to use. If None, uses the experiment's signals.

        Returns:
            numpy.ndarray: Reconstructed image with dimensions (X, Z).
        """
        if signals_AO is None:
            if withTumor:
                signals_AO = self.AOsignal_withTumor
            else:
                signals_AO = self.AOsignal_withoutTumor

        print(f"Original shape of AO signals: {signals_AO.shape}")

        Z = self.params.general['Nz']
        dx = self.params.general['dx']
        c0 = self.params.acoustic['medium']['c0']
        f_saving = self.params.acoustic['f_saving']
        Nt = int(np.ceil(Z * dx / c0 * f_saving))
        print(f"Calculated Nt: {Nt}")

        signals_AO_truncated = signals_AO[5:Nt, :]
        print(f"Shape of AO signals after truncation: {signals_AO_truncated.shape}")

        X = self.params.general['Nx']
        Z = self.params.general['Nz']

        zoom_factor_y = Z / signals_AO_truncated.shape[0]
        zoom_factor_x = X / signals_AO_truncated.shape[1]

        recon_image = zoom(signals_AO_truncated, (zoom_factor_y, zoom_factor_x), order=1)

        recon_image = (recon_image - np.min(recon_image)) / (np.max(recon_image) - np.min(recon_image) + 1e-9)

        return recon_image
    
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
            print(f"Reconstructed image saved at {fig_path}")