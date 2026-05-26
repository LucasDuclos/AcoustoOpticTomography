from AOT_biomaps.Settings import Params
from AOT_biomaps.AOT_Optic._mainOptic import Phantom
from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType, FormatSave
from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave
from AOT_biomaps.AOT_Medium.HomogeneousMedium import HomogeneousMedium
from AOT_biomaps.AOT_Medium.PVAMedium import PVAMedium
from AOT_biomaps.AOT_Medium.MediumEnums import PhantomType
from AOT_biomaps.AOT_Experiment.ExperimentTools import load_AOsignal
from abc import ABC, abstractmethod

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import copy
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

# Optional cupy import for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class Experiment(ABC):
    def __init__(self, params, acousticType=WaveType.StructuredWave, formatSave=FormatSave.HDR_IMG):
        self.params = params
        self.OpticImage = None
        self.medium = None
        self.AcousticFields = None
        self.AOsignal_withTumor = None
        self.AOsignal_withoutTumor = None

        if type(acousticType).__name__ != "WaveType":
            raise TypeError("acousticType must be an instance of the WaveType class")

        self.FormatSave = formatSave
        self.TypeAcoustic = acousticType

        if type(self.params) != Params:
            raise TypeError("params must be an instance of the Params class")

    def copy(self):
        """
    Return une copie profonde de l'objet."""
        return copy.deepcopy(self)
    
    def generate_phantom(self):
        """
        Generate the phantom for the experiment.
        This method initializes the OpticImage attribute with a Phantom instance.
        """
        self.OpticImage = Phantom(params=self.params)
    
    def generate_medium(self):
        """
        Generate the medium for the experiment.
        This method initializes the medium attribute based on the parameters.
        """
        if self.params.acoustic['medium']['type'] == PhantomType.Homogeneous.value:
            self.medium = HomogeneousMedium(params=self.params)
            self.medium.generate_medium()
            print("Medium generated: Homogeneous. -- done.")
        elif self.params.acoustic['medium']['type'] == PhantomType.PVA.value:
            try:
                self.medium = PVAMedium(params=self.params)
                self.medium.generate_medium()
                print("Medium generated: PVA heterogeneous. -- done.")
            except Exception as e:
                
                print(f"Error generating PVA medium: {e}")
                raise
    
    def load_medium(self, folderPath, fileName="medium"):
        """
        Load the medium from a .npy file.
        
        Parameters:
        - folderPath (str): The directory where the file is located.
        - fileName (str): The name of the file (without extension).
        
        Raises:
            ValueError: If medium type is not supported.
            FileNotFoundError: If the file does not exist.
        """
        if self.params.acoustic['medium']['type'] == PhantomType.Homogeneous.value:
            self.medium = HomogeneousMedium(params=self.params)
            self.medium.load_medium(folderPath, fileName)
        elif self.params.acoustic['medium']['type'] == PhantomType.PVA.value:
            self.medium = PVAMedium(params=self.params)
            self.medium.load_medium(folderPath, fileName)
        else:
            raise ValueError(f"Unsupported medium type: {self.params.acoustic['medium']['type']}")

    def save_medium(self, folderPath, fileName="medium"):
        """
        Save the medium to a .npy file.
        
        Parameters:
        - folderPath (str): The directory where the file will be saved.
        - fileName (str): The name of the file (without extension).
        
        Raises:
            ValueError: If medium is not initialized.
        """
        if self.medium is None:
            raise ValueError("Medium is not initialized. Please generate or set the medium before saving.")
        self.medium.save_medium(folderPath, fileName)

    @abstractmethod
    def generate_acoustic_fields(self, fieldDataPath, fieldParamPath, show_log=True):
        """
        Generate the acoustic fields for simulation.
        Args:
            fieldDataPath: Path to save the generated fields.
            fieldParamPath: Path to the field parameters file.
        Returns:
            systemMatrix: A numpy array of the generated fields.
        """
        pass

    def cut_acoustic_fields(self, max_t, min_t=0, show_log=True):
        """
        Cut the acoustic fields to a specified time range.
        Args:
            max_t: Maximum time in SAMPLE to keep in the fields.
            min_t: Minimum time in SAMPLE to keep in the fields (default is 0).
            show_log: Whether to display a progress bar.
        """

        if min_t < 0 or max_t < 0:
            raise ValueError("min_t and max_t must be non-negative integers.")
        if min_t >= max_t:
            raise ValueError("min_t must be less than max_t.")

        if not self.AcousticFields:
            raise ValueError("AcousticFields is empty. Cannot cut fields.")

        iteration = range(len(self.AcousticFields)) if not show_log else trange(len(self.AcousticFields), desc=f"Cutting Acoustic Fields ({min_t} to {max_t} samples)")
        for i in iteration:
            field = self.AcousticFields[i]
            if field.field.shape[0] < max_t:
                raise ValueError(f"Field {field.getName_field()} has an invalid shape: {field.field.shape}. Expected shape to be at least ({max_t},).")
            self.AcousticFields[i].field = field.field[min_t:max_t, :, :]

    def add_noise(self, y=None, noiseType='gaussian', noiseLvl=0.1, dataToUse=None, m=1, withTumor=True, show_log=True):
        """
        Add noise to AO signals with various noise models.

        Supported noise types:
        - 'gaussian': Add Gaussian noise with std = noiseLvl * max(signal)
        - 'poisson': Add Poisson noise proportional to signal amplitude
        - 'experimental': Add noise with same SNR as in experimental dataToUse for m averages

        Parameters:
            y (np.ndarray, optional): Input signal to add noise to. If None, uses self.AOsignal_withTumor or self.AOsignal_withoutTumor.
            noiseType (str): Type of noise ('gaussian', 'poisson', or 'experimental').
            noiseLvl (float): Noise level for gaussian/poisson noise.
            dataToUse (np.ndarray): Experimental data for 'experimental' noise type (shape: (n_repeats, n_signals)).
            m (int): Number of averages for 'experimental' noise type.
            withTumor (bool): If True and y is None, use signal with tumor.
            show_log (bool): If True, displays progress bar.

        Returns:
            np.ndarray: Noisy signal(s) with same shape as input.
        """
        # Select signal source
        if y is None:
            if withTumor:
                if self.AOsignal_withTumor is None:
                    raise ValueError("AO signal with tumor not generated. Generate it first.")
                signals = self.AOsignal_withTumor
            else:
                if self.AOsignal_withoutTumor is None:
                    raise ValueError("AO signal without tumor not generated. Generate it first.")
                signals = self.AOsignal_withoutTumor
        else:
            signals = y

        # For experimental noise, estimate noise parameters from dataToUse
        if noiseType.lower() == 'experimental':
            if dataToUse is None:
                raise ValueError("dataToUse must be provided for experimental noise type.")
            # Estimate noise variance from experimental data (using random pairs)
            n_pairs = min(500, dataToUse.shape[0] // 2)
            random_pairs = np.random.choice(dataToUse.shape[0], size=(n_pairs, 2), replace=False)
            noise_var = 0.0
            for i, k in random_pairs:
                diff = dataToUse[i, :] - dataToUse[k, :]
                noise_var += np.sum(diff**2)
            noise_var /= (2 * dataToUse.shape[1] * n_pairs)
            noise_var *= 0.5  # Because var(n_i - n_k) = 2*σ_n²
            noise_var_for_m = noise_var / m
            mean_signal = np.mean(dataToUse, axis=0)
            amplitude_real = np.std(mean_signal)

        noiseSignals = np.zeros_like(signals)
        n_signals = signals.shape[1]

        # Loop over signals
        iteration = trange(n_signals, desc=f"Adding {noiseType} noise") if show_log else range(n_signals)
        for i in iteration:
            signal = signals[:, i]

            if noiseType.lower() == 'gaussian':
                # Gaussian noise: std = noiseLvl * max(signal)
                noise = np.random.normal(0, noiseLvl * np.max(signal), signal.shape)
                noisy_signal = signal + noise
            elif noiseType.lower() == 'poisson':
                # Poisson noise proportional to signal
                max_signal = np.max(np.abs(signal))
                if max_signal != 0:
                    noise = np.random.poisson(noiseLvl * np.abs(signal)) / (noiseLvl * max_signal)
                    noisy_signal = signal * noise
                else:
                    noisy_signal = signal.copy()
            elif noiseType.lower() == 'experimental':
                # Experimental-based noise with matching SNR
                amplitude_y = np.max(np.abs(signal))
                amplitude_ratio = amplitude_y / amplitude_real
                noise = np.random.randn(signal.shape[0]) * np.sqrt(noise_var_for_m) * amplitude_ratio
                noisy_signal = signal + noise
            else:
                raise ValueError("noiseType must be 'gaussian', 'poisson', or 'experimental'.")

            # Ensure non-negative (shift if needed)
            if np.min(noisy_signal) < 0:
                noisy_signal -= np.min(noisy_signal)

            noiseSignals[:, i] = noisy_signal

        return noiseSignals

    def reduce_dims(self, mode='avg'):
        """
        Reduces the T, X, Z dimensions of a numpy array (T, X, Z) by a factor of 2 using CuPy pooling.
        Falls back to numpy if CuPy is not available.
        Returns a numpy array and updates numerical parameters.
        """
        if not CUPY_AVAILABLE:
            warnings.warn("CuPy not available. Using numpy for downsampling.", UserWarning)
            # Fall back to numpy implementation
            for i in trange(len(self.AcousticFields),
                            desc="Downsampling Acoustic Fields (T, X, Z → T//2, X//2, Z//2)"):
                field = self.AcousticFields[i].field
                if field.ndim != 3:
                    raise ValueError(f"Unsupported shape: {field.shape}. Expected (T, X, Z).")
                # Simple numpy downsampling by slicing
                x_down = field[::2, ::2, ::2]
                self.AcousticFields[i].field = x_down
            return
        
        for i in trange(len(self.AcousticFields),
                        desc="Downsampling Acoustic Fields (T, X, Z → T//2, X//2, Z//2)"):
            # Convert to CuPy array
            field = self.AcousticFields[i].field
            if not isinstance(field, cp.ndarray):
                field = cp.asarray(field)

            # Check shape (must be 3D: T, X, Z)
            if field.ndim != 3:
                raise ValueError(f"Unsupported shape: {field.shape}. Expected (T, X, Z).")

            # Add dimensions for pool3d: (1, 1, T, X, Z)
            x = field[cp.newaxis, cp.newaxis, ...]

            # Downsample using 3D pooling
            if mode == 'avg':
                x_down = cp.nn.pooling.avg_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            else:  # mode == 'max'
                x_down = cp.nn.pooling.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))

            # Convert to numpy array and remove added dimensions
            self.AcousticFields[i].field = cp.asnumpy(x_down.squeeze(0).squeeze(0))

        # Utility function to convert and update a parameter
        def convert_and_update(param_dict, key, operation):
            if key in param_dict:
                if isinstance(param_dict[key], str):
                    param_dict[key] = float(param_dict[key])
                param_dict[key] = operation(param_dict[key])

        # Update parameters
        convert_and_update(self.params.acoustic, 'f_saving', lambda x: x / 2)
        for param in ['dx', 'dy', 'dz']:
            convert_and_update(self.params.general, param, lambda x: x * 2)

    def normalize_AOsignals(self, withTumor=True):
        if withTumor and self.AOsignal_withTumor is None:
            raise ValueError("AO signal with tumor is not generated. Please generate it first.")
        if not withTumor and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signal without tumor is not generated. Please generate it first.")
        if withTumor:
            self.AOsignal_withTumor = self.AOsignal_withTumor - np.min(self.AOsignal_withTumor)/(np.max(self.AOsignal_withTumor)-np.min(self.AOsignal_withTumor))
        else:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor - np.min(self.AOsignal_withoutTumor)/(np.max(self.AOsignal_withoutTumor)-np.min(self.AOsignal_withoutTumor))

    def save_acoustic_fields(self, save_directory):
        progress_bar = trange(len(self.AcousticFields), desc="Saving Acoustic Fields")
        for i in progress_bar:
            progress_bar.set_postfix_str(f"-- {self.AcousticFields[i].getName_field()}")
            self.AcousticFields[i].save_field(save_directory, formatSave=self.FormatSave)

    def show_animated_acoustic(self, wave_name=None, desired_duration_ms=5000, save_dir=None, figsize=(12, 5)):
        """
        Plot synchronized animations of A_matrix slices for selected angles.
        Args:
            wave_name: optional name for labeling the subplots (e.g., "wave1")
            desired_duration_ms: Total duration of the animation in milliseconds.
            save_dir: directory to save the animation gif; if None, animation will not be saved
        Returns:
            ani: Matplotlib FuncAnimation object
        """
        mpl.rcParams['animation.embed_limit'] = 100
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        num_plots = len(self.AcousticFields)
        if num_plots <= 5:
            nrows, ncols = 1, num_plots
        else:
            ncols = 5
            nrows = (num_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        axes = axes.flatten()
        ims = []

        fig.suptitle(f"System Matrix Animation {wave_name}", y=0.98)

        for idx in range(num_plots):
            ax = axes[idx]
            im = ax.imshow(self.AcousticFields[0, :, :, idx],
                        extent=(self.params.general['Xrange'][0], self.params.general['Xrange'][1], self.params.general['Zrange'][1], self.params.general['Zrange'][0]),
                        vmax=1, aspect='equal', cmap='jet', animated=True)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
            ims.append((im, ax, idx))

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        def update(frame):
            artists = []
            for im, ax, idx in ims:
                im.set_array(self.AcousticFields[frame, :, :, idx])
                fig.suptitle(f"System Matrix Animation {wave_name} t = {frame * 25e-6 * 1000:.2f} ms")
                artists.append(im)
            return artists

        interval = desired_duration_ms / self.AcousticFields.shape[0]
        ani = animation.FuncAnimation(
            fig, update,
            frames=range(0, self.AcousticFields.shape[0]),
            interval=interval, blit=True
        )

        if save_dir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            save_filename = f"AcousticField_{wave_name}_{date_str}.gif"
            save_path = os.path.join(save_dir, save_filename)
            ani.save(save_path, writer='pillow', fps=20)
            print(f"Saved: {save_path}")

        plt.close(fig)
        return ani

    def generate_AOsignal(self, withTumor=True, AOsignalDataPath=None):

        if AOsignalDataPath is not None:
            if not os.path.exists(AOsignalDataPath):
                raise FileNotFoundError(f"AO file {AOsignalDataPath} not found.")
            if withTumor:
                self.AOsignal_withTumor = load_AOsignal(AOsignalDataPath)
                if self.AOsignal_withTumor.shape[0] != self.AcousticFields[0].field.shape[0]:
                    print(f"AO signal shape {self.AOsignal_withTumor.shape} does not match the expected shape {self.AcousticFields[0].field.shape}. Resizing Acoustic fields...")
                    self.cut_acoustic_fields(max_t=self.AOsignal_withTumor.shape[0] / float(self.params.acoustic['f_saving']), min_t=0)
            else:
                self.AOsignal_withoutTumor = load_AOsignal(AOsignalDataPath)
                if self.AOsignal_withoutTumor.shape[0] != self.AcousticFields[0].field.shape[0]:
                    print(f"AO signal shape {self.AOsignal_withoutTumor.shape} does not match the expected shape {self.AcousticFields[0].field.shape}. Resizing Acoustic fields...")
                    self.cut_acoustic_fields(max_t=self.AOsignal_withoutTumor.shape[0] / float(self.params.acoustic['f_saving']), min_t=0)
        else:    
            if self.AcousticFields is None:
                raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")

            if self.OpticImage is None:
                raise ValueError("OpticImage is not initialized. Please generate the phantom first.")
            
            if not all(field.field.shape == self.AcousticFields[0].field.shape for field in self.AcousticFields):
                minShape = min([field.field.shape[0] for field in self.AcousticFields])
                self.cut_acoustic_fields(max_t=minShape * self.params.acoustic['f_saving'])
            else:
                shape_field = self.AcousticFields[0].field.shape

            AOsignal = np.zeros((shape_field[0], len(self.AcousticFields)), dtype=np.float32)

            if withTumor:
                description = "Generating AO Signal with Tumor"
            else:
                description = "Generating AO Signal without Tumor"

            for i in trange(len(self.AcousticFields), desc=description):
                for t in range(self.AcousticFields[i].field.shape[0]):
                    if withTumor:
                        interaction = self.OpticImage.phantom * self.AcousticFields[i].field[t, :, :]
                    else:
                        interaction = self.OpticImage.laser.intensity * self.AcousticFields[i].field[t, :, :]
                    AOsignal[t, i] = np.sum(interaction)

            if withTumor:
                self.AOsignal_withTumor = AOsignal
            else:
                self.AOsignal_withoutTumor = AOsignal

    def save_AOsignals_Castor(self, save_directory, withTumor=True):
        if withTumor:
            AO_signal = self.AOsignal_withTumor
            cdf_location = os.path.join(save_directory, "AOSignals_withTumor.cdf")
            cdh_location = os.path.join(save_directory, "AOSignals_withTumor.cdh")
        else:
            AO_signal = self.AOsignal_withoutTumor
            cdf_location = os.path.join(save_directory, "AOSignals_withoutTumor.cdf")
            cdh_location = os.path.join(save_directory, "AOSignals_withoutTumor.cdh")

        info_location = os.path.join(save_directory, "info.txt")
        nScan = AO_signal.shape[1]

        with open(cdf_location, "wb") as fileID:
            for j in range(AO_signal.shape[1]):
                active_list_hex = self.AcousticFields[j].pattern.activeList
                for i in range(0, len(active_list_hex), 2):
                    byte_value = int(active_list_hex[i:i+2], 16)
                    fileID.write(byte_value.to_bytes(1, byteorder='big'))
                angle = self.AcousticFields[j].angle
                fileID.write(np.int8(angle).tobytes())
                fileID.write(AO_signal[:, j].astype(np.float32).tobytes())

        header_content = (
            f"Data filename: {'AOSignals_withTumor.cdf' if withTumor else 'AOSignals_withoutTumor.cdf'}\n"
            f"Number of events: {nScan}\n"
            f"Number of acquisitions per event: {AO_signal.shape[0]}\n"
            f"Start time (s): 0\n"
            f"Duration (s): 1\n"
            f"Acquisition frequency (Hz): {self.params.acoustic['f_saving']}\n"
            f"Data mode: histogram\n"
            f"Data type: AOT\n"
            f"Number of US transducers: {self.params.acoustic['probe']['num_elements']}"
        )

        with open(cdh_location, "w") as fileID:
            fileID.write(header_content)

        with open(info_location, "w") as fileID:
            for field in self.AcousticFields:
                fileID.write(field.getName_field() + "\n")

        print(f"Files .cdf, .cdh and info.txt saved in {save_directory}")

    def show_AOsignal(self, withTumor=True, save_dir=None, wave_name=None, figsize=(12, 5)):
        if withTumor and self.AOsignal_withTumor is None:
            raise ValueError("AO signal with tumor is not generated. Please generate it first.")
        if not withTumor and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signal without tumor is not generated. Please generate it first.")

        if withTumor:
            AOsignal = self.AOsignal_withTumor
        else:
            AOsignal = self.AOsignal_withoutTumor

        time_axis = np.arange(AOsignal.shape[0]) / float(self.params.acoustic['f_saving']) * 1e6

        num_plots = AOsignal.shape[1]
        if num_plots <= 5:
            nrows, ncols = 1, num_plots
        else:
            ncols = 5
            nrows = (num_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        axes = axes.flatten()

        if wave_name is None:
            title = "AO Signal -- all plots"
        else:
            title = f"AO Signal -- {wave_name}"

        fig.suptitle(title, y=0.98)

        for idx in range(num_plots):
            ax = axes[idx]
            ax.plot(time_axis, AOsignal[:, idx])
            ax.set_xlabel("Time (µs)")
            ax.set_ylabel("Value")

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            os.makedirs(save_dir, exist_ok=True)
            save_filename = f"Static_y_Plot{wave_name}_{date_str}.png"
            save_path = os.path.join(save_dir, save_filename)
            plt.savefig(save_path, dpi=200)
            print(f"Saved: {save_path}")

        plt.show()
        plt.close(fig)

    def showAnimatedAll(self, fileOfAcousticField=None, save_dir=None, desired_duration_ms=5000, figsize=(12, 5), wave_name=None):
        mpl.rcParams['animation.embed_limit'] = 100
        pattern_str = StructuredWave.getPattern(fileOfAcousticField)
        angle = StructuredWave.getAngle(fileOfAcousticField)
        fieldToPlot = None

        for field in self.AcousticFields:
            if field.get_path() == fileOfAcousticField:
                fieldToPlot = field
                idx = self.AcousticFields.index(field)
                break
        else:
            raise ValueError(f"Field {fileOfAcousticField} not found in AcousticFields.")

        if wave_name is None:
            wave_name = f"Pattern structure {pattern_str}"

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        if isinstance(axs, plt.Axes):
            axs = np.array([axs])

        fig.suptitle(f"AO Signal Animation {wave_name} | Angle {angle}°", y=0.98)

        axs[0].imshow(self.OpticImage.T, cmap='hot', alpha=1, origin='upper',
                    extent=(self.params.general['Xrange'][0], self.params.general['Xrange'][1], self.params.general['Zrange'][1], self.params.general['Zrange'][0]),
                    aspect='equal')

        im_field = axs[0].imshow(fieldToPlot[0, :, :, idx], cmap='jet', origin='upper',
                                extent=(self.params.general['Xrange'][0], self.params.general['Xrange'][1], self.params.general['Zrange'][1], self.params.general['Zrange'][0]),
                                vmax=1, vmin=0.01, alpha=0.8, aspect='equal')

        axs[0].set_title(f"{wave_name} | Angle {angle}° | t = 0.00 ms")
        axs[0].set_xlabel("x (mm)")
        axs[0].set_ylabel("z (mm)")

        time_axis = np.arange(self.AOsignal.shape[0]) * 25e-6 * 1000
        line_y, = axs[1].plot(time_axis, self.AOsignal[:, idx])
        vertical_line, = axs[1].plot([time_axis[0], time_axis[0]], [0, self.AOsignal[0, idx]], 'r--')

        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Value")
        axs[1].set_title(f"{wave_name} | Angle {angle}° | t = 0.00 ms")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        def update(frame):
            current_time_ms = frame * 25e-6 * 1000
            frame_data = fieldToPlot[frame, :, :, idx]
            masked_data = np.where(frame_data > 0.02, frame_data, np.nan)
            im_field.set_data(masked_data)
            axs[0].set_title(f"{wave_name} | Angle {angle}° | t = {current_time_ms:.2f} ms")

            y_vals = self.AOsignal[:, idx]
            y_copy = np.full_like(y_vals, np.nan)
            y_copy[:frame + 1] = y_vals[:frame + 1]
            line_y.set_data(time_axis, y_copy)

            vertical_line.set_data([time_axis[frame], time_axis[frame]], [0, y_vals[frame]])
            axs[1].set_title(f"{wave_name} | Angle {angle}° | t = {current_time_ms:.2f} ms")

            return [im_field, vertical_line, line_y]

        interval = desired_duration_ms / fieldToPlot.shape[0]
        ani = animation.FuncAnimation(
            fig, update,
            frames=range(0, self.AcousticFields.shape[0]),
            interval=interval, blit=True
        )

        if save_dir is not None:
            now = datetime.now()
            date_str = now.strftime("%Y_%d_%m_%y")
            os.makedirs(save_dir, exist_ok=True)
            save_filename = f"A_y_LAMBDA_overlay_{pattern_str}_{angle}_{date_str}.gif"
            save_path = os.path.join(save_dir, save_filename)
            ani.save(save_path, writer='pillow', fps=20)
            print(f"Saved: {save_path}")

        plt.close(fig)
        return ani

    def showPhantom(self, withROI=False, figsize=(4,4)):
        """
        Displays the optical phantom with absorbers.
        """
        try:
            self.OpticImage.show_phantom(withROI=withROI, figsize=figsize)
        except Exception as e:
            raise RuntimeError(f"Error plotting phantom: {e}")
    
    def showLaser(self, figsize=(4,4)):
        """
        Displays the laser intensity distribution.
        """
        try:
            self.OpticImage.laser.show_laser(figsize=figsize)
        except Exception as e:
            raise RuntimeError(f"Error plotting laser: {e}")
    
    def showMedium(self, figsize=(8,4)):
        """
        Displays the medium properties.
        """
        try:
            self.medium.plot_medium_properties(figsize=figsize)
        except Exception as e:
            raise RuntimeError(f"Error plotting medium: {e}")

    @abstractmethod
    def check(self):
        """
        Check if the experiment is correctly initialized.
        """
        pass
