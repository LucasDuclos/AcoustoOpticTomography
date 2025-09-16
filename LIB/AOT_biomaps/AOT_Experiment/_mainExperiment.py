from AOT_biomaps.Settings import Params
from AOT_biomaps.AOT_Optic._mainOptic import Phantom
from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType, FormatSave
from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave

from abc import ABC, abstractmethod
import os
import numpy as np
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

class Experiment(ABC):
    def __init__(self, params, acousticType=WaveType.StructuredWave, formatSave=FormatSave.HDR_IMG):
        self.params = params
        self.OpticImage = None
        self.AcousticFields = None
        self.AOsignal_withTumor = None
        self.AOsignal_withoutTumor = None

        if type(acousticType).__name__ != "WaveType":
            raise TypeError("acousticType must be an instance of the WaveType class")

        self.FormatSave = formatSave
        self.TypeAcoustic = acousticType

        if type(self.params) != Params:
            raise TypeError("params must be an instance of the Params class")

    def generatePhantom(self):
        """
        Generate the phantom for the experiment.
        This method initializes the OpticImage attribute with a Phantom instance.
        """
        self.OpticImage = Phantom(params=self.params)

    @abstractmethod
    def generateAcousticFields(self, fieldDataPath, fieldParamPath, show_log=True):
        """
        Generate the acoustic fields for simulation.
        Args:
            fieldDataPath: Path to save the generated fields.
            fieldParamPath: Path to the field parameters file.
        Returns:
            systemMatrix: A numpy array of the generated fields.
        """
        pass

    def cutAcousticFields(self, max_t, min_t=0):
        max_t = float(max_t)
        min_t = float(min_t)

        min_sample = int(np.floor(min_t * float(self.params.acoustic['f_saving'])))
        max_sample = int(np.floor(max_t * float(self.params.acoustic['f_saving'])))

        if min_sample < 0 or max_sample < 0:
            raise ValueError("min_sample and max_sample must be non-negative integers.")
        if min_sample >= max_sample:
            raise ValueError("min_sample must be less than max_sample.")

        if not self.AcousticFields:
            raise ValueError("AcousticFields is empty. Cannot cut fields.")

        for i in trange(len(self.AcousticFields), desc=f"Cutting Acoustic Fields ({min_sample} to {max_sample} samples)"):
            field = self.AcousticFields[i]
            if field.field.shape[0] < max_sample:
                raise ValueError(f"Field {field.getName_field()} has an invalid shape: {field.field.shape}. Expected shape to be at least ({max_sample},).")
            self.AcousticFields[i].field = field.field[min_sample:max_sample, :, :]

    def saveAcousticFields(self, save_directory):
        progress_bar = trange(len(self.AcousticFields), desc="Saving Acoustic Fields")
        for i in progress_bar:
            progress_bar.set_postfix_str(f"-- {self.AcousticFields[i].getName_field()}")
            self.AcousticFields[i].save_field(save_directory, formatSave=self.FormatSave)

    def show_animated_Acoustic(self, wave_name=None, desired_duration_ms=5000, save_dir=None):
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

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.3 * nrows))
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        axes = axes.flatten()
        ims = []

        fig.suptitle(f"System Matrix Animation {wave_name}", fontsize=12, y=0.98)

        for idx in range(num_plots):
            ax = axes[idx]
            im = ax.imshow(self.AcousticFields[0, :, :, idx],
                        extent=(self.params['Xrange'][0], self.params['Xrange'][1], self.params['Zrange'][1], self.params['Zrange'][0]),
                        vmax=1, aspect='equal', cmap='jet', animated=True)
            ax.set_xlabel("x (mm)", fontsize=8)
            ax.set_ylabel("z (mm)", fontsize=8)
            ims.append((im, ax, idx))

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        def update(frame):
            artists = []
            for im, ax, idx in ims:
                im.set_array(self.AcousticFields[frame, :, :, idx])
                fig.suptitle(f"System Matrix Animation {wave_name} t = {frame * 25e-6 * 1000:.2f} ms", fontsize=10)
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

    def generateAOsignal(self, withTumor=True, AOsignalDataPath=None):
        if self.AcousticFields is None:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")

        if self.OpticImage is None:
            raise ValueError("OpticImage is not initialized. Please generate the phantom first.")

        if AOsignalDataPath is not None:
            if not os.path.exists(AOsignalDataPath):
                raise FileNotFoundError(f"AO file {AOsignalDataPath} not found.")
            AOmatrix = self._load_AOSignal(AOsignalDataPath)
            if AOmatrix.shape[0] != self.AcousticFields[0].field.shape[0]:
                print(f"AO signal shape {AOmatrix.shape} does not match the expected shape {self.AcousticFields[0].field.shape}. Generating corrected AO signal to match...")
            else:
                return AOmatrix

        if not all(field.field.shape == self.AcousticFields[0].field.shape for field in self.AcousticFields):
            minShape = min([field.field.shape[0] for field in self.AcousticFields])
            self.cutAcousticFields(minShape * self.params['fs_aq'])
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

    @staticmethod
    def _load_AOSignal(cdh_file):
        with open(cdh_file, "r") as file:
            cdh_content = file.readlines()

        n_events = int([line.split(":")[1].strip() for line in cdh_content if "Number of events" in line][0])
        n_acquisitions = int([line.split(":")[1].strip() for line in cdh_content if "Number of acquisitions per event" in line][0])

        AOsignal_matrix = np.zeros((n_events, n_acquisitions), dtype=np.float32)

        with open(cdh_file.replace(".cdh", ".cdf"), "rb") as file:
            for event in range(n_events):
                num_elements = int([line.split(":")[1].strip() for line in cdh_content if "Number of US transducers" in line][0])
                hex_length = (num_elements + 3) // 4
                file.read(hex_length // 2)

                signal = np.frombuffer(file.read(n_acquisitions * 4), dtype=np.float32)
                AOsignal_matrix[event, :] = signal

        return AOsignal_matrix

    def _saveAOsignals_Castor(self, save_directory, withTumor=True):
        if withTumor:
            cdf_location = os.path.join(save_directory, "AOSignals_withTumor.cdf")
            cdh_location = os.path.join(save_directory, "AOSignals_withTumor.cdh")
        else:
            cdf_location = os.path.join(save_directory, "AOSignals_withoutTumor.cdf")
            cdh_location = os.path.join(save_directory, "AOSignals_withoutTumor.cdh")

        info_location = os.path.join(save_directory, "info.txt")
        nScan = self.AOsignal.shape[1]

        with open(cdf_location, "wb") as fileID:
            for j in range(self.AOsignal.shape[1]):
                active_list = self.AcousticFields[j].pattern.activeList
                angle = self.AcousticFields[j].angle
                active_list_str = ''.join(map(str, active_list))
                nb_padded_zeros = (4 - len(active_list_str) % 4) % 4
                active_list_str += '0' * nb_padded_zeros
                active_list_hex = ''.join([hex(int(active_list_str[i:i+4], 2))[2:] for i in range(0, len(active_list_str), 4)])

                for i in range(0, len(active_list_hex), 2):
                    byte_value = int(active_list_hex[i:i + 2], 16)
                    fileID.write(byte_value.to_bytes(1, byteorder='big'))

                fileID.write(np.int8(angle).tobytes())
                fileID.write(self.AOsignal[:, j].astype(np.float32).tobytes())

        header_content = (
            f"Data filename: AOSignals.cdf\n"
            f"Number of events: {nScan}\n"
            f"Number of acquisitions per event: {self.AOsignal.shape[1]}\n"
            f"Start time (s): 0\n"
            f"Duration (s): 1\n"
            f"Acquisition frequency (Hz): {1/self.AcousticFields[0].kgrid.dt}\n"
            f"Data mode: histogram\n"
            f"Data type: AOT\n"
            f"Number of US transducers: {self.params['num_elements']}"
        )

        with open(cdh_location, "w") as fileID:
            fileID.write(header_content)

        with open(info_location, "w") as fileID:
            for field in self.AcousticFields:
                fileID.write(field.getName_field() + "\n")

        print(f"Fichiers .cdf, .cdh et info.txt sauvegardés dans {save_directory}")

    def show_AOsignal(self, withTumor=True, save_dir=None, wave_name=None):
        if withTumor and self.AOsignal_withTumor is None:
            raise ValueError("AO signal with tumor is not generated. Please generate it first.")
        if not withTumor and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signal without tumor is not generated. Please generate it first.")

        if withTumor:
            AOsignal = self.AOsignal_withTumor
        else:
            AOsignal = self.AOsignal_withoutTumor

        time_axis = np.arange(AOsignal.shape[0]) / float(self.params.acoustic['f_AQ']) * 1e6

        num_plots = AOsignal.shape[1]
        if num_plots <= 5:
            nrows, ncols = 1, num_plots
        else:
            ncols = 5
            nrows = (num_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.3 * nrows))
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        axes = axes.flatten()

        if wave_name is None:
            title = "AO Signal -- all plots"
        else:
            title = f"AO Signal -- {wave_name}"

        fig.suptitle(title, fontsize=12, y=0.98)

        for idx in range(num_plots):
            ax = axes[idx]
            ax.plot(time_axis, AOsignal[:, idx])
            ax.set_xlabel("Time (µs)", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)

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

    def show_animated_all(self, fileOfAcousticField=None, save_dir=None, desired_duration_ms=5000):
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

        fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 5.3 * 1))
        if isinstance(axs, plt.Axes):
            axs = np.array([axs])

        fig.suptitle(f"AO Signal Animation {wave_name} | Angle {angle}°", fontsize=12, y=0.98)

        axs[0].imshow(self.OpticImage.T, cmap='hot', alpha=1, origin='upper',
                    extent=(self.params['Xrange'][0], self.params['Xrange'][1], self.params['Zrange'][1], self.params['Zrange'][0]),
                    aspect='equal')

        im_field = axs[0].imshow(fieldToPlot[0, :, :, idx], cmap='jet', origin='upper',
                                extent=(self.params['Xrange'][0], self.params['Xrange'][1], self.params['Zrange'][1], self.params['Zrange'][0]),
                                vmax=1, vmin=0.01, alpha=0.8, aspect='equal')

        axs[0].set_title(f"{wave_name} | Angle {angle}° | t = 0.00 ms", fontsize=10)
        axs[0].set_xlabel("x (mm)", fontsize=8)
        axs[0].set_ylabel("z (mm)", fontsize=8)

        time_axis = np.arange(self.AOsignal.shape[0]) * 25e-6 * 1000
        line_y, = axs[1].plot(time_axis, self.AOsignal[:, idx])
        vertical_line, = axs[1].plot([time_axis[0], time_axis[0]], [0, self.AOsignal[0, idx]], 'r--')

        axs[1].set_xlabel("Time (ms)", fontsize=8)
        axs[1].set_ylabel("Value", fontsize=8)
        axs[1].set_title(f"{wave_name} | Angle {angle}° | t = 0.00 ms", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        def update(frame):
            current_time_ms = frame * 25e-6 * 1000
            frame_data = fieldToPlot[frame, :, :, idx]
            masked_data = np.where(frame_data > 0.02, frame_data, np.nan)
            im_field.set_data(masked_data)
            axs[0].set_title(f"{wave_name} | Angle {angle}° | t = {current_time_ms:.2f} ms", fontsize=10)

            y_vals = self.AOsignal[:, idx]
            y_copy = np.full_like(y_vals, np.nan)
            y_copy[:frame + 1] = y_vals[:frame + 1]
            line_y.set_data(time_axis, y_copy)

            vertical_line.set_data([time_axis[frame], time_axis[frame]], [0, y_vals[frame]])
            axs[1].set_title(f"{wave_name} | Angle {angle}° | t = {current_time_ms:.2f} ms", fontsize=10)

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

    @abstractmethod
    def check(self):
        """
        Check if the experiment is correctly initialized.
        """
        pass
