from ._mainAcoustic import AcousticField
from .AcousticEnums import WaveType

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

try:
    from kwave.utils.signals import tone_burst
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False
    warnings.warn("kWave is not available. Some acoustic simulation features will be disabled.", UserWarning)   

class FocusedWave(AcousticField):
    """
    Class for simulating a focused acoustic wave.
    Applies parabolic time delays to focus the wave at a specified focal line.
    """

    def __init__(self, focal_line, **kwargs):
        """
        Initialize the FocusedWave object.

        Parameters:
            focal_line (float): The x-coordinate of the focal line (in meters).
            **kwargs: Additional keyword arguments for AcousticField initialization.
        """
        super().__init__(**kwargs)
        self.waveType = WaveType.FocusedWave
        self.focal_line = focal_line

    def get_name_field(self):
        """
        Generate the file name for the field based on the focal line.

        Returns:
            str: File name for the field file.
        """
        try:
            return f"field_focused_X{self.focal_line*1000:.2f}_Z{self.params.acoustic['emission']['Foc']*1000:.2f}"
        except Exception as e:
            print(f"[AOT-biomaps] Error generating file name: {e}")
            return None

    def plot_delay(self, figsize=(4,3)):
        """
        Plot the time of the maximum of each delayed signal to visualize the wavefront.
        """
        # Find the index of the maximum for each delayed signal
        max_indices = np.argmax(self.delayedSignal, axis=1)
        element_indices = np.linspace(0, self.params.acoustic['probe']['num_elements'] - 1, self.delayedSignal.shape[0])
        # Convert indices to time (microseconds)
        max_times = max_indices / self.params.acoustic['f_AQ'] * 1e6

        # Determine minimum max time (for active elements)
        min_active_time = np.min(max_times[max_times > 0])

        # Plot the times of the maxima
        plt.figure(figsize=figsize)
        plt.plot(element_indices, max_times, 'o-')
        plt.title('Time of Maximum for Each Delayed Signal')
        plt.xlabel('Transducer Element Index')
        plt.ylabel('Time of Maximum (µs)')
        plt.grid(True)

        # Adjust Y-axis scale to start at minimum active element time
        plt.ylim(bottom=min_active_time * 0.95)  # Add 5% margin for readability
        plt.show()

    def _set_up_source(self, source, Nx, dt, dx, c0, factorT):
        """
        Configure the k-Wave source for a focused wave in 2D.
        Applies parabolic delays and selects active elements around the focal point.

        Args:
            source: k-Wave source object (p_mask and p will be modified).
            Nx (int): Number of grid points in x.
            dt (float): Time step (in seconds).
            dx (float): Spatial step (in meters).
            c0 (float): Speed of sound (in m/s).
            factorT (int): Time downsampling factor.
        """
        num_elements = self.params.acoustic['probe']['num_elements']
        element_width = self.params.acoustic['probe']['element_width']
        element_kerf = self.params.acoustic['probe']['element_kerf']
        pitch = element_width + element_kerf

        f_US = self.params.acoustic['f_US']
        num_cycles = self.params.acoustic['emission']['num_cycles']
        voltage = float(self.params.acoustic['emission']['voltage'])
        sensitivity = float(self.params.acoustic['emission']['sensitivity'])

        focal_z = self.params.acoustic['emission']['Foc']
        focal_x = self.focal_line
        tx_width = focal_z / 2.0 

        probe_physical_width = (num_elements - 1) * pitch + element_width
        grid_center_x = (Nx * dx) / 2.0
        probe_start_x = grid_center_x - (probe_physical_width / 2.0)

        element_x_coords = probe_start_x + np.arange(num_elements) * pitch + (element_width / 2.0)

        distances = np.sqrt((element_x_coords - focal_x)**2 + focal_z**2)

        active_indices = np.where(np.abs(element_x_coords - focal_x) <= tx_width / 2.0)[0]

        if len(active_indices) == 0:
            return source

        active_distances = distances[active_indices]
        delays_sec = (np.max(active_distances) - active_distances) / c0

        delay_samples = np.round(delays_sec / dt).astype(int)
        delay_samples = delay_samples - np.min(delay_samples) + 10

        sampling_freq = 1 / dt
        custom_sig = getattr(self, 'custom_burst', None)

        if custom_sig is not None:
            custom_sig = np.asarray(custom_sig)
            num_active = len(active_indices)
            max_delay = np.max(delay_samples)
            total_len = len(custom_sig) + max_delay + 20
          
            self.burst = np.zeros((num_active, total_len))
            for local_idx in range(num_active):
                shift = delay_samples[local_idx]
                self.burst[local_idx, shift:shift + len(custom_sig)] = custom_sig
        else:
            self.burst = tone_burst(sampling_freq, f_US, num_cycles, signal_offset=delay_samples)

        el_width_px = int(np.round(element_width / dx))
        half_width_px = el_width_px // 2
        active_pixel_signals = []

        for local_idx, global_idx in enumerate(active_indices):
            idx_center = int(np.round(element_x_coords[global_idx] / dx))
            
            idx_start = max(0, idx_center - half_width_px)
            idx_end = min(Nx, idx_start + el_width_px)

            if idx_start < idx_end:
                source.p_mask[idx_start:idx_end, 0] = True
                
                num_pixels_this_element = idx_end - idx_start
                for _ in range(num_pixels_this_element):
                    active_pixel_signals.append(self.burst[local_idx, :])

        source.p = voltage * sensitivity * np.array(active_pixel_signals)
        return source

    def _save2D_HDR_IMG(self, filePath):
        """
        Save the acoustic field to .img and .hdr files.

        Parameters:
            filePath (str): Path to the folder where files will be saved.
        """
        try:
            t_ex = 1 / self.params.acoustic['f_US']
            x_focal = self.focal_line
            z_focal = self.params.acoustic['emission']['Foc']
            file_name = self.getName_field()

            img_path = os.path.join(filePath, file_name + ".img")
            hdr_path = os.path.join(filePath, file_name + ".hdr")

            # Save the acoustic field to the .img file
            with open(img_path, "wb") as f_img:
                self.field.astype('float32').tofile(f_img)

            # Generate global field header
            headerFieldGlob = (
                f"!INTERFILE :=\n"
                f"modality : AOT\n"
                f"voxels number transaxial: {self.field.shape[2]}\n"
                f"voxels number transaxial 2: {self.field.shape[1]}\n"
                f"voxels number axial: {1}\n"
                f"field of view transaxial: {(self.params.general['Xrange'][1] - self.params.general['Xrange'][0]) * 1000}\n"
                f"field of view transaxial 2: {(self.params.general['Zrange'][1] - self.params.general['Zrange'][0]) * 1000}\n"
                f"field of view axial: {1}\n"
            )

            # Generate header
            header = (
                f"!INTERFILE :=\n"
                f"!imaging modality := AOT\n\n"
                f"!GENERAL DATA :=\n"
                f"!data offset in bytes := 0\n"
                f"!name of data file := system_matrix/{file_name}.img\n\n"
                f"!GENERAL IMAGE DATA\n"
                f"!total number of images := {self.field.shape[0]}\n"
                f"imagedata byte order := LITTLEENDIAN\n"
                f"!number of frame groups := 1\n\n"
                f"!STATIC STUDY (General) :=\n"
                f"number of dimensions := 3\n"
                f"!matrix size [1] := {self.field.shape[2]}\n"
                f"!matrix size [2] := {self.field.shape[1]}\n"
                f"!matrix size [3] := {self.field.shape[0]}\n"
                f"!number format := short float\n"
                f"!number of bytes per pixel := 4\n"
                f"scaling factor (mm/pixel) [1] := {self.params.general['dx'] * 1000}\n"
                f"scaling factor (mm/pixel) [2] := {self.params.general['dx'] * 1000}\n"
                f"scaling factor (s/pixel) [3] := {1 / self.params.acoustic['f_AQ']}\n"
                f"first pixel offset (mm) [1] := {self.params.general['Xrange'][0] * 1e3}\n"
                f"first pixel offset (mm) [2] := {self.params.general['Zrange'][0] * 1e3}\n"
                f"first pixel offset (s) [3] := 0\n"
                f"data rescale offset := 0\n"
                f"data rescale slope := 1\n"
                f"quantification units := 1\n\n"
                f"!SPECIFIC PARAMETERS :=\n"
                f"focal point (x, z) := {x_focal}, {z_focal}\n"
                f"number of US transducers := {self.params.acoustic['probe']['num_elements']}\n"
                f"delay (s) := 0\n"
                f"us frequency (Hz) := {self.params.acoustic['f_US']}\n"
                f"excitation duration (s) := {t_ex}\n"
                f"!END OF INTERFILE :=\n"
            )

            # Save the .hdr file
            with open(hdr_path, "w") as f_hdr:
                f_hdr.write(header)

            with open(os.path.join(filePath, "field.hdr"), "w") as f_hdr2:
                f_hdr2.write(headerFieldGlob)
        except Exception as e:
            print(f"[AOT-biomaps] Error saving HDR/IMG files: {e}")

    def _generate_acoustic_field_SIMPLE_SIM(self, show_log=False):
        """
        Generate acoustic field using a simple simulation method.
        (Placeholder for future implementation)
        """
        raise NotImplementedError("[AOT-biomaps] Simple simulation method is not implemented yet.")