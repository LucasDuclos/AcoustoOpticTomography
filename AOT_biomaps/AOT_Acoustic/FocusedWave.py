from ._mainAcoustic import AcousticField
from .AcousticEnums import WaveType

import os
import numpy as np
import warnings

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
        self.delayedSignal = self._apply_delay()

    def get_name_field(self):
        """
        Generate the file name for the field based on the focal line.

        Returns:
            str: File name for the field file.
        """
        try:
            return f"field_focused_X{self.focal_line*1000:.2f}_Z{self.params.acoustic['emission']['Foc']*1000:.2f}"
        except Exception as e:
            print(f"Error generating file name: {e}")
            return None

    def _apply_delay(self, dt=None, dx=None, c0=None):
        """
        Apply correct parabolic time delays for focusing.
        Elements on the edges are activated FIRST (smaller delays).

        Args:
            dt (float, optional): Time step (in seconds). If None, uses self.medium.kgrid.dt.
            dx (float, optional): Spatial step (in meters). If None, uses self.params.general['dx'].
            c0 (float, optional): Speed of sound (in m/s). If None, uses self.params.acoustic['medium']['c0'].

        Returns:
            numpy.ndarray: Array of delayed signals (shape: [total_grid_points, len(burst) + max_delay]).
        """
        try:
            # 1. Initialize parameters
            if dx is None:
                dx = self.params.general['dx']
            if c0 is None:
                c0 = self.params.acoustic['medium']['c0']
            actual_dt = dt if dt is not None else self.medium.kgrid.dt

            # 2. Grid and element setup
            element_width_grid_points = int(round(self.params.acoustic['probe']['element_width'] / dx))
            total_grid_points = self.params.acoustic['probe']['num_elements'] * element_width_grid_points

            # Physical positions of elements (in meters)
            element_positions = np.linspace(
                self.params.general['Xrange'][0] + self.params.acoustic['probe']['element_width'] / 2,
                self.params.general['Xrange'][1] - self.params.acoustic['probe']['element_width'] / 2,
                self.params.acoustic['probe']['num_elements']
            )

            # 3. Select active elements (focusing)
            N_piezoFocal = self.params.acoustic['emission']['N_piezoFocal']

            center_idx = np.argmin(np.abs(element_positions - self.focal_line))
            start_idx = max(0, center_idx - N_piezoFocal)
            end_idx = min(self.params.acoustic['probe']['num_elements'] - 1, center_idx + N_piezoFocal)
            active_elements = np.arange(start_idx, end_idx + 1)
            active_element_positions = element_positions[active_elements]

            # 4. Calculate CORRECT parabolic delays
            # Elements on the edges should have smaller delays
            # Correct law: delay = (Foc - sqrt(Foc^2 + x_rel^2)) / c0
            x_rel = active_element_positions - self.focal_line
            delays = (self.params.acoustic['emission']['Foc'] - np.sqrt(self.params.acoustic['emission']['Foc']**2 + x_rel**2)) / c0

            # 5. Find maximum delay (absolute value)
            max_delay = np.max(np.abs(delays))

            # 6. Convert to samples
            delay_samples = np.round(delays / actual_dt).astype(int)
            max_delay_samples = np.max(np.abs(delay_samples))

            # 7. Initialize delayed signals array
            delayed_signals = np.zeros((total_grid_points, len(self.burst) + max_delay_samples))

            # 8. Apply delays to active elements
            for elem_idx in active_elements:
                start_grid = elem_idx * element_width_grid_points
                end_grid = start_grid + element_width_grid_points
                elem_delay = delay_samples[elem_idx - start_idx]  # Delay for this element

                # Shift in array: max_delay + elem_delay (handles negative delays)
                shift = max_delay_samples + elem_delay

                for grid_idx in range(start_grid, end_grid):
                    if shift >= 0 and shift + len(self.burst) <= delayed_signals.shape[1]:
                        delayed_signals[grid_idx, shift:shift + len(self.burst)] = self.burst

            return delayed_signals

        except Exception as e:
            print(f"Error applying delays: {e}")
            return None

    def plot_delay(self, figsize=(4,3)):
        """
        Plot the time of the maximum of each delayed signal to visualize the wavefront.
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("matplotlib is not available. Cannot plot delay.", UserWarning)
            return
        try:
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
        except Exception as e:
            print(f"Error plotting max times: {e}")

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
        # Element width in pixels
        el_width_px = int(round(self.params.acoustic['probe']['element_width'] / dx))
        total_probe_px = self.params.acoustic['probe']['num_elements'] * el_width_px

        # Medium (PVA) width in pixels
        pva_nx = int(np.round(self.params.acoustic['medium']['width'] / dx))
        air_margin = (Nx - pva_nx) // 2

        # Starting position to center the probe on the medium
        current_position = air_margin + (pva_nx - total_probe_px) // 2

        # ---
        element_positions = np.linspace(
            self.params.general['Xrange'][0] + self.params.acoustic['probe']['element_width'] / 2,
            self.params.general['Xrange'][1] - self.params.acoustic['probe']['element_width'] / 2,
            self.params.acoustic['probe']['num_elements']
        )

        # Active width and active elements (TxWidth = Foc/2)
        TxWidth = self.params.acoustic['emission']['Foc'] / 2  # in meters
        pitch = self.params.acoustic['probe']['element_width']  # in meters
        N_piezoFocal = int(round(TxWidth / pitch))

        center_idx = np.argmin(np.abs(element_positions - self.focal_line))
        start_idx = max(0, center_idx - N_piezoFocal)
        end_idx = min(self.params.acoustic['probe']['num_elements'] - 1, center_idx + N_piezoFocal)
        active_indices = np.arange(start_idx, end_idx + 1)

        # Active element mask (1D grid)
        activeListGrid = np.zeros(total_probe_px, dtype=int)

        # Configure k-Wave mask and mark active indices
        for i in range(self.params.acoustic['probe']['num_elements']):
            if i in active_indices:
                x_start = current_position
                x_end = x_start + el_width_px
                source.p_mask[x_start:x_end, 0] = 1  # Activate in p_mask

                # Mark indices for signal injection
                idx_start = i * el_width_px
                idx_end = idx_start + el_width_px
                activeListGrid[idx_start:idx_end] = 1

            current_position += el_width_px

        # Signal injection (parabolic delays)
        if factorT != 1:
            delayedSignal = self._apply_delay(dt=dt, dx=dx, c0=c0)
        else:
            delayedSignal = self.delayedSignal

        # Apply signal only to active elements
        amplitude = float(self.params.acoustic['emission']['voltage']) * float(self.params.acoustic['emission']['sensitivity'])
        source.p = amplitude * delayedSignal[activeListGrid == 1, :]

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
            print(f"Error saving HDR/IMG files: {e}")

    def _generate_acoustic_field_SIMPLE_SIM(self, show_log=False):
        """
        Generate acoustic field using a simple simulation method.
        (Placeholder for future implementation)
        """
        raise NotImplementedError("Simple simulation method is not implemented yet.")