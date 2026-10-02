from ._mainAcoustic import AcousticField
from .AcousticEnums import WaveType
from .AcousticTools import get_piezo_to_grid_mapping

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
    Applies geometric time delays to focus the wave at a specified focal line.
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

    def plot_delay(self, figsize=(4, 3)):
        """
        Plot the time of the maximum of each delayed signal to visualize the
        wavefront. Times are converted with the SIMULATION time step
        (medium.kgrid.dt), since delayedSignal is built at the simulation
        sampling rate. (f_AQ may be the string "AUTO" before Medium
        resolution, so it is only used as a fallback if numeric.)
        """
        # Find the index of the maximum for each delayed signal
        max_indices = np.argmax(self.delayedSignal, axis=1)
        element_indices = np.linspace(0, self.params.acoustic['probe']['num_elements'] - 1, self.delayedSignal.shape[0])

        # Simulation time step (source of truth)
        if getattr(self, 'medium', None) is not None:
            dt = float(self.medium.kgrid.dt)
        else:
            f_aq = self.params.acoustic.get('f_AQ')
            dt = (1.0 / float(f_aq)) if isinstance(f_aq, (int, float)) else None
        if dt is None:
            print("[AOT-biomaps] No time step available (medium missing and f_AQ='AUTO').")
            return

        # Convert indices to time (microseconds)
        max_times = max_indices * dt * 1e6

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

    def _set_up_source(self, source, burst=None):
        """
        Configure the k-Wave source for a focused wave in 2D.
        All grid/physical parameters are taken from self.medium (SIMULATION
        grid). The source is ALWAYS built at the simulation sampling rate;
        temporal decimation of the saved fields is post-simulation.

        Applies geometric (distance-based) delays to focus the wave at the
        focal line, selects the active elements around the focal line
        (aperture = Foc/2), and maps elements to grid pixels with exact
        fractional coverage (get_piezo_to_grid_mapping), like StructuredWave.

        Args:
            source: k-Wave source object (p_mask and p will be modified).
            burst (optional): 1D emission waveform, assumed sampled at the
                SIMULATION rate (1/dt). Falls back to self.custom_burst.
                If both are None, a tone_burst at f_US is generated.

        Returns:
            source: Configured k-Wave source object.
        """
        Nx = int(self.medium.Nx_reshaped)
        dx = float(self.medium.dx_reshaped)
        dt = float(self.medium.kgrid.dt)
        c0 = float(self.medium.c_mean)

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

        # Geometric distances to the focal point
        distances = np.sqrt((element_x_coords - focal_x) ** 2 + focal_z ** 2)

        # Active aperture: elements within tx_width/2 of the focal line
        active_mask = (np.abs(element_x_coords - focal_x) <= tx_width / 2.0).astype(int)
        if not np.any(active_mask):
            return source

        # Delays: the farthest active element emits first
        delays_sec = (np.max(distances[active_mask == 1]) - distances) / c0
        delay_samples = np.round(np.maximum(delays_sec, 0.0) / dt).astype(int)
        delay_samples = delay_samples - np.min(delay_samples[active_mask == 1]) + 10

        # Emission waveform: explicit burst > custom_burst > tone_burst
        burst_sig = burst if burst is not None else getattr(self, 'custom_burst', None)

        if burst_sig is not None:
            burst_sig = np.asarray(burst_sig)
            num_time_steps = len(burst_sig) + int(np.max(delay_samples)) + 20
            element_signals = np.zeros((num_elements, num_time_steps))
            for i in range(num_elements):
                shift = int(delay_samples[i])
                element_signals[i, shift:shift + len(burst_sig)] = burst_sig
        else:
            element_signals = tone_burst(1 / dt, f_US, num_cycles, signal_offset=delay_samples)

        # Kept for plot_delay (times are in SIMULATION sampling)
        self.delayedSignal = element_signals

        # Element -> grid pixel mapping with fractional coverage
        grid_signals = np.zeros((Nx, element_signals.shape[1]))
        mappings = get_piezo_to_grid_mapping(
            Nx=Nx,
            dx=dx,
            num_elements=num_elements,
            element_width=element_width,
            pitch=pitch,
            probe_start_x=probe_start_x,
            active_list=active_mask,
        )

        for (elem_idx, pixel_idx, weight) in mappings:
            source.p_mask[pixel_idx, 0] = True
            grid_signals[pixel_idx, :] += element_signals[elem_idx, :] * weight

        # source.p rows MUST follow the p_mask pixel order (np.where)
        active_pixels = np.where(source.p_mask[:, 0])[0]
        source.p = voltage * sensitivity * grid_signals[active_pixels, :]

        return source

    def _save2D_HDR_IMG(self, filePath):
        """
        Save the acoustic field to .img and .hdr files.
        The header stores the EFFECTIVE sampling of self.field
        (self.last_decimation), falling back to the parameter schema
        (general.dx/dz and 1/general.ft).

        Safe for save-after-load cycles: if self.field is a read-only
        memmap of the SAME .img path, opening that path in 'wb' mode would
        truncate the file underneath the mapping ("N requested and 0
        written"). The data is therefore snapshotted to RAM BEFORE any file
        is opened, and the .img is written atomically (tmp + os.replace).
        """
        try:
            t_ex = 1 / self.params.acoustic['f_US']
            x_focal = self.focal_line
            z_focal = self.params.acoustic['emission']['Foc']
            file_name = self.getName_field()

            dec = getattr(self, 'last_decimation', None)

            # Effective sampling of self.field
            if dec is not None:
                dx_mm = dec['dx'] * 1000
                dz_mm = dec['dz'] * 1000
                dt_s = dec['dt']
            else:
                dx_mm = self.params.general['dx'] * 1000
                dz_mm = self.params.general['dz'] * 1000
                dt_s = (1.0 / float(self.params.general['ft'])) if self.params.general.get('ft') else None
            time_scaling = f"{dt_s}" if dt_s is not None else "1"

            img_path = os.path.join(filePath, file_name + ".img")
            hdr_path = os.path.join(filePath, file_name + ".hdr")

            # ------------------------------------------------------------------
            # Snapshot BEFORE opening any file: detaches the data from a
            # possible read-only memmap of the destination .img (load_field).
            # np.array(copy=True), NOT np.asarray (which would be a zero-copy
            # view of the memmap and keep the bug).
            # ------------------------------------------------------------------
            field_arr = np.array(self.field, dtype=np.float32, copy=True, order='C')

            # Atomic write: write to a temp file, then swap. If the write
            # fails, the previous .img is untouched.
            tmp_path = img_path + ".tmp"
            with open(tmp_path, "wb") as f_img:
                field_arr.tofile(f_img)
            del field_arr
            os.replace(tmp_path, img_path)

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
                f"scaling factor (mm/pixel) [1] := {dx_mm}\n"
                f"scaling factor (mm/pixel) [2] := {dz_mm}\n"
                f"scaling factor (s/pixel) [3] := {time_scaling}\n"
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

            glob_path = os.path.join(filePath, "field.hdr")
            if not os.path.exists(glob_path):
                with open(glob_path, "w") as f_hdr2:
                    f_hdr2.write(headerFieldGlob)
        except Exception as e:
            print(f"[AOT-biomaps] Error saving HDR/IMG files: {e}")
            raise

    def _generate_acoustic_field_SIMPLE_SIM(self, show_log=False):
        """
        Generate acoustic field using a simple simulation method.
        (Placeholder for future implementation)
        """
        raise NotImplementedError("[AOT-biomaps] Simple simulation method is not implemented yet.")