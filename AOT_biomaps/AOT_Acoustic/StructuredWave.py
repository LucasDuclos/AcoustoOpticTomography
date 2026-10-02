import warnings

from AOT_biomaps.Config import config
from ._mainAcoustic import KWAVE_AVAILABLE, AcousticField
from .AcousticEnums import TypeSim, WaveType
from .AcousticTools import detect_space_0_and_space_1, get_angle, get_frequency, format_angle, compute_field_numba, get_piezo_to_grid_mapping

import os
import numpy as np
import matplotlib.pyplot as plt

# Optional cupy import for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from kwave.utils.signals import tone_burst
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False
    warnings.warn("kWave is not available. Some acoustic simulation features will be disabled.", UserWarning)   


class StructuredWave(AcousticField):

    class PatternParams:
        def __init__(self, space_0, space_1, move_head_0_2tail, move_tail_1_2head, len_hex):
            """
            Initialize the PatternParams object with given parameters.

            Args:
                space_0 (int): Number of zeros in the pattern.
                space_1 (int): Number of ones in the pattern.
                move_head_0_2tail (int): Number of zeros to move from head to tail.
                move_tail_1_2head (int): Number of ones to move from tail to head.
            """
            self.space_0 = space_0
            self.space_1 = space_1
            self.move_head_0_2tail = move_head_0_2tail
            self.move_tail_1_2head = move_tail_1_2head
            self.activeList = None
            self.len_hex = len_hex

        def __str__(self):
            """Return a string representation of the PatternParams object."""
            return f"[AOT-biomaps] PatternParams(space_0={self.space_0}, space_1={self.space_1}, move_head_0_2tail={self.move_head_0_2tail}, move_tail_1_2head={self.move_tail_1_2head}, len_hex={self.len_hex})"

        def generate_pattern(self):
            """
            Generate a binary pattern and return it as a hex string.

            Returns:
                str: Hexadecimal representation of the binary pattern.
            """
            try:
                total_bits = self.len_hex * 4
                unit = "0" * self.space_0 + "1" * self.space_1
                repeat_time = (total_bits + len(unit) - 1) // len(unit)
                pattern = (unit * repeat_time)[:total_bits]

                # Move 0s from head to tail
                if self.move_head_0_2tail > 0:
                    head_zeros = '0' * self.move_head_0_2tail
                    pattern = pattern[self.move_head_0_2tail:] + head_zeros

                # Move 1s from tail to head
                if self.move_tail_1_2head > 0:
                    tail_ones = '1' * self.move_tail_1_2head
                    pattern = tail_ones + pattern[:-self.move_tail_1_2head]

                # Convert to hex
                hex_output = hex(int(pattern, 2))[2:]
                hex_output = hex_output.zfill(self.len_hex)

                return hex_output
            except Exception as e:
                print(f"[AOT-biomaps] Error generating pattern: {e}")
                return None
        
        def generate_paths(self, angles, base_path):
            """Generate the list of system matrix .hdr file paths for this wave."""
            pattern_str = self.generate_pattern()
            return [f"{base_path}/field_{pattern_str}_{format_angle(a)}.hdr" for a in angles]

        def to_string(self):
            """
            Format the pattern parameters into a string like '0_48_0_0'.

            Returns:
                str: Formatted string of pattern parameters.
            """
            return f"{self.space_0}_{self.space_1}_{self.move_head_0_2tail}_{self.move_tail_1_2head}"

        def describe(self):
            """
            Return a readable description of the pattern parameters.

            Returns:
                str: Description of the pattern parameters.
            """
            return f"Pattern structure: {self.to_string()}"

    def __init__(self, fileName=None, angle=None, space_0=None, space_1=None,
                 move_head_0_2tail=None, move_tail_1_2head=None, **kwargs):
        """
        Initialize the StructuredWave object.

        Args:
            angle (float): Angle in degrees.
            fileName (str): Name of the file containing the hexadecimal active
                list and the angle (format: activelistHEX_Angle).
            space_0 (int): Number of zeros in the pattern.
            space_1 (int): Number of ones in the pattern.
            move_head_0_2tail (int): Number of zeros to move from head to tail.
            move_tail_1_2head (int): Number of ones to move from tail to head.
            **kwargs: Additional keyword arguments (params, medium).
        """
        try:
            super().__init__(**kwargs)
            self.waveType = WaveType.StructuredWave

            # NOTE (new schema): the old "Nt is None -> setTime(Nt*1.5)"
            # branch is removed. Medium.__init__ ALWAYS computes general['Nt']
            # from the geometry, and silently mutating medium.kgrid here
            # would desynchronize it from the schema. If steered patterns
            # need a longer record, increase the margin in Medium.

            if space_0 is not None and space_1 is not None and move_head_0_2tail is not None and move_tail_1_2head is not None and angle is not None:
                self.pattern = self.PatternParams(space_0, space_1, move_head_0_2tail, move_tail_1_2head, self.params.acoustic['probe']['num_elements'] // 4)
                self.angle = angle
                self.pattern.activeList = self.pattern.generate_pattern()
            elif fileName is not None:
                self.pattern = self.PatternParams(0, 0, 0, 0, self.params.acoustic['probe']['num_elements'] // 4)
                self.pattern.space_0, self.pattern.space_1 = detect_space_0_and_space_1(fileName.split('_')[0])
                self.angle = get_angle(fileName)
                self.pattern.activeList = fileName.split('_')[0]
            else:
                raise ValueError("[AOT-biomaps] Invalid pattern parameters, must provide either fileName or all space/move parameters.")

            self.pattern.len_hex = self.params.acoustic['probe']['num_elements'] // 4

            # The pattern's spatial frequency is a PHYSICAL quantity: it is
            # computed on the SIMULATION grid (dx_sim), where the probe
            # geometry lives. It must not change when the SAVE resolution
            # changes (old schema: dx == dx_sim, values are unchanged).
            dx_sim = float(self.params.acoustic.get('dx_sim', self.params.general['dx']))
            self.f_s = get_frequency(self.pattern.activeList, self.params.acoustic['probe']['num_elements'], dx_sim)

            if len(self.pattern.activeList) != self.params.acoustic['probe']['num_elements'] // 4:
                raise ValueError(f"[AOT-biomaps] Active list string must be {self.params.acoustic['probe']['num_elements'] // 4} characters long.")

        except Exception as e:
            print(f"[AOT-biomaps] Error initializing StructuredWave: {e}")

    def get_name_field(self):
        """
        Generate the list of system matrix .hdr file paths for this wave.

        Returns:
            str: File path for the system matrix .hdr file.
        """
        try:
            pattern_str = self.pattern.activeList
            angle_str = format_angle(self.angle)
            return f"field_{pattern_str}_{angle_str}"
        except Exception as e:
            print(f"[AOT-biomaps] Error generating file path: {e}")
            return None
    
    def plot_delay(self, figsize=(4, 3)):
        """
        Plot the time of the maximum of each delayed signal to visualize the wavefront.
        Times are converted with the SIMULATION time step (medium.kgrid.dt),
        since delayedSignal is built at the simulation sampling rate.
        """
        try:
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

            # Plot the times of the maxima
            plt.figure(figsize=figsize)
            plt.plot(element_indices, max_indices * dt, 'o-')
            plt.title('Time of Maximum for Each Delayed Signal')
            plt.xlabel('Transducer Element Index')
            plt.ylabel('Time of Maximum (s)')
            plt.grid(True)
            plt.show()
        except AttributeError:
            print("[AOT-biomaps] delayedSignal not set yet: run _set_up_source first.")

    def _set_up_source(self, source, burst=None):
        """
        Set up the k-Wave source for the acoustic field simulation.
        All grid/physical parameters are taken from self.medium (SIMULATION
        grid). The source is ALWAYS built at the simulation sampling rate;
        temporal decimation of the saved fields is post-simulation.

        Parameters:
            source: k-Wave source object (p_mask and p will be modified).
            burst (optional): 1D emission waveform, assumed sampled at the
                SIMULATION rate (1/dt). Resample beforehand if not.

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

        active_list = np.array([int(char) for char in ''.join(f"{int(self.pattern.activeList[i:i+2], 16):08b}" for i in range(0, len(self.pattern.activeList), 2))])

        probe_physical_width = (num_elements - 1) * pitch + element_width
        grid_center_x = (Nx * dx) / 2.0
        probe_start_x = grid_center_x - (probe_physical_width / 2.0)

        element_indices = np.arange(num_elements) - (num_elements - 1) / 2.0
        delay_sec = element_indices * pitch * np.sin(np.deg2rad(self.angle)) / c0

        delay_samples = np.round(delay_sec / dt).astype(int)
        delay_samples = delay_samples - np.min(delay_samples) + 10

        if burst is not None:
            burst_sig = np.asarray(burst)
            num_time_steps = len(burst_sig) + np.max(delay_samples) + 20
            element_signals = np.zeros((num_elements, num_time_steps))
            for i in range(num_elements):
                shift = delay_samples[i]
                element_signals[i, shift:shift + len(burst_sig)] = burst_sig
        else:
            element_signals = tone_burst(1 / dt, f_US, num_cycles, signal_offset=delay_samples)

        self.delayedSignal = element_signals

        num_time_steps = element_signals.shape[1]

        el_width_px = int(np.round(element_width / dx))
        if el_width_px < 1:
            el_width_px = 1

        grid_signals = np.zeros((Nx, num_time_steps))
        mappings = get_piezo_to_grid_mapping(
            Nx=Nx,
            dx=dx,
            num_elements=num_elements,
            element_width=element_width,
            pitch=pitch,
            probe_start_x=probe_start_x,
            active_list=active_list
        )

        for (elem_idx, pixel_idx, weight) in mappings:
            source.p_mask[pixel_idx, 0] = True
            grid_signals[pixel_idx, :] += element_signals[elem_idx, :] * weight

        active_indices = np.where(source.p_mask[:, 0])[0]
        source.p = voltage * sensitivity * grid_signals[active_indices, :]

        return source
    
    def _save2D_HDR_IMG(self, pathFolder):
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
            angle_sign = '1' if self.angle < 0 else '0'
            formatted_angle = f"{angle_sign}{abs(self.angle):02d}"
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

            file_name = f"field_{self.pattern.activeList}_{formatted_angle}"
            img_path = os.path.join(pathFolder, file_name + ".img")
            hdr_path = os.path.join(pathFolder, file_name + ".hdr")

            # ------------------------------------------------------------------
            # Snapshot BEFORE opening any file: detaches the data from a
            # possible read-only memmap of the destination .img (load_field).
            # ------------------------------------------------------------------
            field_arr = np.array(self.field, dtype=np.float32, copy=True, order='C')

            # Atomic write: write to a temp file, then swap. If the write
            # fails, the previous .img is untouched.
            tmp_path = img_path + ".tmp"
            with open(tmp_path, "wb") as f_img:
                field_arr.tofile(f_img)
            del field_arr
            os.replace(tmp_path, img_path)

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
                f"angle (degree) := {self.angle}\n"
                f"activation list := {''.join(f'{int(self.pattern.activeList[i:i+2], 16):08b}' for i in range(0, len(self.pattern.activeList), 2))}\n"
                f"number of US transducers := {self.params.acoustic['probe']['num_elements']}\n"
                f"delay (s) := 0\n"
                f"us frequency (Hz) := {self.params.acoustic['f_US']}\n"
                f"excitation duration (s) := {t_ex}\n"
                f"!END OF INTERFILE :=\n"
            )

            with open(hdr_path, "w") as f_hdr:
                f_hdr.write(header)

            glob_path = os.path.join(pathFolder, "field.hdr")
            if not os.path.exists(glob_path):
                with open(glob_path, "w") as f_hdr2:
                    f_hdr2.write(headerFieldGlob)

        except Exception as e:
            print(f"[AOT-biomaps] Error saving HDR/IMG files: {e}")
            raise

    def _generate_acoustic_field_SIMPLE_SIM(self, show_log=False):
        """
        Analytic (Numba) field, produced DIRECTLY on the SAVE grid
        (general.dx/dz) at the SAVE time step (1/ft), so SIMPLE_SIM fields
        are geometrically consistent with the decimated k-Wave fields
        (same Nt', Nz', Nx') and can be mixed in the same S-matrix.

        The internal refinement factor is chosen so that the FINE grid
        stays at dx_sim/4 (the historical internal resolution), regardless
        of the save/sim ratio.

        Parameters:
            show_log (bool): Whether to display simulation logs.

        Returns:
            numpy.ndarray: Simulated acoustic field, shape (Nt', Nz', Nx').
        """
        # --- SAVE grid / cadence (from the schema) ---
        Nx = int(self.params.general['Nx'])          # save grid
        Nz = int(self.params.general['Nz'])
        dx = float(self.params.general['dx'])
        dz = float(self.params.general['dz'])

        # --- simulation record duration (source of truth: medium kgrid) ---
        Nt_sim = int(self.medium.kgrid.Nt)
        dt_sim = float(self.medium.kgrid.dt)
        duration = Nt_sim * dt_sim

        dt_save = getattr(self.medium, 'dt_save', None) or dt_sim
        Nt = max(2, int(np.floor(duration / dt_save)))
        dt = duration / Nt

        # Uniform sampling metadata for the header (same contract as the
        # k-Wave pipeline: last_decimation carries the effective steps)
        self.last_decimation = {'Nt': Nt, 'dt': dt,
                                'Nz': Nz, 'dz': dz,
                                'Nx': Nx, 'dx': dx}

        c0 = float(self.params.acoustic['medium']['c0'])
        f0 = float(self.params.acoustic['f_US'])
        num_cycles = float(self.params.acoustic['emission']['num_cycles'])

        # Internal refinement: keep the fine grid at dx_sim/4 (historical
        # behaviour: old code used dx = dx_sim with factor 4)
        dx_sim = float(self.params.acoustic.get('dx_sim', dx))
        factor = 4 * max(1, int(np.round(dx / dx_sim)))
        Nx_fine, Nz_fine = Nx * factor, Nz * factor
        dx_fine = dx / factor

        # Temporal envelope (Hanning)
        burst_duration = num_cycles / f0
        n_t_burst = int(round(burst_duration / dt))
        enveloppe_t = (np.sin(np.linspace(0, np.pi, n_t_burst)) ** 2).astype(np.float32)

        # Probe setup
        num_elements = int(self.params.acoustic['probe']['num_elements'])

        if self.params.acoustic.get('useApod', False):
            from scipy.signal.windows import tukey
            alpha = np.clip(self.params.acoustic.get('apodStrength', 0.5), 1e-3, 1.0)
            apod_window = tukey(num_elements, alpha=alpha).astype(np.float32)
        else:
            apod_window = np.ones(num_elements, dtype=np.float32)

        active_hex = self.pattern.activeList
        active_list = np.array([int(char) for char in ''.join(f"{int(active_hex[i:i+2], 16):08b}" for i in range(0, len(active_hex), 2))])

        el_width_px_fine = int(round(self.params.acoustic['probe']['element_width'] / dx_fine))
        pva_nx_fine = int(np.round(self.params.acoustic['medium']['width'] / dx_fine))

        # Standard centering
        x_start_probe_fine = ((Nx_fine - pva_nx_fine) // 2) + (pva_nx_fine - (num_elements * el_width_px_fine)) // 2
        x_pivot_px_fine = x_start_probe_fine if self.angle >= 0 else x_start_probe_fine + (num_elements * el_width_px_fine)

        angle_rad = float(np.deg2rad(self.angle))
        cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))

        # Initialization
        field = np.zeros((Nt, Nz, Nx), dtype=np.float32)
        t = (np.arange(Nt) * dt).astype(np.float32)

        active_indices = np.where(active_list == 1)[0].astype(np.int32)
        weight_base = float(1.0 / (factor * factor))

        if show_log:
            print(f"[SIM] Starting parallel Numba computation for {len(active_indices)} active elements...")

        compute_field_numba(
            field, t, active_indices, apod_window, weight_base,
            x_start_probe_fine, x_pivot_px_fine, dx_fine, c0, angle_rad,
            n_t_burst, enveloppe_t, el_width_px_fine, cos_a, sin_a,
            factor, Nt, Nz, Nx, Nx_fine, Nz_fine
        )

        return field
