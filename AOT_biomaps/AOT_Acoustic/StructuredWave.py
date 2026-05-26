from AOT_biomaps.Config import config
from ._mainAcoustic import AcousticField
from .AcousticEnums import TypeSim, WaveType
from .AcousticTools import detect_space_0_and_space_1, get_angle, get_frequency, format_angle

import os
import numpy as np
import matplotlib.pyplot as plt

# Optional cupy import for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


    


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
            return f"PatternParams(space_0={self.space_0}, space_1={self.space_1}, move_head_0_2tail={self.move_head_0_2tail}, move_tail_1_2head={self.move_tail_1_2head}, len_hex={self.len_hex})"

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
                print(f"Error generating pattern: {e}")
                return None
        
        def generate_paths(self, base_path):
            """Generate the list of system matrix .hdr file paths for this wave."""
            #pattern_str = self.pattern_params.to_string()
            pattern_str = self.generate_pattern()
            paths = []
            for angle in self.angles:
                angle_str = self.format_angle(angle)
                paths.append(f"{base_path}/field_{pattern_str}_{angle_str}.hdr")
            return paths

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

    def __init__(self, fileName = None, angle = None, space_0 = None, space_1 = None, move_head_0_2tail = None, move_tail_1_2head = None, **kwargs):
        """
        Initialize the StructuredWave object.

        Args:
            angle (float): Angle in degrees.
            fileName (str): Name of the file containing the hexadecimal active list and the angle (format : activelisthEXA_Angle)
            space_0 (int): Number of zeros in the pattern.
            space_1 (int): Number of ones in the pattern.
            move_head_0_2tail (int): Number of zeros to move from head to tail.
            move_tail_1_2head (int): Number of ones to move from tail to head.
            **kwargs: Additional keyword arguments.
        """
        try:
            super().__init__(**kwargs)
            self.waveType = WaveType.StructuredWave
            if  self.params.general['Nt'] is None:
                self.medium.kgrid.setTime(int(self.medium.kgrid.Nt*1.5),self.medium.kgrid.dt) # Extend the time grid to allow for delays
            if space_0 is not None and space_1 is not None and move_head_0_2tail is not None and move_tail_1_2head is not None and angle is not None:
                self.pattern = self.PatternParams(space_0, space_1, move_head_0_2tail, move_tail_1_2head, self.params.acoustic['probe']['num_elements'] // 4)
                self.angle = angle
                self.pattern.activeList = self.pattern.generate_pattern()
            elif fileName is not None:
                self.pattern = self.PatternParams(0,0,0,0,self.params.acoustic['probe']['num_elements'] // 4)
                self.pattern.space_0, self.pattern.space_1 = detect_space_0_and_space_1(fileName.split('_')[0])
                self.angle = get_angle(fileName)
                self.pattern.activeList = fileName.split('_')[0]
            else:
                raise ValueError("Invalid pattern parameters, must provide either fileName or all space/move parameters.")
            
            self.pattern.len_hex = self.params.acoustic['probe']['num_elements'] // 4
            self.f_s = get_frequency(self.pattern.activeList, self.params.acoustic['probe']['num_elements'], self.params.general['dx'])

            if len(self.pattern.activeList) != self.params.acoustic['probe']['num_elements'] // 4:
                raise ValueError(f"Active list string must be {self.params.acoustic['probe']['num_elements'] // 4} characters long.")
            if self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
                self.delayedSignal = self._apply_delay()
        except Exception as e:
            print(f"Error initializing StructuredWave: {e}")

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
            print(f"Error generating file path: {e}")
            return None
    
    def plot_delay(self,figsize=(4,3)):
        """
        Plot the time of the maximum of each delayed signal to visualize the wavefront.
        """
        try:
            # Find the index of the maximum for each delayed signal
            max_indices = np.argmax(self.delayedSignal, axis=1)
            element_indices = np.linspace(0, self.params.acoustic['probe']['num_elements'] - 1, self.delayedSignal.shape[0])
            # Convert indices to time
            max_times = max_indices / self.params.acoustic['f_AQ']

            # Plot the times of the maxima
            plt.figure(figsize=figsize)
            plt.plot(element_indices, max_times, 'o-')
            plt.title('Time of Maximum for Each Delayed Signal')
            plt.xlabel('Transducer Element Index')
            plt.ylabel('Time of Maximum (s)')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting max times: {e}")


    ## PRIVATE METHODS ##


    def _apply_delay(self,dt=None,dx=None,c0=None):
        """
        Apply a temporal delay to the signal for each transducer element.

        Returns:
            ndarray: Array of delayed signals.
        """
        try:
            is_positive = self.angle >= 0
            if dx is None:
                dx = self.params.general['dx']
            if c0 is None:
                c0 = self.params.acoustic['medium']['c0']
            actual_dt = dt if dt is not None else self.medium.kgrid.dt
            # Calculate the total number of grid points for all elements
            total_grid_points = self.params.acoustic['probe']['num_elements'] * int(round(self.params.acoustic['probe']['element_width'] / dx))

            # Initialize delays array with size total_grid_points
            delays = np.zeros(total_grid_points)

            # Calculate the physical positions of the elements starting from Xrange[0]
            element_positions = np.linspace(0, total_grid_points * dx, total_grid_points)

            # Calculate delays based on physical positions
            for i in range(total_grid_points):
                delays[i] = (element_positions[i] * np.tan(np.deg2rad(abs(self.angle)))) / c0  # Delay in seconds

            delay_samples = np.round(delays / actual_dt).astype(int)
            max_delay = np.max(np.abs(delay_samples))

            delayed_signals = np.zeros((total_grid_points, len(self.burst) + max_delay))
            for i in range(total_grid_points):
                shift = delay_samples[i]

                if is_positive:
                    delayed_signals[i, shift:shift + len(self.burst)] = self.burst  # Right shift
                else:
                    delayed_signals[i, max_delay - shift:max_delay - shift + len(self.burst)] = self.burst  # Left shift

            return delayed_signals
        except Exception as e:
            print(f"Error applying delay: {e}")
            return None

    def _save2D_HDR_IMG(self, pathFolder):
        """
        Save the acoustic field to .img and .hdr files.

        Args:
            pathFolder (str): Path to the folder where files will be saved.
        """
        try:
            t_ex = 1 / self.params.acoustic['f_US']
            angle_sign = '1' if self.angle < 0 else '0'
            formatted_angle = f"{angle_sign}{abs(self.angle):02d}"

            # Define file names (img and hdr)
            file_name = f"field_{self.pattern.activeList}_{formatted_angle}"

            img_path = os.path.join(pathFolder, file_name + ".img")
            hdr_path = os.path.join(pathFolder, file_name + ".hdr")

            # Save the acoustic field to the .img file
            with open(img_path, "wb") as f_img:
                self.field.astype('float32').tofile(f_img)  # Save in float32 format (equivalent to "single" in MATLAB)

            # Generate headerFieldGlob
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
                f"scaling factor (mm/pixel) [2] := {self.params.general['dz'] * 1000}\n"
                f"scaling factor (s/pixel) [3] := {1 / self.params.acoustic['f_saving']}\n"
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
            # Save the .hdr file
            with open(hdr_path, "w") as f_hdr:
                f_hdr.write(header)

            with open(os.path.join(pathFolder, "field.hdr"), "w") as f_hdr2:
                f_hdr2.write(headerFieldGlob)
        except Exception as e:
            print(f"Error saving HDR/IMG files: {e}")

    def _set_up_source(self, source, Nx, dt, dx, c0, factorT):
        """
        Set up the k-Wave source for the acoustic field simulation.
        Configures the source mask and applies delayed signals to active elements.

        Parameters:
            source: k-Wave source object (p_mask and p will be modified).
            Nx (int): Number of grid points in x.
            dt (float): Time step (in seconds).
            dx (float): Spatial step (in meters).
            c0 (float): Speed of sound (in m/s).
            factorT (int): Time downsampling factor.

        Returns:
            source: Configured k-Wave source object.
        """
        # Convert hex activeList to binary array
        active_list = np.array([int(char) for char in ''.join(f"{int(self.pattern.activeList[i:i+2], 16):08b}" for i in range(0, len(self.pattern.activeList), 2))])

        # Element width in pixels
        el_width_px = int(round(self.params.acoustic['probe']['element_width'] / dx))
        # Total probe width in pixels
        total_probe_px = self.params.acoustic['probe']['num_elements'] * el_width_px

        # Get PVA width in pixels or recalculate
        pva_nx = int(np.round(self.params.acoustic['medium']['width'] / dx))
        air_margin = (Nx - pva_nx) // 2

        # Starting position to center the probe on the PHANTOM
        # Start at the end of the air margin, and center the probe within pva_nx
        current_position = air_margin + (pva_nx - total_probe_px) // 2

        activeListGrid = np.zeros(total_probe_px, dtype=int)

        for i in range(self.params.acoustic['probe']['num_elements']):
            if active_list[i] == 1:
                x_start = current_position
                x_end = x_start + el_width_px

                # Fill k-Wave mask (z=0)
                source.p_mask[x_start:x_end, 0] = 1

                # Mark active indices for delayed signal injection
                idx_start = i * el_width_px
                idx_end = idx_start + el_width_px
                activeListGrid[idx_start:idx_end] = 1

            current_position += el_width_px  # No variable "spacing", elements are contiguous

        # Signal injection
        if factorT != 1:
            delayedSignal = self._apply_delay(dt=dt, dx=dx, c0=c0)
        else:
            delayedSignal = self.delayedSignal

        # Inject only where p_mask == 1
        source.p = float(self.params.acoustic['emission']['voltage']) * float(self.params.acoustic['emission']['sensitivity']) * delayedSignal[activeListGrid == 1, :]
        return source

    def _generate_acoustic_field_SIMPLE_SIM(self, show_log=False):
        """
        Simulate the acoustic field (Nt, Nz, Nx) without internal padding.
        Spatial apodization and temporal envelope are preserved.

        Parameters:
            show_log (bool): Whether to display simulation logs. Default is False.

        Returns:
            numpy.ndarray: Simulated acoustic field with shape (Nt, Nz, Nx).
        """
        # 1. Base parameters (Initial grid)
        Nx = int(self.params.general['Nx'])
        Nz = int(self.params.general['Nz'])
        Nt = int(self.params.general['Nt'] * self.params.acoustic['f_saving'] / self.params.acoustic['f_AQ'])

        dx = self.params.general['dx']  # in meters
        dt = 1 / self.params.acoustic['f_saving']
        c0 = self.params.acoustic['medium']['c0']
        f0 = self.params.acoustic['f_US']
        num_cycles = self.params.acoustic['emission']['num_cycles']

        factor = 4
        Nx_fine, Nz_fine = Nx * factor, Nz * factor
        dx_fine = dx / factor

        # Temporal Envelope (Hanning)
        burst_duration = num_cycles / f0
        n_t_burst = int(round(burst_duration / dt))
        enveloppe_t = np.sin(np.linspace(0, np.pi, n_t_burst))**2

        # ---
        num_elements = self.params.acoustic['probe']['num_elements']
        if self.params.acoustic.get('useApod', False):
            from scipy.signal.windows import tukey
            alpha = np.clip(self.params.acoustic.get('apodStrength', 0.5), 1e-3, 1.0)
            apod_window = tukey(num_elements, alpha=alpha)
        else:
            apod_window = np.ones(num_elements)

        # Probe setup (Centered on the Nx grid)
        active_hex = self.pattern.activeList
        active_list = np.array([int(char) for char in ''.join(f"{int(active_hex[i:i+2], 16):08b}" for i in range(0, len(active_hex), 2))])

        el_width_px_fine = int(round(self.params.acoustic['probe']['element_width'] / dx_fine))
        pva_nx_fine = int(np.round(self.params.acoustic['medium']['width'] / dx_fine))

        # Standard centering
        x_start_probe_fine = ((Nx_fine - pva_nx_fine) // 2) + (pva_nx_fine - (num_elements * el_width_px_fine)) // 2
        x_pivot_px_fine = x_start_probe_fine if self.angle >= 0 else x_start_probe_fine + (num_elements * el_width_px_fine)

        angle_rad = np.deg2rad(self.angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # 2. Initialization and Simulation
        field = np.zeros((Nt, Nz, Nx), dtype=np.float32)
        t = np.arange(Nt) * dt
        active_indices = np.where(active_list == 1)[0]
        weight_base = 1.0 / (factor * factor)

        for i in active_indices:
            val_i = weight_base * apod_window[i]

            x_i_px_fine = x_start_probe_fine + (i * el_width_px_fine)
            dist_to_pivot = (x_i_px_fine - x_pivot_px_fine) * dx_fine
            delay_i = (abs(dist_to_pivot) * np.sin(abs(angle_rad))) / c0

            t_eff = t - delay_i
            mask = (t_eff > 0) & (t_eff < t[-1])
            if not np.any(mask): continue

            dist_travelled = c0 * t_eff[mask]
            t_indices = np.where(mask)[0]

            z_px_fine = np.floor((dist_travelled * cos_a) / dx_fine).astype(int)
            x_px_fine_base = np.floor((x_i_px_fine * dx_fine + dist_travelled * sin_a) / dx_fine).astype(int)

            for b_shift in range(n_t_burst):
                st = t_indices + b_shift
                v_t = st < Nt

                curr_t, curr_z, curr_xb = st[v_t], z_px_fine[v_t], x_px_fine_base[v_t]
                val_final = enveloppe_t[b_shift] * val_i

                for offset_x in range(el_width_px_fine):
                    curr_x = curr_xb + offset_x
                    m = (curr_z >= 0) & (curr_z < Nz_fine) & (curr_x >= 0) & (curr_x < Nx_fine)

                    if np.any(m):
                        zf, xf, tf = curr_z[m]//factor, curr_x[m]//factor, curr_t[m]
                        flat_idx = tf.astype(np.int64) * (Nz * Nx) + zf * Nx + xf
                        np.add.at(field.ravel(), flat_idx, val_final)

        return field