from AOT_biomaps.Config import config
from ._mainAcoustic import AcousticField
from .AcousticEnums import WaveType
from .AcousticTools import detect_space_0_and_space_1, getAngle

import os
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, fileName = None, angle_deg = None, space_0 = None, space_1 = None, move_head_0_2tail = None, move_tail_1_2head = None, **kwargs):
        """
        Initialize the StructuredWave object.

        Args:
            angle_deg (float): Angle in degrees.
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
            self.kgrid.setTime(int(self.kgrid.Nt*1.5),self.kgrid.dt) # Extend the time grid to allow for delays
            if space_0 is not None and space_1 is not None and move_head_0_2tail is not None and move_tail_1_2head is not None and angle_deg is not None:
                self.pattern = self.PatternParams(space_0, space_1, move_head_0_2tail, move_tail_1_2head, self.params['num_elements'] // 4)
                self.angle = angle_deg
                self.pattern.activeList = self.pattern.generate_pattern()
            elif fileName is not None:
                self.pattern = self.PatternParams(0,0,0,0,self.params['num_elements'] // 4)
                self.pattern.space_0, self.pattern.space_1 = detect_space_0_and_space_1(fileName.split('_')[0])
                self.angle = getAngle(fileName)
                self.pattern.activeList = fileName.split('_')[0]
            else:
                raise ValueError("Invalid pattern parameters, must provide either fileName or all space/move parameters.")
            
            self.pattern.len_hex = self.params['num_elements'] // 4
            self.f_s = self._getDecimationFrequency()

            if self.angle < -20 or self.angle > 20:
                raise ValueError("Angle must be between -20 and 20 degrees.")

            if len(self.pattern.activeList) != self.params["num_elements"] // 4:
                raise ValueError(f"Active list string must be {self.params['num_elements'] // 4} characters long.")
            self.delayedSignal = self._apply_delay()
        except Exception as e:
            print(f"Error initializing StructuredWave: {e}")

    def getName_field(self):
        """
        Generate the list of system matrix .hdr file paths for this wave.

        Returns:
            str: File path for the system matrix .hdr file.
        """
        try:
            pattern_str = self.pattern.activeList
            angle_str = self._format_angle()
            return f"field_{pattern_str}_{angle_str}"
        except Exception as e:
            print(f"Error generating file path: {e}")
            return None

    def _getDecimationFrequency(self):
            """
            Calculate the decimation frequency based on the pattern parameters.

            Returns:
                int: Decimation frequency.
            """
            try:
                return 1/(self.pattern.space_0 + self.pattern.space_1)/self.params['element_width']
            except Exception as e:
                print(f"Error calculating decimation frequency: {e}")
                return None

    ## PRIVATE METHODS ##

    def _format_angle(self):
        """
        Format an angle into a 3-digit code like '120' for -20°, '020' for +20°.

        Args:
            angle (float): Angle in degrees.

        Returns:
            str: Formatted angle string.
        """
        return f"{'1' if self.angle < 0 else '0'}{abs(self.angle):02d}"

    def _apply_delay(self,dx=None):
        """
        Apply a temporal delay to the signal for each transducer element.

        Returns:
            ndarray: Array of delayed signals.
        """
        try:
            is_positive = self.angle >= 0
            if dx is None:
                dx = self.params['dx']
            # Calculate the total number of grid points for all elements
            total_grid_points = self.params['num_elements'] * int(round(self.params['element_width'] / dx))

            # Initialize delays array with size total_grid_points
            delays = np.zeros(total_grid_points)

            # Calculate the physical positions of the elements starting from Xrange[0]
            element_positions = np.linspace(0, total_grid_points * dx, total_grid_points)

            # Calculate delays based on physical positions
            for i in range(total_grid_points):
                delays[i] = (element_positions[i] * np.tan(np.deg2rad(abs(self.angle)))) / self.params['c0']  # Delay in seconds


            delay_samples = np.round(delays / self.kgrid.dt).astype(int)
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

    def plot_delay(self):
        """
        Plot the time of the maximum of each delayed signal to visualize the wavefront.
        """
        try:
            # Find the index of the maximum for each delayed signal
            max_indices = np.argmax(self.delayedSignal, axis=1)
            element_indices = np.linspace(0, self.params['num_elements'] - 1, self.delayedSignal.shape[0])
            # Convert indices to time
            max_times = max_indices / self.params['f_AQ']

            # Plot the times of the maxima
            plt.figure(figsize=(10, 6))
            plt.plot(element_indices, max_times, 'o-')
            plt.title('Time of Maximum for Each Delayed Signal')
            plt.xlabel('Transducer Element Index')
            plt.ylabel('Time of Maximum (s)')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting max times: {e}")

    def _save2D_HDR_IMG(self, pathFolder):
        """
        Save the acoustic field to .img and .hdr files.

        Args:
            pathFolder (str): Path to the folder where files will be saved.
        """
        try:
            t_ex = 1 / self.params['f_US']
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
                f"field of view transaxial: {(self.params['Xrange'][1] - self.params['Xrange'][0]) * 1000}\n"
                f"field of view transaxial 2: {(self.params['Zrange'][1] - self.params['Zrange'][0]) * 1000}\n"
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
                f"scaling factor (mm/pixel) [1] := {self.params['dx'] * 1000}\n"
                f"scaling factor (mm/pixel) [2] := {self.params['dx'] * 1000}\n"
                f"scaling factor (s/pixel) [3] := {1 / self.params['f_AQ']}\n"
                f"first pixel offset (mm) [1] := {self.params['Xrange'][0] * 1e3}\n"
                f"first pixel offset (mm) [2] := {self.params['Zrange'][0] * 1e3}\n"
                f"first pixel offset (s) [3] := 0\n"
                f"data rescale offset := 0\n"
                f"data rescale slope := 1\n"
                f"quantification units := 1\n\n"
                f"!SPECIFIC PARAMETERS :=\n"
                f"angle (degree) := {self.angle}\n"
                f"activation list := {''.join(f'{int(self.pattern.activeList[i:i+2], 16):08b}' for i in range(0, len(self.pattern.activeList), 2))}\n"
                f"number of US transducers := {self.params['num_elements']}\n"
                f"delay (s) := 0\n"
                f"us frequency (Hz) := {self.params['f_US']}\n"
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
    
    def _SetUpSource(self, source, Nx, dx, factorT):
        """
        Set up source for both 2D and 3D structured waves.
        """
        active_list = np.array([int(char) for char in ''.join(f"{int(self.pattern.activeList[i:i+2], 16):08b}" for i in range(0, len(self.pattern.activeList), 2))])
        element_width_grid_points = int(round(self.params['element_width'] / dx))

        if source.p_mask.ndim == 2:
            element_width_grid_points = int(round(self.params['element_width'] / dx))
            total_elements_width = self.params['num_elements'] * element_width_grid_points

            # Vérifier que les éléments rentrent dans le grid
            if total_elements_width > Nx:
                raise ValueError(f"La largeur totale des éléments ({total_elements_width}) dépasse Nx ({Nx}).")

            remaining_space = Nx - total_elements_width
            if remaining_space < 0:
                raise ValueError(f"Pas assez d'espace pour placer les éléments: total_elements_width ({total_elements_width}) > Nx ({Nx}).")

            spacing = remaining_space // (self.params['num_elements'] + 1)
            center_index = np.argmin(np.abs(np.linspace(self.params['Xrange'][0], self.params['Xrange'][1], Nx)))

            activeListGrid = np.zeros(total_elements_width, dtype=int)
            current_position = center_index - (total_elements_width + (self.params['num_elements'] - 1) * spacing) // 2

            # Placement des éléments actifs
            for i in range(self.params['num_elements']):
                if active_list[i] == 1:
                    x_pos = max(0, current_position)  # Éviter les indices négatifs
                    x_end = x_pos + element_width_grid_points
                    if x_end > Nx:
                        x_end = Nx  # Limiter à Nx
                    source.p_mask[x_pos:x_end,0] = 1

                    start_idx = i * element_width_grid_points
                    end_idx = start_idx + element_width_grid_points
                    if end_idx > total_elements_width:
                        end_idx = total_elements_width
                    activeListGrid[start_idx:end_idx] = 1

                current_position += element_width_grid_points + spacing

            # Chargement des signaux retardés
            if factorT != 1:
                delayedSignal = self._apply_delay(dx=dx)
            else:
                delayedSignal = self.delayedSignal

            # Vérification de la taille de delayedSignal
            num_active_elements = np.sum(activeListGrid == 1)
            if delayedSignal.shape[0] < num_active_elements:
                raise ValueError(f"delayedSignal a une taille insuffisante: {delayedSignal.shape[0]} < {num_active_elements}.")

            # Assigner source.p
            source.p =  float(self.params['voltage']) * float(self.params['sensitivity']) * delayedSignal[activeListGrid == 1, :]


        elif source.p_mask.ndim == 3:
            # --- 3D ---
            center_index_x = Nx // 2
            center_index_y = self.params['Ny'] // 2
            spacing = (Nx - self.params['num_elements'] * element_width_grid_points) // (self.params['num_elements'] + 1)
            current_position = center_index_x - (self.params['num_elements'] * element_width_grid_points + (self.params['num_elements'] - 1) * spacing) // 2

            for i in range(self.params['num_elements']):
                if active_list[i] == 1:
                    x_pos = current_position
                    source.p_mask[x_pos:x_pos + element_width_grid_points, center_index_y, 0] = 1
                current_position += element_width_grid_points + spacing

            delayed_signals = self._apply_delay()
            source.p =  float(self.params['voltage']) * float(self.params['sensitivity']) * delayed_signals.T