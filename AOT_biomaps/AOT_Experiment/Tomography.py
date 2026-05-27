from AOT_biomaps.AOT_Acoustic.AcousticTools import format_angle, get_angle, get_frequency
from AOT_biomaps.AOT_Acoustic.AcousticEnums import TypeSim, WaveType
from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Experiment.ExperimentTools import calc_mat_os, convert_to_hex_list, get_phase_deterministic, hex_to_binary_profile, binary_to_hex_profile, load_AOsignal
from AOT_biomaps.AOT_Experiment._mainExperiment import Experiment
import os
import numpy as np
from tqdm import trange
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import h5py
    


class Tomography(Experiment):
    """
    Tomography experiment class for acousto-optic imaging.
    Handles structured wave patterns, acoustic field generation, and signal processing.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patterns = None
        self.theta = []
        self.decimations = []
        self.ActiveList = []
        self.DelayLaw = []

    # PUBLIC METHODS
    def check(self):
        """
        Check if the experiment is correctly initialized.

        Returns:
            tuple: (bool, str) - (True, "success message") if valid, (False, "error message") otherwise
        """
        if self.TypeAcoustic is None or self.TypeAcoustic.value == WaveType.FocusedWave.value:
            return False, "acousticType must be provided and cannot be FocusedWave for Tomography experiment"
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
                return False, f"Field {field.get_name_field()} has an invalid Time shape: {field.field.shape[0]}. Expected time shape to be {self.AOsignal_withTumor.shape[0]}."
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

    def generate_acoustic_fields(self, fieldDataPath=None, show_log=True, nameBlock=None):
        """
        Generate the acoustic fields for simulation.

        Parameters:
            fieldDataPath (str): Path to save the generated fields.
            show_log (bool): Whether to show progress logs.
            nameBlock (str): Optional name for the block when saving.

        Returns:
            list: List of generated FocusedWave objects.
        """
        if self.medium is None:
            raise ValueError("Medium is not initialized. Please generate the medium first.")
        if self.TypeAcoustic.value == WaveType.StructuredWave.value:
            self.AcousticFields = self._generate_acousticFields_STRUCT_CPU(fieldDataPath, show_log, nameBlock)
        else:
            raise ValueError("Unsupported wave type.")

    def show_pattern(self,figsize=(5, 4)):
        """
        Display the transducer activation patterns.
        """
        if self.AcousticFields is None:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")

        # Collect and sort entries
        entries = []
        for field in self.AcousticFields:
            if field.waveType != WaveType.StructuredWave:
                raise TypeError("AcousticFields must be of type StructuredWave to plot pattern.")
            pattern = field.pattern
            entries.append((
                (pattern.space_0, pattern.space_1, pattern.move_head_0_2tail, pattern.move_tail_1_2head),
                pattern.activeList,
                field.angle
            ))

        entries.sort(key=lambda x: (
            -(x[0][0] + x[0][1]),
            -max(x[0][0], x[0][1]),
            -x[0][0],
            -x[0][2],
            x[0][3]
        ))

        # Extract data
        hex_list = [hex_str for _, hex_str, _ in entries]
        angle_list = [angle for _, _, angle in entries]

        # Use hex_to_binary_profile instead of hex_string_to_binary_column
        bit_columns = [hex_to_binary_profile(h, n_piezos=len(h)*4).reshape(-1, 1) for h in hex_list]
        image = np.hstack(bit_columns)

        height, width = image.shape

        # Create figure with compact size
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

        # Plot binary pattern
        im = ax.imshow(image, cmap='binary', aspect='auto', interpolation='none', vmin=0, vmax=1)

        ax.set_title("Scan Configuration", fontsize=12, pad=10, weight='bold')
        ax.set_xlabel("Wave Index", fontsize=10, labelpad=8)
        ax.set_ylabel("Transducer Activation", fontsize=10, labelpad=8)
        yticks_positions = np.arange(0, height)
        yticks_labels = np.arange(1, height + 1)

        ax.set_yticks(yticks_positions)
        ax.set_yticklabels(yticks_labels, fontsize=8)

        # Plot angle markers
        angle_min, angle_max = -20.2, 20.2
        center = height / 2
        scale = height / (angle_max - angle_min)
        for i, angle in enumerate(angle_list):
            y = round(center - angle * scale)
            if 0 <= y < height:
                ax.plot(i, y - 0.5, 'ro', markersize=4, alpha=0.7)

        ax.set_ylim(height - 0.5, -0.5)

        # Twin axis for angles
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        yticks_angle = np.linspace(20, -20, 5)
        yticks_pos = np.interp(yticks_angle, [angle_min, angle_max], [height - 0.5, -0.5])
        ax2.set_yticks(yticks_pos)
        ax2.set_yticklabels([f"{a:.1f}°" for a in yticks_angle], fontsize=9, color='r')
        ax2.set_ylabel("Angle [°]", fontsize=11, color='r', labelpad=10)
        ax2.tick_params(axis='y', colors='r', labelsize=9, width=1.5, length=5)

        # Make axes thicker
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax2.spines['right'].set_linewidth(1.5)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.5)
        ax.set_xticks(np.linspace(0, width-1, 6))
        ax.set_yticks(np.linspace(0, height-1, 6))
        ax.tick_params(axis='both', which='both', labelsize=8, width=1.5, length=4)

        plt.tight_layout()
        plt.show()

    def plot_angle_frequency_distribution(self, figsize=(12, 5)):
        """
        Plot the distribution of angles and spatial frequencies in the patterns.
        """
        if self.patterns is None:
            raise ValueError("patterns is not initialized. Please load or generate the active list first.")

        num_elements = self.params.acoustic['probe']['num_elements']
        # Find all even divisors of num_elements (including num_elements itself)
        divs = sorted([d for d in range(2, num_elements + 1) if num_elements % d == 0 and d % 2 == 0])
        if num_elements not in divs:
            divs.append(num_elements)
        divs.sort()

        angles = []
        freqs = []

        for p in self.patterns:
            # Extract "hexa_XXX" from the dictionary
            file_name = p["fileName"]
            hex_part, angle_str = file_name.split('_')

            # Get the angle
            sign = -1 if angle_str[0] == '1' else 1
            angle = sign * int(angle_str[1:])
            angles.append(angle)

            # Get the spatial frequency
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])
            if np.all(bits == 1):  # All elements active case
                freqs.append(num_elements)
                continue

            for block_size in divs:
                half_block = block_size // 2
                block = np.array([0] * half_block + [1] * half_block)
                reps = num_elements // block_size
                pattern_check = np.tile(block, reps)
                if any(np.array_equal(np.roll(pattern_check, shift), bits) for shift in range(block_size)):
                    freqs.append(block_size)
                    break
            else:
                freqs.append(None)

        freqs = [f for f in freqs if f is not None]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Angle histogram
        axes[0].hist(angles, bins=np.arange(-20.5, 21.5, 1), color='skyblue', edgecolor='black', rwidth=0.8)
        axes[0].set_xlabel("Angle (°)")
        axes[0].set_ylabel("Number of patterns")
        axes[0].set_title("Angle Distribution")
        axes[0].set_xticks(np.arange(-20, 21, 2))

        # Spatial frequency histogram
        unique_freqs, freq_counts = np.unique(freqs, return_counts=True)
        x_pos = np.arange(len(divs))
        for freq, count in zip(unique_freqs, freq_counts):
            idx = divs.index(freq)
            axes[1].bar(x_pos[idx], count, color='salmon', edgecolor='black', width=0.8)

        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(divs)
        axes[1].set_xlabel("Block size (spatial frequency)")
        axes[1].set_ylabel("Number of patterns")
        axes[1].set_title("Spatial Frequency Distribution")

        plt.tight_layout()
        plt.show()

    def load_activeList(self, fieldParamPath):
        """
        Load the active list patterns from a parameter file.

        Parameters:
            fieldParamPath (str): Path to the file containing pattern parameters.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(fieldParamPath):
            raise FileNotFoundError(f"Field parameter file {fieldParamPath} not found.")
        patterns = []
        with open(fieldParamPath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "_" in line and all(c in "0123456789abcdefABCDEF" for c in line.split("_")[0]):
                    patterns.append({"fileName": line})
                    self.theta.append(get_angle(line))
                    profile = hex_to_binary_profile(line.split('_')[0], self.params.acoustic['probe']['num_elements'])
                    self.ActiveList.append(profile)
                    new_Delay = 1000 * (1/self.params.acoustic['medium']['c0']) * np.sin(np.deg2rad(self.theta[-1])) * np.arange(1, self.params.acoustic['probe']['num_elements'] + 1) * self.params.acoustic['probe']['element_width']
                    self.DelayLaw.append(new_Delay - np.min(new_Delay))
                    self.decimations.append(get_frequency(line, self.params.acoustic['probe']['num_elements'], self.params.acoustic['probe']['element_width']))
                    continue
                try:
                    parsed = eval(line, {"__builtins__": None})
                    if isinstance(parsed, tuple) and len(parsed) == 2:
                        coords, angles = parsed
                        for angle in angles:
                            patterns.append({
                                "space_0": coords[0],
                                "space_1": coords[1],
                                "move_head_0_2tail": coords[2],
                                "move_tail_1_2head": coords[3],
                                "angle": angle
                            })
                    else:
                        raise ValueError("Unexpected line (not a tuple of two elements)")
                except Exception as e:
                    print(f"Parsing error on line: {line}\n{e}")
        self.patterns = patterns

    def save_activeList(self, filePath):
        """
        Save the list of patterns to a text file.

        Parameters:
            filePath (str): Path to the output file.
        """
        with open(filePath, 'w') as file:
            for pattern in self.patterns:
                if "fileName" in pattern:
                    # Case 1: Simple pattern (format "hexa_XXX")
                    file.write(f"{pattern['fileName']}\n")
                else:
                    # Case 2: Pattern with parameters (tuple format)
                    coords = (
                        pattern["space_0"],
                        pattern["space_1"],
                        pattern["move_head_0_2tail"],
                        pattern["move_tail_1_2head"]
                    )
                    angles = [pattern["angle"]]
                    line = f"({coords}, {angles})\n"
                    file.write(line)

    def generate_activeList(self, N=None, decimations=None, angles=None):
        """
        Generate a list of balanced and regular activation patterns.

        Parameters:
            N (int): Number of patterns to generate.
            decimations (list): List of decimation factors.
            angles (list): List of angles to use.

        Raises:
            ValueError: If N < 2 and decimations/angles are not provided.
        """
        if decimations is not None and angles is not None:
            self.patterns = self._generate_patterns_from_decimations(decimations, angles)
        elif N is not None and N > 1:
            self.patterns = self._generate_patterns(N)
            if not self._check_patterns(self.patterns):
                raise ValueError("Generated patterns failed validation.")
        else:
            raise ValueError("Either N (>=2) or both decimations and angles must be provided for pattern generation.")

    def save_AOsignals_matlab(self, filePath):
        """
        Save AO signals to a MATLAB .mat file.

        Parameters:
            filePath (str): Path to save the .mat file.
        """
        savemat(filePath, {
            'data': self.AOsignal_withTumor,
            'thetas': self.theta,
            'decimations': self.decimations,
            'ActiveList': self.ActiveList,
            'DelayLaw': self.DelayLaw
        })

    def select_angles(self, angles):
        """
        Select acoustic fields and AO signals based on specified angles.

        Parameters:
            angles (list): List of angles to select.

        Raises:
            ValueError: If AO signals or AcousticFields are not initialized.
        """
        if self.AOsignal_withTumor is None and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signals are not initialized. Please load or generate the AO signals first.")
        if self.AcousticFields is None or len(self.AcousticFields) == 0:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")
        newAcousticFields = []
        index = []
        for i, field in enumerate(self.AcousticFields):
            if field.angle in angles:
                newAcousticFields.append(field)
                index.append(i)
        if self.AOsignal_withTumor is not None:
            self.AOsignal_withTumor = self.AOsignal_withTumor[:, index]
        if self.AOsignal_withoutTumor is not None:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, index]
        self.AcousticFields = newAcousticFields
        self.theta = [field.angle for field in newAcousticFields]
        self.decimations = [field.f_s for field in newAcousticFields]
        self.DelayLaw = [self.DelayLaw[i] for i in index]
        self.ActiveList = [self.ActiveList[i] for i in index]

    def select_shifts(self, shifts):
        """
        Select patterns based on their phase shift parameters.
        Possible values for shifts: "0", "pi/2", "pi", "3pi/2" or "0", "90", "180", "270" (in degrees).

        Parameters:
            shifts (list): List of shift values to select.

        Raises:
            ValueError: If AO signals or AcousticFields are not initialized.
        """
        if self.AOsignal_withTumor is None and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signals are not initialized. Please load or generate the AO signals first.")
        if self.AcousticFields is None or len(self.AcousticFields) == 0:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")

        # Convert shifts to radians if needed
        shift_rads = []
        for shift in shifts:
            if isinstance(shift, str):
                if shift in ["0", "90", "180", "270"]:
                    shift_rads.append(np.deg2rad(int(shift)))
                elif shift in ["0", "pi/2", "pi", "3pi/2"]:
                    shift_rads.append(float(shift.split('/')[0])/2 if '/' in shift else float(shift))
                else:
                    raise ValueError(f"Invalid shift value: {shift}")
            else:
                shift_rads.append(shift)

        newAcousticFields = []
        index = []
        for i, field in enumerate(self.AcousticFields):
            phase = get_phase_deterministic(hex_to_binary_profile(field.get_name_field()[6:-4], self.params.acoustic['probe']['num_elements']))
            if phase in shift_rads:
                newAcousticFields.append(field)
                index.append(i)

        if self.AOsignal_withTumor is not None:
            self.AOsignal_withTumor = self.AOsignal_withTumor[:, index]
        if self.AOsignal_withoutTumor is not None:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, index]

        self.AcousticFields = newAcousticFields
        self.theta = [field.angle for field in newAcousticFields]
        self.decimations = [field.f_s for field in newAcousticFields]
        self.DelayLaw = [self.DelayLaw[i] for i in index]
        self.ActiveList = [self.ActiveList[i] for i in index]

    def select_decimations(self, decimations):
        """
        Select acoustic fields and AO signals based on specified decimation factors.

        Parameters:
            decimations (list): List of decimation factors to select.

        Raises:
            ValueError: If AO signals or AcousticFields are not initialized.
        """
        if self.AOsignal_withTumor is None and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signals are not initialized. Please load or generate the AO signals first.")
        if self.AcousticFields is None or len(self.AcousticFields) == 0:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")
        newAcousticFields = []
        index = []
        for i, field in enumerate(self.AcousticFields):
            if field.f_s in decimations:
                newAcousticFields.append(field)
                index.append(i)
        if self.AOsignal_withTumor is not None:
            self.AOsignal_withTumor = self.AOsignal_withTumor[:, index]
        if self.AOsignal_withoutTumor is not None:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, index]
        self.AcousticFields = newAcousticFields
        self.decimations = [field.f_s for field in newAcousticFields]
        self.theta = [field.angle for field in newAcousticFields]
        self.DelayLaw = [self.DelayLaw[i] for i in index]
        self.ActiveList = [self.ActiveList[i] for i in index]

    def select_patterns(self, pattern_names):
        """
        Select acoustic fields and AO signals based on specified pattern names.

        Parameters:
            pattern_names (list): List of pattern names to select.

        Raises:
            ValueError: If AO signals or AcousticFields are not initialized.
        """
        if self.AOsignal_withTumor is None and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signals are not initialized. Please load or generate the AO signals first.")
        if self.AcousticFields is None or len(self.AcousticFields) == 0:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")
        newAcousticFields = []
        index = []
        for i, field in enumerate(self.AcousticFields):
            if field.pattern.activeList in pattern_names:
                newAcousticFields.append(field)
                index.append(i)
        if self.AOsignal_withTumor is not None:
            self.AOsignal_withTumor = self.AOsignal_withTumor[:, index]
        if self.AOsignal_withoutTumor is not None:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, index]
        self.AcousticFields = newAcousticFields
        self.decimations = [field.f_s for field in newAcousticFields]
        self.theta = [field.angle for field in newAcousticFields]
        self.DelayLaw = [self.DelayLaw[i] for i in index]
        self.ActiveList = [self.ActiveList[i] for i in index]

    def select_random(self, N):
        """
        Randomly select N acoustic fields and corresponding AO signals.

        Parameters:
            N (int): Number of fields to select.

        Raises:
            ValueError: If AO signals or AcousticFields are not initialized, or if N > number of available fields.
        """
        if self.AOsignal_withTumor is None and self.AOsignal_withoutTumor is None:
            raise ValueError("AO signals are not initialized. Please load or generate the AO signals first.")
        if self.AcousticFields is None or len(self.AcousticFields) == 0:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")
        if N > len(self.AcousticFields):
            raise ValueError("N is larger than the number of available AcousticFields.")
        indices = np.random.choice(len(self.AcousticFields), size=N, replace=False)
        newAcousticFields = [self.AcousticFields[i] for i in indices]
        if self.AOsignal_withTumor is not None:
            self.AOsignal_withTumor = self.AOsignal_withTumor[:, indices]
        if self.AOsignal_withoutTumor is not None:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, indices]
        self.AcousticFields = newAcousticFields
        self.decimations = [field.f_s for field in newAcousticFields]
        self.theta = [field.angle for field in newAcousticFields]
        self.DelayLaw = [self.DelayLaw[i] for i in indices]
        self.ActiveList = [self.ActiveList[i] for i in indices]

    def _generate_patterns_from_decimations(self, decimations, angles):
        """
        Generate patterns from specified decimations and angles.

        Parameters:
            decimations (array): Array of decimation factors.
            angles (array): Array of angles in degrees.

        Returns:
            list: List of pattern dictionaries.
        """
        if isinstance(decimations, list):
            decimations = np.array(decimations)
        if isinstance(angles, list):
            angles = np.array(angles)

        angles = np.sort(angles)
        decimations = np.sort(decimations)
        self.DelayLaw = []
        self.theta = []
        self.decimations = []
        self.ActiveList = []
        self.patterns = []

        num_elements = self.params.acoustic['probe']['num_elements']
        Width = self.params.acoustic['probe']['element_width']
        kerf = self.params.acoustic['probe'].get('kerf', 0.00000)
        Nactuators = num_elements

        # ---
        has_zero = 0 in decimations
        if has_zero:
            Nscans = 4 * len(angles) * (len(decimations) - 1) + len(angles)
        else:
            Nscans = 4 * len(angles) * len(decimations)

        ActiveLIST = np.ones((num_elements, Nscans))

        # ---
        Xc = (Width + (Nactuators - 1) * (kerf + Width)) / 2
        Xm = np.array([Width * (i - 1) + Width / 2 - Xc for i in range(1, Nactuators + 1)])

        # ---
        # If there's a 0, modulated patterns start after the plane wave
        # Otherwise, they start at index 0
        current_offset = len(angles) if has_zero else 0

        if has_zero:
            I_plane = np.arange(len(angles))
            ActiveLIST[:, I_plane] = 1

        # ---
        active_decimations = decimations[decimations != 0]
        dFx = 1 / (Nactuators * Width)

        for i_dec in range(len(active_decimations)):
            # Calculate indices relative to start offset
            I = np.arange(len(angles)) + current_offset + (i_dec * 4 * len(angles))

            Icos = I
            Incos = I + 1 * len(angles)
            Insin = I + 2 * len(angles)  # Insin before Isin to match storage order
            Isin = I + 3 * len(angles)

            fx = dFx * active_decimations[i_dec]

            # Apply modulated patterns
            ActiveLIST[:, Icos] = calc_mat_os(Xm, fx, ActiveLIST[:, Icos[:1]], 'cos')
            ActiveLIST[:, Incos] = 1 - ActiveLIST[:, Icos]
            ActiveLIST[:, Isin] = calc_mat_os(Xm, fx, ActiveLIST[:, Isin[:1]], 'sin')
            ActiveLIST[:, Insin] = 1 - ActiveLIST[:, Isin]

        # ---
        hexa_list = convert_to_hex_list(ActiveLIST)

        patterns = []
        print(f"Generating {Nscans} patterns...")
        for i in range(Nscans):
            angle_val = angles[i % len(angles)]
            hex_pattern = hexa_list[i]
            fileName = f"{hex_pattern}_{format_angle(angle_val)}"
            patterns.append({"fileName": fileName})
            self.theta.append(get_angle(fileName))
            profile = hex_to_binary_profile(fileName.split('_')[0], self.params.acoustic['probe']['num_elements'])
            self.ActiveList.append(profile)
            new_Delay = 1000 * (1/self.params.acoustic['medium']['c0']) * np.sin(np.deg2rad(self.theta[-1])) * np.arange(1, self.params.acoustic['probe']['num_elements'] + 1) * self.params.acoustic['probe']['element_width']
            self.DelayLaw.append(new_Delay - np.min(new_Delay))
            self.decimations.append(get_frequency(fileName, self.params.acoustic['probe']['num_elements'], self.params.acoustic['probe']['element_width']))

        return patterns

    def _generate_patterns(self, N, angles=None):
        """
        Generate N random balanced patterns with random angles.

        Parameters:
            N (int): Number of patterns to generate.
            angles (list): Optional list of angles to use. If None, uses range(-20, 21).

        Returns:
            list: List of pattern dictionaries.
        """
        self.DelayLaw = []
        self.theta = []
        self.decimations = []
        self.ActiveList = []
        self.patterns = []

        num_elements = self.params.acoustic['probe']['num_elements']
        if angles is None:
            angle_choices = list(range(-20, 21))
        else:
            if isinstance(angles, np.ndarray):
                angles = angles.tolist()
            angle_choices = angles

        # 1. Find ALL even divisors of num_elements (including num_elements itself)
        divs = [d for d in range(2, num_elements + 1) if num_elements % d == 0 and d % 2 == 0]
        if not divs:
            print(f"No even divisors found for num_elements = {num_elements}")
            return []

        # 2. Use a set to track unique patterns
        unique_patterns = set()

        # 3. Generate until N unique patterns are found
        while len(unique_patterns) < N:
            # Randomly select a divisor (including num_elements)
            block_size = np.random.choice(divs)

            if block_size == num_elements:
                # Special case: "all active" pattern
                pattern_bits = np.ones(num_elements, dtype=int)
            else:
                # General case: balanced pattern
                half_block = block_size // 2
                block = np.array([0] * half_block + [1] * half_block)
                reps = num_elements // block_size
                base_pattern = np.tile(block, reps)
                # Randomly select a shift
                shift = np.random.randint(0, block_size)
                pattern_bits = np.roll(base_pattern, shift)

            # Convert to hex and choose a random angle
            hex_pattern = binary_to_hex_profile(pattern_bits)
            angle = np.random.choice(angle_choices)
            pair = f"{hex_pattern}_{format_angle(angle)}"

            # Add to set (duplicates are automatically ignored)
            unique_patterns.add(pair)

        # 4. Convert to list of dictionaries with "fileName" key
        patterns = [{"fileName": pair} for pair in unique_patterns]
        for i in range(N):
            self.theta.append(get_angle(patterns[i]["fileName"]))
            profile = hex_to_binary_profile(patterns[i]["fileName"].split('_')[0], self.params.acoustic['probe']['num_elements'])
            self.ActiveList.append(profile)
            new_Delay = 1000 * (1/self.params.acoustic['medium']['c0']) * np.sin(np.deg2rad(self.theta[-1])) * np.arange(1, self.params.acoustic['probe']['num_elements'] + 1) * self.params.acoustic['probe']['element_width']
            self.DelayLaw.append(new_Delay - np.min(new_Delay))
            self.decimations.append(get_frequency(patterns[i]["fileName"], self.params.acoustic['probe']['num_elements'], self.params.acoustic['probe']['element_width']))

        # 5. Return exactly N patterns
        return patterns[:N]

    def _check_patterns(self, patterns):
        """
        Check if the patterns are valid (no duplicates, correct length, balanced, regular).

        Parameters:
            patterns (list): List of pattern dictionaries to check.

        Returns:
            bool: True if all patterns are valid, False otherwise.
        """
        # 1. Check for duplicates (based on "fileName")
        file_names = [p["fileName"] for p in patterns]
        if len(file_names) != len(set(file_names)):
            from collections import Counter
            file_counts = Counter(file_names)
            duplicates = [fn for fn, count in file_counts.items() if count > 1]
            for dup in duplicates:
                print(f"Error: Duplicate detected for {dup}")
            return False

        # 2. Check each pattern individually
        num_elements = self.params.acoustic['probe']['num_elements']
        for pattern in patterns:
            hex_part, angle_str = pattern["fileName"].split('_')
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])

            # Check length
            if len(bits) != num_elements:
                print(f"Error length: {pattern['fileName']}")
                return False

            # Special case: "all active" pattern
            if np.all(bits == 1):
                continue

            # Check 0/1 balance
            if np.sum(bits) != num_elements // 2:
                print(f"Error 0/1 balance: {pattern['fileName']}")
                return False

            # Check regularity
            valid = False
            divs = [d for d in range(2, num_elements + 1) if num_elements % d == 0 and d % 2 == 0]
            for block_size in divs:
                half_block = block_size // 2
                block = np.array([0] * half_block + [1] * half_block)
                reps = num_elements // block_size
                expected_pattern = np.tile(block, reps)
                if any(np.array_equal(np.roll(expected_pattern, shift), bits) for shift in range(block_size)):
                    valid = True
                    break
            if not valid:
                print(f"Error regularity: {pattern['fileName']}")
                return False

        return True

    def apply_apodisation(self, alpha=0.3, divergence_deg=0.5):
        """
        Apply dynamic apodization on stored acoustic fields.
        Apodization follows the emission angle and natural beam divergence to
        suppress diffraction lobes (edge artifacts) without affecting the useful signal.

        Parameters:
            alpha (float): Tukey parameter (0.0=rectangle, 1.0=hann). 0.3 is a good compromise.
            divergence_deg (float): Opening angle of the mask to follow beam broadening. 0.0 = Straight, 0.5 = Slight opening (recommended).
        """
        print(f"Applying apodization (Alpha={alpha}, Div={divergence_deg}°) on {len(self.AcousticFields)} fields...")

        probe_width = self.params.acoustic['probe']['num_elements'] * self.params.acoustic['probe']['element_width']

        for i in trange(len(self.AcousticFields), desc="Apodization"):
            # 1. Retrieve data and angle
            field = self.AcousticFields[i].field  # Can be (Z, X) or (Time, Z, X)
            angle = self.AcousticFields[i].angle  # Plane wave angle

            # 2. Retrieve or create physical axes
            nz, nx = field.shape[-2:]

            if hasattr(self, 'x_axis') and self.x_axis is not None:
                x_axis = self.x_axis
            else:
                # Default generation centered on 0 (e.g., -20mm to +20mm)
                x_axis = np.linspace(-probe_width/2, probe_width/2, nx)

            if hasattr(self, 'z_axis') and self.z_axis is not None:
                z_axis = self.z_axis
            else:
                # Default generation (e.g., 0 to 40mm, based on standard pitch or arbitrary)
                estimated_depth = 40e-3
                z_axis = np.linspace(0, estimated_depth, nz)

            # 3. Prepare grids for the mask
            Z, X = np.meshgrid(z_axis, x_axis, indexing='ij')

            # 4. Calculate aligned geometry (Steering)
            angle_rad = np.deg2rad(angle)
            X_aligned = X - Z * np.tan(angle_rad)

            # 5. Calculate dynamic mask width (Divergence)
            div_rad = np.deg2rad(divergence_deg)
            current_half_width = (probe_width / 2.0) + Z * np.tan(div_rad)

            # 6. Normalization and Tukey mask creation
            X_norm = np.divide(X_aligned, current_half_width, out=np.zeros_like(X_aligned), where=current_half_width!=0)

            mask = np.zeros_like(X_norm)
            plateau_threshold = 1.0 * (1 - alpha)

            # Central zone (plateau = 1)
            mask[np.abs(X_norm) <= plateau_threshold] = 1.0

            # Transition zone (cosine)
            transition_indices = (np.abs(X_norm) > plateau_threshold) & (np.abs(X_norm) <= 1.0)
            if np.any(transition_indices):
                x_trans = np.abs(X_norm[transition_indices]) - plateau_threshold
                width_trans = 1.0 * alpha
                mask[transition_indices] = 0.5 * (1 + np.cos(np.pi * x_trans / width_trans))

            # 7. Apply mask (Handle 2D vs 3D)
            if field.ndim == 3:
                field_apodized = field * mask[np.newaxis, :, :]
            else:
                field_apodized = field * mask

            # 8. Update object
            self.AcousticFields[i].field = field_apodized

        print("Apodization done.")

    # PRIVATE METHODS
    def _generate_acousticFields_STRUCT_CPU(self, fieldDataPath=None, show_log=False, nameBlock=None):
        """
        Generate acoustic fields for structured waves using CPU-based simulation.

        Parameters:
            fieldDataPath (str): Path to save generated fields.
            show_log (bool): Whether to show progress logs.
            nameBlock (str): Optional name for the block when saving.

        Returns:
            list: List of generated StructuredWave objects.
        """
        if self.patterns is None:
            raise ValueError("patterns is not initialized. Please load or generate the active list first.")
        listAcousticFields = []
        progress_bar = trange(0, len(self.patterns), desc="Generating acoustic fields")
        for i in progress_bar:
            pattern = self.patterns[i]
            if "fileName" in pattern:
                AcousticField = StructuredWave(fileName=pattern["fileName"], params=self.params, medium=self.medium)
            else:
                AcousticField = StructuredWave(
                    angle_deg=pattern["angle"],
                    space_0=pattern["space_0"],
                    space_1=pattern["space_1"],
                    move_head_0_2tail=pattern["move_head_0_2tail"],
                    move_tail_1_2head=pattern["move_tail_1_2head"],
                    params=self.params,
                    medium=self.medium
                )
            if fieldDataPath is None:
                pathField = None
            else:
                pathField = os.path.join(fieldDataPath, AcousticField.get_name_field() + self.FormatSave.value)
            if pathField is not None and os.path.exists(pathField) and self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
                progress_bar.set_postfix_str(f"Loading field - {AcousticField.get_name_field()}")
                try:
                    AcousticField.load_field(fieldDataPath, self.FormatSave, nameBlock)
                except:
                    progress_bar.set_postfix_str(f"Error loading field -> Generating field - {AcousticField.get_name_field()} ---- processing on {config.get_process().upper()} ----")
                    AcousticField.generate_field(show_log=show_log)
                    if not os.path.exists(pathField):
                        progress_bar.set_postfix_str(f"Saving field - {AcousticField.get_name_field()}")
                        os.makedirs(os.path.dirname(pathField), exist_ok=True)
                        AcousticField.save_field(fieldDataPath)
            else:
                progress_bar.set_postfix_str(f"Generating field - {AcousticField.get_name_field()} ---- processing on {config.get_process().upper()} ----")
                AcousticField.generate_field(show_log=show_log)
                if pathField is not None and not os.path.exists(pathField) and self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
                    progress_bar.set_postfix_str(f"Saving field - {AcousticField.get_name_field()}")
                    os.makedirs(os.path.dirname(pathField), exist_ok=True)
                    AcousticField.save_field(fieldDataPath)
            listAcousticFields.append(AcousticField)
            progress_bar.set_postfix_str("")

        return listAcousticFields

    def load_experimentalAO(self, pathAO, withTumor=True, h5name='AOsignal'):
        """
        Load experimental AO signals from specified file paths.

        Parameters:
            pathAO (str): Path to the AO signal file.
            withTumor (bool): If True, load as signal with tumor. If False, load as signal without tumor.
            h5name (str): Name of the dataset in HDF5/MAT files. Default is 'AOsignal'.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If the dataset is not found in the file.
            ValueError: If the file format is not supported.
        """
        if not os.path.exists(pathAO):
            raise FileNotFoundError(f"File {pathAO} not found.")

        if pathAO.endswith('.npy'):
            AOsignal = np.load(pathAO)
        elif pathAO.endswith('.h5'):
            with h5py.File(pathAO, 'r') as f:
                if h5name not in f:
                    raise KeyError(f"Dataset '{h5name}' not found in the HDF5 file.")
                AOsignal = f[h5name][:]
        elif pathAO.endswith('.mat'):
            mat_data = loadmat(pathAO)
            if h5name not in mat_data:
                raise KeyError(f"Dataset '{h5name}' not found in the .mat file.")
            AOsignal = mat_data[h5name]
        elif pathAO.endswith('.hdr'):
            AOsignal = load_AOsignal(pathAO)
        else:
            raise ValueError("Unsupported file format. Supported formats are: .npy, .h5, .mat, .hdr")

        if withTumor:
            self.AOsignal_withTumor = AOsignal
        else:
            self.AOsignal_withoutTumor = AOsignal

    def check_experimentalAO(self, activeListPath, withTumor=True):
        """
        Check if the experimental AO signals are correctly initialized.

        Parameters:
            activeListPath (str): Path to the active list file for validation.
            withTumor (bool): If True, check signal with tumor. If False, check signal without tumor.

        Raises:
            ValueError: If signals or fields are not properly initialized.
        """
        if withTumor:
            if self.AOsignal_withTumor is None:
                raise ValueError("Experimental AOsignal with tumor is not initialized. Please load the experimental AO signal with tumor first.")
        else:
            if self.AOsignal_withoutTumor is None:
                raise ValueError("Experimental AOsignal without tumor is not initialized. Please load the experimental AO signal without tumor first.")
        if self.AcousticFields is not None:
            if self.AcousticFields[0].field.shape[0] > self.AOsignal_withTumor.shape[0]:
                self.cutAcousticFields(max_t=self.AOsignal_withTumor.shape[0]/float(self.params.acoustic['f_saving']))
            else:
                min_time_shape = min(field.field.shape[0] for field in self.AcousticFields)
                if withTumor:
                    self.AOsignal_withTumor = self.AOsignal_withTumor[:min_time_shape, :]
                else:
                    self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:min_time_shape, :]

            for field in self.AcousticFields:
                if activeListPath is not None:
                    with open(activeListPath, 'r') as file:
                        lines = file.readlines()
                        expected_name = lines[self.AcousticFields.index(field)].strip()
                        nameField = field.get_name_field()
                        if nameField.startswith("field_"):
                            nameField = nameField[len("field_"):]
                        if nameField != expected_name:
                            raise ValueError(f"Field name {nameField} does not match the expected name {expected_name} from the active list.")
        print("Experimental AO signals are correctly initialized.")

    def parse_and_demodulate(self, withTumor=True):
        """
        Parse and demodulate AO signals into complex-valued data.
        Groups signals by (spatial frequency, angle) and applies phase-based demodulation.

        Parameters:
            withTumor (bool): If True, use signals with tumor. If False, use signals without tumor.

        Returns:
            dict: Dictionary with keys (fs, theta) and values as complex arrays.
        """
        if withTumor:
            AOsignal = self.AOsignal_withTumor
        else:
            AOsignal = self.AOsignal_withoutTumor
        delta_x = self.params.general['dx']  # in meters
        n_piezos = self.params.acoustic['probe']['num_elements']
        demodulated_data = {}
        structured_buffer = {}

        for i in trange(AOsignal.shape[1], desc="Demodulating AO signals"):
            hex_pattern = self.patterns[i]["fileName"]
            fs_key = self.decimations[i]
            angle_rad = np.deg2rad(self.theta[i])

            # Plane wave (f_s = 0)
            if fs_key == 0:
                demodulated_data[(fs_key, angle_rad)] = np.array(AOsignal[:,i])
                continue

            # Structured wave
            profile = hex_to_binary_profile(hex_pattern, n_piezos)

            # Calculate spatial frequency (FS)
            ft_prof = np.fft.fft(profile)
            # Only consider positive non-DC part
            idx_max = np.argmax(np.abs(ft_prof[1:len(profile)//2])) + 1
            freqs = np.fft.fftfreq(len(profile), d=delta_x)

            # freqs is in m^-1 because delta_x is in meters
            fs_m_inv = abs(freqs[idx_max])

            # CORRECTION: Convert fs from m^-1 to mm^-1 (mm^-1 is used in iRadon)
            fs_key = float(np.round(fs_m_inv / 1000.0, 5))
            angle_rad = float(np.round(angle_rad, 5))

            if fs_key == 0: continue

            # Calculate Phase (Shift)
            phase = get_phase_deterministic(profile)

            # Store by (fs, theta) and phase
            key = (fs_key, angle_rad)
            if key not in structured_buffer:
                structured_buffer[key] = {}

            # Averaging is needed if multiple acquisitions have the same phase (for SNR)
            if phase in structured_buffer[key]:
                structured_buffer[key][phase] = (structured_buffer[key][phase] + np.array(AOsignal[:,i])) / 2
            else:
                structured_buffer[key][phase] = np.array(AOsignal[:,i])

        for (fs, theta), phases in structured_buffer.items():
            s0 = phases.get(0.0, 0)
            s_pi_2 = phases.get(np.pi/2, 0)
            s_pi = phases.get(np.pi, 0)
            s_3pi_2 = phases.get(3*np.pi/2, 0)

            # Ensure zeros are arrays of the correct size
            example = next(val for val in phases.values() if not isinstance(val, int))
            if isinstance(s0, int): s0 = np.zeros_like(example)
            if isinstance(s_pi, int): s_pi = np.zeros_like(example)
            if isinstance(s_pi_2, int): s_pi_2 = np.zeros_like(example)
            if isinstance(s_3pi_2, int): s_3pi_2 = np.zeros_like(example)

            real = s0 - s_pi
            imag = s_pi_2 - s_3pi_2

            demodulated_data[(fs, theta)] = (real - 1j * imag) / (2/np.pi)

        return demodulated_data

    def demodulate_acoustic_fields(self):
        """
        Demodulate acoustic fields into a flat dictionary: {(fs, theta): complex_field}.
        Identical structure to parse_and_demodulate.

        Returns:
            dict: Dictionary with keys (fs, theta) and values as complex fields.
        """
        n_piezos = self.params.acoustic['probe']['num_elements']
        delta_x = self.params.general['dx']

        # buffer[(fs, theta)][phase] = real field
        buffer = {}

        # 1. Grouping and Averaging
        for i in trange(len(self.AcousticFields), desc="Organizing Acoustic Fields"):
            field_obj = self.AcousticFields[i]
            label = field_obj.get_name_field()
            parts = label.split("_")
            hex_pattern = parts[1]
            angle_code = parts[-1]

            # Extract Angle and Frequency
            angle_deg = -int(angle_code[1:]) if angle_code.startswith("1") else int(angle_code)
            angle_rad = np.round(np.deg2rad(angle_deg), 5)

            if set(hex_pattern.lower().replace(" ", "")) == {'f'}:
                fs_key = 0.0
                phase = 0.0
            else:
                profile = hex_to_binary_profile(hex_pattern, n_piezos)
                ft_prof = np.fft.fft(profile)
                idx_max = np.argmax(np.abs(ft_prof[1:n_piezos//2])) + 1
                freqs = np.fft.fftfreq(n_piezos, d=delta_x)
                fs_key = np.round(abs(freqs[idx_max]) / 1000.0, 5)
                phase = get_phase_deterministic(profile)

            # FLAT KEY (fs, theta)
            key = (fs_key, angle_rad)
            if key not in buffer: buffer[key] = {}

            current_f = field_obj.field
            if phase in buffer[key]:
                buffer[key][phase] = (buffer[key][phase] + current_f) / 2
            else:
                buffer[key][phase] = current_f

        # 2. Quadrature
        demodulated_fields = {}
        keys = list(buffer.keys())

        for i in trange(len(keys), desc="Computing Complex Operator"):
            key = keys[i]  # key is (fs, theta)
            phases = buffer[key]
            fs = key[0]

            if fs == 0.0:
                demodulated_fields[key] = next(iter(phases.values())).astype(np.complex64)
            else:
                s0 = phases.get(0.0)
                s_pi_2 = phases.get(np.pi/2)
                s_pi = phases.get(np.pi)
                s_3pi_2 = phases.get(3*np.pi/2)

                example = next(iter(phases.values()))
                s0 = s0 if s0 is not None else np.zeros_like(example)
                s_pi = s_pi if s_pi is not None else np.zeros_like(example)
                s_pi_2 = s_pi_2 if s_pi_2 is not None else np.zeros_like(example)
                s_3pi_2 = s_3pi_2 if s_3pi_2 is not None else np.zeros_like(example)

                real = s0 - s_pi
                imag = s_pi_2 - s_3pi_2

                # Store with key (fs, theta)
                demodulated_fields[key] = ((real - 1j * imag) / (2/np.pi)).astype(np.complex64)

        print(f"Acoustic Operator complete: {len(demodulated_fields)} configurations processed.")
        return demodulated_fields

    def flip_probe(self, flipPattern=True, flipAngle=True):
        """
        Flip the probe (binary pattern and/or angle) for all acoustic fields and AO signals.

        Parameters:
            flipPattern (bool): If True, reverse the order of active elements in the binary pattern.
            flipAngle (bool): If True, invert the sign of the angle.
        """
        if self.AcousticFields is None:
            print("Warning: AcousticFields is not initialized. No fields to flip, only AO signals.")
            available_fields = False
        else:
            available_fields = True

        num_elements = self.params.acoustic['probe']['num_elements']
        hex_chars_expected = (num_elements + 3) // 4  # Number of hex chars expected for num_elements bits

        new_AcousticFields = []
        new_ActiveList = []
        new_DelayLaw = []
        new_theta = []
        new_decimations = []

        # Flip AO signals if they exist
        if self.AOsignal_withTumor is not None:
            if flipPattern:
                self.AOsignal_withTumor = self.AOsignal_withTumor[:, ::-1]  # Reverse column order (N)
            numScans = self.AOsignal_withTumor.shape[1]
        if self.AOsignal_withoutTumor is not None:
            if flipPattern:
                self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, ::-1]  # Reverse column order (N)
            numScans = self.AOsignal_withoutTumor.shape[1]

        for i in trange(numScans, desc="Flipping AO signals"):
            fileName = self.patterns[i]["fileName"].split('_')[0]  # Extract the hex pattern part
            # Extract the current pattern and angle from the field name
            angle = self.theta[i]

            # Flip the binary pattern if requested
            if flipPattern:
                bits = bin(int(fileName, 16))[2:].zfill(num_elements)
                flipped_bits = bits[::-1]  # Reverse the bits
                flipped_hex = f"{int(flipped_bits, 2):0{hex_chars_expected}x}"
            else:
                flipped_hex = fileName  # Keep the original pattern

            if flipAngle:
                new_angle = -angle  # Invert the angle
            else:
                new_angle = angle  # Keep the original angle

            # Create a new StructuredWave with the flipped pattern and/or angle
            if available_fields:
                new_angle_str = format_angle(new_angle)
                new_field_name = f"field_{flipped_hex}_{new_angle_str}"
                new_field = StructuredWave(
                    fileName=new_field_name,
                    params=self.params,
                    medium=self.medium
                )

            # Copy the field data (flipping columns if flipPattern is True)
            if flipPattern:
                if available_fields:
                    new_field.field = self.AcousticFields[i].field[:, ::-1, :]  # Reverse the order of elements (columns)
            else:
                if available_fields:
                    new_field.field = self.AcousticFields[i].field.copy()  # Keep the original field data
            if available_fields:
                new_AcousticFields.append(new_field)
            new_theta.append(new_angle)

            # Update ActiveList (binary profile)
            new_profile = hex_to_binary_profile(flipped_hex, num_elements)
            new_ActiveList.append(new_profile)

            # Update DelayLaw
            new_Delay = 1000 * (1/self.params.acoustic['medium']['c0']) * np.sin(np.deg2rad(new_angle)) * np.arange(1, num_elements + 1) * self.params.acoustic['probe']['element_width']
            new_DelayLaw.append(new_Delay - np.min(new_Delay))

            # Update decimations (recalculate if pattern is flipped)
            if flipPattern:
                if set(flipped_hex.lower().replace(" ", "")) == {'f'}:
                    fs_key = 0.0  # fs_key in mm^-1 (0.0 mm^-1 for all elements active)
                else:
                    ft_prof = np.fft.fft(new_profile)
                    idx_max = np.argmax(np.abs(ft_prof[1:len(new_profile)//2])) + 1
                    freqs = np.fft.fftfreq(len(new_profile), d=self.params.general['dx'])
                    fs_m_inv = abs(freqs[idx_max])
                    fs_key = fs_m_inv  # Spatial frequency in mm^-1
                new_decimations.append(int(fs_key / (1/(len(new_profile)*self.params.general['dx']))))
            else:
                new_decimations.append(self.AcousticFields[i].f_s)  # Keep the original decimation

        # Update the attributes
        if available_fields:
            self.AcousticFields = new_AcousticFields
        self.ActiveList = new_ActiveList
        self.DelayLaw = new_DelayLaw
        self.theta = new_theta
        self.decimations = new_decimations
        if flipPattern and flipAngle:
            print(f"Flipped both probe and AO signals (pattern and angle).")
        elif flipPattern and not flipAngle:
            print(f"Flipped probe and AO signals (pattern).")
        elif not flipPattern and flipAngle:
            print(f"Flipped probe and AO signals (angle).")
        else:
            print(f"No flipping applied.")