from AOT_biomaps.AOT_Acoustic.AcousticTools import format_angle, get_angle, get_frequency
from AOT_biomaps.AOT_Acoustic.AcousticEnums import TypeSim, WaveType
from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Experiment.ExperimentTools import calc_mat_os, convert_to_hex_list, get_phase_deterministic, hex_to_binary_profile, binary_to_hex_profile, load_AOsignal
from AOT_biomaps.AOT_Experiment._mainExperiment import Experiment
import os
import concurrent.futures
from tqdm import tqdm
from tqdm import trange
import numpy as np
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
        """
        if self.TypeAcoustic is None or self.TypeAcoustic.value == WaveType.FocusedWave.value:
           raise ValueError("[AOT-biomaps] acousticType must be provided and cannot be FocusedWave for Tomography experiment")
        if self.AcousticFields is None:
            raise ValueError("[AOT-biomaps] AcousticFields is not initialized. Please generate the system matrix first.")
        if self.AOsignal_withTumor is None:
            raise ValueError("[AOT-biomaps] AOsignal with tumor is not initialized. Please generate the AO signal with tumor first.")
        if self.AOsignal_withoutTumor is None:
            raise ValueError("[AOT-biomaps] AOsignal without tumor is not initialized. Please generate the AO signal without tumor first.")
        if self.OpticImage is None:
            raise ValueError("[AOT-biomaps] OpticImage is not initialized. Please generate the optic image first.")
        if self.AOsignal_withoutTumor.shape != self.AOsignal_withTumor.shape:
            raise ValueError("[AOT-biomaps] AOsignal with and without tumor must have the same shape.")
        for field in self.AcousticFields:
            if field.field.shape[0] != self.AOsignal_withTumor.shape[0]:
                raise ValueError(f"[AOT-biomaps] Field {field.get_name_field()} has an invalid Time shape: {field.field.shape[0]}. Expected time shape to be {self.AOsignal_withTumor.shape[0]}.")
        if not all(field.field.shape == self.AcousticFields[0].field.shape for field in self.AcousticFields):
            raise ValueError("[AOT-biomaps] All AcousticFields must have the same shape.")
        if self.OpticImage is None:
            raise ValueError("[AOT-biomaps] OpticImage is not initialized. Please generate the optic image first.")
        if self.OpticImage.phantom is None:
            raise ValueError("[AOT-biomaps] OpticImage phantom is not initialized. Please generate the phantom first.")
        if self.OpticImage.laser is None:
            raise ValueError("[AOT-biomaps] OpticImage laser is not initialized. Please generate the laser first.")
        if self.OpticImage.laser.shape != self.OpticImage.phantom.shape:
            raise ValueError("[AOT-biomaps] OpticImage laser and phantom must have the same shape.")
        if self.OpticImage.phantom.shape[0] != self.AcousticFields[0].field.shape[1] or self.OpticImage.phantom.shape[1] != self.AcousticFields[0].field.shape[2]:
            raise ValueError(f"[AOT-biomaps] OpticImage phantom shape {self.OpticImage.phantom.shape} does not match AcousticFields shape {self.AcousticFields[0].field.shape[1:]}.")

        print("[AOT-biomaps] Experiment is correctly initialized.")

    def generate_acoustic_fields(self, isGPU=None, GPUdevice=None, fieldDataPath=None, tempFieldName="Kwave", nameBlock=None, generation_type="envelope_squared", show_log=True, max_workers=None, backend=None, materialize=False):
        """
        Generate the acoustic fields for simulation.

        Parameters:
            isGPU (bool): Whether to use GPU for simulation. (If None, the default setting will be used.)
            GPUdevice (int): The GPU device to use. (If None, the default GPU will be used.)
            fieldDataPath (str): Path to save the generated fields.
            tempFieldName (str): Name for the temporary field files. Mainly used for multithreading to avoid multiple threads writing to the same file.
            nameBlock (str): Optional name for h5 file.
            generation_type (str): The type of field generation to perform. Must be one of "envelope_squared", "envelope", or "field".
            show_log (bool): Whether to show progress logs.
            max_workers (int): Max PARALLEL GENERATIONS. Default: one worker per GPU (GPU mode) or 4 (CPU mode).
            backend (str): "python" (default) or "cpp". Use "cpp" with a GPUdevice LIST for true multi-GPU scaling.
            materialize (bool): If True, promote loaded fields from file-backed memmap (or views over a memmap) to resident anonymous RAM ndarrays, guarded by a RAM budget check (80% of MemAvailable). 
                The disk cost is paid once, visibly, during the loading step; every downstream pass (e.g. generate_AOsignal) then runs at RAM bandwidth and becomes immune to page-cache eviction by unrelated I/O. Recommended for analysis/reconstruction sessions.
                Leave False for one-shot passes or oversized SetMixtes.

        Returns:
            list: List of generated FocusedWave objects.
        """
        if self.medium is None:
            raise ValueError("[AOT-biomaps] Medium is not initialized. Please generate the medium first.")
        if self.TypeAcoustic.value == WaveType.StructuredWave.value:
            self.AcousticFields = self._generate_acousticFields_STRUCT(isGPU=isGPU, GPUdevice=GPUdevice, fieldDataPath=fieldDataPath, tempFieldName=tempFieldName, nameBlock=nameBlock, generation_type=generation_type, show_log=show_log, max_workers=max_workers, backend=backend, materialize=materialize)
        else:
            raise ValueError("[AOT-biomaps] Unsupported wave type.")

    def show_pattern(self,figsize=(5, 4)):
        """
        Display the transducer activation patterns.
        """
        if self.AcousticFields is None:
            raise ValueError("[AOT-biomaps] AcousticFields is not initialized. Please generate the system matrix first.")

        # Collect and sort entries
        entries = []
        for field in self.AcousticFields:
            if field.waveType != WaveType.StructuredWave:
                raise TypeError("[AOT-biomaps] AcousticFields must be of type StructuredWave to plot pattern.")
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
            raise ValueError("[AOT-biomaps] patterns is not initialized. Please load or generate the active list first.")

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
            raise FileNotFoundError(f"[AOT-biomaps] Field parameter file {fieldParamPath} not found.")
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
                        raise ValueError("[AOT-biomaps] Unexpected line (not a tuple of two elements)")
                except Exception as e:
                    print(f"[AOT-biomaps] Parsing error on line: {line}\n{e}")
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
                raise ValueError("[AOT-biomaps] Generated patterns failed validation.")
        else:
            raise ValueError("[AOT-biomaps] Either N (>=2) or both decimations and angles must be provided for pattern generation.")

    def generate_activeList_from_exp(self):
        if self.expParams is None:
            raise ValueError("[AOT-biomaps] expParams is not initialized. Please load the experiment data first.")
        active_elements = convert_to_hex_list(self.expParams['ActiveListMatrix'])
        self.DelayLaw = []
        self.theta = []
        self.decimations = []
        self.ActiveList = []
        self.patterns = []
        for i in range(len(active_elements)):
            self.patterns.append({"fileName": f"{active_elements[i]}_{format_angle(self.expParams['AngleMatrix'][i])}"})
            self.theta.append(self.expParams['AngleMatrix'][i])
            self.ActiveList = active_elements
            new_Delay = 1000 * (1/self.params.acoustic['medium']['c0']) * np.sin(np.deg2rad(self.theta[-1])) * np.arange(1, self.params.acoustic['probe']['num_elements'] + 1) * self.params.acoustic['probe']['element_width']
            self.DelayLaw.append(new_Delay - np.min(new_Delay))
            self.decimations.append(get_frequency(active_elements[i], self.params.acoustic['probe']['num_elements'], self.params.acoustic['probe']['element_width']))

    def check_ActiveList(self, activeList_path):
        with open(activeList_path, 'r') as f:
            for i,line in enumerate(f):
                if line.strip() != self.patterns[i]["fileName"]:
                    print(f"[AOT-biomaps] Mismatch at line {i+1}: file has '{line.strip()}', but generated list has '{self.patterns[i]['fileName']}'")
                    return False    
        return True

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
        Select acoustic fields and patterns based on specified angles.
        Works even if AcousticFields or AO signals are None.
        """
        if self.patterns is None:
            raise ValueError("[AOT-biomaps] patterns is not initialized.")

        index = []
        for i, p in enumerate(self.patterns):
            angle = get_angle(p["fileName"]) if "fileName" in p else p.get("angle")
            if angle in angles:
                index.append(i)

        self._apply_selection_indices(index)

    def select_shifts(self, shifts):
        """
        Select patterns based on their phase shift parameters (in radians or degrees).
        """
        if self.patterns is None:
            raise ValueError("[AOT-biomaps] patterns is not initialized.")

        # Convert shifts to radians if needed
        shift_rads = []
        for shift in shifts:
            if isinstance(shift, str):
                if shift in ["0", "90", "180", "270"]:
                    shift_rads.append(np.deg2rad(int(shift)))
                elif shift in ["0", "pi/2", "pi", "3pi/2"]:
                    shift_rads.append(float(shift.split('/')[0])/2 if '/' in shift else float(shift))
                else:
                    raise ValueError(f"[AOT-biomaps] Invalid shift value: {shift}")
            else:
                shift_rads.append(shift)

        n_piezos = self.params.acoustic['probe']['num_elements']
        index = []
        for i, p in enumerate(self.patterns):
            hex_part = p["fileName"].split('_')[0] if "fileName" in p else None
            if hex_part:
                profile = hex_to_binary_profile(hex_part, n_piezos)
                phase = get_phase_deterministic(profile)
                if any(np.isclose(phase, sr, atol=1e-3) for sr in shift_rads):
                    index.append(i)

        self._apply_selection_indices(index)

    def select_decimations(self, decimations):
        """
        Select patterns based on decimation factors (spatial frequencies).
        """
        if self.patterns is None:
            raise ValueError("[AOT-biomaps] patterns is not initialized.")

        n_piezos = self.params.acoustic['probe']['num_elements']
        width = self.params.acoustic['probe']['element_width']
        
        index = []
        for i, p in enumerate(self.patterns):
            if "fileName" in p:
                f_s = get_frequency(p["fileName"], n_piezos, width)
                if f_s in decimations:
                    index.append(i)

        self._apply_selection_indices(index)

    def select_patterns(self, pattern_names):
        """
        Select patterns based on a specified list of file names or identifiers.
        """
        if self.patterns is None:
            raise ValueError("[AOT-biomaps] patterns is not initialized.")

        index = []
        for i, p in enumerate(self.patterns):
            fname = p.get("fileName")
            if fname in pattern_names:
                index.append(i)

        self._apply_selection_indices(index)

    def select_random(self, N):
        """
        Randomly select N patterns and associated data.
        """
        if self.patterns is None:
            raise ValueError("[AOT-biomaps] patterns is not initialized.")
        if N > len(self.patterns):
            raise ValueError("[AOT-biomaps] N is larger than the number of available patterns.")

        indices = np.random.choice(len(self.patterns), size=N, replace=False)
        indices = sorted(indices.tolist())
        self._apply_selection_indices(indices)

    def _apply_selection_indices(self, index):
        """
        Internal method to filter all class attributes according to a list of indices.
        Gracefully handles potentially None attributes.
        """
        # 1. Filter base pattern attributes
        self.patterns = [self.patterns[i] for i in index]
        self.theta = [self.theta[i] for i in index] if self.theta else []
        self.decimations = [self.decimations[i] for i in index] if self.decimations else []
        self.ActiveList = [self.ActiveList[i] for i in index] if self.ActiveList else []
        self.DelayLaw = [self.DelayLaw[i] for i in index] if self.DelayLaw else []

        # 2. Filter AcousticFields if initialized
        if self.AcousticFields is not None:
            if len(self.AcousticFields) >= max(index, default=-1) + 1:
                self.AcousticFields = [self.AcousticFields[i] for i in index]
            else:
                self.AcousticFields = None
                print("[AOT-biomaps] Warning: AcousticFields has been reset because indices no longer match.")

        # 3. Filter AO signals if initialized
        if self.AOsignal_withTumor is not None:
            self.AOsignal_withTumor = self.AOsignal_withTumor[:, index]
        if self.AOsignal_withoutTumor is not None:
            self.AOsignal_withoutTumor = self.AOsignal_withoutTumor[:, index]
            
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
        print(f"[AOT-biomaps] Generating {Nscans} patterns...")
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
            print(f"[AOT-biomaps] No even divisors found for num_elements = {num_elements}")
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
                print(f"[AOT-biomaps] Error: Duplicate detected for {dup}")
            return False

        # 2. Check each pattern individually
        num_elements = self.params.acoustic['probe']['num_elements']
        for pattern in patterns:
            hex_part, angle_str = pattern["fileName"].split('_')
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])

            # Check length
            if len(bits) != num_elements:
                print(f"[AOT-biomaps] Error length: {pattern['fileName']}")
                return False

            # Special case: "all active" pattern
            if np.all(bits == 1):
                continue

            # Check 0/1 balance
            if np.sum(bits) != num_elements // 2:
                print(f"[AOT-biomaps] Error 0/1 balance: {pattern['fileName']}")
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
                print(f"[AOT-biomaps] Error regularity: {pattern['fileName']}")
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
        print(f"[AOT-biomaps] Applying apodization (Alpha={alpha}, Div={divergence_deg}°) on {len(self.AcousticFields)} fields...")

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

        print("[AOT-biomaps] Apodization done.")

    # PRIVATE METHODS
    def _generate_acousticFields_STRUCT(self, fieldDataPath=None, isGPU=None, GPUdevice=None, tempFieldName="Kwave", nameBlock=None, generation_type="envelope_squared", show_log=False, max_workers=None, backend=None, materialize=False):
        """
        Generate acoustic fields for structured waves.

        Concurrency model:
        - LOADING is cheap (np.memmap) and stays parallel.
        - GENERATION is VRAM-bound: one k-Wave simulation + fused post-processing peaks at ~10-12 GB VRAM per field (sensor data alone = Nx_sim*Nz_sim*Nt*4 B). An unbounded ThreadPoolExecutor (~32 workers) saturates a 48 GB GPU -> OutOfMemoryError.
            Default: one generation worker PER GPU.
        - backend "cpp": the simulation runs as a subprocess -> the GIL is released during the sim -> multiple workers truly scale.
        - backend "python": the kwave-python time loop is GIL-bound -> multiple GPU workers do NOT scale (each field ~2x slower).

        Memory model (materialize):
        - load_field(HDR_IMG) returns a np.memmap: the .img file is only MAPPED (zero bytes read). The kernel reads pages on first touch, so the disk cost silently reappears later (e.g. in generate_AOsignal), and the page cache can be evicted at any time by unrelated I/O.
        - materialize=True promotes every loaded field from page-cache-backed memmap (or any ndarray view whose .base chain reaches a memmap) to an anonymous ndarray (guaranteed resident RAM). The disk cost moves (visibly) into the loading progress bar, and every downstream pass runs at RAM bandwidth.
        - A safety budget check (80% of MemAvailable, which includes the reclaimable page cache, unlike MemFree) refuses promotion if the estimated total size does not fit.

        Parameters:
            fieldDataPath (str): Path to save generated fields.
            isGPU (bool): Whether to use GPU for simulation. (Default: config.)
            GPUdevice (int or list of int): GPU device index. A LIST pins one worker per GPU (e.g. [0, 1] on a 2-GPU machine); an int pins all workers to one device; None uses the best GPU.
            tempFieldName (str): Prefix for the per-field temporary scratch dirs (cpp backend). Kept unique per field internally.
            nameBlock (str): Optional name for the block when saving.
            generation_type (str): "envelope_squared", "envelope" or "field".
            show_log (bool): Whether to show progress logs.
            max_workers (int): Max PARALLEL GENERATIONS. Default: one per GPU (GPU mode) or 4 (CPU mode). Do not raise this above the number of GPUs.
            backend (str): "python" (default) or "cpp". Use "cpp" with a GPUdevice LIST for true multi-GPU scaling.
            materialize (bool): If True, promote loaded fields from np.memmap to resident RAM ndarrays (guarded by a RAM budget check).
                Recommended for analysis/reconstruction sessions that call generate_AOsignal repeatedly. Leave False for one-shot passes or when RAM cannot hold the whole SetMixte.

        Returns:
            list: List of generated StructuredWave objects.
        """
        if self.patterns is None:
            raise ValueError("[AOT-biomaps] patterns is not initialized. Please load or generate the active list first.")

        if isGPU is None:
            isGPU = True if config.get_process() == 'gpu' else False

        backend = backend or "python"

        # --- Device pool: one generation worker per GPU ---------------------
        if isGPU:
            if isinstance(GPUdevice, (list, tuple)) and len(GPUdevice) > 0:
                gpu_devices = [int(d) for d in GPUdevice]
            elif isinstance(GPUdevice, int):
                gpu_devices = [GPUdevice]
            else:
                gpu_devices = [config.select_best_gpu()]
        else:
            gpu_devices = [None]

        if max_workers is None:
            max_workers = len(gpu_devices) if isGPU else min(4, os.cpu_count() or 1)
        if isGPU and max_workers > len(gpu_devices):
            print(f"[AOT-biomaps] Warning: max_workers={max_workers} > number of GPUs "
                    f"({len(gpu_devices)}). Capping to {len(gpu_devices)} to avoid VRAM OOM.")
            max_workers = len(gpu_devices)

        # One-time warning: multi-GPU only pays off with the cpp backend.
        if isGPU and len(gpu_devices) > 1 and backend == "python":
            print(f"[AOT-biomaps] Warning: {len(gpu_devices)} GPUs pinned with "
                    f"backend='python': the kwave-python time loop is GIL-bound, "
                    f"multiple workers will NOT scale (each field ~2x slower). "
                    f"Use backend='cpp' for true multi-GPU scaling.")

        # 1. Pre-check step: Instantiation and sorting
        to_load = []
        to_generate = []

        # Absolute mapping: pre-allocation to guarantee output order
        listAcousticFields = [None] * len(self.patterns)

        for i, pattern in enumerate(self.patterns):
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

            pathField = None
            if fieldDataPath is not None:
                pathField = os.path.join(fieldDataPath, AcousticField.get_name_field() + self.FormatSave.value)

            # Sorting: a field file counts as "on disk" only if it is NOT EMPTY. Zero-byte .img files (e.g. from a previously interrupted or truncated save) must be regenerated AND overwritten, not blindly trusted.
            if pathField is not None and os.path.exists(pathField) and os.path.getsize(pathField) > 0 and self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
                to_load.append((i, AcousticField, pathField))
            else:
                to_generate.append((i, AcousticField, pathField))

        print(f"[AOT-biomaps] Pre-check complete: {len(to_load)} fields to load, {len(to_generate)} fields to generate ({max_workers} generation worker(s)).")

        # --- RAM budget, inlined: MemAvailable (includes reclaimable page cache), NOT MemFree (SC_AVPHYS_PAGES). Promoting a cached memmap to an anonymous array mostly re-labels existing page-cache pages, so MemAvailable is the honest budget. sysconf fallback if /proc/meminfo is not readable (non-Linux).
        ram_available = None
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        ram_available = int(line.split()[1]) * 1024   # kB -> bytes
                        break
        except OSError:
            pass
        if ram_available is None:
            ram_available = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')   # fallback

        # --- Materialization state (closure-shared, GIL-protected) -----------
        # materialize_enabled[0] may be flipped OFF by the budget check below. bytes_per_field[0] is filled once, on the first successful load.
        materialize_enabled = [bool(materialize)]
        bytes_per_field = [None]

        # 2. Loading step (memmap -> cheap, stays parallel)
        def do_load(task):
            index, AcousticField, pathField = task
            try:
                AcousticField.load_field(fieldDataPath, self.FormatSave, nameBlock)
                F = AcousticField.field

                # --- File-backed detection, inlined: walk the .base chain; a view over a memmap is still file-backed (evictable page cache) whatever its ndarray subclass. -------------------
                b = F
                is_file_backed = False
                while b is not None:
                    if isinstance(b, np.memmap):
                        is_file_backed = True
                        break
                    b = b.base

                # --- Materialization: ANY file-backed array (memmap subclass OR ndarray view whose base chain reaches a memmap) is promoted to a fresh ANONYMOUS copy -> guaranteed RAM. ---
                if materialize_enabled[0] and is_file_backed:
                    if bytes_per_field[0] is None:
                        bytes_per_field[0] = F.nbytes
                        est_total = bytes_per_field[0] * len(to_load)
                        if est_total > 0.8 * ram_available:
                            materialize_enabled[0] = False
                            print(f"[AOT-biomaps] Warning: materialize refused: "
                                    f"~{est_total / 2**30:.0f} Go needed for {len(to_load)} fields "
                                    f"vs {ram_available / 2**30:.0f} Go RAM available. "
                                    f"Falling back to lazy memmap.")
                        else:
                            print(f"[AOT-biomaps] Materializing fields to RAM: "
                                    f"~{est_total / 2**30:.0f} Go for {len(to_load)} fields "
                                    f"(RAM available: {ram_available / 2**30:.0f} Go).")
                    if materialize_enabled[0]:
                        # np.array(copy=True) is the ONLY construct that always forces a real data copy; it also drops any view/memmap subclass -> anonymous pages, base is None.
                        Fc = np.array(F, dtype=F.dtype, order='C', copy=True)
                        assert Fc.base is None   # fresh anonymous array, not file-backed
                        AcousticField.field = Fc

                return index, AcousticField, True
            except Exception:
                return index, AcousticField, False

        if to_load:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures_load = [executor.submit(do_load, task) for task in to_load]

                # First distinct progress bar for loading
                for future in tqdm(concurrent.futures.as_completed(futures_load), total=len(to_load), desc="[AOT-biomaps] Loading fields", mininterval=0.0):
                    index, AcousticField, success = future.result()
                    if success:
                        listAcousticFields[index] = AcousticField
                    else:
                        pathField = os.path.join(fieldDataPath, AcousticField.get_name_field() + self.FormatSave.value)
                        to_generate.append((index, AcousticField, pathField))

        # 3. Generation step (VRAM-bound -> bounded, one worker per GPU)
        # NOTE: freshly generated fields are already resident ndarrays
        # (save_field writes a copy to disk, it does not de-materialize the in-memory array), so no materialization is needed on this path.
        def do_generate(task):
            index, AcousticField, pathField = task
            safe_tempFieldName = f"{tempFieldName}_{AcousticField.get_name_field()}"

            # Round-robin device pinning: worker index -> GPU index
            device = gpu_devices[index % len(gpu_devices)]

            AcousticField.generate_field(isGPU=isGPU, GPUdevice=device, tempFieldName=safe_tempFieldName, generation_type=generation_type, show_log=show_log, backend=backend)

            # Save only if missing or EMPTY (overwrite corrupted zero-byte files)
            if pathField is not None and self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
                if (not os.path.exists(pathField)) or os.path.getsize(pathField) == 0:
                    os.makedirs(os.path.dirname(pathField), exist_ok=True)
                    AcousticField.save_field(fieldDataPath)

            return index, AcousticField

        if to_generate:
            # Progress bar description: built ONCE, then actually used.
            if len(gpu_devices) > 1:
                devices_str = "multi-GPU " + ",".join(str(d) for d in gpu_devices)
            elif gpu_devices[0] is not None:
                devices_str = f"GPU {gpu_devices[0]}"
            else:
                devices_str = "CPU"
            desc = f"[AOT-biomaps] Generating fields (backend: {backend}, {devices_str})"

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures_gen = [executor.submit(do_generate, task) for task in to_generate]

                # Second distinct progress bar for generation
                for future in tqdm(concurrent.futures.as_completed(futures_gen), total=len(to_generate), desc=desc):
                    index, AcousticField = future.result()
                    listAcousticFields[index] = AcousticField

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
            raise FileNotFoundError(f"[AOT-biomaps] File {pathAO} not found.")

        if pathAO.endswith('.npy'):
            AOsignal = np.load(pathAO)
        elif pathAO.endswith('.h5'):
            with h5py.File(pathAO, 'r') as f:
                if h5name not in f:
                    raise KeyError(f"[AOT-biomaps] Dataset '{h5name}' not found in the HDF5 file.")
                AOsignal = f[h5name][:]
        elif pathAO.endswith('.mat'):
            mat_data = loadmat(pathAO)
            if h5name not in mat_data:
                raise KeyError(f"[AOT-biomaps] Dataset '{h5name}' not found in the .mat file.")
            AOsignal = mat_data[h5name]
        elif pathAO.endswith('.hdr'):
            AOsignal = load_AOsignal(pathAO)
        else:
            raise ValueError("[AOT-biomaps] Unsupported file format. Supported formats are: .npy, .h5, .mat, .hdr")

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
                raise ValueError("[AOT-biomaps] Experimental AOsignal with tumor is not initialized. Please load the experimental AO signal with tumor first.")
        else:
            if self.AOsignal_withoutTumor is None:
                raise ValueError("[AOT-biomaps] Experimental AOsignal without tumor is not initialized. Please load the experimental AO signal without tumor first.")
        if self.AcousticFields is not None:
            if self.AcousticFields[0].field.shape[0] > self.AOsignal_withTumor.shape[0]:
                self.cutAcousticFields(max_t=self.AOsignal_withTumor.shape[0]/float(self.params.general['ft']))
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
                            raise ValueError(f"[AOT-biomaps] Field name {nameField} does not match the expected name {expected_name} from the active list.")
        print("Experimental AO signals are correctly initialized.")

    def demodulate_AOsignal(self, withTumor=True):
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

        for i in trange(AOsignal.shape[1], desc="[AOT-biomaps] Demodulating AO signals (4-phases quadrature)"):
            field_obj = self.AcousticFields[i]
            label = field_obj.get_name_field()
            parts = label.split("_")
            hex_pattern = parts[1]
            angle_code = parts[-1]

            angle_deg = -int(angle_code[1:]) if angle_code.startswith("1") else int(angle_code)
            angle_rad = float(np.round(np.deg2rad(angle_deg), 6))

            if set(hex_pattern.lower().replace(" ", "")) == {'f'}:
                fs_key = 0.0
                phase = 0.0
            else:
                profile = hex_to_binary_profile(hex_pattern, n_piezos)
                ft_prof = np.fft.fft(profile)
                idx_max = np.argmax(np.abs(ft_prof[1:n_piezos//2])) + 1
                freqs = np.fft.fftfreq(n_piezos, d=delta_x)
                fs_key = float(np.round(abs(freqs[idx_max]) / 1000.0, 6))  # in mm⁻¹
                phase = get_phase_deterministic(profile)

            if fs_key == 0.0: # plane wave, no demodulation needed
                demodulated_data[(fs_key, angle_rad)] = np.array(AOsignal[:, i], dtype=np.complex64)
                continue

            key = (fs_key, angle_rad)
            if key not in structured_buffer:
                structured_buffer[key] = {}

            sig = np.array(AOsignal[:, i])
            if phase in structured_buffer[key]:
                structured_buffer[key][phase] = (structured_buffer[key][phase] + sig) / 2
            else:
                structured_buffer[key][phase] = sig

        for (fs, theta), phases in structured_buffer.items():
            required_phases = [0.0, np.pi/2, np.pi, 3*np.pi/2]
            if not all(p in phases for p in required_phases):
                example = next(iter(phases.values()))
                s0 = phases.get(0.0, np.zeros_like(example))
                s_pi_2 = phases.get(np.pi/2, np.zeros_like(example))
                s_pi = phases.get(np.pi, np.zeros_like(example))
                s_3pi_2 = phases.get(3*np.pi/2, np.zeros_like(example))
            else:
                s0 = phases[0.0]
                s_pi_2 = phases[np.pi/2]
                s_pi = phases[np.pi]
                s_3pi_2 = phases[3*np.pi/2]

            real = s0 - s_pi
            imag = s_pi_2 - s_3pi_2
            demodulated_data[(fs, theta)] = (real - 1j * imag) / (2/np.pi)

        return demodulated_data

    def demodulate_acoustic_fields(self, max_workers=None):
        """
        Demodulate acoustic fields into a flat dictionary: {(fs, theta): complex_field}.
        Identical structure to parse_and_demodulate, optimized with thread-safe multithreading.

        Returns:
            dict: Dictionary with keys (fs, theta) and values as complex fields.
        """
        n_piezos = self.params.acoustic['probe']['num_elements']
        delta_x = self.params.general['dx']

        # buffer[(fs, theta)][phase] = real field
        buffer = {}

        # 1. Grouping and Averaging (Sequential to build keys in deterministic order)
        for i in range(len(self.AcousticFields)):
            field_obj = self.AcousticFields[i]
            label = field_obj.get_name_field()
            parts = label.split("_")
            hex_pattern = parts[1]
            angle_code = parts[-1]

            # Extract Angle and Frequency
            angle_deg = -int(angle_code[1:]) if angle_code.startswith("1") else int(angle_code)
            angle_rad = float(np.round(np.deg2rad(angle_deg), 5))

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
            if key not in buffer: 
                buffer[key] = {}

            current_f = field_obj.field
            if phase in buffer[key]:
                buffer[key][phase] = (buffer[key][phase] + current_f) / 2
            else:
                buffer[key][phase] = current_f

        # 2. Quadrature (Multithreaded with strict order preservation)
        demodulated_fields = {}
        keys = list(buffer.keys())

        def process_quadrature(key):
            phases = buffer[key]
            fs = key[0]

            if fs == 0.0:
                return key, next(iter(phases.values())).astype(np.complex64)

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

            complex_field = ((real - 1j * imag) / (2/np.pi)).astype(np.complex64)
            return key, complex_field

        # Execute parallel quadrature computations
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map processes in parallel but yields in the EXACT order of 'keys'
            results = executor.map(process_quadrature, keys)
            
            for key, complex_field in tqdm(results, total=len(keys), desc="[AOT-biomaps] Demodulating Acoustic Fields", mininterval=0.0):
                demodulated_fields[key] = complex_field

        print(f"[AOT-biomaps] Acoustic Operator complete: {len(demodulated_fields)} configurations processed.")
        return demodulated_fields
    
    def flip_probe(self, flipPattern=True, flipAngle=True):
        """
        Flip the probe (binary pattern and/or angle) for all acoustic fields and AO signals.

        Parameters:
            flipPattern (bool): If True, reverse the order of active elements in the binary pattern.
            flipAngle (bool): If True, invert the sign of the angle.
        """
        if self.AcousticFields is None:
            print("[AOT-biomaps] Warning: AcousticFields is not initialized. No fields to flip, only AO signals.")
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
                new_field_name = f"{flipped_hex}_{new_angle_str}"
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
            print(f"[AOT-biomaps] Flipped both probe and AO signals (pattern and angle).")
        elif flipPattern and not flipAngle:
            print(f"[AOT-biomaps] Flipped probe and AO signals (pattern).")
        elif not flipPattern and flipAngle:
            print(f"[AOT-biomaps] Flipped probe and AO signals (angle).")
        else:
            print(f"[AOT-biomaps] No flipping applied.")