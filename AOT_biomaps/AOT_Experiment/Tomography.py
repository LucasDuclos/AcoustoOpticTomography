from ._mainExperiment import Experiment
from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType
from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave
from AOT_biomaps.Config import config

import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class Tomography(Experiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # PUBLIC METHODS
        
    def check(self):
        """
        Check if the experiment is correctly initialized.
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
                return False, f"Field {field.getName_field()} has an invalid Time shape: {field.field.shape[0]}. Expected time shape to be {self.AOsignal_withTumor.shape[0]}."
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

    def generateAcousticFields(self, fieldDataPath = None, show_log = True):
        """
        Generate the acoustic fields for simulation.

        Args:
            fieldDataPath: Path to save the generated fields.
            fieldParamPath: Path to the field parameters file.

        Returns:
            systemMatrix: A numpy array of the generated fields.
        """
        if self.TypeAcoustic.value == WaveType.StructuredWave.value:
            self.AcousticFields = self._generateAcousticFields_STRUCT_CPU(fieldDataPath,show_log)
        else:
            raise ValueError("Unsupported wave type.")

    def show_pattern(self):
        if self.AcousticFields is None:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")

        # Collect entries as a list of tuples
        entries = []
        for field in self.AcousticFields:
            if field.waveType != WaveType.StructuredWave:
                raise TypeError("AcousticFields must be of type StructuredWave to plot pattern.")
            pattern = field.pattern
            entries.append((
                (pattern.space_0, pattern.space_1, pattern.move_head_0_2tail, pattern.move_tail_1_2head),
                pattern.activeList,  # hex_str
                field.angle
            ))

        # Sort entries (same logic as before)
        entries.sort(
            key=lambda x: (
                -(x[0][0] + x[0][1]),  # Total length descending
                -max(x[0][0], x[0][1]), # Max(space_0, space_1) descending
                -x[0][0],              # space_0 descending
                -x[0][2],              # move_head_0_2tail descending
                x[0][3]                # move_tail_1_2head ascending
            )
        )

        # Extract data without Pandas
        hex_list = [hex_str for _, hex_str, _ in entries]
        angle_list = [angle for _, _, angle in entries]
        space_data = [t for t, _, _ in entries]  # List of (space_0, space_1, move_head_0_2tail, move_tail_1_2head)

        # Convert hex strings to binary columns (NumPy)
        def hex_string_to_binary_column(hex_str):
            bits = ''.join(f'{int(c, 16):04b}' for c in hex_str)
            return np.array([int(b) for b in bits], dtype=np.uint8).reshape(-1, 1)

        bit_columns = [hex_string_to_binary_column(h) for h in hex_list]
        image = np.hstack(bit_columns)
        height = image.shape[0]

        # Plot
        _, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image, cmap='gray', aspect='auto')
        ax.set_title("Scan configuration", fontsize='large')
        ax.set_xlabel("Wave", fontsize='medium')
        ax.set_ylabel("Transducer activation", fontsize='medium')

        # Plot angle markers
        angle_min = -20.2
        angle_max = 20.2
        center = height / 2
        scale = height / (angle_max - angle_min)
        for i, angle in enumerate(angle_list):
            y = round(center - angle * scale)
            if 0 <= y < height:
                ax.plot(i, y - 0.5, 'r.', markersize=5)

        ax.set_ylim(height - 0.5, -0.5)

        # Twin axis for angle labels
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        yticks_angle = np.linspace(20, -20, 9)
        yticks_pos = np.interp(yticks_angle, [angle_min, angle_max], [height - 0.5, -0.5])
        ax2.set_yticks(yticks_pos)
        ax2.set_yticklabels([f"{a:.1f}°" for a in yticks_angle])
        ax2.set_ylabel("Angle [degree]", fontsize='medium', color='r')
        ax2.tick_params(axis='y', colors='r')

        plt.show()
     
    def plot_angle_frequency_distribution(patterns, num_elements=192):
        """
        Plots the distribution of angles and spatial frequencies from a list of patterns.
        Args:
            patterns (list): List of strings in the format "hex_angle".
            num_elements (int): Number of elements in each pattern (default is 192).
        """
        freq_bins = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]

        angles = []
        freqs = []

        for p in patterns:
            hex_part, angle_str = p.split('_')
            # Récupérer l'angle
            sign = -1 if angle_str[0] == '1' else 1
            angle = sign * int(angle_str[1:])
            angles.append(angle)

            # Récupérer fréquence spatiale
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])

            # Cas spécial : pattern "192 on" (tous les bits à 1)
            if np.all(bits == 1):
                freqs.append(192)
                continue

            # Chercher la plus petite taille de bloc divisant num_elements
            for block_size in freq_bins:
                half_block = block_size // 2
                block = np.array([0]*half_block + [1]*half_block)
                reps = num_elements // block_size
                pattern_check = np.tile(block, reps)
                if any(np.array_equal(np.roll(pattern_check, shift), bits) for shift in range(block_size)):
                    freqs.append(block_size)
                    break
            else:
                # Si aucun bloc n'est trouvé (ne devrait pas arriver si les patterns sont valides)
                freqs.append(None)

        # Filtrer les valeurs None (si un pattern n'a pas de fréquence détectée)
        freqs = [f for f in freqs if f is not None]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogramme des angles
        axes[0].hist(angles, bins=np.arange(-20.5, 21.5, 1), color='skyblue', edgecolor='black', rwidth=0.8)
        axes[0].set_xlabel("Angle (°)")
        axes[0].set_ylabel("Nombre de patterns")
        axes[0].set_title("Distribution des angles")
        axes[0].set_xticks(np.arange(-20, 21, 2))

        # Histogramme des fréquences spatiales
        # Ajouter 193 pour inclure 192 dans le dernier bin
        bins = np.append(freq_bins, 193)
        axes[1].hist(freqs, bins=bins, color='salmon', edgecolor='black', rwidth=0.8)
        axes[1].set_xscale('log')
        axes[1].set_xticks(freq_bins)
        axes[1].set_xticklabels(freq_bins)
        axes[1].set_xlabel("Taille du bloc / fréquence spatiale")
        axes[1].set_ylabel("Nombre de patterns")
        axes[1].set_title("Distribution des fréquences spatiales")

        plt.tight_layout()
        plt.show()

    def loadActiveList(self, fieldParamPath):
        if not os.path.exists(fieldParamPath):
            raise FileNotFoundError(f"Field parameter file {fieldParamPath} not found.")
        
        patternList = []

        with open(fieldParamPath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                # 🔍 Tentative de lecture comme string type fileName
                if "_" in line and all(c in "0123456789abcdefABCDEF" for c in line.split("_")[0]):
                    patternList.append({"fileName": line})
                    continue

                # 🔍 Sinon, tentative de parsing classique
                try:
                    parsed = eval(line, {"__builtins__": None})
                    if isinstance(parsed, tuple) and len(parsed) == 2:
                        coords, angles = parsed
                        for angle in angles:
                            patternList.append({
                                "space_0": coords[0],
                                "space_1": coords[1],
                                "move_head_0_2tail": coords[2],
                                "move_tail_1_2head": coords[3],
                                "angle": angle
                            })
                    else:
                        raise ValueError("Ligne inattendue (pas un tuple de deux éléments)")
                except Exception as e:
                    print(f"Erreur de parsing sur la ligne : {line}\n{e}")
        self.patternList = patternList

    def generateActiveList(self,N):
        """
        Génère une liste de patterns d'activation équilibrés et réguliers.

        Args:
            N (int): Nombre de patterns à générer.

        Returns:
            list: Liste de strings au format "hex_angle".
        """
        if N < 1:
            raise ValueError("N must be a positive integer.")
        patterns = self._generate_patterns(N)
        if not self._check_patterns(patterns):
            raise ValueError("Generated patterns failed validation.")
        return patterns

    def _generate_patterns(N, num_elements=192):
        def format_angle(a):
            return f"{'1' if a < 0 else '0'}{abs(a):02d}"

        def bits_to_hex(bits):
            bit_string = ''.join(str(b) for b in bits)
            return f"{int(bit_string, 2):0{num_elements//4}x}"

        # 1. Générer le pattern "192 on" pour tous les angles
        all_on_bits = np.ones(num_elements, dtype=int)
        all_on_hex = bits_to_hex(all_on_bits)
        all_on_pairs = [f"{all_on_hex}_{format_angle(angle)}" for angle in range(-20, 21)]

        # 2. Initialiser results avec ces paires
        results = set(all_on_pairs)

        # 3. Calculer combien de patterns équilibrés il reste à générer
        remaining_N = N - len(all_on_pairs)
        if remaining_N <= 0:
            # Si N <= 41 (nombre d'angles), on retourne juste les "192 on"
            return list(results)[:N]

        # 4. Générer les autres patterns équilibrés
        divs = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]  # On exclut 192
        angle_choices = list(range(-20, 21))
        nb_freq = len(divs)
        max_per_block = int(np.ceil(remaining_N / nb_freq))

        for block_size in divs:
            half_block = block_size // 2
            block = np.array([0]*half_block + [1]*half_block)
            reps = num_elements // block_size
            base_pattern = np.tile(block, reps)

            n_shifts = 1 if block_size == num_elements else block_size
            all_shifted = [np.roll(base_pattern, shift) for shift in range(n_shifts)]
            np.random.shuffle(all_shifted)

            count = 0
            for pattern_bits in all_shifted:
                available_angles = angle_choices.copy()
                np.random.shuffle(available_angles)

                for angle in available_angles:
                    hex_pattern = bits_to_hex(pattern_bits)
                    pair = f"{hex_pattern}_{format_angle(angle)}"
                    if pair not in results:
                        results.add(pair)
                        count += 1
                        if count >= max_per_block:
                            break
                if count >= max_per_block:
                    break

        # 5. Compléter si nécessaire
        results = list(results)
        if len(results) < N:
            extra_needed = N - len(results)
            results += list(np.random.choice(results, extra_needed, replace=False))

        np.random.shuffle(results)
        return results[:N]

    def _check_patterns(patterns, num_elements=192):
        for p in patterns:
            hex_part = p.split('_')[0]
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])

            # Vérifier longueur
            if len(bits) != num_elements:
                print(f"Erreur longueur: {p}")
                return False

            # Cas spécial : pattern "192 on"
            if np.all(bits == 1):
                continue

            # Vérifier équilibre 0/1 (sauf pour "all on")
            if np.sum(bits) != num_elements // 2:
                print(f"Erreur équilibre 0/1: {p}")
                return False

            # Vérifier régularité (sauf pour "all on")
            valid = False
            for block_size in range(2, num_elements+1, 2):
                if num_elements % block_size != 0:
                    continue
                half_block = block_size // 2
                block = np.array([0]*half_block + [1]*half_block)
                reps = num_elements // block_size
                expected_pattern = np.tile(block, reps)
                if any(np.array_equal(np.roll(expected_pattern, shift), bits) for shift in range(block_size)):
                    valid = True
                    break
            if not valid:
                print(f"Erreur régularité: {p}")
                return False

        return True


    # PRIVATE METHODS

    def _generateAcousticFields_STRUCT_CPU(self,fieldDataPath =None, show_log=False):
        
        if self.patternList is None:
            raise ValueError("patternList is not initialized. Please load or generate the active list first.")
        
        listAcousticFields = []

        progress_bar = trange(0, len(self.patternList), desc="Generating acoustic fields")

        for i in progress_bar:
            memory = psutil.virtual_memory()
            pattern = self.patternList[i]

            # Cas 1 : format avec fileName (hex_angle)
            if "fileName" in pattern:
                AcousticField = StructuredWave(fileName=pattern["fileName"],params=self.params)
            # Cas 2 : format structuré classique
            else:
                AcousticField = StructuredWave(
                    angle_deg=pattern["angle"],
                    space_0=pattern["space_0"],
                    space_1=pattern["space_1"],
                    move_head_0_2tail=pattern["move_head_0_2tail"],
                    move_tail_1_2head=pattern["move_tail_1_2head"],
                    params=self.params
                )

            # Déterminer chemin de sauvegarde
            if fieldDataPath is None:
                pathField = None
            else:
                pathField = os.path.join(fieldDataPath, AcousticField.getName_field() + self.FormatSave.value)

            # Charger ou générer
            if pathField is not None and os.path.exists(pathField):
                progress_bar.set_postfix_str(f"Loading field - {AcousticField.getName_field()} -- Memory used :{memory.percent}%")
                try:
                    AcousticField.load_field(fieldDataPath, self.FormatSave)
                except:
                    progress_bar.set_postfix_str(f"Error loading field -> Generating field - {AcousticField.getName_field()} -- Memory used :{memory.percent}% ---- processing on {config.get_process().upper()} ----")
                    AcousticField.generate_field(show_log=show_log)
                    if not os.path.exists(pathField):
                        progress_bar.set_postfix_str(f"Saving field - {AcousticField.getName_field()} -- Memory used :{memory.percent}%")
                        os.makedirs(os.path.dirname(pathField), exist_ok=True)
                        AcousticField.save_field(fieldDataPath)

            elif pathField is None or not os.path.exists(pathField):
                progress_bar.set_postfix_str(f"Generating field - {AcousticField.getName_field()} -- Memory used :{memory.percent}% ---- processing on {config.get_process().upper()} ----")
                AcousticField.generate_field(show_log=show_log)
                if pathField is not None and not os.path.exists(pathField):
                    progress_bar.set_postfix_str(f"Saving field - {AcousticField.getName_field()} -- Memory used :{memory.percent}%")
                    os.makedirs(os.path.dirname(pathField), exist_ok=True)
                    AcousticField.save_field(fieldDataPath)

            listAcousticFields.append(AcousticField)
            progress_bar.set_postfix_str("")

        return listAcousticFields


