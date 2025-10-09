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
        self.patterns = None

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

    def generateAcousticFields(self, fieldDataPath=None, show_log=True):
        """
        Generate the acoustic fields for simulation.
        Args:
            fieldDataPath: Path to save the generated fields.
            show_log: Whether to show progress logs.
        Returns:
            systemMatrix: A numpy array of the generated fields.
        """
        if self.TypeAcoustic.value == WaveType.StructuredWave.value:
            self.AcousticFields = self._generateAcousticFields_STRUCT_CPU(fieldDataPath, show_log)
        else:
            raise ValueError("Unsupported wave type.")

    def show_pattern(self):
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

        def hex_string_to_binary_column(hex_str):
            bits = ''.join(f'{int(c, 16):04b}' for c in hex_str)
            return np.array([int(b) for b in bits], dtype=np.uint8).reshape(-1, 1)

        bit_columns = [hex_string_to_binary_column(h) for h in hex_list]
        image = np.hstack(bit_columns)
        print(image)  # Doit être un tableau de 1 partout

        height, width = image.shape

        # Create figure with compact size
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

        # Plot binary pattern
        im = ax.imshow(image, cmap='binary', aspect='auto', interpolation='none', vmin=0, vmax=1)

        ax.set_title("Scan Configuration", fontsize=12, pad=10, weight='bold')
        ax.set_xlabel("Wave Index", fontsize=10, labelpad=8)
        ax.set_ylabel("Transducer Activation", fontsize=10, labelpad=8)
        yticks_positions = np.arange(0, height)  # Positions des ticks (0 à 191)
        yticks_labels = np.arange(1, height + 1)  # Labels de 1 à 192

        ax.set_yticks(yticks_positions)
        ax.set_yticklabels(yticks_labels, fontsize=8)
        # Plot angle markers (bigger and bolder)
        angle_min, angle_max = -20.2, 20.2
        center = height / 2
        scale = height / (angle_max - angle_min)
        for i, angle in enumerate(angle_list):
            y = round(center - angle * scale)
            if 0 <= y < height:
                ax.plot(i, y - 0.5, 'ro', markersize=4, alpha=0.7)  # Points rouges plus gros

        ax.set_ylim(height - 0.5, -0.5)

        # Twin axis for angles (with larger font)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        yticks_angle = np.linspace(20, -20, 5)  # 5 ticks pour plus de détails
        yticks_pos = np.interp(yticks_angle, [angle_min, angle_max], [height - 0.5, -0.5])
        ax2.set_yticks(yticks_pos)
        ax2.set_yticklabels([f"{a:.1f}°" for a in yticks_angle], fontsize=9, color='r')
        ax2.set_ylabel("Angle [°]", fontsize=11, color='r', labelpad=10)
        ax2.tick_params(axis='y', colors='r', labelsize=9, width=1.5, length=5)

        # Make axes thicker
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax2.spines['right'].set_linewidth(1.5)

        # Add grid (thicker lines)
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.5)
        ax.set_xticks(np.linspace(0, width-1, 6))  # Plus de ticks sur l'axe x
        ax.set_yticks(np.linspace(0, height-1, 6))  # Plus de ticks sur l'axe y
        ax.tick_params(axis='both', which='both', labelsize=8, width=1.5, length=4)

        plt.tight_layout()
        plt.show()

    def plot_angle_frequency_distribution(self):
        if self.patterns is None:
            raise ValueError("patterns is not initialized. Please load or generate the active list first.")

        num_elements = self.params.acoustic['num_elements']
        divs = sorted([d for d in range(2, num_elements + 1) if num_elements % d == 0 and d % 2 == 0])
        if num_elements not in divs:
            divs.append(num_elements)
        divs.sort()

        angles = []
        freqs = []

        for p in self.patterns:
            # Extraire la chaîne "hexa_XXX" depuis le dictionnaire
            file_name = p["fileName"]
            hex_part, angle_str = file_name.split('_')  # Split sur le dictionnaire corrigé

            # Récupérer l'angle
            sign = -1 if angle_str[0] == '1' else 1
            angle = sign * int(angle_str[1:])
            angles.append(angle)

            # Récupérer la fréquence spatiale
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])
            if np.all(bits == 1):  # Cas "tous activés"
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogramme des angles
        axes[0].hist(angles, bins=np.arange(-20.5, 21.5, 1), color='skyblue', edgecolor='black', rwidth=0.8)
        axes[0].set_xlabel("Angle (°)")
        axes[0].set_ylabel("Nombre de patterns")
        axes[0].set_title("Distribution des angles")
        axes[0].set_xticks(np.arange(-20, 21, 2))

        # Histogramme des fréquences spatiales
        unique_freqs, freq_counts = np.unique(freqs, return_counts=True)
        x_pos = np.arange(len(divs))
        for freq, count in zip(unique_freqs, freq_counts):
            idx = divs.index(freq)
            axes[1].bar(x_pos[idx], count, color='salmon', edgecolor='black', width=0.8)

        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(divs)
        axes[1].set_xlabel("Taille du bloc (fréquence spatiale)")
        axes[1].set_ylabel("Nombre de patterns")
        axes[1].set_title("Distribution des fréquences spatiales")

        plt.tight_layout()
        plt.show()

    def loadActiveList(self, fieldParamPath):
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
                        raise ValueError("Ligne inattendue (pas un tuple de deux éléments)")
                except Exception as e:
                    print(f"Erreur de parsing sur la ligne : {line}\n{e}")
        self.patterns = patterns

    def saveActiveList(self, filePath):
        """
        Sauvegarde la liste des patterns dans un fichier texte.
        Args:
            filePath (str): Chemin du fichier de sortie.
        """
        with open(filePath, 'w') as file:
            for pattern in self.patterns:
                if "fileName" in pattern:
                    # Cas 1 : Pattern simple (format "hexa_XXX")
                    file.write(f"{pattern['fileName']}\n")
                else:
                    # Cas 2 : Pattern avec paramètres (format tuple)
                    coords = (
                        pattern["space_0"],
                        pattern["space_1"],
                        pattern["move_head_0_2tail"],
                        pattern["move_tail_1_2head"]
                    )
                    angles = [pattern["angle"]]  # Supposons que chaque pattern a un seul angle
                    line = f"({coords}, {angles})\n"
                    file.write(line)

    def generateActiveList(self, N):
        """
        Génère une liste de patterns d'activation équilibrés et réguliers.
        Args:
            N (int): Nombre de patterns à générer.
        Returns:
            list: Liste de strings au format "hex_angle".
        """
        if N < 1:
            raise ValueError("N must be a positive integer.")
        self.patterns = self._generate_patterns(N)
        if not self._check_patterns(self.patterns):
            raise ValueError("Generated patterns failed validation.")

    def _generate_patterns(self, N):
        def format_angle(a):
            return f"{'1' if a < 0 else '0'}{abs(a):02d}"

        def bits_to_hex(bits):
            bit_string = ''.join(str(b) for b in bits)
            bit_string = bit_string.zfill(len(bits))
            hex_string = ''.join([f"{int(bit_string[i:i+4], 2):x}" for i in range(0, len(bit_string), 4)])
            return hex_string

        num_elements = self.params.acoustic['num_elements']
        angle_choices = list(range(-20, 21))

        # 1. Trouver TOUS les diviseurs PAIRS de num_elements (y compris num_elements)
        divs = [d for d in range(2, num_elements + 1) if num_elements % d == 0 and d % 2 == 0]
        if not divs:
            print(f"Aucun diviseur pair trouvé pour num_elements = {num_elements}")
            return []

        # 2. Utiliser un ensemble pour suivre les patterns uniques
        unique_patterns = set()

        # 3. Générer jusqu'à N patterns uniques
        while len(unique_patterns) < N:
            # Tirer un diviseur aléatoire (y compris num_elements)
            block_size = np.random.choice(divs)

            if block_size == num_elements:
                # Cas spécial : pattern "tous activés"
                pattern_bits = np.ones(num_elements, dtype=int)
            else:
                # Cas général : pattern équilibré
                half_block = block_size // 2
                block = np.array([0] * half_block + [1] * half_block)
                reps = num_elements // block_size
                base_pattern = np.tile(block, reps)
                # Tirer un décalage aléatoire
                shift = np.random.randint(0, block_size)
                pattern_bits = np.roll(base_pattern, shift)

            # Convertir en hex et choisir un angle aléatoire
            hex_pattern = bits_to_hex(pattern_bits)
            angle = np.random.choice(angle_choices)
            pair = f"{hex_pattern}_{format_angle(angle)}"

            # Ajouter à l'ensemble (les doublons sont automatiquement ignorés)
            unique_patterns.add(pair)

        # 4. Convertir en liste de dictionnaires avec la clé "fileName"
        patterns = [{"fileName": pair} for pair in unique_patterns]

        # 5. Retourner exactement N patterns (on a déjà vérifié la taille avec while)
        return patterns[:N]  # Par sécurité, même si len(unique_patterns) == N

    def _check_patterns(self, patterns):
        # 1. Vérifier les doublons (basé sur "fileName")
        file_names = [p["fileName"] for p in patterns]
        if len(file_names) != len(set(file_names)):
            # Trouver les doublons
            from collections import Counter
            file_counts = Counter(file_names)
            duplicates = [fn for fn, count in file_counts.items() if count > 1]
            for dup in duplicates:
                print(f"Erreur : Doublon détecté pour {dup}")
            return False

        # 2. Vérifier chaque pattern individuellement
        num_elements = self.params.acoustic['num_elements']
        for pattern in patterns:
            hex_part, angle_str = pattern["fileName"].split('_')
            bits = np.array([int(b) for b in bin(int(hex_part, 16))[2:].zfill(num_elements)])

            # Vérifier la longueur
            if len(bits) != num_elements:
                print(f"Erreur longueur: {pattern['fileName']}")
                return False

            # Cas spécial : pattern "tous activés"
            if np.all(bits == 1):
                continue

            # Vérifier l'équilibre 0/1
            if np.sum(bits) != num_elements // 2:
                print(f"Erreur équilibre 0/1: {pattern['fileName']}")
                return False

            # Vérifier la régularité
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
                print(f"Erreur régularité: {pattern['fileName']}")
                return False

        return True

    # PRIVATE METHODS
    def _generateAcousticFields_STRUCT_CPU(self, fieldDataPath=None, show_log=False):
        if self.patterns is None:
            raise ValueError("patterns is not initialized. Please load or generate the active list first.")
        listAcousticFields = []
        progress_bar = trange(0, len(self.patterns), desc="Generating acoustic fields")
        for i in progress_bar:
            memory = psutil.virtual_memory()
            pattern = self.patterns[i]
            if "fileName" in pattern:
                AcousticField = StructuredWave(fileName=pattern["fileName"], params=self.params)
            else:
                AcousticField = StructuredWave(
                    angle_deg=pattern["angle"],
                    space_0=pattern["space_0"],
                    space_1=pattern["space_1"],
                    move_head_0_2tail=pattern["move_head_0_2tail"],
                    move_tail_1_2head=pattern["move_tail_1_2head"],
                    params=self.params
                )
            if fieldDataPath is None:
                pathField = None
            else:
                pathField = os.path.join(fieldDataPath, AcousticField.getName_field() + self.FormatSave.value)
            if pathField is not None and os.path.exists(pathField):
                progress_bar.set_postfix_str(f"Loading field - {AcousticField.getName_field()} -- Memory used: {memory.percent}%")
                try:
                    AcousticField.load_field(fieldDataPath, self.FormatSave)
                except:
                    progress_bar.set_postfix_str(f"Error loading field -> Generating field - {AcousticField.getName_field()} -- Memory used: {memory.percent}% ---- processing on {config.get_process().upper()} ----")
                    AcousticField.generate_field(show_log=show_log)
                    if not os.path.exists(pathField):
                        progress_bar.set_postfix_str(f"Saving field - {AcousticField.getName_field()} -- Memory used: {memory.percent}%")
                        os.makedirs(os.path.dirname(pathField), exist_ok=True)
                        AcousticField.save_field(fieldDataPath)
            elif pathField is None or not os.path.exists(pathField):
                progress_bar.set_postfix_str(f"Generating field - {AcousticField.getName_field()} -- Memory used: {memory.percent}% ---- processing on {config.get_process().upper()} ----")
                AcousticField.generate_field(show_log=show_log)
                if pathField is not None and not os.path.exists(pathField):
                    progress_bar.set_postfix_str(f"Saving field - {AcousticField.getName_field()} -- Memory used: {memory.percent}%")
                    os.makedirs(os.path.dirname(pathField), exist_ok=True)
                    AcousticField.save_field(fieldDataPath)
            listAcousticFields.append(AcousticField)
            progress_bar.set_postfix_str("")
        return listAcousticFields
