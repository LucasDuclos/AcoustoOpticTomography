from ._mainExperiment import Experiment
from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType
from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave
from AOT_biomaps.Config import config

import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

    def generateAcousticFields(self, fieldDataPath, fieldParamPath, show_log = True):
        """
        Generate the acoustic fields for simulation.

        Args:
            fieldDataPath: Path to save the generated fields.
            fieldParamPath: Path to the field parameters file.

        Returns:
            systemMatrix: A numpy array of the generated fields.
        """
        if self.TypeAcoustic.value == WaveType.StructuredWave.value:
            self.AcousticFields = self._generateAcousticFields_STRUCT_CPU(fieldDataPath, fieldParamPath,show_log)
        else:
            raise ValueError("Unsupported wave type.")

    def show_pattern(self):
        if self.AcousticFields is None:
            raise ValueError("AcousticFields is not initialized. Please generate the system matrix first.")

        entries = []
        for field in self.AcousticFields:
            if field.waveType != WaveType.StructuredWave:
                raise TypeError("AcousticFields must be of type StructuredWave to plot pattern.")
            entries.append(((field.pattern.space_0, field.pattern.space_1, field.pattern.move_head_0_2tail, field.pattern.move_tail_1_2head), field.pattern.activeList, field.angle))
        print("Entries:", entries)
        

        # Sorting rule
        entries.sort(
            key=lambda x: (
                -(x[0][0] + x[0][1]),  # Total length descending
                -max(x[0][0], x[0][1]), # Max(space_0, space_1) descending
                -x[0][0],              # space_0 descending
                -x[0][2],              # move_head_0_2tail descending
                x[0][3]                # move_tail_1_2head ascending
            )
        )

        df = pd.DataFrame([
            {
                "hex": hex_str,
                "space_0": t[0],
                "space_1": t[1],
                "move_head_0_2tail": t[2],
                "move_tail_1_2head": t[3],
                "angles": angles
            }
            for t, hex_str, angles in entries
        ])

        def hex_string_to_binary_column(hex_str):
            bits = ''.join(f'{int(c, 16):04b}' for c in hex_str)
            return np.array([int(b) for b in bits], dtype=np.uint8).reshape(-1, 1)

        hex_list = df['hex'].tolist()
        angle_list = df['angles'].tolist()
        bit_columns = [hex_string_to_binary_column(h) for h in hex_list]
        image = np.hstack(bit_columns)
        height = image.shape[0]

        _, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image, cmap='gray', aspect='auto')
        ax.set_title("Scan configuration", fontsize='large')
        ax.set_xlabel("Wave", fontsize='medium')
        ax.set_ylabel("Transducer activation", fontsize='medium')

        angle_min = -20.2
        angle_max = 20.2
        center = height / 2
        scale = height / (angle_max - angle_min)

        for i, angle in enumerate(angle_list):
            y = round(center - angle * scale)
            if 0 <= y <= height:
                ax.plot(i, y-0.5, 'r.', markersize=5)

        ax.set_ylim(height - 0.5, -0.5)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

        yticks_angle = np.linspace(20, -20, 9)
        yticks_pos = np.interp(yticks_angle, [angle_min, angle_max], [height - 0.5, -0.5])
        ax2.set_yticks(yticks_pos)
        ax2.set_yticklabels([f"{a:.1f}°" for a in yticks_angle])
        ax2.set_ylabel("Angle [degree]", fontsize='medium', color='r')
        ax2.tick_params(axis='y', colors='r')

        plt.show()
     
    # PRIVATE METHODS

    def _generateAcousticFields_STRUCT_CPU(self, fieldDataPath, fieldParamPath, show_log):
        if not os.path.exists(fieldParamPath):
            raise FileNotFoundError(f"Field parameter file {fieldParamPath} not found.")
        if fieldDataPath is not None:
            os.makedirs(fieldDataPath, exist_ok=True)

        listAcousticFields = []
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

        progress_bar = trange(0, len(patternList), desc="Generating acoustic fields")

        for i in progress_bar:
            memory = psutil.virtual_memory()
            pattern = patternList[i]

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

    # def _generateAcousticFields_STRUCT_CPU(self, fieldDataPath, fieldParamPath, show_log):
    #     if not os.path.exists(fieldParamPath):
    #         raise FileNotFoundError(f"Field parameter file {fieldParamPath} not found.")
    #     if not fieldDataPath is None:
    #         os.makedirs(fieldDataPath, exist_ok=True)
    #     listAcousticFields = []
    #     patternList = []
    #     with open(fieldParamPath, 'r') as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             line = line.strip()
    #             if not line:
    #                 continue  # skip empty lines

    #             try:
    #                 # Sécurise l'évaluation en supprimant accès à builtins
    #                 parsed = eval(line, {"__builtins__": None})

    #                 if isinstance(parsed, tuple) and len(parsed) == 2:
    #                     coords, angles = parsed
    #                     for angle in angles:
    #                         patternList.append([*coords, angle])
    #                 else:
    #                     raise ValueError("Ligne inattendue (pas un tuple de deux éléments)")

    #             except Exception as e:
    #                 print(f"Erreur de parsing sur la ligne : {line}\n{e}")

    #     progress_bar = trange(0,len(patternList), desc="Generating acoustic fields")

    #     for i in progress_bar:
    #         memory = psutil.virtual_memory()
    #         pattern = patternList[i]
    #         if len(pattern) != 5:
    #             raise ValueError(f"Invalid pattern format: {pattern}. Expected 5 values.")
    #         # Initialisation de l'objet AcousticField
    #         AcousticField = StructuredWave(
    #             angle_deg=pattern[4],
    #             space_0=pattern[0],
    #             space_1=pattern[1],
    #             move_head_0_2tail=pattern[2],
    #             move_tail_1_2head=pattern[3],
    #             params=self.params
    #         )

    #         if fieldDataPath is None:
    #             pathField = None
    #         else:
    #             pathField = os.path.join(fieldDataPath, os.path.basename(AcousticField.getName_field() + self.FormatSave.value))

    #         if not pathField is None and os.path.exists(pathField):
    #             if progress_bar is not None:
    #                 progress_bar.set_postfix_str(f"Loading field - {AcousticField.getName_field()} -- Memory used :{memory.percent}%")
    #                 try:
    #                     AcousticField.load_field(fieldDataPath,  self.FormatSave)
    #                 except:
    #                     progress_bar.set_postfix_str(f"Error loading field -> Generating field - {AcousticField.getName_field()} -- Memory used :{memory.percent}% ---- processing on {config.get_process().upper()} ----")
    #                     AcousticField.generate_field(show_log = show_log)
    #                     if not pathField is None and not os.path.exists(pathField):
    #                         progress_bar.set_postfix_str(f"Saving field - {AcousticField.getName_field()} -- Memory used :{memory.percent}%")
    #                         os.makedirs(os.path.dirname(pathField), exist_ok=True) 
    #                         AcousticField.save_field(fieldDataPath)

    #         elif pathField is None or not os.path.exists(pathField):
    #             progress_bar.set_postfix_str(f"Generating field - {AcousticField.getName_field()} -- Memory used :{memory.percent}% ---- processing on {config.get_process().upper()} ----")
    #             AcousticField.generate_field(show_log = show_log)
            
    #         if not pathField is None and not os.path.exists(pathField):
    #             progress_bar.set_postfix_str(f"Saving field - {AcousticField.getName_field()} -- Memory used :{memory.percent}%")
    #             os.makedirs(os.path.dirname(pathField), exist_ok=True) 
    #             AcousticField.save_field(fieldDataPath)

    #         listAcousticFields.append(AcousticField)
    #         # Réinitialiser le texte de la barre de progression pour l'itération suivante
    #         progress_bar.set_postfix_str("")
   
    #     return listAcousticFields
    
