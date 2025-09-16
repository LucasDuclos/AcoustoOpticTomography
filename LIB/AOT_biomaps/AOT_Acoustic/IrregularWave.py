from AOT_biomaps.AOT_Acoustic._mainAcoustic import AcousticField
from .AcousticEnums import WaveType, TypeSim

import numpy as np

class IrregularWave(AcousticField):
    """
    Class for irregular wave types, inheriting from AcousticField.
    This class is a placeholder for future implementation of irregular wave types.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.waveType = WaveType.IrregularWave
        self.params = {
            'typeSim': TypeSim.IRREGULAR.value,
        }

    def getName_field(self):
        raise NotImplementedError("getName_field method not implemented for IrregularWave.")
    
    def  _generate_diverse_structurations(self,num_elements, num_sequences, num_frequencies):
        """
        Génère num_sequences structurations irrégulières ON/OFF pour une sonde de num_elements éléments.
        Chaque structuration contient exactement num_frequencies fréquences spatiales distinctes.

        :param num_elements: Nombre total d'éléments piézoélectriques de la sonde.
        :param num_sequences: Nombre total de structurations générées.
        :param num_frequencies: Nombre de fréquences spatiales distinctes par structuration.
        :return: Matrice de structuration de taille (num_sequences, num_elements)
        """
        
        # Définition des fréquences spatiales disponibles
        max_freq = num_elements // 2  # Nyquist limit
        available_frequencies = np.arange(1, max_freq + 1)  # Fréquences possibles
        
        # Matrice des structurations
        structurations = np.zeros((num_sequences, num_elements), dtype=int)
        
        # Sélectionner des fréquences uniques pour chaque structuration
        chosen_frequencies = []
        for _ in range(num_sequences):
            freqs = np.random.choice(available_frequencies, size=num_frequencies, replace=False)
            chosen_frequencies.append(freqs)

            # Construire la structuration correspondante
            structuration = np.zeros(num_elements)
            for f in freqs:
                structuration += np.cos(2 * np.pi * f * np.arange(num_elements) / num_elements)  # Ajouter la fréquence
            
            structuration = np.where(structuration >= 0, 1, 0)  # Binarisation ON/OFF
            structurations[_] = structuration
        
        return structurations, chosen_frequencies
    
    def getName_field(self):
        raise NotImplementedError("getName_field method not implemented for IrregularWave.")

    def _generate_2Dacoustic_field_KWAVE(self):
        raise NotImplementedError("2D acoustic field generation not implemented for IrregularWave.")

    def _generate_3Dacoustic_field_KWAVE(self):
        raise NotImplementedError("3D acoustic field generation not implemented for IrregularWave.")

    def _save2D_HDR_IMG(self, filePath):
        raise NotImplementedError("HDR/IMG saving not implemented for IrregularWave.")
