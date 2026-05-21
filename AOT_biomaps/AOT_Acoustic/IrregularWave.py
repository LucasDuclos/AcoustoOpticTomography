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

    def getName_field(self):
        """
        Generate the name for the field file.
        Not implemented for IrregularWave.
        """
        raise NotImplementedError("getName_field method not implemented for IrregularWave.")

    def _generate_diverse_structurations(self, num_elements, num_sequences, num_frequencies):
        """
        Generate num_sequences irregular ON/OFF structurations for a probe with num_elements elements.
        Each structuration contains exactly num_frequencies distinct spatial frequencies.

        Parameters:
            num_elements (int): Total number of piezoelectric elements in the probe.
            num_sequences (int): Total number of structurations to generate.
            num_frequencies (int): Number of distinct spatial frequencies per structuration.

        Returns:
            tuple: (structurations, chosen_frequencies)
                - structurations: Matrix of structurations with shape (num_sequences, num_elements)
                - chosen_frequencies: List of selected frequencies for each structuration
        """
        # Define available spatial frequencies
        max_freq = num_elements // 2  # Nyquist limit
        available_frequencies = np.arange(1, max_freq + 1)  # Possible frequencies

        # Structurations matrix
        structurations = np.zeros((num_sequences, num_elements), dtype=int)

        # Select unique frequencies for each structuration
        chosen_frequencies = []
        for idx in range(num_sequences):
            freqs = np.random.choice(available_frequencies, size=num_frequencies, replace=False)
            chosen_frequencies.append(freqs)

            # Build the corresponding structuration
            structuration = np.zeros(num_elements)
            for f in freqs:
                structuration += np.cos(2 * np.pi * f * np.arange(num_elements) / num_elements)  # Add frequency

            structuration = np.where(structuration >= 0, 1, 0)  # Binarize to ON/OFF
            structurations[idx] = structuration  # Fixed: use idx instead of _

        return structurations, chosen_frequencies

    def _generate_2Dacoustic_field_KWAVE(self):
        """
        Generate a 2D acoustic field using k-Wave.
        Not implemented for IrregularWave.
        """
        raise NotImplementedError("2D acoustic field generation not implemented for IrregularWave.")

    def _generate_3Dacoustic_field_KWAVE(self):
        """
        Generate a 3D acoustic field using k-Wave.
        Not implemented for IrregularWave.
        """
        raise NotImplementedError("3D acoustic field generation not implemented for IrregularWave.")

    def _save2D_HDR_IMG(self, filePath):
        """
        Save the acoustic field to HDR/IMG files.
        Not implemented for IrregularWave.
        """
        raise NotImplementedError("HDR/IMG saving not implemented for IrregularWave.")