from .OpticEnums import OpticFieldType

import numpy as np

class Laser:
    def __init__(self, params):
        """
        Initializes the laser with the given parameters.
        :param params: Configuration parameters for the laser.
        """
        try:
            self.x = np.arange(params.general['Xrange'][0], params.general['Xrange'][1], params.general['dx']) * 1000
            self.z = np.arange(params.general['Zrange'][0], params.general['Zrange'][1], params.general['dz']) * 1000
            self.shape = OpticFieldType(params.optic['laser']['shape'].capitalize())
            self.center = params.optic['laser']['center']
            self.w0 = params.optic['laser']['w0'] * 1000
            self._set_intensity()
        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid laser shape: {e}")

    def _set_intensity(self):
        """
        Sets the intensity of the beam based on its shape.
        """
        try:
            if self.shape == OpticFieldType.GAUSSIAN:
                self.intensity = self._gaussian_beam()
            elif self.shape == OpticFieldType.UNIFORM:
                raise NotImplementedError("Uniform beam not implemented yet.")
            elif self.shape == OpticFieldType.SPHERICAL:
                raise NotImplementedError("Spherical beam not implemented yet.")
            else:
                raise ValueError("Unknown beam shape.")
        except Exception as e:
            raise RuntimeError(f"Error setting intensity: {e}")

    def _gaussian_beam(self):
        """
        Generates a Gaussian laser beam in the XZ plane.
        :return: Intensity matrix of the Gaussian beam.
        """
        try:
            if self.center == 'center':
                x0 = (self.x[0] + self.x[-1]) / 2
                z0 = (self.z[0] + self.z[-1]) / 2
            else:
                x0 = self.center[0] * 1000
                z0 = self.center[1] * 1000
            X, Z = np.meshgrid(self.x, self.z, indexing='ij')
            return np.exp(-2 * ((X - x0)**2 + (Z - z0)**2) / self.w0**2)
        except Exception as e:
            raise RuntimeError(f"Error generating Gaussian beam: {e}")

    def show_laser(self):
        """
        Displays the laser intensity distribution.
        """
        try:
            import matplotlib.pyplot as plt
            plt.imshow(self.intensity, extent=(self.x[0], self.x[-1] + 1, self.z[-1], self.z[0]), aspect='auto', cmap='hot')
            plt.colorbar(label='Intensity')
            plt.xlabel('X (mm)', fontsize=20)
            plt.ylabel('Z (mm)', fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.title('Laser Intensity Distribution')
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error plotting laser intensity: {e}")
