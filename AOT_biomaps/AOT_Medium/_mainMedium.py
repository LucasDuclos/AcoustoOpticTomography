from abc import abstractmethod
import os
import numpy as np
import warnings

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Optional kwave imports
try:
    from kwave.kgrid import kWaveGrid
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False


class Medium:
    
    def __init__(self, params):
        self.params = params
        self.medium = None
        self.factorX = None
        self.factorZ = None
        self.factorT = None
        self.c_mean = None
        self.Nx_reshaped = None
        self.Nz_reshaped = None
        self.dx_reshaped = None
        self.medium_properties = None

        if KWAVE_AVAILABLE:
            self.kgrid = kWaveGrid([self.params.general["Nx"], self.params.general["Nz"]], [self.params.general["dx"], self.params.general["dz"]])

            if self.params.acoustic['f_AQ'] is None:
                self.kgrid.makeTime(self.params.acoustic['medium']['c0'])
                self.params.acoustic['f_AQ'] = int(1/self.kgrid.dt)
            else:
                if self.params.general['Nt'] is None or self.params.general['Nt'] == "None":
                    Nt = int(1.25*np.ceil((self.params.general['Zrange'][1] - self.params.general['Zrange'][0])*float(self.params.acoustic['f_AQ']) / self.params.acoustic['medium']['c0']))
                    self.params.general['Nt'] = Nt
                else:
                    Nt = self.params.general['Nt']
                self.kgrid.setTime(Nt, 1/float(self.params.acoustic['f_AQ']))
            self.Nt_reshaped = self.kgrid.Nt
        else:
            self.kgrid = None
            self.Nt_reshaped = self.params.general.get('Nt', 100)
            warnings.warn("kWave is not available. Using default values for grid parameters.", UserWarning)

    @abstractmethod
    def generate_medium(self):
        """
        Abstract method to generate the medium properties.
        This method should be implemented by subclasses.
        """
        pass

    def save_medium(self, folderPath, fileName="medium"):
        """
        Save the medium properties to a .npy file.

        Parameters:
        - folderPath (str): The directory where the file will be saved.
        - fileName (str): The name of the file (without extension).
        """
        try:
            os.makedirs(folderPath, exist_ok=True)
            filePath = os.path.join(folderPath, fileName)
            if not os.path.splitext(fileName)[1]:
                filePath += '.npy'
            else:
                raise ValueError("The fileName should not contain an extension; .npy will be added automatically.")
            
            # Save medium properties as a dictionary
            if hasattr(self, 'medium_properties') and self.medium_properties is not None:
                np.save(filePath, self.medium_properties)
            else:
                raise ValueError("Medium properties are not available. Please generate the medium first.")
        except Exception as e:
            print(f"Error in save_medium method: {e}")
            raise

    def load_medium(self, folderPath, fileName="medium", isAbsorbingMedium=None):
        """
        Load the medium properties from a .npy file.

        Parameters:
        - folderPath (str): The directory where the file will be loaded from.
        - fileName (str): The name of the file (without extension).
        """
        try:
            if not os.path.splitext(fileName)[1]:
                fileName += '.npy'
            else:
                raise ValueError("The fileName should not contain an extension; .npy will be added automatically.")
            filePath = os.path.join(folderPath, fileName)
            if not os.path.exists(filePath):
                raise FileNotFoundError(f"The file {filePath} does not exist.")
            
            self.medium_properties = np.load(filePath, allow_pickle=True).item()
            
            # Restore properties
            if 'sound_speed' in self.medium_properties:
                if KWAVE_AVAILABLE:
                    self.kmedium.sound_speed = self.medium_properties['sound_speed']
                self.c_map = self.medium_properties['sound_speed']
            if 'density' in self.medium_properties:
                if KWAVE_AVAILABLE:
                    self.kmedium.density = self.medium_properties['density']
                self.rho_map = self.medium_properties['density']
            
            self.factorX = self.medium_properties.get('factorX')
            self.factorZ = self.medium_properties.get('factorZ')
            self.factorT = self.medium_properties.get('factorT')
            self.c_mean = self.medium_properties.get('c_mean')
            self.Nx_reshaped = self.medium_properties.get('Nx_reshaped')
            self.Nz_reshaped = self.medium_properties.get('Nz_reshaped')
            self.dx_reshaped = self.medium_properties.get('dx_reshaped')

            if isAbsorbingMedium is not None:
                self.params.acoustic['medium']['isAbsorbingMedium'] = isAbsorbingMedium
            if self.params.acoustic['medium']['isAbsorbingMedium']:
                print("Warning: The loaded medium is set to be absorbing")
            else:
                if KWAVE_AVAILABLE and hasattr(self, 'kmedium') and self.kmedium is not None:
                    self.kmedium.alpha_coeff = np.zeros((self.Nx_reshaped, self.Nz_reshaped), dtype=np.float32)
                    self.kmedium.alpha_mode = 'no_absorption'
                print("Warning: The loaded medium is set to be non-absorbing")
        except Exception as e:
            print(f"Error in load_medium method: {e}")
            raise

    def plot_medium_properties(self):
        if not KWAVE_AVAILABLE:
            warnings.warn("kWave is not available. Cannot plot medium properties.", UserWarning)
            return
        
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("matplotlib is not available. Cannot plot medium properties.", UserWarning)
            return
        
        if self.kmedium is None:
            raise ValueError("Medium properties are not available. Please generate or load the medium first.")
        vmin_speed = np.min(self.kmedium.sound_speed)
        vmax_speed = np.max(self.kmedium.sound_speed)
        vmin_density = np.min(self.kmedium.density)
        vmax_density = np.max(self.kmedium.density)
        extent = [self.params.general['Xrange'][0]*1e3, self.params.general['Xrange'][1]*1e3, self.params.general['Zrange'][1]*1e3, self.params.general['Zrange'][0]*1e3]
        plt.figure(figsize=(13,6))
        plt.subplot(121)
        plt.imshow(self.kmedium.sound_speed.T, vmin=vmin_speed, vmax=vmax_speed, cmap='autumn', extent=extent)
        plt.title('Sound speed map (m/s)')
        plt.xlabel('X (mm)')
        plt.ylabel('Z (mm)')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(self.kmedium.density.T, vmin=vmin_density, vmax=vmax_density, cmap='summer', extent=extent)
        plt.title('Density map (kg/m^3)')
        plt.xlabel('X (mm)')
        plt.ylabel('Z (mm)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
