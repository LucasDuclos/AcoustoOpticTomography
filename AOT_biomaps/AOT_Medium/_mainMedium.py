from abc import ABC, abstractmethod
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Optional kwave imports
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False


class Medium(ABC):
    
    def __init__(self, params):
        self.params = params
        self.factorX = None
        self.factorZ = None
        self.factorT = None
        self.c_mean = None
        self.Nx_reshaped = None
        self.Nz_reshaped = None
        self.dx_reshaped = None
        self.dz_reshaped = None
        self.medium_properties = None

        if KWAVE_AVAILABLE:
            self.kgrid = kWaveGrid([self.params.general["Nx"], self.params.general["Nz"]], 
                                   [self.params.general["dx"], self.params.general["dz"]])
            
            if self.params.acoustic['f_AQ'] is None or self.params.acoustic['f_AQ'] == "AUTO":
                self.kgrid.makeTime(self.params.acoustic['medium']['c0'])
                self.params.acoustic['f_AQ'] = int(1/self.kgrid.dt)
                
            if self.params.general['Nt'] is None or self.params.general['Nt'] == "None":
                Lx = self.params.general["Nx"] * self.params.general["dx"]
                Lz = self.params.general['Zrange'][1] - self.params.general['Zrange'][0]
                theta = np.radians(20) 
                distance_max = (Lx * np.sin(theta)) + (Lz * np.cos(theta))
                f_aq = float(self.params.acoustic['f_AQ'])
                c0 = float(self.params.acoustic['medium']['c0'])
                Nt_strict = distance_max * f_aq / c0
                margin = 1.05 
                Nt = int(np.ceil(Nt_strict * margin))
                
                self.params.general['Nt'] = Nt
            else:
                Nt = self.params.general['Nt']

            self.kgrid.setTime(Nt, 1/float(self.params.acoustic['f_AQ']))
            
            if self.params.acoustic['f_saving'] is None or self.params.acoustic['f_saving'] == "AUTO":
                self.params.acoustic['f_saving'] = self.params.acoustic['f_AQ']
            else:
                self.params.acoustic['f_saving'] = int(float(self.params.acoustic['f_saving']))
                
        else:
            self.kgrid = None
            self.Nt_reshaped = self.params.general.get('Nt', 400)
            print("[AOT-biomaps] Warning: kWave is not available. Using default values for grid parameters.")

    @abstractmethod
    def generate_medium(self):
        """
        Abstract method to generate the medium properties.
        This method should be implemented by subclasses.
        """
        pass

    def save_medium(self, folderPath, fileName="medium"):
        """
        Save the entire medium properties to a .npy file.
        Universally handles any subclass (PVAMedium, BubbleMedium, etc.)
        """
        if os.path.splitext(fileName)[1]:
            raise ValueError("[AOT-biomaps] The fileName should not contain an extension; .npy will be added automatically.")
        
        os.makedirs(folderPath, exist_ok=True)
        filePath = os.path.join(folderPath, fileName + '.npy')
        
        if os.path.isdir(filePath):
            raise IsADirectoryError(f"[AOT-biomaps] Cannot save medium: {filePath} is a directory.")
        
        state_to_save = {}
        kmedium_data = {}
        
        state_to_save['__class_name__'] = self.__class__.__name__
        
        for key, value in self.__dict__.items():
            if key == 'kgrid':
                continue
            elif key == 'kmedium':
                if value is not None:
                    # CORRECTION : Extraction exhaustive de tous les paramètres kWave
                    props = ['sound_speed', 'density', 'alpha_coeff', 'alpha_mode', 
                             'BonA', 'alpha_power', 'absorbing', 'sound_speed_ref', 'stokes']
                    for p in props:
                        kmedium_data[p] = getattr(value, p, None)
                continue
            else:
                state_to_save[key] = value
                
        state_to_save['__kmedium_data__'] = kmedium_data
        np.save(filePath, state_to_save, allow_pickle=True)

    def load_medium(self, folderPath, fileName="medium", isAbsorbingMedium=False):
        """
        Load the medium properties from a .npy file for ANY subclass.
        Rebuilds kWave objects by injecting saved physical tensors directly 
        into their constructors.
        """
        if os.path.splitext(fileName)[1]:
            raise ValueError("[AOT-biomaps] The fileName should not contain an extension; .npy will be added automatically.")

        filePath = os.path.join(folderPath, fileName + '.npy')
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"[AOT-biomaps] The file {filePath} does not exist.")
        
        loaded_state = np.load(filePath, allow_pickle=True).item()
        
        saved_class = loaded_state.pop('__class_name__', 'Medium')
        if saved_class != self.__class__.__name__ and saved_class != 'Medium':
            print(f"[AOT-biomaps] Warning: Loading a file generated by '{saved_class}' into '{self.__class__.__name__}'.")

        kmedium_data = loaded_state.pop('__kmedium_data__', {})
        
        self.__dict__.update(loaded_state)
        
        if getattr(self, 'medium_properties', None) is not None:
            if 'sound_speed' in self.medium_properties:
                self.c_map = self.medium_properties['sound_speed']
            if 'density' in self.medium_properties:
                self.rho_map = self.medium_properties['density']

        if KWAVE_AVAILABLE and hasattr(self, 'params'):
            Nx = getattr(self, 'Nx_reshaped', self.params.general.get("Nx"))
            Nz = getattr(self, 'Nz_reshaped', self.params.general.get("Nz"))
            dx = getattr(self, 'dx_reshaped', self.params.general.get("dx"))
            dz = getattr(self, 'dz_reshaped', self.params.general.get("dz")) # Optionnel, souvent égal à dx
            
            self.kgrid = kWaveGrid([Nx, Nz], [dx, dz])
            
            if getattr(self, 'Nt_reshaped', None) is not None and 'f_AQ' in self.params.acoustic:
                self.kgrid.setTime(self.Nt_reshaped, 1/float(self.params.acoustic['f_AQ']))
            
            kmedium_kwargs = {k: v for k, v in kmedium_data.items() if v is not None}
            
            if 'sound_speed' in kmedium_kwargs:
                self.kmedium = kWaveMedium(**kmedium_kwargs)
            else:
                self.kmedium = None
        else:
            self.kgrid = None
            self.kmedium = None

        if isAbsorbingMedium is True:
            self.params.acoustic['medium']['isAbsorbingMedium'] = isAbsorbingMedium
            
        if self.params.acoustic['medium'].get('isAbsorbingMedium', True):
            print("[AOT-biomaps] Info: The loaded medium is set to be absorbing.")
        else:
            if KWAVE_AVAILABLE and getattr(self, 'kmedium', None) is not None:
                self.kmedium.alpha_coeff = np.zeros((self.Nx_reshaped, self.Nz_reshaped), dtype=np.float32)
                self.kmedium.alpha_mode = 'no_absorption'
            print("[AOT-biomaps] Info: The loaded medium is set to be non-absorbing.")
            
    def plot_medium_properties(self, figsize=(12, 5),vmin_speed=None, vmax_speed=None, vmin_density=None, vmax_density=None):
        if not KWAVE_AVAILABLE:
            print("[AOT-biomaps] Warning: kWave is not available. Cannot plot medium properties.")
            return
        
        if getattr(self, 'kmedium', None) is None:
            raise ValueError("[AOT-biomaps] Medium properties are not available. Please generate or load the medium first.")
            
        if vmin_speed is None:
            vmin_speed = np.min(self.kmedium.sound_speed)
        if vmax_speed is None:
            vmax_speed = np.max(self.kmedium.sound_speed)
        if vmin_density is None:
            vmin_density = np.min(self.kmedium.density)
        if vmax_density is None:
            vmax_density = np.max(self.kmedium.density)

        extent = [self.params.general['Xrange'][0]*1e3, self.params.general['Xrange'][1]*1e3, 
                  self.params.general['Zrange'][1]*1e3, self.params.general['Zrange'][0]*1e3]
        
        plt.figure(figsize=figsize)
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