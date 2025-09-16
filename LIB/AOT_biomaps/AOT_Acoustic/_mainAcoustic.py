import AOT_biomaps.Settings
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Acoustic.AcousticTools import calculate_envelope_squared, CPU_hilbert

import scipy.io
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import ceil
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.signals import tone_burst
from abc import ABC, abstractmethod
from .AcousticEnums import TypeSim, Dim, FormatSave, WaveType


####### ABSTRACT CLASS #######

class AcousticField(ABC):
    """
    Abstract class to generate and manipulate acoustic fields for ultrasound imaging.
    Provides methods to initialize parameters, generate fields, save and load data, and calculate envelopes.

    Principal parameters:
    - field: Acoustic field data.
    - burst: Burst signal used for generating the field for each piezo elements.
    - delayedSignal: Delayed burst signal for each piezo element.
    - medium: Medium properties for k-Wave simulation. Because field2 and Hydrophone simulation are not implemented yet, this attribute is set to None for these types of simulation.
    """

    def __init__(self, params):
        """
        Initialize global properties of the AcousticField object.

        Parameters:
        - typeSim (TypeSim): Type of simulation to be performed. Options include KWAVE, FIELD2, and HYDRO. Default is TypeSim.KWAVE.
        - dim (Dim): Dimension of the acoustic field. Can be 2D or 3D. Default is Dim.D2.
        - c0 (float): Speed of sound in the medium, specified in meters per second (m/s). Default is 1540 m/s.
        - f_US (float): Frequency of the ultrasound signal, specified in Hertz (Hz). Default is 6 MHz.
        - f_AQ (float): Frequency of data acquisition, specified in Hertz (Hz). Default is 180 MHz.
        - f_saving (float): Frequency at which the acoustic field data is saved, specified in Hertz (Hz). Default is 10 MHz.
        - num_cycles (int): Number of cycles in the burst signal. Default is 4 cycles.
        - num_elements (int): Number of elements in the transducer array. Default is 192 elements.
        - element_width (float): Width of each transducer element, specified in meters (m). Default is 0.2 mm.
        - element_height (float): Height of each transducer element, specified in meters (m). Default is 6 mm.
        - Xrange (list of float): Range of X coordinates for the acoustic field, specified in meters (m). Default is from -20 mm to 20 mm.
        - Yrange (list of float, optional): Range of Y coordinates for the acoustic field, specified in meters (m). Default is None, indicating no specific Y range.
        - Zrange (list of float): Range of Z coordinates for the acoustic field, specified in meters (m). Default is from 0 m to 37 mm.
        """
        required_keys = [
            'c0', 'f_US', 'f_AQ', 'f_saving', 'num_cycles', 'num_elements',
            'element_width', 'element_height', 'Xrange', 'Zrange', 'dim',
            'typeSim', 'dx', 'dz'
        ]

        # Verify required keys
        try:
            if params != None:
                for key in required_keys:
                    if key not in params.acoustic and key not in params.general:
                        raise ValueError(f"{key} must be provided in the parameters.")
        except ValueError as e:
            print(f"Initialization error: {e}")
            raise
        if params != None:
            if type(params) != AOT_biomaps.Settings.Params:
                raise TypeError("params must be an instance of the Params class")

            self.params = {
                'c0': params.acoustic['c0'],
                'f_US': int(float(params.acoustic['f_US'])),
                'f_AQ': params.acoustic['f_AQ'],
                'f_saving': int(float(params.acoustic['f_saving'])),
                'num_cycles': params.acoustic['num_cycles'],
                'num_elements': params.acoustic['num_elements'],
                'element_width': params.acoustic['element_width'],
                'element_height': params.acoustic['element_height'],
                'Xrange': params.general['Xrange'],
                'Yrange': params.general['Yrange'],
                'Zrange': params.general['Zrange'],
                'dim': params.acoustic['dim'],
                'typeSim': params.acoustic['typeSim'],
                'dx': params.general['dx'],
                'dy': params.general['dy'] if params.general['Yrange'] is not None else None,
                'dz': params.general['dz'],
                'Nx': int(np.round((params.general['Xrange'][1] - params.general['Xrange'][0])/params.general['dx'])),
                'Ny': int(np.round((params.general['Yrange'][1] - params.general['Yrange'][0])/params.general['dy']))  if params.general['Yrange'] is not None else 1,
                'Nz': int(np.round((params.general['Zrange'][1] - params.general['Zrange'][0])/params.general['dz'])),
                'probeWidth': params.acoustic['num_elements'] * params.acoustic['element_width'],
                'IsAbsorbingMedium': params.acoustic['isAbsorbingMedium'],
            }
            self.kgrid = kWaveGrid([self.params["Nx"], self.params["Nz"]], [self.params["dx"], self.params["dz"]])
            if params.acoustic['f_AQ'] == "AUTO":

                self.kgrid.makeTime(self.params['c0'])

                self.params['f_AQ'] = int(1/self.kgrid.dt)
            else:
                Nt = ceil((self.params['Zrange'][1] - self.params['Zrange'][0])*float(params.acoustic['f_AQ']) / self.params['c0'])

                self.kgrid.setTime(Nt,1/float(params.acoustic['f_AQ']))
                self.params['f_AQ'] = int(float(params.acoustic['f_AQ']))

            self._generate_burst_signal()
            if self.params["dim"] == Dim.D3 and self.params["Yrange"] is None:
                raise ValueError("Yrange must be provided for 3D fields.")
            if self.params['typeSim'] == TypeSim.KWAVE.value:
                if self.params ['IsAbsorbingMedium'] == True:
                    self.medium = kWaveMedium(
                        sound_speed=self.params['c0'],
                        density=params.acoustic['Absorption']['density'],    
                        alpha_coeff=params.acoustic['Absorption']['alpha_coeff'],  # dB/(MHz·cm)
                        alpha_power=params.acoustic['Absorption']['alpha_power'],  # 0.5
                        absorbing=True
                    )
                else:
                    self.medium = kWaveMedium(sound_speed=self.params['c0'])
            elif self.params['typeSim'] == TypeSim.FIELD2.value:
                self.medium = None
        else:
            self.medium = None

        self.waveType = None
        self.field = None   

    def __str__(self):
        """
        Returns a string representation of the AcousticField object, including its parameters and attributes.
        The string is formatted in a table-like structure for better readability.
        """
        try:
            # Get all attributes of the instance
            attrs = {**self.params, **{k: v for k, v in vars(self).items() if k not in self.params}}

            # Base attributes of AcousticField
            base_attrs_keys = ['c0', 'f_US', 'f_AQ', 'f_saving', 'num_cycles', 'num_elements',
                            'element_width', 'element_height',
                            'Xrange', 'Yrange', 'Zrange', 'dim', 'typeSim', 'Nx', 'Ny', 'Nz',
                            'dx', 'dy', 'dz', 'probeWidth']
            base_attrs = {key: value for key, value in attrs.items() if key in base_attrs_keys}

            # Attributes specific to the derived class, excluding 'params'
            derived_attrs = {key: value for key, value in attrs.items() if key not in base_attrs_keys and key != 'params'}

            # Create lines for base and derived attributes
            base_attr_lines = [f"  {key}: {value}" for key, value in base_attrs.items()]

            derived_attr_lines = []
            for key, value in derived_attrs.items():
                if key in {'burst', 'delayedSignal'}:
                    continue
                elif key == 'pattern':
                    # Inspect the pattern object
                    try:
                        pattern_attrs = vars(value)
                        pattern_str = ", ".join([f"{k}={v}" for k, v in pattern_attrs.items()])
                        derived_attr_lines.append(f"  pattern: {{{pattern_str}}}")
                    except Exception as e:
                        derived_attr_lines.append(f"  pattern: <unreadable: {e}>")
                else:
                    try:
                        derived_attr_lines.append(f"  {key}: {value}")
                    except Exception as e:
                        derived_attr_lines.append(f"  {key}: <unprintable: {e}>")

            # Add shapes for burst and delayedSignal
            if 'burst' in derived_attrs:
                derived_attr_lines.append(f"  burst: shape={self.burst.shape}")
            if 'delayedSignal' in derived_attrs:
                derived_attr_lines.append(f"  delayedSignal: shape={self.delayedSignal.shape}")

            # Define borders and titles
            border = "+" + "-" * 40 + "+"
            title = f"|Type : {self.__class__.__name__} wave |"
            base_title = "| AcousticField Attributes |"
            derived_title = f"| {self.__class__.__name__} Specific Attributes |" if derived_attrs else ""

            # Convert attributes to strings
            base_attr_str = "\n".join(base_attr_lines)
            derived_attr_str = "\n".join(derived_attr_lines)

            # Assemble the final result
            result = f"{border}\n{title}\n{border}\n{base_title}\n{border}\n{base_attr_str}\n"
            if derived_attrs:
                result += f"\n{border}\n{derived_title}\n{border}\n{derived_attr_str}\n"
            result += border

            return result
        except Exception as e:
            print(f"Error in __str__ method: {e}")
            raise

    def __del__(self):
        """
        Destructor for the AcousticField class. Cleans up the field and envelope attributes.
        """
        try:
            self.field = None
            self.burst = None
            self.delayedSignal = None
        except Exception as e:
            print(f"Error in __del__ method: {e}")
            raise

    ## TOOLS METHODS ##

    def generate_field(self, isGpu=config.get_process() == 'gpu',show_log = True):
        """
        Generate the acoustic field based on the specified simulation type and parameters.
        """
        try:
            if self.params['typeSim'] == TypeSim.FIELD2.value:
                raise NotImplementedError("FIELD2 simulation is not implemented yet.")
            elif self.params['typeSim'] == TypeSim.KWAVE.value:
                if self.params["dim"] == Dim.D2.value:
                    try:
                        field = self._generate_2Dacoustic_field_KWAVE(isGpu, show_log)
                    except Exception as e:
                        raise RuntimeError(f"Failed to generate 2D acoustic field: {e}")
                    self.field = calculate_envelope_squared(field, isGpu)
                elif self.params["dim"] == Dim.D3.value:
                    field = self._generate_3Dacoustic_field_KWAVE(isGpu, show_log)
                    self.field = calculate_envelope_squared(field, isGpu)
            elif self.params['typeSim'] == TypeSim.HYDRO.value:
                raise ValueError("Cannot generate field for Hydrophone simulation, load exciting acquisitions.")
            else:
                raise ValueError("Invalid simulation type. Supported types are: FIELD2, KWAVE, HYDRO.")
        except Exception as e:
            print(f"Error in generate_field method: {e}")
            raise

    def save_field(self, filePath, formatSave=FormatSave.HDR_IMG):
        """
        Save the acoustic field to a file in the specified format.

        Parameters:
        - filePath (str): The path where the file will be saved.
        """
        try:
            if formatSave.value == FormatSave.HDR_IMG.value:
                self._save2D_HDR_IMG(filePath)
            elif formatSave.value == FormatSave.H5.value:
                self._save2D_H5(filePath)
            elif formatSave.value == FormatSave.NPY.value:
                self._save2D_NPY(filePath)
            else:
                raise ValueError("Unsupported format. Supported formats are: HDR_IMG, H5, NPY.")
        except Exception as e:
            print(f"Error in save_field method: {e}")
            raise

    def load_field(self, folderPath, formatSave=FormatSave.HDR_IMG):
        """
        Load the acoustic field from a file in the specified format.

        Parameters:
        - filePath (str): The folder path from which to load the file.
        """
        try:
            if str(type(formatSave)) != str(AOT_biomaps.AOT_Acoustic.FormatSave):
                    raise ValueError(f"Unsupported file format: {formatSave}. Supported formats are: HDR_IMG, H5, NPY.")

            if self.params['typeSim'] == TypeSim.FIELD2.value:
                raise NotImplementedError("FIELD2 simulation is not implemented yet.")
            elif self.params['typeSim'] == TypeSim.KWAVE.value:
                if formatSave.value == FormatSave.HDR_IMG.value: 
                    if self.params["dim"] == Dim.D2.value:
                        self._load_fieldKWAVE_XZ(os.path.join(folderPath,self.getName_field()+formatSave.value))
                    elif self.params["dim"] == Dim.D3.value:
                        raise NotImplementedError("3D KWAVE field loading is not implemented yet.")
                elif formatSave.value == FormatSave.H5.value:
                    if self.params["dim"] == Dim.D2.value:
                         self._load_field_h5(folderPath)
                    elif self.params["dim"] == Dim.D3.value:
                        raise NotImplementedError("H5 KWAVE field loading is not implemented yet.")
                elif formatSave.value == FormatSave.NPY.value:
                    if self.params["dim"] == Dim.D2.value:
                        self.field = np.load(os.path.join(folderPath,self.getName_field()+formatSave.value))
                    elif self.params["dim"] == Dim.D3.value:
                        raise NotImplementedError("3D NPY KWAVE field loading is not implemented yet.")
            elif self.params['typeSim'] == TypeSim.HYDRO.value:
                print("Loading Hydrophone field...")
                if formatSave.value == FormatSave.HDR_IMG.value:
                    raise ValueError("HDR_IMG format is not supported for Hydrophone acquisition.")
                if formatSave.value == FormatSave.H5.value:
                    if self.params["dim"] == Dim.D2.value:
                        self.field, self.params['Xrange'], self.params['Zrange'] = self._load_fieldHYDRO_XZ(os.path.join(folderPath, self.getName_field() + '.h5'),  os.path.join(folderPath, "PARAMS_" +self.getName_field() + '.mat'))
                    elif self.params["dim"] == Dim.D3.value: 
                        self._load_fieldHYDRO_XYZ(os.path.join(folderPath, self.getName_field() + '.h5'),  os.path.join(folderPath, "PARAMS_" +self.getName_field() + '.mat'))
                elif formatSave.value == FormatSave.NPY.value:
                    if self.params["dim"] == Dim.D2.value:
                        self.field = np.load(folderPath)
                    elif self.params["dim"] == Dim.D3.value:
                        raise NotImplementedError("3D NPY Hydrophone field loading is not implemented yet.")
            else:
                raise ValueError("Invalid simulation type. Supported types are: FIELD2, KWAVE, HYDRO.")
           
        except Exception as e:
            print(f"Error in load_field method: {e}")
            raise

    @abstractmethod
    def getName_field(self):
        pass

    ## DISPLAY METHODS ##

    def plot_burst_signal(self):
        """
        Plot the burst signal used for generating the acoustic field.
        """
        try:
            time2plot = np.arange(0, len(self.burst)) / self.params['f_AQ'] * 1000000  # Convert to microseconds
            plt.figure(figsize=(8, 8))
            plt.plot(time2plot, self.burst)
            plt.title('Excitation burst signal')
            plt.xlabel('Time (µs)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.show()
        except Exception as e:
            print(f"Error in plot_burst_signal method: {e}")
            raise

    def animated_plot_AcousticField(self, desired_duration_ms = 5000, save_dir=None):
        """
        Plot synchronized animations of A_matrix slices for selected angles.

        Args:
            step (int): Time step between frames (default is every 10 frames).
            save_dir (str): Directory to save the animation gif; if None, animation will not be saved.

        Returns:
            ani: Matplotlib FuncAnimation object.
        """
        try:

            maxF = np.max(self.field[:,20:,:])
            minF = np.min(self.field[:,20:,:])
            # Set the maximum embedded animation size to 100 MB
            plt.rcParams['animation.embed_limit'] = 100

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Set main title
            if self.waveType.value == WaveType.FocusedWave.value:
                fig.suptitle("[System Matrix Animation] Focused Wave", fontsize=12, y=0.98)
            elif self.waveType.value == WaveType.PlaneWave.value:
                fig.suptitle(f"[System Matrix Animation] Plane Wave | Angles {self.angle}°", fontsize=12, y=0.98)
            elif self.waveType.value == WaveType.StructuredWave.value:
                fig.suptitle(f"[System Matrix Animation] Structured Wave | Pattern structure: {self.pattern.activeList} | Angles {self.angle}°", fontsize=12, y=0.98)
            else:

                raise ValueError("Invalid wave type. Supported types are: FocusedWave, PlaneWave, StructuredWave.")

            # Initial plot
            im = ax.imshow(
                self.field[0, :, :],
                extent=(self.params['Xrange'][0] * 1000, self.params['Xrange'][-1] * 1000, self.params['Zrange'][-1] * 1000, self.params['Zrange'][0] * 1000),
                vmin = 1.2*minF,
                vmax=0.8*maxF,
                aspect='equal',
                cmap='jet',
                animated=True
            )
            ax.set_title(f"t = 0 ms", fontsize=10)
            ax.set_xlabel("x (mm)", fontsize=8)
            ax.set_ylabel("z (mm)", fontsize=8)


            # Unified update function for all subplots
            def update(frame):
                im.set_data(self.field[frame, :, :])
                ax.set_title(f"t = {frame / self.params['f_AQ'] * 1000:.2f} ms", fontsize=10)
                return [im]  # Return a list of artists that were modified

            interval = desired_duration_ms / self.AcousticFields.shape[0]

            # Create animation
            ani = animation.FuncAnimation(
                fig, update,
                frames=range(0, self.field.shape[0]),
                interval=interval, blit=True
            )

            # Save animation if needed
            if save_dir is not None:
                if self.waveType == WaveType.FocusedWave:
                    save_filename = f"Focused_Wave_.gif"
                elif self.waveType == WaveType.PlaneWave:
                    save_filename = f"Plane_Wave_{self._format_angle()}.gif"
                else:
                    save_filename = f"Structured_Wave_PatternStructure_{self.pattern.activeList}_{self._format_angle()}.gif"
                save_path = os.path.join(save_dir, save_filename)
                ani.save(save_path, writer='pillow', fps=20)
                print(f"Saved: {save_path}")

            plt.close(fig)

            return ani
        except Exception as e:
            print(f"Error creating animation: {e}")
            return None


    ## PRIVATE METHODS ##

    def _generate_burst_signal(self):
        if self.params['typeSim'] == TypeSim.FIELD2.value:
            raise NotImplementedError("FIELD2 simulation is not implemented yet.")
        elif self.params['typeSim'] == TypeSim.KWAVE.value:
            self._generate_burst_signalKWAVE()
        elif self.params['typeSim'] == TypeSim.HYDRO.value:
            raise ValueError("Cannot generate burst signal for Hydrophone simulation.")

    def _generate_burst_signalKWAVE(self):
        """
        Private method to generate a burst signal based on the specified parameters.
        """
        try:
            self.burst = tone_burst(1/self.kgrid.dt, self.params['f_US'], self.params['num_cycles']).squeeze()
        except Exception as e:
            print(f"Error in __generate_burst_signal method: {e}")
            raise

    @abstractmethod
    def _generate_2Dacoustic_field_KWAVE(self, isGpu):
        """
        Generate a 2D acoustic field using k-Wave simulation.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def _generate_3Dacoustic_field_KWAVE(self, isGpu):
        """
        Generate a 3D acoustic field using k-Wave simulation.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def _save2D_HDR_IMG(self, filePath):
        """
        Save the 2D acoustic field as an HDR_IMG file.
        Must be implemented in subclasses.
        """
        pass

    def _load_field_h5(self, filePath):
        """
        Load the 2D acoustic field from an H5 file.

        Parameters:
        - filePath (str): The path to the H5 file.

        Returns:
        - field (numpy.ndarray): The loaded acoustic field.
        """
        try:
            with h5py.File(filePath+self.getName_field()+".h5", 'r') as f:
                self.field = f['data'][:]
        except Exception as e:
            print(f"Error in _load_field_h5 method: {e}")
            raise

    def _save2D_H5(self, filePath):
        """
        Save the 2D acoustic field as an H5 file.

        Parameters:
        - filePath (str): The path where the file will be saved.
        """
        try:
            with h5py.File(filePath+self.getName_field()+"h5", 'w') as f:
                for key, value in self.__dict__.items():
                    if key != 'field':
                        f.create_dataset(key, data=value)
                f.create_dataset('data', data=self.field, compression='gzip')
        except Exception as e:
            print(f"Error in _save2D_H5 method: {e}")
            raise

    def _save2D_NPY(self, filePath):
        """
        Save the 2D acoustic field as a NPY file.

        Parameters:
        - filePath (str): The path where the file will be saved.
        """
        try:
            np.save(filePath+self.getName_field()+"npy", self.field)
        except Exception as e:
            print(f"Error in _save2D_NPY method: {e}")
            raise

    def _load_fieldKWAVE_XZ(self, hdr_path):
        """
        Read an Interfile (.hdr) and its binary file (.img) to reconstruct an acoustic field.

        Parameters:
        - hdr_path (str): The path to the .hdr file.

        Returns:
        - field (numpy.ndarray): The reconstructed acoustic field with dimensions reordered to (X, Z, time).
        - header (dict): A dictionary containing the metadata from the .hdr file.
        """
        try:
            header = {}
            # Read the .hdr file
            with open(hdr_path, 'r') as f:
                for line in f:
                    if ':=' in line:
                        key, value = line.split(':=', 1)
                        key = key.strip().lower().replace('!', '')
                        value = value.strip()
                        header[key] = value

            # Get the associated .img file name
            data_file = header.get('name of data file') or header.get('name of date file')
            if data_file is None:
                raise ValueError(f"Cannot find the data file associated with the header file {hdr_path}")
            img_path = os.path.join(os.path.dirname(hdr_path), os.path.basename(data_file))

            # Determine the field size from metadata
            shape = [int(header[f'matrix size [{i}]']) for i in range(1, 3) if f'matrix size [{i}]' in header]
            if not shape:
                raise ValueError("Cannot determine the shape of the acoustic field from metadata.")

            # Data type
            data_type = header.get('number format', 'short float').lower()
            dtype_map = {
                'short float': np.float32,
                'float': np.float32,
                'int16': np.int16,
                'int32': np.int32,
                'uint16': np.uint16,
                'uint8': np.uint8
            }
            dtype = dtype_map.get(data_type)
            if dtype is None:
                raise ValueError(f"Unsupported data type: {data_type}")

            # Byte order (endianness)
            byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
            endianess = '<' if 'little' in byte_order else '>'

            # Verify the actual size of the .img file
            fileSize = os.path.getsize(img_path)
            timeDim = int(fileSize / (np.dtype(dtype).itemsize * np.prod(shape)))
            shape = shape + [timeDim]

            # Read binary data
            with open(img_path, 'rb') as f:
                data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)

            # Reshape data to (time, Z, X)
            field = data.reshape(shape[::-1])  # NumPy interprets in C order (opposite of MATLAB)

            # Apply scaling factors if available
            rescale_slope = float(header.get('data rescale slope', 1))
            rescale_offset = float(header.get('data rescale offset', 0))
            field = field * rescale_slope + rescale_offset

            self.field = field
        except Exception as e:
            print(f"Error in _load_fieldKWAVE_XZ method: {e}")
            raise

    def _load_fieldHYDRO_XZ(self, file_path_h5, param_path_mat):
        """
        Load the 2D acoustic field for Hydrophone simulation from H5 and MAT files.

        Parameters:
        - file_path_h5 (str): The path to the H5 file.
        - param_path_mat (str): The path to the MAT file.

        Returns:
        - envelope_transposed (numpy.ndarray): The transposed envelope of the acoustic field.
        """
        try:
            # Load parameters from the .mat file
            param = scipy.io.loadmat(param_path_mat)

            # Load the ranges for x and z
            x_test = param['x'].flatten()
            z_test = param['z'].flatten()

            x_range = np.arange(-23, 21.2, 0.2)
            z_range = np.arange(0, 37.2, 0.2)
            X, Z = np.meshgrid(x_range, z_range)

            # Load the data from the .h5 file
            with h5py.File(file_path_h5, 'r') as file:
                data = file['data'][:]

            # Initialize a matrix to store the acoustic data
            acoustic_field = np.zeros((len(z_range), len(x_range), data.shape[1]))

            # Fill the grid with acoustic data
            index = 0
            for i in range(len(z_range)):
                if i % 2 == 0:
                    # Traverse left to right
                    for j in range(len(x_range)):
                        acoustic_field[i, j, :] = data[index]
                        index += 1
                else:
                    # Traverse right to left
                    for j in range(len(x_range) - 1, -1, -1):
                        acoustic_field[i, j, :] = data[index]
                        index += 1

            # Calculate the analytic envelope
            envelope = np.abs(CPU_hilbert(acoustic_field, axis=2))
            # Reorganize the array to have the shape (Times, Z, X)
            envelope_transposed = np.transpose(envelope, (2, 0, 1)).T

            self.field = envelope_transposed
            self.params['Xrange'] = x_range
            self.params['Zrange'] = z_range

        except Exception as e:
            print(f"Error in _load_fieldHYDRO_XZ method: {e}")
            raise

    def _load_fieldHYDRO_YZ(self, file_path_h5, param_path_mat):
        """
        Load the 2D acoustic field for Hydrophone simulation from H5 and MAT files.

        Parameters:
        - file_path_h5 (str): The path to the H5 file.
        - param_path_mat (str): The path to the MAT file.

        Returns:
        - envelope_transposed (numpy.ndarray): The transposed envelope of the acoustic field.
        - y_range (numpy.ndarray): The range of y values.
        - z_range (numpy.ndarray): The range of z values.
        """
        try:
            # Load parameters from the .mat file
            param = scipy.io.loadmat(param_path_mat)

            # Extract the ranges for y and z
            y_range = param['y'].flatten()
            z_range = param['z'].flatten()

            # Load the data from the .h5 file
            with h5py.File(file_path_h5, 'r') as file:
                data = file['data'][:]

            # Calculate the number of scans
            Ny = len(y_range)
            Nz = len(z_range)

            # Create the scan positions
            positions_y = []
            positions_z = []

            for i in range(Nz):
                if i % 2 == 0:
                    # Traverse top to bottom for even rows
                    positions_y.extend(y_range)
                else:
                    # Traverse bottom to top for odd rows
                    positions_y.extend(y_range[::-1])
                positions_z.extend([z_range[i]] * Ny)

            Positions = np.column_stack((positions_y, positions_z))

            # Initialize a matrix to store the reorganized data
            reorganized_data = np.zeros((Ny, Nz, data.shape[1]))

            # Reorganize the data according to the scan positions
            for index, (j, k) in enumerate(Positions):
                y_idx = np.where(y_range == j)[0][0]
                z_idx = np.where(z_range == k)[0][0]
                reorganized_data[y_idx, z_idx, :] = data[index, :]

            # Calculate the analytic envelope
            envelope = np.abs(CPU_hilbert(reorganized_data, axis=2))
            # Reorganize the array to have the shape (Times, Z, Y)
            envelope_transposed = np.transpose(envelope, (2, 0, 1))
            return envelope_transposed, y_range, z_range
        except Exception as e:
            print(f"Error in _load_fieldHYDRO_YZ method: {e}")
            raise

    def _load_fieldHYDRO_XYZ(self, file_path_h5, param_path_mat):
        """
        Load the 3D acoustic field for Hydrophone simulation from H5 and MAT files.

        Parameters:
        - file_path_h5 (str): The path to the H5 file.
        - param_path_mat (str): The path to the MAT file.

        Returns:
        - EnveloppeField (numpy.ndarray): The envelope of the acoustic field.
        - x_range (numpy.ndarray): The range of x values.
        - y_range (numpy.ndarray): The range of y values.
        - z_range (numpy.ndarray): The range of z values.
        """
        try:
            # Load parameters from the .mat file
            param = scipy.io.loadmat(param_path_mat)

            # Extract the ranges for x, y, and z
            x_range = param['x'].flatten()
            y_range = param['y'].flatten()
            z_range = param['z'].flatten()

            # Create a meshgrid for x, y, and z
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

            # Load the data from the .h5 file
            with h5py.File(file_path_h5, 'r') as file:
                data = file['data'][:]

            # Calculate the number of scans
            Nx = len(x_range)
            Ny = len(y_range)
            Nz = len(z_range)
            Nscans = Nx * Ny * Nz

            # Create the scan positions
            if Ny % 2 == 0:
                X = np.tile(np.concatenate([x_range[:, np.newaxis], x_range[::-1, np.newaxis]]), (Ny // 2, 1))
                Y = np.repeat(y_range, Nx)
            else:
                X = np.concatenate([x_range[:, np.newaxis], np.tile(np.concatenate([x_range[::-1, np.newaxis], x_range[:, np.newaxis]]), ((Ny - 1) // 2, 1))])
                Y = np.repeat(y_range, Nx)

            XY = np.column_stack((X.flatten(), Y))

            if Nz % 2 == 0:
                XYZ = np.tile(np.concatenate([XY, np.flipud(XY)]), (Nz // 2, 1))
                Z = np.repeat(z_range, Nx * Ny)
            else:
                XYZ = np.concatenate([XY, np.tile(np.concatenate([np.flipud(XY), XY]), ((Nz - 1) // 2, 1))])
                Z = np.repeat(z_range, Nx * Ny)

            Positions = np.column_stack((XYZ, Z))

            # Initialize a matrix to store the reorganized data
            reorganized_data = np.zeros((Nx, Ny, Nz, data.shape[1]))

            # Reorganize the data according to the scan positions
            for index, (i, j, k) in enumerate(Positions):
                x_idx = np.where(x_range == i)[0][0]
                y_idx = np.where(y_range == j)[0][0]
                z_idx = np.where(z_range == k)[0][0]
                reorganized_data[x_idx, y_idx, z_idx, :] = data[index, :]

            EnveloppeField = np.zeros_like(reorganized_data)

            for y in range(reorganized_data.shape[1]):
                for z in range(reorganized_data.shape[2]):
                    EnveloppeField[:, y, z, :] = np.abs(CPU_hilbert(reorganized_data[:, y, z, :], axis=1))
            self.field = np.transpose(EnveloppeField,  (3, 2, 1, 0))
            self.params['Xrange'] = [x_range[0], x_range[-1]]
            self.params['Yrange'] = [y_range[0], y_range[-1]]
            self.params['Zrange'] = [z_range[0], z_range[-1]]
            self.params['Nx'] = Nx
            self.params['Ny'] = Ny
            self.params['Nz'] = Nz
        except Exception as e:
            print(f"Error in _load_fieldHYDRO_XYZ method: {e}")
            raise
