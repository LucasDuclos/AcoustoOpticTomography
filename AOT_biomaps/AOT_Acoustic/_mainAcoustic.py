import copy
import AOT_biomaps
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Acoustic.AcousticTools import calculate_envelope_squared, loadmat, reshape_field
from AOT_biomaps.AOT_Acoustic.AcousticEnums import TypeSim, Dim, FormatSave, WaveType
from AOT_biomaps.AOT_Medium import Medium


import os
import numpy as np
from scipy.io import loadmat as scipy_loadmat

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from tempfile import gettempdir
from abc import ABC, abstractmethod
import logging
import warnings
import sys
import platform

# Optional kwave imports - will be None if kwave is not installed
KWAVE_AVAILABLE = False
KWAVE_BINARIES_AVAILABLE = False

try:
    from kwave.utils.signals import tone_burst
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    KWAVE_AVAILABLE = True
    
    # Check if kwave binaries are available and executable
    import subprocess
    import sys
    try:
        # Try to check if the CUDA binary exists and is executable
        import kwave
        bin_path = os.path.join(os.path.dirname(kwave.__file__), 'bin')
        if sys.platform.startswith('linux'):
            cuda_bin = os.path.join(bin_path, 'linux', 'kspaceFirstOrder-CUDA')
        elif sys.platform == 'darwin':
            cuda_bin = os.path.join(bin_path, 'mac', 'kspaceFirstOrder-CUDA')
        elif sys.platform == 'win32':
            cuda_bin = os.path.join(bin_path, 'windows', 'kspaceFirstOrder-CUDA.exe')
        else:
            cuda_bin = None
        
        if cuda_bin and os.path.exists(cuda_bin):
            # Try to check if we can execute it (this will fail if dependencies are missing)
            result = subprocess.run([cuda_bin, '-h'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=5)
            KWAVE_BINARIES_AVAILABLE = (result.returncode == 0)
        else:
            KWAVE_BINARIES_AVAILABLE = False
    except Exception:
        KWAVE_BINARIES_AVAILABLE = False
    
    if not KWAVE_BINARIES_AVAILABLE:
        system = platform.system().lower()
        message = "kWave binaries are not available or cannot be executed. Some acoustic simulation features will be disabled."

        if system == "linux":
            message += " On Linux, you may need to install: libaec0 libaec-dev libfftw3-dev"
        elif system == "windows":
            message += " On Windows, ensure Visual C++ Redistributable is installed."
        else:
            message += " Check system dependencies for kWave."

        print(message, file=sys.stderr)  # Clean output without file path
        KWAVE_AVAILABLE = False
            
except ImportError:
    KWAVE_AVAILABLE = False
    warnings.warn("kWave is not available. Some acoustic simulation features will be disabled.", UserWarning)

from AOT_biomaps.Settings import Params

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

    def __init__(self, params, medium):
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
        if type(params) != Params:
            raise TypeError("params must be an instance of the Params class")
        if not isinstance(medium, Medium):
            raise TypeError("medium must be an instance of the Medium class")

        self.medium = medium
        self.params = params
        if self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
            self.generate_burst_signal()
        if self.params.acoustic["dim"] == Dim.D3 and self.params.general["Yrange"] is None:
            raise ValueError("Yrange must be provided for 3D fields.")
            
        self.waveType = None
        self.field = None  

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
            logging.getLogger('root').setLevel(logging.ERROR)
            if self.params.acoustic['typeSim'] == TypeSim.FIELD2.value:
                raise NotImplementedError("FIELD2 simulation is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.SIMPLE_SIM.value:
                self.field = self._generate_acoustic_field_SIMPLE_SIM(show_log)
            elif self.params.acoustic['typeSim'] == TypeSim.KWAVE.value:
                if self.params.acoustic["dim"] == Dim.D2.value:
                    try:
                        field = self._generate_acoustic_field_KWAVE_2D(isGpu, show_log)
                    except Exception as e:
                        raise RuntimeError(f"Failed to generate 2D acoustic field: {e}")
                    self.field = calculate_envelope_squared(field)
                elif self.params.acoustic["dim"] == Dim.D3.value:
                    field = self._generate_acoustic_field_KWAVE_3D(isGpu, show_log)
                    self.field = calculate_envelope_squared(field)
            elif self.params.acoustic['typeSim'] == TypeSim.HYDRO.value:
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

    def load_field(self, folderPath, formatSave=FormatSave.HDR_IMG, nameBlock=None):
        """
        Load the acoustic field from a file in the specified format.

        Parameters:
        - filePath (str): The folder path from which to load the file.
        """
        try:
            if str(type(formatSave)) != str(AOT_biomaps.AOT_Acoustic.FormatSave):
                    raise ValueError(f"Unsupported file format: {formatSave}. Supported formats are: HDR_IMG, H5, NPY.")

            if self.params.acoustic['typeSim'] == TypeSim.FIELD2.value:
                raise NotImplementedError("FIELD2 simulation is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.KWAVE.value or self.params.acoustic['typeSim'] == TypeSim.SIMPLE_SIM.value:
                if formatSave.value == FormatSave.HDR_IMG.value: 
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self._load_fieldKWAVE_XZ(os.path.join(folderPath,self.get_name_field()+formatSave.value))
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("3D KWAVE field loading is not implemented yet.")
                elif formatSave.value == FormatSave.H5.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                         self._load_field_h5(folderPath,nameBlock)
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("H5 KWAVE field loading is not implemented yet.")
                elif formatSave.value == FormatSave.NPY.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self.field = np.load(os.path.join(folderPath,self.get_name_field()+formatSave.value))
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("3D NPY KWAVE field loading is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.HYDRO.value:
                print("Loading Hydrophone field...")
                if formatSave.value == FormatSave.HDR_IMG.value:
                    raise ValueError("HDR_IMG format is not supported for Hydrophone acquisition.")
                if formatSave.value == FormatSave.H5.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self.field, self.params.general['Xrange'], self.params.general['Zrange'] = self._load_fieldHYDRO_XZ(os.path.join(folderPath, self.get_name_field() + '.h5'),  os.path.join(folderPath, "PARAMS_" +self.get_name_field() + '.mat'))
                    elif self.params.acoustic["dim"] == Dim.D3.value: 
                        self._load_fieldHYDRO_XYZ(os.path.join(folderPath, self.get_name_field() + '.h5'),  os.path.join(folderPath, "PARAMS_" +self.get_name_field() + '.mat'))
                elif formatSave.value == FormatSave.NPY.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self.field = np.load(folderPath)
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("3D NPY Hydrophone field loading is not implemented yet.")
            else:
                raise ValueError("Invalid simulation type. Supported types are: FIELD2, KWAVE, HYDRO.")
           
        except Exception as e:
            print(f"Error in load_field method: {e}")
            raise

    @abstractmethod
    def get_name_field(self):
        pass

    ## DISPLAY METHODS ##

    def plot_burst_signal(self, figsize=(4,3)):
        """
        Plot the burst signal used for generating the acoustic field.
        """
        try:
            time2plot = np.arange(0, len(self.burst)) / self.params.acoustic['f_AQ'] * 1000000  # Convert to microseconds
            plt.figure(figsize=figsize)
            plt.plot(time2plot, self.burst)
            plt.title('Excitation burst signal')
            plt.xlabel('Time (µs)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.show()
        except Exception as e:
            print(f"Error in plot_burst_signal method: {e}")
            raise

    def animated_plot_AcousticField(self, desired_duration_ms = 5000, save_dir=None,figsize=(4,3)):
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
            fig, ax = plt.subplots(figsize=figsize)

            # Set main title
            if self.waveType.value == WaveType.FocusedWave.value:
                fig.suptitle("[System Matrix Animation] Focused Wave", y=0.98)
            elif self.waveType.value == WaveType.PlaneWave.value:
                fig.suptitle(f"[System Matrix Animation] Plane Wave | Angles {self.angle}°", y=0.98)
            elif self.waveType.value == WaveType.StructuredWave.value:
                fig.suptitle(f"[System Matrix Animation] Structured Wave | Pattern structure: {self.pattern.activeList} | Angles {self.angle}°", y=0.98)
            else:

                raise ValueError("Invalid wave type. Supported types are: FocusedWave, PlaneWave, StructuredWave.")

            # Initial plot
            im = ax.imshow(
                self.field[0, :, :],
                extent=(self.params.general['Xrange'][0] * 1000, self.params.general['Xrange'][-1] * 1000, self.params.general['Zrange'][-1] * 1000, self.params.general['Zrange'][0] * 1000),
                vmin = 1.2*minF,
                vmax=0.8*maxF,
                aspect='equal',
                cmap='jet',
                animated=True
            )
            ax.set_title(f"t = 0 ms")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")

            # Unified update function for all subplots
            def update(frame):
                im.set_data(self.field[frame, :, :])
                ax.set_title(f"t = {frame / self.params.acoustic['f_AQ'] * 1000:.2f} ms")
                return [im]  # Return a list of artists that were modified

            interval = desired_duration_ms / self.field.shape[0]

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

            try:
                from IPython.display import HTML
                return HTML(ani.to_jshtml())
            except ImportError:
                print("IPython not available. Returning animation object without HTML wrapper.")
                return ani
        except Exception as e:
            print(f"Error creating animation: {e}")
            return None

    def show(self, use_dB=False, reference=1e6,Vmax=None, figsize=(4,3)):
        """
        Display the maximum intensity projection of the acoustic field envelope.

        Parameters:
        - use_dB (bool): If True, display in dB relative to the reference pressure.
        - reference (float): Reference pressure in Pa for dB calculation (default: 1 MPa).
        """
        try:
            if self.field is None:
                raise ValueError("Field data is not available. Please generate or load the field first.")
            if self.field.min() < 0:
                raise ValueError("Calculation of the envelope has not been performed. Please generate the envelope first.")

            # Convertir l'enveloppe au carré en amplitude (Pa) en prenant la racine carrée
            envelope_amplitude = np.sqrt(self.field)

            if use_dB:
                # Convertir en dB re reference (Pa)
                envelope_dB = 20 * np.log10(envelope_amplitude / reference)
                data_to_show = envelope_dB
                unit_label = f'dB re {reference / 1e6} MPa'
                if Vmax is not None:
                    vmax = Vmax
                else:
                    vmax = 0

            else:
                # Convertir en MPa
                envelope_amplitude_mpa = envelope_amplitude / 1e6
                data_to_show = envelope_amplitude_mpa
                unit_label = 'MPa'
                if Vmax is not None:
                    vmax = Vmax
                else:
                    vmax = 0.85*np.max(envelope_amplitude_mpa)

            plt.figure(figsize=figsize)
            plt.imshow(data_to_show.max(axis=0),
                    extent=(self.params.general['Xrange'][0] * 1000, self.params.general['Xrange'][1] * 1000,
                            self.params.general['Zrange'][1] * 1000, self.params.general['Zrange'][0] * 1000),
                    aspect='equal', cmap='jet', vmin=0, vmax=vmax)
            plt.colorbar(label=f'Envelope Amplitude ({unit_label})')
            plt.title('Maximum Intensity Projection of Acoustic Field Envelope')
            plt.xlabel('X (mm)')
            plt.ylabel('Z (mm)')
            plt.show()
        except Exception as e:
            print(f"Error in show method: {e}")
            raise

    ## PRIVATE METHODS ##

    @abstractmethod
    def _generate_acoustic_field_SIMPLE_SIM(self, show_log=False):
        pass

    def generate_burst_signal(self):
        if self.params.acoustic['typeSim'] == TypeSim.FIELD2.value:
            raise NotImplementedError("FIELD2 simulation is not implemented yet.")
        elif self.params.acoustic['typeSim'] == TypeSim.KWAVE.value:
            self._generate_burst_signalKWAVE()
        elif self.params.acoustic['typeSim'] == TypeSim.HYDRO.value:
            raise ValueError("Cannot generate burst signal for Hydrophone simulation.")

    def _generate_burst_signalKWAVE(self):
        """
        Private method to generate a burst signal based on the specified parameters.
        """
        try:
            self.burst = tone_burst(1/self.medium.kgrid.dt, self.params.acoustic['f_US'], self.params.acoustic['emission']['num_cycles']).squeeze()
        except Exception as e:
            print(f"Error in __generate_burst_signal method: {e}")
            raise

    def _generate_acoustic_field_KWAVE_2D(self, isGPU=True if config.get_process() == 'gpu' else False, show_log=True):
        """
        Base function to generate a 2D acoustic field using k-Wave.
        Handles common setup, simulation, and post-processing.
        """
        source = kSource()
        source.p_mask = np.zeros(( self.medium.Nx_reshaped, self.medium.Nz_reshaped))
        # Appel à la méthode spécialisée
        source = self._set_up_source(source, self.medium.Nx_reshaped, self.medium.kgrid.dt, self.medium.dx_reshaped, self.medium.c_mean,self.medium.factorT)  # factorT=1 pour simplifier

        # ---
        sensor = kSensor()
        sensor.mask = np.ones((self.medium.Nx_reshaped, self.medium.Nz_reshaped))
        # ---
        pml_size = 50 

        # ---
        simulation_options = SimulationOptions(
        pml_inside=False, # PML ajoutée autour de la grille Air+PVA
        pml_size=[1, pml_size],
        use_sg=False,
        save_to_disk=True,
        input_filename=os.path.join(gettempdir(), "KwaveIN.h5"),
        output_filename=os.path.join(gettempdir(), "KwaveOUT.h5"),
        smooth_c0 = True,
        smooth_rho0 = True,
        smooth_p0 = True,
        scale_source_terms=True,       # INDISPENSABLE pour source.p
         use_kspace=True,               # Améliore la précision de propagation

        )

        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=config.get_process() == 'gpu' and isGPU,
            device_num=config.bestGPU,
            show_sim_log=show_log
        )

        medium_copy = copy.deepcopy(self.medium) # Avoid in-place modifications of the medium properties during simulation, which can affect subsequent simulations if the same medium object is reused.

        # ---
        sensor_data = kspaceFirstOrder2D(
            kgrid=medium_copy.kgrid,
            medium=medium_copy.kmedium,
            source=source,
            sensor=sensor,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )

        # ---
        data = sensor_data['p'].reshape(self.medium.kgrid.Nt, self.medium.Nz_reshaped, self.medium.Nx_reshaped    )
        if self.medium.factorT != 1 or self.medium.factorX != 1 or self.medium.factorZ != 1:
            data = reshape_field(data, [self.medium.factorT, self.medium.factorX, self.medium.factorZ])
            xStart = (self.medium.Nx_reshaped//2)//self.medium.factorX - (self.params.general['Nx']//2)
            return data[:, :self.params.general['Nz'], xStart:xStart+self.params.general['Nx']]
        else:
            return data[:, :self.params.general['Nz'], xStart:xStart+self.params.general['Nx']]

    # def _generate_acoustic_field_KWAVE_3D(self, isGPU=True, show_log=True):
    #     """
    #     Generate a 3D acoustic field using k-Wave.
    #     """
    #     try:
    #         # ---
    #         dx = self.params['dx']
    #         if dx >= self.params['element_width']:
    #             dx = self.params['element_width'] / 2
    #             if self.params['width_phantom'] is not None:
    #                 Nx = int(np.round((self.params['width_phantom'])/dx))
    #             else:
    #                 Nx = int(round((self.params['Xrange'][1] - self.params['Xrange'][0]) / dx))
    #             if self.params['height_phantom'] is not None:
    #                 Nz = int(np.round((self.params['height_phantom'])/dx))
    #             else:
    #                 Nz = int(round((self.params['Zrange'][1] - self.params['Zrange'][0]) / dx))
    #         else:
    #             if self.params['width_phantom'] is not None:
    #                 Nx = int(np.round((self.params['width_phantom'])/self.params['dx']))
    #             else:
    #                 Nx = int(round((self.params['Xrange'][1] - self.params['Xrange'][0]) / self.params['dx']))
    #             if self.params['height_phantom'] is not None:
    #                 Nz = int(np.round((self.params['height_phantom'])/self.params['dz']))
    #             else:
    #                 Nz = int(round((self.params['Zrange'][1] - self.params['Zrange'][0]) / self.params['dz']))

    #         # ---
    #         factorT = int(np.ceil(self.params['f_AQ'] / self.params['f_saving']))
    #         factorX = int(np.ceil(Nx / self.params['Nx']))
    #         factorZ = int(np.ceil(Nz / self.params['Nz']))

    #         kgrid = kWaveGrid([Nx, Nz], [dx, dx])
    #         kgrid.setTime(self.kgrid.Nt, 1 / self.params['f_AQ'])

    #         source = kSource()
    #         source.p_mask = np.zeros((self.params['Nx'], self.params['Ny'], self.params['Nz']))

    #         # Appel à la méthode spécialisée
    #         self._set_up_source(source, self.params['Nx'], self.params['dx'], factorT)  # factorT=1 pour simplifier

    #         sensor = kSensor()
    #         sensor.mask = np.ones((self.params['Nx'], self.params['Ny'], self.params['Nz']))

    #         simulation_options = SimulationOptions(
    #             pml_inside=False,
    #             pml_auto=True,
    #             use_sg=False,
    #             save_to_disk=True,
    #             input_filename=os.path.join(gettempdir(), "KwaveIN.h5"),
    #             output_filename=os.path.join(gettempdir(), "KwaveOUT.h5")
    #         )

    #         execution_options = SimulationExecutionOptions(
    #             is_gpu_simulation=config.get_process() == 'gpu' and isGPU,
    #             device_num=config.bestGPU,
    #             show_sim_log=show_log
    #         )

    #         sensor_data = kspaceFirstOrder3D(
    #             kgrid=kgrid,
    #             medium=self.medium,
    #             source=source,
    #             sensor=sensor,
    #             simulation_options=simulation_options,
    #             execution_options=execution_options,
    #         )

    #         data = sensor_data['p'].reshape(kgrid.Nt, Nz, Nx)
    #         if factorT != 1 or factorX != 1 or factorZ != 1:
    #             return reshape_field(data, [factorT, factorX, factorZ])
    #         else:
    #             return data

    #     except Exception as e:
    #         print(f"Error generating 3D acoustic field: {e}")
    #         return None
        
    @abstractmethod
    def _set_up_source(self, source, Nx, dt, dx, c0, factorT):
        """
        Abstract method: each subclass must implement its own source setup.
        """
        pass

    @abstractmethod
    def _save2D_HDR_IMG(self, filePath):
        """
        Save the 2D acoustic field as an HDR_IMG file.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def get_name_field(self):
        """
        Abstract method to get the name of the field for saving and loading.
        Must be implemented in subclasses.
        """
        pass

    def _load_field_h5(self, filePath,nameBlock):
        """
        Load the 2D acoustic field from an H5 file.

        Parameters:
        - filePath (str): The path to the H5 file.

        Returns:
        - field (numpy.ndarray): The loaded acoustic field.
        """
        try:
            if nameBlock is None:
                nameBlock = 'data'
            with h5py.File(os.path.join(filePath, self.get_name_field()+".h5"), 'r') as f:
                self.field = f[nameBlock][:]
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
            with h5py.File(filePath+self.get_name_field()+"h5", 'w') as f:
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
            np.save(filePath+self.get_name_field()+"npy", self.field)
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
            param = loadmat(param_path_mat)

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
            self.params.general['Xrange'] = x_range
            self.params.general['Zrange'] = z_range

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
            param = loadmat(param_path_mat)

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
            param = loadmat(param_path_mat)

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
            self.params.general['Xrange'] = [x_range[0], x_range[-1]]
            self.params.general['Yrange'] = [y_range[0], y_range[-1]]
            self.params.general['Zrange'] = [z_range[0], z_range[-1]]
            self.params.general['Nx'] = Nx
            self.params.general['Ny'] = Ny
            self.params.general['Nz'] = Nz
        except Exception as e:
            print(f"Error in _load_fieldHYDRO_XYZ method: {e}")
            raise
