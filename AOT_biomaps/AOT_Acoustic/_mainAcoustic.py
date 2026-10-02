import copy
import AOT_biomaps
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Acoustic.AcousticTools import calculate_envelope, calculate_envelope_squared, loadmat, sensor_data_to_grid, compute_target_sizes, resample_field, to_pipeline_layout
from AOT_biomaps.AOT_Acoustic.AcousticEnums import TypeSim, Dim, FormatSave, WaveType
from AOT_biomaps.AOT_Medium import Medium


import os
import numpy as np
import shutil
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
import uuid


# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Optional kwave imports - will be None if kwave is not installed
KWAVE_AVAILABLE = False
KWAVE_BINARIES_AVAILABLE = False

try:
    from kwave.utils.signals import tone_burst
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    KWAVE_AVAILABLE = True
    
    # correct kwave issue with subprocess.Popen on Windows and Linux (encoding)
    import subprocess
    import sys

    _original_popen = subprocess.Popen

    class PatchedPopen(_original_popen):
        def __init__(self, *args, **kwargs):
            if kwargs.get('text', False) or kwargs.get('universal_newlines', False):
                kwargs.setdefault('encoding', 'utf-8')
                kwargs.setdefault('errors', 'replace') # Remplace les caractères impossibles à lire
            super().__init__(*args, **kwargs)

    subprocess.Popen = PatchedPopen

    # Check if kwave binaries are available and executable
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
        message = "[AOT-biomaps] Warning: kWave binaries are not available or cannot be executed. Some acoustic simulation features will be disabled."

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
    print("[AOT-biomaps] Warning: kWave is not available. Some acoustic simulation features will be disabled.")

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
            raise TypeError(f"[AOT-biomaps] params must be an instance of the Params class")
        if not isinstance(medium, Medium):
            raise TypeError(f"[AOT-biomaps] medium must be an instance of the Medium class")

        self.medium = medium
        self.params = params
        if self.params.acoustic['typeSim'] != TypeSim.SIMPLE_SIM.value:
            self.generate_burst_signal()
        if self.params.acoustic["dim"] == Dim.D3 and self.params.general["Yrange"] is None:
            raise ValueError(f"[AOT-biomaps] Yrange must be provided for 3D fields.")
            
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
            print(f"[AOT-biomaps] Error in __del__ method: {e}")
            raise

    def generate_field(self, burst=None, isGPU=None, GPUdevice=None, tempFieldName=None, generation_type="envelope_squared", show_log=False, backend=None):
        """
        Generate the acoustic field.

        k-Wave path: simulation + envelope + band-limited resampling to the
        SAVE grid (general.dx/dz, 1/ft), fused in a single GPU pass.
        SIMPLE_SIM path: analytic field produced directly on the save grid.
        In both cases self.field has layout (Nt', Nz', Nx') and the effective
        sampling is stored in self.last_decimation.

        backend: None/"python" (default) or "cpp" (compiled binaries, temp
        h5 files named after tempFieldName -> required for multi-GPU
        threading; tempFileName must be UNIQUE per concurrent field).
        """
        try:
            if isGPU is None:
                isGPU = True if config.get_process() == 'gpu' else False

            if self.medium.medium_properties is None:
                raise ValueError("[AOT-biomaps] Medium properties are not defined. Please generate or load a valid Medium object.")

            logging.getLogger('root').setLevel(logging.ERROR)

            if self.params.acoustic['typeSim'] == TypeSim.FIELD2.value:
                raise NotImplementedError("[AOT-biomaps] FIELD2 simulation is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.SIMPLE_SIM.value:
                self.field = self._generate_acoustic_field_SIMPLE_SIM(burst=burst, show_log=show_log)
            elif self.params.acoustic['typeSim'] == TypeSim.KWAVE.value:
                if self.params.acoustic["dim"] == Dim.D2.value:
                    self.field = self._generate_acoustic_field_KWAVE_2D(
                        burst=burst, isGPU=isGPU, GPUdevice=GPUdevice,
                        generation_type=generation_type, show_log=show_log,
                        backend=backend, tempFileName=tempFieldName)
                elif self.params.acoustic["dim"] == Dim.D3.value:
                    raise NotImplementedError("[AOT-biomaps] 3D k-Wave generation is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.HYDRO.value:
                raise ValueError("[AOT-biomaps] Cannot generate field for Hydrophone simulation, load existing acquisitions.")
            else:
                raise ValueError("[AOT-biomaps] Invalid simulation type. Supported types are: FIELD2, KWAVE, SIMPLE_SIM, HYDRO.")
        except Exception as e:
            print(f"[AOT-biomaps] Error in generate_field method: {e}")
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
                raise ValueError(f"[AOT-biomaps] Unsupported format. Supported formats are: HDR_IMG, H5, NPY.")
        except Exception as e:
            print(f"[AOT-biomaps] Error in save_field method: {e}")
            raise

    def load_field(self, folderPath, formatSave=FormatSave.HDR_IMG, nameBlock=None):
        """
        Load the acoustic field from a file in the specified format.

        Parameters:
        - filePath (str): The folder path from which to load the file.
        """
        try:
            if str(type(formatSave)) != str(AOT_biomaps.AOT_Acoustic.FormatSave):
                    raise ValueError(f"[AOT-biomaps] Unsupported file format: {formatSave}. Supported formats are: HDR_IMG, H5, NPY.")

            if self.params.acoustic['typeSim'] == TypeSim.FIELD2.value:
                raise NotImplementedError("[AOT-biomaps] FIELD2 simulation is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.KWAVE.value or self.params.acoustic['typeSim'] == TypeSim.SIMPLE_SIM.value:
                if formatSave.value == FormatSave.HDR_IMG.value: 
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self._load_fieldKWAVE_XZ(os.path.join(folderPath,self.get_name_field()+formatSave.value))
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("[AOT-biomaps] 3D KWAVE field loading is not implemented yet.")
                elif formatSave.value == FormatSave.H5.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                         self._load_field_h5(folderPath,nameBlock)
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("[AOT-biomaps] H5 KWAVE field loading is not implemented yet.")
                elif formatSave.value == FormatSave.NPY.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self.field = np.load(os.path.join(folderPath,self.get_name_field()+formatSave.value))
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("[AOT-biomaps] 3D NPY KWAVE field loading is not implemented yet.")
            elif self.params.acoustic['typeSim'] == TypeSim.HYDRO.value:
                print("[AOT-biomaps] Loading Hydrophone field...")
                if formatSave.value == FormatSave.HDR_IMG.value:
                    raise ValueError("[AOT-biomaps] HDR_IMG format is not supported for Hydrophone acquisition.")
                if formatSave.value == FormatSave.H5.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self.field, self.params.general['Xrange'], self.params.general['Zrange'] = self._load_fieldHYDRO_XZ(os.path.join(folderPath, self.get_name_field() + '.h5'),  os.path.join(folderPath, "PARAMS_" +self.get_name_field() + '.mat'))
                    elif self.params.acoustic["dim"] == Dim.D3.value: 
                        self._load_fieldHYDRO_XYZ(os.path.join(folderPath, self.get_name_field() + '.h5'),  os.path.join(folderPath, "PARAMS_" +self.get_name_field() + '.mat'))
                elif formatSave.value == FormatSave.NPY.value:
                    if self.params.acoustic["dim"] == Dim.D2.value:
                        self.field = np.load(folderPath)
                    elif self.params.acoustic["dim"] == Dim.D3.value:
                        raise NotImplementedError("[AOT-biomaps] 3D NPY Hydrophone field loading is not implemented yet.")
            else:
                raise ValueError("[AOT-biomaps] Invalid simulation type. Supported types are: FIELD2, KWAVE, HYDRO.")

        except Exception as e:
            print(f"[AOT-biomaps] Error in load_field method: {e}")
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
            print(f"[AOT-biomaps] Error in plot_burst_signal method: {e}")
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

            # Effective time step of self.field (decimated fields are NOT
            # sampled at 1/f_AQ)
            dec = getattr(self, 'last_decimation', None)
            dt_frame = dec['dt'] if dec else 1.0 / float(self.params.acoustic['f_AQ'])

            # Unified update function for all subplots
            def update(frame):
                im.set_data(self.field[frame, :, :])
                ax.set_title(f"t = {frame * dt_frame * 1000:.2f} ms")
                return [im]

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
                print(f"[AOT-biomaps] Saved: {save_path}")

            plt.close(fig)

            try:
                from IPython.display import HTML
                return HTML(ani.to_jshtml())
            except ImportError:
                print("[AOT-biomaps] IPython not available. Returning animation object without HTML wrapper.")
                return ani
        except Exception as e:
            print(f"[AOT-biomaps] Error creating animation: {e}")
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
            print(f"[AOT-biomaps] Error in show method: {e}")
            raise

    ## PRIVATE METHODS ##

    @abstractmethod
    def _generate_acoustic_field_SIMPLE_SIM(self, burst=None, show_log=False):
        pass

    def generate_burst_signal(self):
        if self.params.acoustic['typeSim'] == TypeSim.FIELD2.value:
            raise NotImplementedError("[AOT-biomaps] FIELD2 simulation is not implemented yet.")
        elif self.params.acoustic['typeSim'] == TypeSim.KWAVE.value:
            self._generate_burst_signalKWAVE()
        elif self.params.acoustic['typeSim'] == TypeSim.HYDRO.value:
            raise ValueError("[AOT-biomaps] Cannot generate burst signal for Hydrophone simulation.")

    def _generate_burst_signalKWAVE(self):
        """
        Private method to generate a burst signal based on the specified parameters.
        """
        try:
            self.burst = tone_burst(1/self.medium.kgrid.dt, self.params.acoustic['f_US'], self.params.acoustic['emission']['num_cycles']).squeeze()
        except Exception as e:
            print(f"[AOT-biomaps] Error in _generate_burst_signal method: {e}")
            raise

    def _generate_acoustic_field_KWAVE_2D(self, burst=None, isGPU=None, GPUdevice=None, generation_type="envelope_squared", show_log=True, keep_on_gpu=False, free_vram=True, target_dt=None, target_dx=None, target_dz=None, backend=None, tempFileName=None):
        """
        k-Wave simulation + post-processing, all on GPU, in one pass:
            1. simulation on the SIMULATION grid (medium.kgrid),
            2. envelope / envelope squared (Hilbert, time axis, FULL rate),
            3. band-limited resampling to the TARGET resolutions
               (defaults: medium.dx_save/dz_save/dt_save, i.e. the schema).

        Backends:
            - "python": Native kwave-python solver, in-memory, single-GPU/GIL-bound.
            - "cpp": Compiled binary via subprocess (releases GIL, enables multi-GPU scaling).
                     HDF5 scratch files are placed in /dev/shm and cleaned up immediately.

        Returns:
            Acoustic field in pipeline layout (Nt', Nz', Nx').
        """
        if target_dt is None:
            target_dt = getattr(self.medium, "dt_save", None)
        if target_dx is None:
            target_dx = getattr(self.medium, "dx_save", None)
        if target_dz is None:
            target_dz = getattr(self.medium, "dz_save", None)
        if target_dz is None:
            target_dz = target_dx

        if backend is None:
            backend = "python"
        if backend == "cpp" and not KWAVE_BINARIES_AVAILABLE:
            print(
                "[AOT-biomaps] Warning: k-Wave binaries unavailable. Falling back to python backend."
            )
            backend = "python"

        data_path = None
        if backend == "cpp":
            # Fix kwave-python 0.6.2 issue where unspecified source mode maps to 2 (invalid for C++)
            try:
                import kwave.solvers.cpp_simulation as cs

                for key in (None, "", "default"):
                    if key not in cs._SOURCE_MODE_MAP:
                        cs._SOURCE_MODE_MAP[key] = 0
            except Exception:
                pass

            # Unique scratch directory per worker in RAM (/dev/shm)
            base_tmp = "/dev/shm" if os.path.isdir("/dev/shm") else gettempdir()
            unique = tempFileName if tempFileName else uuid.uuid4().hex
            data_path = os.path.join(base_tmp, f"AOT_kwave_{unique}")
            
            if os.path.exists(data_path):
                shutil.rmtree(data_path, ignore_errors=True)
                
            os.makedirs(data_path, exist_ok=True)

        try:
            if isGPU is None:
                isGPU = True if config.get_process() == "gpu" else False
            if GPUdevice is None:
                GPUdevice = config.select_best_gpu()

            if isGPU and not CUPY_AVAILABLE:
                print(
                    "[AOT-biomaps] Warning: CuPy not available -> CPU fallback."
                )
                isGPU = False
            if isGPU and CUPY_AVAILABLE:
                cp.cuda.Device(GPUdevice).use()
            xp = cp if (isGPU and CUPY_AVAILABLE) else np

            # --- 1. Source and sensor setup (simulation grid) ------------
            source = kSource()
            source.p_mask = np.zeros(
                (self.medium.Nx_reshaped, self.medium.Nz_reshaped), dtype=bool
            )
            source = self._set_up_source(source, burst=burst)

            sensor = kSensor()
            sensor.mask = np.ones(
                (self.medium.Nx_reshaped, self.medium.Nz_reshaped), dtype=bool
            )

            pml_val = self.params.acoustic["medium"].get("pml_size", 0)
            pml_size = (
                [pml_val, pml_val]
                if isinstance(pml_val, int)
                else list(pml_val)
            )

            medium_copy = copy.deepcopy(self.medium)

            # =============================================================
            # FIX 1: Exact C++ source normalization by 2*c0^2
            # =============================================================
            if backend == "cpp" and getattr(source, "p", None) is not None:
                c0_val = getattr(medium_copy.kmedium, "sound_speed", None)
                if c0_val is None:
                    c0_val = getattr(medium_copy.kmedium, "c0", 1480.0)
                c_ref = float(np.max(np.asarray(c0_val)))

                G = 2.0 * (c_ref**2)
                source.p = (source.p / G).astype(np.float32)

                if show_log:
                    print(
                        f"[AOT-biomaps] C++ scaling applied: source divided by 2*c0^2 (G={G:.4e})"
                    )

            # =============================================================
            # FIX EXPERT : Neutraliser alpha_mode='no_absorption' pour le C++
            # Le binaire C++ bloque bêtement là-dessus alors qu'il gère les zéros nativement.
            # =============================================================
            if backend == "cpp" and hasattr(medium_copy, "kmedium") and medium_copy.kmedium is not None:
                if getattr(medium_copy.kmedium, "alpha_mode", None) is not None:
                    medium_copy.kmedium.alpha_mode = None

            # --- 2. Simulation execution ---------------------------------
            sim_kwargs = dict(
                pml_size=pml_size,
                use_sg=False,
                use_kspace=True,
                smooth_p0=True,
                backend=backend,
                device="gpu" if isGPU else "cpu",
                quiet=not show_log,
            )
            if isGPU:
                sim_kwargs["device_num"] = GPUdevice
            if backend == "python":
                sim_kwargs["dtype"] = np.float32
            else:
                sim_kwargs["data_path"] = data_path

            sensor_data = kspaceFirstOrder(
                medium_copy.kgrid,
                medium_copy.kmedium,
                source,
                sensor,
                **sim_kwargs,
            )

            # =============================================================
            # FIX 2: Early NaN detection guard
            # =============================================================
            p_raw = sensor_data["p"]
            if CUPY_AVAILABLE and isinstance(p_raw, cp.ndarray):
                n_nan = int(cp.count_nonzero(cp.isnan(p_raw)))
            else:
                n_nan = int(np.count_nonzero(np.isnan(p_raw)))

            if n_nan > 0:
                raise ValueError(
                    f"[AOT-biomaps] k-Wave ({backend}) returned {n_nan} NaNs. "
                    f"Simulation diverged -- field {tempFileName} was not saved."
                )

            # --- 3. Layout reconstruction and memory cleanup -------------
            Nx, Nz = self.medium.Nx_reshaped, self.medium.Nz_reshaped

            # =============================================================
            # FIX 3: Backend-specific layout handling (C-order vs Fortran)
            # =============================================================
            g = sensor_data_to_grid(
                sensor_data["p"], Nx, Nz, backend=backend, xp=xp
            )
            Nt = g.shape[-1]
            del sensor_data

            if data_path is not None:
                shutil.rmtree(data_path, ignore_errors=True)
                data_path = None

            # --- 4. Post-processing (Envelope and Resampling) -------------
            dx = float(self.medium.dx_reshaped)
            dz = float(getattr(self.medium, "dz_reshaped", dx))
            dt = float(medium_copy.kgrid.dt)

            sizes, steps = compute_target_sizes(
                Nx, Nz, Nt, dx, dz, dt, target_dt, target_dx, target_dz
            )
            self.last_decimation = dict(zip(("Nt", "Nz", "Nx"), sizes)) | dict(
                zip(("dt", "dz", "dx"), steps)
            )

            if show_log:
                print(
                    f"Resampling: ({Nx},{Nz},{Nt}) -> {tuple(reversed(sizes))} | "
                    f"dt {dt*1e9:.1f}->{steps[0]*1e9:.1f} ns, "
                    f"dx {dx*1e6:.0f}->{steps[2]*1e6:.1f} um, "
                    f"dz {dz*1e6:.0f}->{steps[1]*1e6:.1f} um"
                )

            if generation_type == "envelope_squared":
                out = calculate_envelope_squared(g, xp=xp)
            elif generation_type == "envelope":
                out = calculate_envelope(g, xp=xp)
            elif generation_type == "field":
                out = g
            else:
                raise ValueError(
                    f"[AOT-biomaps] Invalid generation_type: {generation_type}. "
                    "Supported: 'envelope_squared', 'envelope', 'field'."
                )

            if generation_type != "field":
                del g

            out = resample_field(out, sizes, xp=xp)
            out = to_pipeline_layout(out, xp=xp)

            if keep_on_gpu:
                return out

            if CUPY_AVAILABLE and isinstance(out, cp.ndarray):
                out_np = cp.asnumpy(out)
            else:
                out_np = np.asarray(out)
            del out

            if free_vram and CUPY_AVAILABLE and isGPU:
                cp.get_default_memory_pool().free_all_blocks()

            return out_np

        except Exception as e:
            if data_path is not None:
                shutil.rmtree(data_path, ignore_errors=True)
            print(f"[AOT-biomaps] Error in _generate_acoustic_field_KWAVE_2D: {e}")
            raise

    @abstractmethod
    def _set_up_source(self, source, burst=None):
        """
        Abstract method: each subclass must implement its own source setup.
        All grid/physical parameters come from self.medium (SIMULATION grid);
        the source is always built at the simulation sampling rate.
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
            print(f"[AOT-biomaps] Error in _load_field_h5 method: {e}")
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
            print(f"[AOT-biomaps] Error in _save2D_H5 method: {e}")
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
            print(f"[AOT-biomaps] Error in _save2D_NPY method: {e}")
            raise

    def _load_fieldKWAVE_XZ(self, hdr_path):
        """
        Read an Interfile (.hdr) and its binary file (.img).
        Loads the field lazily via np.memmap (read-only) and restores the
        EFFECTIVE sampling metadata (last_decimation) from the header, so
        that display/save/reconstruction use the correct grid even after
        loading a decimated field.
        """
        try:
            header = {}
            with open(hdr_path, 'r') as f:
                for line in f:
                    if ':=' in line:
                        key, value = line.split(':=', 1)
                        key = key.strip().lower().replace('!', '')
                        header[key] = value.strip()

            data_file = header.get('name of data file') or header.get('name of date file')
            if data_file is None:
                raise ValueError(f"Cannot find the data file associated with the header file {hdr_path}")
            img_path = os.path.join(os.path.dirname(hdr_path), os.path.basename(data_file))

            shape = [int(header[f'matrix size [{i}]']) for i in range(1, 3) if f'matrix size [{i}]' in header]
            if not shape:
                raise ValueError("Cannot determine the shape of the acoustic field from metadata.")

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

            byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
            endianess = '<' if 'little' in byte_order else '>'

            fileSize = os.path.getsize(img_path)
            timeDim = int(fileSize / (np.dtype(dtype).itemsize * np.prod(shape)))
            if f'matrix size [3]' in header and int(header['matrix size [3]']) != timeDim:
                raise ValueError(f"[AOT-biomaps] Truncated .img: header declares "
                                 f"{header['matrix size [3]']} time samples, file has {timeDim}.")
            shape = shape + [timeDim]

            self.field = np.memmap(img_path, dtype=endianess + np.dtype(dtype).char,
                                   mode='r', shape=tuple(shape[::-1]))

            # Apply scaling factors if available (materializes a copy only if needed)
            rescale_slope = float(header.get('data rescale slope', 1))
            rescale_offset = float(header.get('data rescale offset', 0))
            if rescale_slope != 1.0 or rescale_offset != 0.0:
                self.field = self.field * rescale_slope + rescale_offset

            # ------------------------------------------------------------------
            # Effective sampling metadata, FROM THE HEADER (survives
            # generation, save, load, re-save cycles)
            # ------------------------------------------------------------------
            self.last_decimation = None
            try:
                dt_h = float(header.get('scaling factor (s/pixel) [3]'))
                dx_h = float(header.get('scaling factor (mm/pixel) [1]')) / 1000.0
                dz_h = float(header.get('scaling factor (mm/pixel) [2]')) / 1000.0
                self.last_decimation = {'Nt': timeDim, 'dt': dt_h,
                                        'Nx': shape[0], 'dx': dx_h,
                                        'Nz': shape[1], 'dz': dz_h}
            except (TypeError, ValueError):
                print("[AOT-biomaps] Warning: incomplete sampling metadata in the "
                      "header; last_decimation not restored.")

        except Exception as e:
            print(f"[AOT-biomaps] Error in _load_fieldKWAVE_XZ method: {e}")
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
            print(f"[AOT-biomaps] Error in _load_fieldHYDRO_XZ method: {e}")
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
            print(f"[AOT-biomaps] Error in _load_fieldHYDRO_YZ method: {e}")
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
            print(f"[AOT-biomaps] Error in _load_fieldHYDRO_XYZ method: {e}")
            raise
