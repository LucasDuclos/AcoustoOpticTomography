from ._mainAcoustic import AcousticField
from .AcousticEnums import WaveType
from AOT_biomaps.Config import config
from .AcousticTools import next_power_of_2, reshape_field

import ctypes
import os
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from kwave.kgrid import kWaveGrid
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from tempfile import gettempdir


class FocusedWave(AcousticField):

    def __init__(self, focal_line, **kwargs):
        """
        Initialize the FocusedWave object.

        Parameters:
        - focal_line (tuple): The focal line coordinates (x) in meters.
        - **kwargs: Additional keyword arguments for AcousticField initialization.
        """
        super().__init__(**kwargs)
        self.waveType = WaveType.FocusedWave
        self.kgrid.setTime(int(self.kgrid.Nt*2),self.kgrid.dt) # Extend the time grid to allow for delays
        self.focal_line = focal_line
        self.delayedSignal = self._apply_delay()

    def getName_field(self):
        """
        Generate the name for the field file based on the focal line.

        Returns:
            str: File name for the system matrix file.
        """
        try:
            return f"field_focused_X{self.focal_line*1000:.2f}"
        except Exception as e:
            print(f"Error generating file name: {e}")
            return None

    def _apply_delay(self, dx=None):
        """
        Apply temporal delays to focus the wave at (self.focal_line, self.params['Foc']).
        Only active elements around focal_line are considered (N_piezoFocal elements).
        Returns:
            ndarray: Delayed signals of shape (total_grid_points, len(burst) + max_delay).
        """
        try:
            if dx is None:
                dx = self.params['dx']

            # 1. Calculate total grid points and points per element
            element_width_grid_points = int(round(self.params['element_width'] / dx))
            total_grid_points = self.params['num_elements'] * element_width_grid_points

            # 2. Calculate element center positions
            element_positions = np.linspace(
                self.params['Xrange'][0] + self.params['element_width']/2,
                self.params['Xrange'][1] - self.params['element_width']/2,
                self.params['num_elements']
            )

            # 3. Determine active elements around focal_line
            center_idx = np.argmin(np.abs(element_positions - self.focal_line))
            start_idx = max(0, center_idx - self.params['N_piezoFocal'] // 2)
            end_idx = min(self.params['num_elements'] - 1, start_idx + self.params['N_piezoFocal'] - 1)
            active_elements = np.arange(start_idx, end_idx + 1)

            # 4. Calculate all grid positions
            grid_positions = np.linspace(
                self.params['Xrange'][0],
                self.params['Xrange'][1] - dx,
                total_grid_points
            )

            # 5. Calculate maximum possible delay
            max_distance = np.sqrt(
                max((self.params['Xrange'][0] - self.focal_line)**2,
                    (self.params['Xrange'][1] - self.focal_line)**2) +
                self.params['Foc']**2
            )
            max_delay_samples = int(np.ceil(max_distance / self.params['c0'] / self.kgrid.dt))

            # 6. Initialize output array
            delayed_signals = np.zeros((total_grid_points, len(self.burst) + max_delay_samples))

            # 7. Calculate and apply delays for active elements only
            for elem_idx in active_elements:
                # Get grid indices for this element
                start_grid = elem_idx * element_width_grid_points
                end_grid = start_grid + element_width_grid_points

                # Calculate delays for each grid point of this element
                for grid_idx in range(start_grid, end_grid):
                    # Distance to focal point (focal_line, Foc)
                    distance = np.sqrt(
                        (grid_positions[grid_idx] - self.focal_line)**2 +
                        self.params['Foc']**2
                    )

                    # Time delay = (max_distance - distance) / speed_of_sound
                    delay_samples = int(np.round((max_distance - distance) / self.params['c0'] / self.kgrid.dt))

                    # Apply delay if valid
                    if delay_samples >= 0 and delay_samples + len(self.burst) <= delayed_signals.shape[1]:
                        delayed_signals[grid_idx, delay_samples:delay_samples + len(self.burst)] = self.burst

            return delayed_signals

        except Exception as e:
            print(f"Error applying delay: {e}")
            return None

  
    def plot_delay(self):
        """
        Plot the time of the maximum of each delayed signal to visualize the wavefront.
        """
        try:
            # Find the index of the maximum for each delayed signal
            max_indices = np.argmax(self.delayedSignal, axis=1)
            element_indices = np.linspace(0, self.params['num_elements'] - 1, self.delayedSignal.shape[0])
            # Convert indices to time
            max_times = max_indices / self.params['f_AQ']*1e6

            # Plot the times of the maxima
            plt.figure(figsize=(10, 6))
            plt.plot(element_indices, max_times, 'o-')
            plt.title('Time of Maximum for Each Delayed Signal')
            plt.xlabel('Transducer Element Index')
            plt.ylabel('Time of Maximum (µs)')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting max times: {e}")


    def _generate_2Dacoustic_field_KWAVE(self, isGPU=True if config.get_process() == 'gpu' else False, show_log=True):
        """
        Generate a 2D acoustic field using k-Wave simulation for a focused wave.
        """
        try:
            # 1. Adjust grid spacing if necessary
            dx = self.params['dx']
            if dx >= self.params['element_width']:
                dx = self.params['element_width'] / 2
                Nx = int(round((self.params['Xrange'][1] - self.params['Xrange'][0]) / dx))
                Nz = int(round((self.params['Zrange'][1] - self.params['Zrange'][0]) / dx))
            else:
                Nx = self.params['Nx']
                Nz = self.params['Nz']

            # 2. Calculate element positions and select active elements
            element_positions = np.linspace(
                self.params['Xrange'][0] + self.params['element_width'] / 2,
                self.params['Xrange'][1] - self.params['element_width'] / 2,
                self.params['num_elements']
            )
            center_idx = np.argmin(np.abs(element_positions - self.focal_line))
            start_idx = max(0, center_idx - self.params['N_piezoFocal'] // 2)
            end_idx = min(self.params['num_elements'] - 1, start_idx + self.params['N_piezoFocal'] - 1)
            selected_indices = np.arange(start_idx, end_idx + 1)

            # 3. Calculate grid factors for downsampling
            factorT = int(np.ceil(self.params['f_AQ'] / self.params['f_saving']))
            factorX = int(np.ceil(Nx / self.params['Nx']))
            factorZ = int(np.ceil(Nz / self.params['Nz']))

            # 4. Calculate element width in grid points
            element_width_grid_points = int(round(self.params['element_width'] / dx))
            total_elements_width = self.params['num_elements'] * element_width_grid_points

            # 5. Create source mask and active grid points list
            source = kSource()
            source.p_mask = np.zeros((Nx, Nz))
            kgrid = kWaveGrid([Nx, Nz], [dx, dx])
            kgrid.setTime(self.kgrid.Nt, 1 / self.params['f_AQ'])

            # Calculate starting position to center the transducer
            current_position = (Nx - total_elements_width) // 2
            active_grid_indices = []

            # 6. Set active elements in source mask and collect active grid indices
            for i in range(self.params['num_elements']):
                x_start = current_position
                x_end = current_position + element_width_grid_points

                if i in selected_indices:
                    source.p_mask[x_start:x_end, 0] = 1  # Mark all grid points of this element
                    active_grid_indices.extend(range(x_start, x_end))  # Collect all grid points indices

                current_position += element_width_grid_points

            # 7. Prepare delayed signals
            if factorT != 1:
                delayedSignal = self._apply_delay(dx=dx)  # Should return (total_grid_points, ...)
            else:
                delayedSignal = self.delayedSignal

            # Verify delayedSignal has correct shape
            if delayedSignal.shape[0] != total_elements_width:
                raise ValueError(f"Expected delayedSignal to have {total_elements_width} rows, got {delayedSignal.shape[0]}")

            # 8. Select signals for active grid points only
            source.p = delayedSignal[active_grid_indices, :]

            # Verify the number of time series matches the number of active grid points
            if len(active_grid_indices) != source.p.shape[0]:
                raise ValueError(f"Mismatch: {len(active_grid_indices)} active grid points vs {source.p.shape[0]} time series")

            # 9. Define sensor mask
            sensor = kSensor()
            sensor.mask = np.ones((Nx, Nz))

            # 10. Calculate PML sizes
            total_size_x = next_power_of_2(Nx)
            total_size_z = next_power_of_2(Nz)
            pml_x_size = (total_size_x - Nx) // 2
            pml_z_size = (total_size_z - Nz) // 2

            # 11. Simulation options
            simulation_options = SimulationOptions(
                pml_inside=False,
                pml_size=[pml_x_size, pml_z_size],
                use_sg=False,
                save_to_disk=True,
                input_filename=os.path.join(gettempdir(), "KwaveIN.h5"),
                output_filename=os.path.join(gettempdir(), "KwaveOUT.h5")
            )

            execution_options = SimulationExecutionOptions(
                is_gpu_simulation=config.get_process() == 'gpu' and isGPU,
                device_num=config.bestGPU,
                show_sim_log=show_log
            )

            # 12. Run simulation
            sensor_data = kspaceFirstOrder2D(
                kgrid=kgrid,
                medium=self.medium,
                source=source,
                sensor=sensor,
                simulation_options=simulation_options,
                execution_options=execution_options,
            )

            # 13. Process results
            data = sensor_data['p'].reshape(kgrid.Nt, Nz, Nx)
            if factorT != 1 or factorX != 1 or factorZ != 1:
                return reshape_field(data, [factorT, factorX, factorZ])
            else:
                return data

        except Exception as e:
            print(f"Error generating 2D acoustic field: {e}")
            return None




    def _generate_3Dacoustic_field_KWAVE(self, isGPU=True if config.get_process() == 'gpu' else False, show_log = True):
        """
        Generate a 3D acoustic field using k-Wave simulation for a focused wave.

        Parameters:
        - isGpu (bool): Flag indicating whether to use GPU for simulation.

        Returns:
            ndarray: Simulated acoustic field data.
        """
        try:
            # Create a source mask for the transducer
            source = kSource()
            source.p_mask = np.zeros((self.params['Nx'], self.params['Ny'], self.params['Nz']))

            # Calculate the center of the transducer
            center_index_x = self.params['Nx'] // 2
            center_index_y = self.params['Ny'] // 2

            # Set the active elements in the source mask
            element_width_grid_points = int(round(self.params['element_width'] / self.params['dx']))
            for i in range(self.params['num_elements']):
                x_pos = center_index_x - (self.params['num_elements'] // 2) * element_width_grid_points + i * element_width_grid_points
                source.p_mask[x_pos, center_index_y, 0] = 1

            # Apply delays to the burst signal using the _apply_delay method
            delayed_signals = self._apply_delay()

            source.p = delayed_signals.T

            # Define sensors to observe acoustic fields
            sensor = kSensor()
            sensor.mask = np.ones((self.params['Nx'], self.params['Ny'], self.params['Nz']))

            # Simulation options
            simulation_options = SimulationOptions(
                pml_inside=False,
                pml_auto=True,
                use_sg=False,
                save_to_disk=True,
                input_filename=os.path.join(gettempdir(), "KwaveIN.h5"),
                output_filename=os.path.join(gettempdir(), "KwaveOUT.h5")
            )

            execution_options = SimulationExecutionOptions(
                is_gpu_simulation=config.get_process() == 'gpu' and isGPU,
                device_num=config.bestGPU,
                show_sim_log= show_log
            )

            # Run the simulation
            print("Starting simulation...")
            sensor_data = kspaceFirstOrder3D(
                kgrid=self.kgrid,
                medium=self.medium,
                source=source,
                sensor=sensor,
                simulation_options=simulation_options,
                execution_options=execution_options,
            )
            print("Simulation completed successfully.")

            return sensor_data['p'].reshape(self.kgrid.Nt, self.params['Nz'], self.params['Ny'], self.params['Nx'])
        except Exception as e:
            print(f"Error generating 3D acoustic field: {e}")
            return None

    def _save2D_HDR_IMG(self, filePath):
        """
        Save the acoustic field to .img and .hdr files.

        Parameters:
        - filePath (str): Path to the folder where files will be saved.
        """
        try:
            t_ex = 1 / self.params['f_US']
            x_focal, z_focal = self.focal_point

            # Define file names (img and hdr)
            file_name = f"field_focused_{x_focal:.2f}_{z_focal:.2f}"

            img_path = os.path.join(filePath, file_name + ".img")
            hdr_path = os.path.join(filePath, file_name + ".hdr")

            # Save the acoustic field to the .img file
            with open(img_path, "wb") as f_img:
                self.field.astype('float32').tofile(f_img)

            # Generate headerFieldGlob
            headerFieldGlob = (
                f"!INTERFILE :=\n"
                f"modality : AOT\n"
                f"voxels number transaxial: {self.field.shape[2]}\n"
                f"voxels number transaxial 2: {self.field.shape[1]}\n"
                f"voxels number axial: {1}\n"
                f"field of view transaxial: {(self.params['Xrange'][1] - self.params['Xrange'][0]) * 1000}\n"
                f"field of view transaxial 2: {(self.params['Zrange'][1] - self.params['Zrange'][0]) * 1000}\n"
                f"field of view axial: {1}\n"
            )

            # Generate header
            header = (
                f"!INTERFILE :=\n"
                f"!imaging modality := AOT\n\n"
                f"!GENERAL DATA :=\n"
                f"!data offset in bytes := 0\n"
                f"!name of data file := system_matrix/{file_name}.img\n\n"
                f"!GENERAL IMAGE DATA\n"
                f"!total number of images := {self.field.shape[0]}\n"
                f"imagedata byte order := LITTLEENDIAN\n"
                f"!number of frame groups := 1\n\n"
                f"!STATIC STUDY (General) :=\n"
                f"number of dimensions := 3\n"
                f"!matrix size [1] := {self.field.shape[2]}\n"
                f"!matrix size [2] := {self.field.shape[1]}\n"
                f"!matrix size [3] := {self.field.shape[0]}\n"
                f"!number format := short float\n"
                f"!number of bytes per pixel := 4\n"
                f"scaling factor (mm/pixel) [1] := {self.params['dx'] * 1000}\n"
                f"scaling factor (mm/pixel) [2] := {self.params['dx'] * 1000}\n"
                f"scaling factor (s/pixel) [3] := {1 / self.params['f_AQ']}\n"
                f"first pixel offset (mm) [1] := {self.params['Xrange'][0] * 1e3}\n"
                f"first pixel offset (mm) [2] := {self.params['Zrange'][0] * 1e3}\n"
                f"first pixel offset (s) [3] := 0\n"
                f"data rescale offset := 0\n"
                f"data rescale slope := 1\n"
                f"quantification units := 1\n\n"
                f"!SPECIFIC PARAMETERS :=\n"
                f"focal point (x, z) := {x_focal}, {z_focal}\n"
                f"number of US transducers := {self.params['num_elements']}\n"
                f"delay (s) := 0\n"
                f"us frequency (Hz) := {self.params['f_US']}\n"
                f"excitation duration (s) := {t_ex}\n"
                f"!END OF INTERFILE :=\n"
            )

            # Save the .hdr file
            with open(hdr_path, "w") as f_hdr:
                f_hdr.write(header)

            with open(os.path.join(filePath, "field.hdr"), "w") as f_hdr2:
                f_hdr2.write(headerFieldGlob)
        except Exception as e:
            print(f"Error saving HDR/IMG files: {e}")
 
