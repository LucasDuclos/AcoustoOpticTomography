from ._mainAcoustic import AcousticField
from .AcousticEnums import WaveType

import os
import numpy as np
import matplotlib.pyplot as plt



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
        try:
            if dx is None:
                dx = self.params['dx']

            # 1. Grid and element setup (utilise les paramètres utilisateur)
            element_width_grid_points = int(round(self.params['element_width'] / dx))
            total_grid_points = self.params['num_elements'] * element_width_grid_points
            element_positions = np.linspace(
                self.params['Xrange'][0] + self.params['element_width']/2,
                self.params['Xrange'][1] - self.params['element_width'],
                self.params['num_elements']
            )

            # 2. Select active elements (utilise les paramètres utilisateur)
            center_idx = np.argmin(np.abs(element_positions - self.focal_line))
            half_N = self.params['N_piezoFocal'] // 2
            start_idx = max(0, center_idx - half_N)
            end_idx = min(self.params['num_elements'] - 1, center_idx + half_N - 1)
            active_elements = np.arange(start_idx, end_idx + 1)
            active_element_positions = element_positions[active_elements]

            # 3. Calculate relative positions
            x_rel = active_element_positions - self.focal_line
            a = np.max(np.abs(x_rel))

            # 4. Calculate the parabolic delays (utilise les paramètres utilisateur)
            tau_max = (a**2) / (2 * self.params['Foc'] * self.params['c0'])
            delays = tau_max * (1 - (x_rel / a)**2)

            # 5. Force the delays at the edges to be zero
            tau_edge = delays[0]
            delays -= tau_edge

            # 6. Convert delays to samples
            delay_samples = np.round(delays / self.kgrid.dt).astype(int)
            max_delay_samples = np.max(delay_samples)
            print(f"Max delay (samples): {max_delay_samples}")

            # 7. Initialize output
            delayed_signals = np.zeros((total_grid_points, len(self.burst) + max_delay_samples))

            # 8. Apply delays
            for elem_idx, elem_delay in zip(active_elements, delay_samples):
                start_grid = elem_idx * element_width_grid_points
                end_grid = start_grid + element_width_grid_points
                for grid_idx in range(start_grid, end_grid):
                    if elem_delay >= 0 and elem_delay + len(self.burst) <= delayed_signals.shape[1]:
                        delayed_signals[grid_idx, elem_delay:elem_delay + len(self.burst)] = self.burst

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
            max_times = max_indices / self.params['f_AQ'] * 1e6

            # Détermine la valeur minimale des temps de maximum (pour les éléments actifs)
            min_active_time = np.min(max_times[max_times > 0])

            # Plot the times of the maxima
            plt.figure(figsize=(10, 6))
            plt.plot(element_indices, max_times, 'o-')
            plt.title('Time of Maximum for Each Delayed Signal')
            plt.xlabel('Transducer Element Index')
            plt.ylabel('Time of Maximum (µs)')
            plt.grid(True)

            # Ajuste l'échelle de l'axe Y pour commencer à la valeur minimale des éléments actifs
            plt.ylim(bottom=min_active_time * 0.95)  # Ajoute une marge de 5% pour plus de lisibilité
            plt.show()
        except Exception as e:
            print(f"Error plotting max times: {e}")

    def _SetUpSource(self, source, Nx, dx, factorT):
        """
        Set up source for both 2D and 3D focused waves.
        """
        element_width_grid_points = int(round(self.params['element_width'] / dx))

        if source.p_mask.ndim == 2:
            # --- 2D ---
            element_positions = np.linspace(
                self.params['Xrange'][0] + self.params['element_width'] / 2,
                self.params['Xrange'][1] - self.params['element_width'] / 2,
                self.params['num_elements']
            )
            center_idx = np.argmin(np.abs(element_positions - self.focal_line))
            start_idx = max(0, center_idx - self.params['N_piezoFocal'] // 2)
            end_idx = min(self.params['num_elements'] - 1, start_idx + self.params['N_piezoFocal'] - 1)
            selected_indices = np.arange(start_idx, end_idx + 1)

            current_position = (Nx - self.params['num_elements'] * element_width_grid_points) // 2
            active_grid_indices = []
            for i in range(self.params['num_elements']):
                x_start = current_position
                x_end = current_position + element_width_grid_points
                if i in selected_indices:
                    source.p_mask[x_start:x_end, 0] = 1
                    active_grid_indices.extend(range(x_start, x_end))
                current_position += element_width_grid_points

            delayed_signals = self._apply_delay(dx=dx) if factorT != 1 else self.delayedSignal
            source.p = float(self.params['voltage']) * float(self.params['sensitivity']) * delayed_signals[active_grid_indices, :]

        elif source.p_mask.ndim == 3:
            # --- 3D ---
            center_index_x = Nx // 2
            center_index_y = self.params['Ny'] // 2

            for i in range(self.params['num_elements']):
                x_pos = center_index_x - (self.params['num_elements'] // 2) * element_width_grid_points + i * element_width_grid_points
                source.p_mask[x_pos, center_index_y, 0] = 1

            delayed_signals = self._apply_delay()
            source.p = float(self.params['voltage']) * float(self.params['sensitivity']) * delayed_signals.T

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
 
