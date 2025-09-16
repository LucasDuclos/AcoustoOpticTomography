from .Laser import Laser
from .Absorber import Absorber

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Phantom:
    """
    Class to apply absorbers to a laser field in the XZ plane.
    """
    def __init__(self, params):
        """
        Initializes the phantom with the given parameters.
        :param params: Configuration parameters for the phantom.
        """
        try:
            absorber_params = params.optic['absorbers']
            self.absorbers = [Absorber(**a) for a in absorber_params]
            self.laser = Laser(params)
            self.phantom = self._apply_absorbers()
            self.phantom = np.transpose(self.phantom)
        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}")
        except Exception as e:
            raise RuntimeError(f"Error initializing Phantom: {e}")

    def _apply_absorbers(self):
        """
        Applies the absorbers to the laser field.
        :return: Intensity matrix of the phantom with applied absorbers.
        """
        try:
            X, Z = np.meshgrid(self.laser.x, self.laser.z, indexing='ij')
            intensity = np.copy(self.laser.intensity)
            for absorber in self.absorbers:
                r2 = (X - absorber.center[0] * 1000)**2 + (Z - absorber.center[1] * 1000)**2
                absorption = -absorber.amplitude * np.exp(-r2 / (absorber.radius * 1000)**2)
                intensity += absorption
            return np.clip(intensity, 0, None)
        except Exception as e:
            raise RuntimeError(f"Error applying absorbers: {e}")

    def __str__(self):
        """
        Returns a string representation of the Phantom object,
        including its laser and absorber parameters.
        :return: String representing the Phantom object.
        """
        try:
            # Laser attributes
            laser_attrs = {
                'shape': self.laser.shape.name.capitalize(),
                'center': self.laser.center,
                'w0': self.laser.w0,
            }
            laser_attr_lines = [f"  {k}: {v}" for k, v in laser_attrs.items()]
            # Absorber attributes
            absorber_lines = []
            for absorber in self.absorbers:
                absorber_lines.append(f"  - name: \"{absorber.name}\"")
                absorber_lines.append(f"    type: \"{absorber.type}\"")
                absorber_lines.append(f"    center: {absorber.center}")
                absorber_lines.append(f"    radius: {absorber.radius}")
                absorber_lines.append(f"    amplitude: {absorber.amplitude}")
            # Define borders and titles
            border = "+" + "-" * 40 + "+"
            title = f"| Type : {self.__class__.__name__} |"
            laser_title = "| Laser Parameters |"
            absorber_title = "| Absorbers |"
            # Assemble the final result
            result = f"{border}\n{title}\n{border}\n{laser_title}\n{border}\n"
            result += "\n".join(laser_attr_lines)
            result += f"\n{border}\n{absorber_title}\n{border}\n"
            result += "\n".join(absorber_lines)
            result += f"\n{border}"
            return result
        except Exception as e:
            raise RuntimeError(f"Error generating string representation: {e}")

    def show_phantom(self):
        """
        Displays the optical phantom with absorbers.
        """
        try:
            plt.imshow(self.phantom, extent=(self.laser.x[0], self.laser.x[-1] + 1, self.laser.z[-1], self.laser.z[0]), aspect='equal', cmap='hot')
            plt.colorbar(label='Intensity')
            plt.xlabel('X (mm)', fontsize=20)
            plt.ylabel('Z (mm)', fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.title('Optical Phantom with Absorbers')
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error plotting phantom: {e}")

    def show_ROI(self):
        """
        Displays the optical image with regions of interest (ROIs) highlighted.
        This method overlays dashed green circles on the optical image to indicate the regions of interest defined by the absorbers.
        It also calculates and prints the average intensity within these regions.
        """
        X, Z = np.meshgrid(self.laser.x, self.laser.z, indexing='ij')
        # --- Plot the optical image and overlay the regions of interest ---
        _, ax = plt.subplots(figsize=(5, 5))
        # Plot the optical intensity image
        _ = ax.imshow(self.phantom,
                        extent=(self.laser.x[0]*1000, self.laser.x[-1]*1000, self.laser.z[-1]*1000, self.laser.z[0]*1000),
                        aspect='equal', cmap='hot')
        # Overlay dashed green circles for each region of interest
        for region in self.absorbers:
            circle = patches.Circle(
                (region.center[0] * 1000, region.center[1] * 1000),  # Convert to mm
                region.radius * 1000,
                edgecolor='limegreen', facecolor='none', linewidth=0.8, linestyle='--', alpha=0.8
            )
            ax.add_patch(circle)
        ax.set_title("Optical image with regions of interest")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.tick_params(axis='both', which='major')
        plt.show()
        # --- Create a mask for the regions of interest and calculate average intensity ---
        # Initialize an empty mask for the regions
        ROI_mask = np.zeros_like(X, dtype=bool)
        # Iterate over the regions to create and combine masks
        for region in self.absorbers:
            cx, cz = region.center
            r = region.radius
            # Calculate squared distance from the center
            dist_sq = (X - cx)**2 + (Z - cz)**2
            # Create mask for points within the current region
            current_mask = dist_sq <= r**2
            # Combine with the overall region mask
            ROI_mask = np.logical_or(ROI_mask, current_mask)
        # Extract intensity values in the regions of interest and compute the average
        region_intensity_values = self.phantom[ROI_mask]
        average_intensity = np.mean(region_intensity_values)
        print("Average intensity in regions of interest:", average_intensity)
