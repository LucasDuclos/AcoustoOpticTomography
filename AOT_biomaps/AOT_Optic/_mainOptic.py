from .Laser import Laser
from .Absorber import Absorber
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            self.absorbers = [Absorber(**a) for a in params.optic['absorbers']] if params.optic['absorbers'] else []
            self.laser = Laser(params)
            self.phantom = self._apply_absorbers()
            self.phantom = np.transpose(self.phantom)
            self.laser.intensity = np.transpose(self.laser.intensity)
            self.find_ROI() 
        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}")
        except Exception as e:
            raise RuntimeError(f"Error initializing Phantom: {e}")

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
            title = f"| Type: {self.__class__.__name__} |"
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
        
    def find_ROI(self):
        """
        Computes binary masks for each ROI and stores them in self.maskList.
        :return: True if pixels are detected in any ROI, False otherwise.
        """
        try:
            X_mm, Z_mm = np.meshgrid(self.laser.x, self.laser.z, indexing='xy')
            assert self.phantom.shape == X_mm.shape, (
                f"Shape mismatch: phantom={self.phantom.shape}, grid={X_mm.shape}"
            )
            self.maskList = []  # Reset the list

            for absorber in self.absorbers:
                center_x_mm = absorber.center[0] * 1000  # Convert to mm
                center_z_mm = absorber.center[1] * 1000  # Convert to mm
                radius_mm = absorber.radius * 1000  # Convert to mm

                # Create mask for this ROI
                mask_i = (X_mm - center_x_mm)**2 + (Z_mm - center_z_mm)**2 <= radius_mm**2
                self.maskList.append(mask_i)

        except Exception as e:
            raise RuntimeError(f"Error in find_ROI: {e}")

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

    def show_phantom(self, withROI=False, figsize=(4,4)):
        """
        Displays the optical phantom, optionally with ROIs and average intensities.

        Parameters:
            withROI (bool): If True, displays ROIs, absorber numbers, and average intensities.
                            If False, displays only the phantom with a simple colorbar.
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                self.phantom,
                extent=(
                    np.min(self.laser.x),
                    np.max(self.laser.x),
                    np.max(self.laser.z),
                    np.min(self.laser.z)
                ),
                aspect='equal',
                cmap='hot'
            )

            if withROI:
                if not self.maskList:
                    self.find_ROI()

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='Intensity')

                # Draw ROIs
                for i, absorber in enumerate(self.absorbers):
                    center_x_mm = absorber.center[0] * 1000  # Convert to mm
                    center_z_mm = absorber.center[1] * 1000  # Convert to mm
                    radius_mm = absorber.radius * 1000  # Convert to mm

                    circle = patches.Circle(
                        (center_x_mm, center_z_mm),
                        radius_mm,
                        edgecolor='limegreen',
                        facecolor='none',
                        linewidth=2
                    )
                    ax.add_patch(circle)
                    ax.text(
                        center_x_mm,
                        center_z_mm - 2,
                        str(i + 1),
                        color='limegreen',
                        ha='center',
                        va='center',
                        fontsize=3*figsize[0],
                        fontweight='bold'
                    )

                # Global mask (union of all ROIs)
                ROI_mask = np.zeros_like(self.phantom, dtype=bool)
                for mask in self.maskList:
                    ROI_mask |= mask

                roi_values = self.phantom[ROI_mask]
                if roi_values.size == 0:
                    print("❌ NO PIXELS IN ROIs! Check positions:")
                    for i, absorber in enumerate(self.absorbers):
                        print(f"  Absorber {i}: center=({absorber.center[0]*1000:.3f}, {absorber.center[1]*1000:.3f}) mm")
                        print(f"          radius={absorber.radius*1000:.3f} mm")
                else:
                    print(f" Average intensity in ROIs: {np.mean(roi_values):.4f}")

                ax.set_xlabel('x (mm)')
                ax.set_ylabel('z (mm)')
                ax.set_title('Optical Phantom \n (with ROIs)')
            else:
                plt.colorbar(im, label='Intensity')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Z (mm)')
                ax.set_title('Optical Phantom')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error in show_phantom: {e}")
