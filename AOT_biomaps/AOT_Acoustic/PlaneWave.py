from AOT_biomaps.AOT_Acoustic.StructuredWave import StructuredWave
from AOT_biomaps.AOT_Acoustic.AcousticEnums import WaveType
from AOT_biomaps.AOT_Acoustic.AcousticTools import format_angle

class PlaneWave(StructuredWave):
    def __init__(self, angle, **kwargs):
        """
        Initialize the PlaneWave object.

        Args:
            angle (float): Angle in degrees.
            **kwargs: Additional keyword arguments.
        """
        try:
            super().__init__(angle=angle, fileName=None, space_0=0, space_1=192, move_head_0_2tail=0, move_tail_1_2head=0, **kwargs)
            self.waveType = WaveType.PlaneWave
            self._check_angle()
        except Exception as e:
            print(f"Error initializing PlaneWave: {e}")
            raise 

    def _check_angle(self):
        """
        Check if the angle is within the valid range.

        Raises:
            ValueError: If the angle is not between -20 and 20 degrees.
        """
        if self.angle < -20 or self.angle > 20:
            raise ValueError("Angle must be between -20 and 20 degrees.")

    def get_name_field(self):
        """
        Generate the list of system matrix .hdr file paths for this wave.

        Returns:
            str: File path for the system matrix .hdr file.
        """
        try:
            angle_str = format_angle(self.angle)
            return f"field_{self.pattern.activeList}_{angle_str}"
        except Exception as e:
            print(f"Error generating file path: {e}")
            return None
