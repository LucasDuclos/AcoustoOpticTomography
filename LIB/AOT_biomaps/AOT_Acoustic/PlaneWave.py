from .StructuredWave import StructuredWave
from .AcousticEnums import WaveType


class PlaneWave(StructuredWave):
    def __init__(self, angle, space_0 = 0, space_1 = 192, move_head_0_2tail = 0, move_tail_1_2head = 0, **kwargs):
        """
        Initialize the PlaneWave object.

        Args:
            angle_deg (float): Angle in degrees.
            **kwargs: Additional keyword arguments.
        """
        try:
            super().__init__(angle, space_0, space_1, move_head_0_2tail, move_tail_1_2head, **kwargs)
            self.waveType = WaveType.PlaneWave
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

    def getName_field(self):
        """
        Generate the list of system matrix .hdr file paths for this wave.

        Returns:
            str: File path for the system matrix .hdr file.
        """
        try:
            angle_str = self._format_angle()
            return f"field_{self.pattern.activeList}_{angle_str}"
        except Exception as e:
            print(f"Error generating file path: {e}")
            return None
