from enum import Enum

class OpticFieldType(Enum):
    """
    Enumeration of available optic field types.

    Selection of optic field types:
    - GAUSSIAN: A Gaussian optic field type.
    - UNIFORM: A uniform optic field type.
    - SPHERICAL: A spherical optic field type.
    """
    GAUSSIAN = "Gaussian"
    """A Gaussian optic field type."""
    UNIFORM = "Uniform"
    """A uniform optic field type."""
    SPHERICAL = "Spherical"
    """A spherical optic field type."""
