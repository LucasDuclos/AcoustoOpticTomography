from enum import Enum

class PhantomType(Enum):
    """
    Enum for different types of acoustic phantoms.

    Selection of phantom types:
    """
    PVA = 'PVA'
    """Polyvinyl Alcohol phantom.""" 
    Homogeneous = 'Homogeneous'   
    """Homogeneous phantom (water or uniform medium)."""
    Bubble = 'Bubble'
    """Phantom with bubbles."""