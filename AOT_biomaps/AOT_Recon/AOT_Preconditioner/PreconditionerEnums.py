
from enum import Enum

class PreconditionerType(Enum):
    """
    Enum for preconditioning types used in iterative reconstruction.
    
    Available preconditioners:
    - NONE: No preconditioning (identity)
    - DIAGONAL: Diagonal preconditioning using inverse of diagonal elements
    """
    NONE = 'NONE'
    """No preconditioning applied."""
    DIAGONAL = 'DIAGONAL'
    """Diagonal preconditioning: M^-1 where M is diagonal matrix with A^T*1."""
