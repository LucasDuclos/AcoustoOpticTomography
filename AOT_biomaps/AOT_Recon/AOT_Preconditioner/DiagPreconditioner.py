from AOT_biomaps.AOT_Recon.AOT_Preconditioner._mainPreconditioner import Preconditioner
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.PreconditionerEnums import PreconditionerType

class DiagPreconditioner(Preconditioner):

    def __init__(self, damping_factor=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.diagonal = None
        self.precondType = PreconditionerType.DIAGONAL
        self.damping_factor = damping_factor  # Damping factor to avoid division by zero
    
    def get_name(self):
        return "Diagonal Preconditioner"

    def build(self):
        xp = self.get_array_module()
        with self.get_device_context():
            self.diagonal = self.SMatrix.compute_hessian_diagonal().astype(xp.float32)
            max_val = xp.max(self.diagonal)
            self.diagonal = self.diagonal + max_val * self.damping_factor

    def apply_inverse(self, x):

        return x / self.diagonal