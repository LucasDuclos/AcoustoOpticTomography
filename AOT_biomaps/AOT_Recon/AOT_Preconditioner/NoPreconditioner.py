from AOT_biomaps.AOT_Recon.AOT_Preconditioner._mainPreconditioner import Preconditioner
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.PreconditionerEnums import PreconditionerType

class NoPreconditioner(Preconditioner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.precondType = PreconditionerType.NONE
    
    def get_name(self):
        return "No Preconditioner"

    def build(self):
        pass

    def apply_inverse(self, x):
        return x