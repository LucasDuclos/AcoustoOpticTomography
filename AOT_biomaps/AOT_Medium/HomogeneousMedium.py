import numpy as np
import warnings
from AOT_biomaps.AOT_Medium._mainMedium import Medium

# Optional kwave imports
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False


class HomogeneousMedium(Medium):
    """
    Class representing a homogeneous medium for acoustic wave propagation.
    Models a uniform medium with constant acoustic properties.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_medium(self):
        """
        Generate a homogeneous medium based on defined parameters.
        """
        dx = self.params.general['dx']
        dz = self.params.general['dz']

        width = self.params.acoustic['medium'].get('width', self.params.general['Xrange'][1] - self.params.general['Xrange'][0])
        height = self.params.acoustic['medium'].get('height', self.params.general['Zrange'][1] - self.params.general['Zrange'][0])
        pva_nx = int(np.round(width / dx))
        pva_nz = int(np.round(height / dz))

        air_margin = 20 if self.params.acoustic['medium']['isAirReflection'] else 0
        Nx, Nz = pva_nx + 2 * air_margin, pva_nz

        x_start = air_margin
        x_end = x_start + pva_nx

        c_map = np.full((Nx, Nz), 343.0 if air_margin else self.params.acoustic['medium']['c0'], dtype=np.float32)
        rho_map = np.full((Nx, Nz), 1.2 if air_margin else self.params.acoustic['medium']['density'], dtype=np.float32)
        alpha_coeff_map = np.zeros((Nx, Nz), dtype=np.float32)
        BonA_map = np.zeros((Nx, Nz), dtype=np.float32)

        c_map[x_start:x_end, :] = self.params.acoustic['medium']['c0']
        rho_map[x_start:x_end, :] = self.params.acoustic['medium']['density']

        is_absorbing = self.params.acoustic['medium']['isAbsorbingMedium']

        if is_absorbing:
            alpha_coeff_map[x_start:x_end, :] = self.params.acoustic['medium']['alpha_coeff']
            alpha_power = self.params.acoustic['medium']['alpha_power']
            alpha_mode = 'no_dispersion'
        else:
            alpha_power = 1.5
            alpha_mode = 'no_absorption'

        BonA_map[x_start:x_end, :] = self.params.acoustic['medium']['BonA']

        c_map = c_map.astype(np.float32)
        rho_map = rho_map.astype(np.float32)
        alpha_coeff_map = alpha_coeff_map.astype(np.float32)
        BonA_map = BonA_map.astype(np.float32)

        self.medium_properties = {
            'sound_speed': c_map,
            'density': rho_map,
            'alpha_coeff': alpha_coeff_map,
            'alpha_power': alpha_power,
            'alpha_mode': alpha_mode,
            'BonA': BonA_map,
            'absorbing': is_absorbing
        }

        if KWAVE_AVAILABLE:
            self.kmedium = kWaveMedium(
                sound_speed=c_map,
                density=rho_map,
                alpha_coeff=alpha_coeff_map,
                alpha_power=alpha_power,
                alpha_mode=alpha_mode,
                BonA=BonA_map,
                absorbing=is_absorbing,
                stokes=False
            )

            self.kgrid = kWaveGrid([Nx, Nz], [dx, dx])
            dt = 1/(self.params.acoustic['f_AQ'])
            self.kgrid.setTime(self.Nt_reshaped, dt)
        else:
            self.kmedium = None
            self.kgrid = None
            warnings.warn("kWave is not available. Medium properties stored in medium_properties dictionary.", UserWarning)

        self.factorX = int(np.ceil(self.params.general['dx'] / dx))
        self.factorZ = self.factorX
        if KWAVE_AVAILABLE and self.kgrid is not None:
            self.factorT = int(np.ceil((1/self.kgrid.dt) / (self.params.acoustic['f_saving'])))
        else:
            self.factorT = 1
        self.c_mean = np.mean(c_map[:, 0])
        self.Nx_reshaped = Nx
        self.Nz_reshaped = Nz
        self.dx_reshaped = dx
